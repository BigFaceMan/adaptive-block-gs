#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
import shutil
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import camera_to_JSON
from utils.partition_utils import (
    all_camera_infos,
    apply_partition_to_scene_info,
    mark_camera_infos_as_test,
    points_in_partition_bbox,
    write_basic_point_cloud_ply,
    write_block_training_metadata,
)


def load_scene_info_from_path(source_path, images, depths, normals, eval_split, train_test_exp, white_background):
    if os.path.exists(os.path.join(source_path, "sparse")):
        return sceneLoadTypeCallbacks["Colmap"](source_path, images, depths, normals, eval_split, train_test_exp)
    if os.path.exists(os.path.join(source_path, "transforms_train.json")):
        print("Found transforms_train.json file, assuming Blender data set!")
        return sceneLoadTypeCallbacks["Blender"](source_path, white_background, depths, normals, eval_split)
    raise RuntimeError(f"Could not recognize scene type: {source_path}")


def load_external_test_scene_info(args):
    test_source_path = getattr(args, "test_source_path", "")
    if not test_source_path:
        return None
    test_images = getattr(args, "test_images", "") or args.images
    test_depths = getattr(args, "test_depths", "")
    test_normals = getattr(args, "test_normals", "")
    return load_scene_info_from_path(
        test_source_path,
        test_images,
        test_depths,
        test_normals,
        False,
        getattr(args, "train_test_exp", False),
        getattr(args, "white_background", False),
    )

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.block_metadata = None

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        scene_info = load_scene_info_from_path(
            args.source_path,
            args.images,
            args.depths,
            getattr(args, "normals", ""),
            args.eval,
            args.train_test_exp,
            args.white_background,
        )
        test_scene_info = load_external_test_scene_info(args)

        partition_path = getattr(args, "partition_path", "")
        block_id = getattr(args, "block_id", "")
        partition_init_mode = getattr(args, "partition_init_mode", "cropped")
        if partition_path or block_id:
            if not partition_path or not block_id:
                raise ValueError("--partition_path and --block_id must be set together")
            if partition_init_mode not in {"cropped", "coarse"}:
                raise ValueError("--partition_init_mode must be 'cropped' or 'coarse'")
            scene_info, self.block_metadata = apply_partition_to_scene_info(
                scene_info,
                partition_path=partition_path,
                block_id=block_id,
                bbox_mode=getattr(args, "partition_bbox_mode", "expanded"),
                crop_initial_points=partition_init_mode == "cropped",
                load_test_cameras=getattr(args, "partition_load_test_cameras", False),
                test_scene_info=test_scene_info,
            )
            print(
                "[Partition] "
                f"block_id={block_id}, "
                f"bbox_mode={self.block_metadata['_partition_bbox_mode']}, "
                f"init_mode={self.block_metadata['_partition_init_mode']}, "
                f"train_cameras={self.block_metadata['_num_train_cameras_loaded']}, "
                f"test_cameras={self.block_metadata['_num_test_cameras_loaded']}, "
                f"initial_points={self.block_metadata['_num_initial_points_loaded']}"
            )
            write_block_training_metadata(self.model_path, self.block_metadata)
        elif test_scene_info is not None:
            test_cameras = mark_camera_infos_as_test(all_camera_infos(test_scene_info))
            scene_info = scene_info._replace(test_cameras=test_cameras)
            print(f"[External Test] loaded test_cameras={len(test_cameras)} from {args.test_source_path}")

        if not self.loaded_iter:
            input_ply_path = os.path.join(self.model_path, "input.ply")
            if self.block_metadata is not None:
                if partition_init_mode == "coarse":
                    init_ply_path = self.block_metadata.get("_partition_coarse_model", "")
                    if not init_ply_path:
                        raise ValueError("partition_tree.json has no coarse_model for --partition_init_mode coarse")
                    if os.path.abspath(init_ply_path) != os.path.abspath(input_ply_path):
                        shutil.copyfile(init_ply_path, input_ply_path)
                else:
                    write_basic_point_cloud_ply(input_ply_path, scene_info.point_cloud)
            else:
                shutil.copyfile(scene_info.ply_path, input_ply_path)
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.is_nerf_synthetic = scene_info.is_nerf_synthetic
        self.train_camera_infos = scene_info.train_cameras
        self.test_camera_infos = scene_info.test_cameras

        print(
            "[Scene] Using CameraInfo lists. "
            "Images are loaded by GSCameraDataset/CameraDataLoader during train/render."
        )

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
        elif self.block_metadata is not None and partition_init_mode == "coarse":
            init_ply_path = self.block_metadata.get("_partition_coarse_model", "")
            if not init_ply_path:
                raise ValueError("partition_tree.json has no coarse_model for --partition_init_mode coarse")
            print(f"[Partition] Loading full coarse Gaussian init from {init_ply_path}")
            self.gaussians.load_ply(init_ply_path, False)
            self.gaussians.prepare_loaded_ply_for_training(scene_info.train_cameras, self.cameras_extent)
            self.block_metadata["_num_initial_gaussians_loaded"] = int(self.gaussians.get_xyz.shape[0])
            write_block_training_metadata(self.model_path, self.block_metadata)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        save_mask = None
        if self.block_metadata is not None:
            xyz = self.gaussians.get_xyz.detach().cpu().numpy()
            save_mask = points_in_partition_bbox(xyz, self.block_metadata, "core_bbox")
            kept = int(save_mask.sum())
            if kept == 0:
                raise RuntimeError(
                    f"Block {self.block_metadata.get('id', '<unknown>')} core bbox selected zero Gaussians at save"
                )
            self.block_metadata["_last_saved_iteration"] = int(iteration)
            self.block_metadata["_last_saved_gaussians"] = kept
            self.block_metadata["_last_discarded_gaussians"] = int(save_mask.shape[0] - kept)
            write_block_training_metadata(self.model_path, self.block_metadata)
            print(
                "[Partition] "
                f"saving core Gaussians for {self.block_metadata.get('id', '<unknown>')}: "
                f"{kept}/{save_mask.shape[0]}"
            )

        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"), mask=save_mask)
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        raise RuntimeError("Scene no longer preloads Camera objects; use getTrainCameraInfos() with GSCameraDataset")

    def getTestCameras(self, scale=1.0):
        raise RuntimeError("Scene no longer preloads Camera objects; use getTestCameraInfos() with GSCameraDataset")

    def getTrainCameraInfos(self):
        return self.train_camera_infos

    def getTestCameraInfos(self):
        return self.test_camera_infos
