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

import numpy as np
from utils.graphics_utils import fov2focal

WARNED = False

def get_camera_resolution(args, orig_w, orig_h, resolution_scale):
    if args.resolution in [1, 2, 4, 8]:
        return round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))

    if args.resolution == -1:
        if orig_w > 1600:
            global WARNED
            if not WARNED:
                print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                    "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                WARNED = True
            global_down = orig_w / 1600
        else:
            global_down = 1
    else:
        global_down = orig_w / args.resolution

    scale = float(global_down) * float(resolution_scale)
    return int(orig_w / scale), int(orig_h / scale)

def camera_to_JSON(id, camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
