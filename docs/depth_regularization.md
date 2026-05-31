# Depth Regularization Run Script

本文档记录如何使用 `scripts/prepare_depth_regularization.sh` 为 COLMAP/SfM 数据集生成 3DGS depth regularization 所需的数据。

该脚本会完成两件事：

1. 使用 Depth Anything V2 为每张输入图生成单目深度先验。
2. 使用 COLMAP sparse points 生成 `sparse/0/depth_params.json`，把单目反深度对齐到 SfM 坐标尺度。

## Input Dataset

数据集需要满足 COLMAP 格式：

```text
DATASET_ROOT/
  images/
    00001.jpg
    00002.jpg
  sparse/0/
    cameras.bin
    images.bin
    points3D.bin
```

也支持 COLMAP txt 格式：

```text
DATASET_ROOT/sparse/0/
  cameras.txt
  images.txt
  points3D.txt
```

默认图像目录名是 `images`。如果图像目录不是 `images`，可以作为第二个参数传入。

## Preflight Check

先只检查路径、COLMAP 格式和 Depth Anything checkpoint，不生成文件：

```bash
cd /lfs1/users/spsong/Code/project/gaussian-splatting-dev

CHECK_ONLY=1 bash scripts/prepare_depth_regularization.sh \
  /lfs3/users/spsong/dataset/tandt/train
```

预期会看到类似输出：

```text
Preparing depth regularization data
  dataset:      /lfs3/users/spsong/dataset/tandt/train
  images:       /lfs3/users/spsong/dataset/tandt/train/images
  depths:       /lfs3/users/spsong/dataset/tandt/train/depths_any
  colmap type:  bin
  DA root:      /lfs1/users/spsong/Code/Depth-Anything-V2
  encoder:      vitl
CHECK_ONLY=1, preflight checks passed; no files were generated.
```

## Generate Depth Priors

正式生成 depth PNG 和对齐参数：

```bash
cd /lfs1/users/spsong/Code/project/gaussian-splatting-dev

bash scripts/prepare_depth_regularization.sh \
  /lfs3/users/spsong/dataset/tandt/train
```

脚本默认使用：

```text
Depth Anything V2 repo: /lfs1/users/spsong/Code/Depth-Anything-V2
Depth checkpoint:       checkpoints/depth_anything_v2_vitl.pth
Depth output dir:       DATASET_ROOT/depths_any
COLMAP model type:      auto
```

生成完成后，数据集中会新增：

```text
DATASET_ROOT/
  depths_any/
    00001.png
    00002.png
  sparse/0/
    depth_params.json
```

其中 `depths_any/*.png` 是 16-bit 单通道 depth prior，`depth_params.json` 保存每张图的 `scale` 和 `offset`。

## Use Another Dataset

之后换数据集只需要替换最后一个路径：

```bash
bash scripts/prepare_depth_regularization.sh /path/to/another/colmap_scene
```

如果图像目录不是 `images`，例如是 `rgb`：

```bash
bash scripts/prepare_depth_regularization.sh /path/to/another/colmap_scene rgb
```

训练时也要保持同样的 image directory：

```bash
python train.py \
  -s /path/to/another/colmap_scene \
  -i rgb \
  -d depths_any \
  -m output/another_scene_depth \
  --data_device cpu
```

## Environment Overrides

常用覆盖项：

```bash
DA_ROOT=/lfs1/users/spsong/Code/Depth-Anything-V2 \
DA_PYTHON=/lfs1/users/spsong/Code/Depth-Anything-V2/.venv/bin/python \
GS_PYTHON=python \
DEPTH_DIR_NAME=depths_any \
ENCODER=vitl \
INPUT_SIZE=518 \
bash scripts/prepare_depth_regularization.sh /path/to/colmap_scene
```

其他开关：

```bash
FORCE_DEPTH=1  # 重新生成已存在的 depth PNG
RUN_DEPTH=0    # 跳过 Depth Anything，只重新生成 depth_params.json
RUN_ALIGN=0    # 只生成 depth PNG，不生成 depth_params.json
MODEL_TYPE=txt # 手动指定 COLMAP txt 格式
```

## Train With Depth Regularization

准备完成后，直接在当前 3DGS repo 中训练：

```bash
python train.py \
  -s /lfs3/users/spsong/dataset/tandt/train \
  -i images \
  -d depths_any \
  -m output/tandt_train_depth \
  --data_device cpu
```

`-d depths_any` 是打开 depth regularization 的关键参数。训练时会自动读取：

```text
DATASET_ROOT/depths_any/<image_basename>.png
DATASET_ROOT/sparse/0/depth_params.json
```

如果深度约束过强导致质量下降，可以降低权重：

```bash
python train.py \
  -s /lfs3/users/spsong/dataset/tandt/train \
  -i images \
  -d depths_any \
  -m output/tandt_train_depth_w01 \
  --data_device cpu \
  --depth_l1_weight_init 0.1 \
  --depth_l1_weight_final 0.001
```

## Use With Block Training

分块训练可以复用同一套 depth regularization 逻辑。`Scene` 会先按 `dataset.depths`
读取带 `depth_path` 和 `depth_params` 的 `CameraInfo`，之后 `apply_partition_to_scene_info`
只过滤相机列表，不会丢掉这些字段，所以每个 block 训练时仍会读取对应 depth map。

配置方式是在 recursive/block YAML 中设置：

```yaml
dataset:
  depths: depths_any

optimization:
  depth_reg_mask_mode: block_projection
  depth_reg_mask_bbox_mode: expanded
  depth_reg_mask_dilate_px: 16
  depth_reg_mask_min_pixels: 2048
  depth_reg_mask_max_points: 100000
```

然后继续使用现有分块流程：

```bash
python partition.py --config config/mc_aerial_recursive_d3_c8.yaml
CONFIG=config/mc_aerial_recursive_d3_c8.yaml bash scripts/matrix_city_aerial_train_blocks.sh
```

如果使用无 YAML 的旧脚本入口，也可以直接传：

```bash
DEPTHS=depths_any bash scripts/matrix_city_aerial_train_blocks.sh
```

## Quick Validation

手动检查输出：

```bash
ls /lfs3/users/spsong/dataset/tandt/train/depths_any | head
ls /lfs3/users/spsong/dataset/tandt/train/sparse/0/depth_params.json
```

文件名需要和输入图像 basename 对齐：

```text
images/00001.jpg
depths_any/00001.png
```
