# Project Architecture

本项目基于 GraphDECO/Inria 的 3D Gaussian Splatting 代码，当前主要用于 MatrixCity 等大场景重建实验。核心流程是：读取 COLMAP/Blender 数据，构建 `Scene` 和 `GaussianModel`，通过 differentiable rasterizer 渲染当前视角，计算图像/深度损失，反向优化 Gaussian 参数，并在训练中执行 densify/prune。

## Main Entry Points

- `train.py`
  - 训练入口。
  - 创建 `Scene`、`GaussianModel`、优化器和 SwanLab logger。
  - 每轮随机采样一个训练相机，调用 `gaussian_renderer.render()`，计算 RGB loss、SSIM loss 和可选 depth loss。
  - 负责 densification、checkpoint、PLY 保存、训练/验证日志。
  - 已加入更多 SwanLab 观测指标：点云数量、可见点数量、显存、学习率、densify 前后点数、OOM 信息。

- `render.py`
  - 渲染入口。
  - 从 `output/<exp>/point_cloud/iteration_x/point_cloud.ply` 加载训练好的 Gaussian。
  - 对 train/test split 渲染结果和 GT，保存到 `train/ours_x/` 或 `test/ours_x/`。

- `metrics.py`
  - 指标评估入口。
  - 读取 render 输出的 `renders/` 和 `gt/`，计算 SSIM、PSNR、LPIPS。

- `convert.py`
  - 数据转换入口，主要服务 COLMAP 数据准备。

## Core Modules

- `arguments/`
  - 统一定义命令行参数。
  - `ModelParams`：数据路径、图像目录、分辨率、数据设备、相机加载 worker 数等。
  - `PipelineParams`：renderer 相关开关、SwanLab 配置。
  - `OptimizationParams`：迭代数、学习率、densify/prune、loss 权重等训练超参。

- `scene/`
  - 场景、相机、数据读取和 Gaussian 模型。
  - `scene/__init__.py`：`Scene` 类，识别 COLMAP/Blender 数据，加载 train/test cameras，创建或加载 Gaussian。
  - `scene/cameras.py`：`Camera` 类，保存相机内外参、原图 tensor、alpha mask、depth map、投影矩阵。
  - `scene/dataset_readers.py`：读取 COLMAP/Blender 数据，生成 `SceneInfo`。
  - `scene/gaussian_model.py`：核心 Gaussian 参数和优化器状态，负责 save/load、densify、prune、opacity reset。

- `gaussian_renderer/`
  - Differentiable Gaussian rasterization 封装。
  - `render()` 输入 `Camera`、`GaussianModel`、pipeline 参数和背景色，输出渲染图、viewspace points、visibility filter、radii、depth。
  - 底层依赖 `submodules/diff-gaussian-rasterization`。

- `utils/`
  - 通用工具。
  - `camera_utils.py`：`loadCam()` 和 `cameraList_from_camInfos()`，支持 `--camera_load_workers` 多线程加载相机图像。
  - `loss_utils.py`：L1、SSIM 等 loss。
  - `image_utils.py`：PSNR 等图像指标。
  - `graphics_utils.py`：相机矩阵、FOV、投影相关工具。
  - `read_write_model.py`、`colmap_loader.py`：COLMAP 模型读取。

- `scripts/`
  - 实验脚本。
  - `matrix_city_aerial_block_all.sh`：MatrixCity aerial block_all baseline/续训脚本。
  - `matrix_city_aerial_block_all_lowmem.sh`：低显存版本，提前停止/降低 densification。
  - `matrix_city_aerial_block_all_test_load.sh`：相机加载测试脚本。
  - 其他 `matrix_city_*` 脚本用于 street/depth/single image 等实验。

- `submodules/`
  - 外部 CUDA/C++ 扩展和第三方模块。
  - 不应提交生成物或本地修改，通常通过 `.gitmodules` 管理。

- `output/`
  - 训练输出目录。
  - 包含 `cfg_args`、checkpoint、PLY、render 结果、metrics 结果、SwanLab 本地日志。
  - 不提交到 Git。

## Training Data Flow

1. `train.py` 解析参数，初始化 SwanLab logger。
2. `Scene(dataset, gaussians)` 根据 `source_path` 识别数据类型：
   - 有 `sparse/`：按 COLMAP 数据读取。
   - 有 `transforms_train.json`：按 Blender 数据读取。
3. `cameraList_from_camInfos()` 构建 train/test camera list。
   - 默认单线程加载。
   - `--camera_load_workers N` 可并行加载图像和构造 Camera。
   - 大图会按 `--resolution` 自动或显式缩放。
4. `GaussianModel.create_from_pcd()` 从初始点云创建 Gaussian 参数。
5. 每轮训练：
   - 随机选一个 train camera。
   - `gaussian_renderer.render()` 渲染当前视角。
   - 和 `viewpoint_cam.original_image` 计算 RGB/SSIM loss。
   - 如果有可靠 depth，则加入 depth regularization。
   - `loss.backward()` 后更新 Gaussian 参数。
6. densification 阶段：
   - 统计可见 Gaussian 的 screen-space gradient。
   - 按阈值 clone/split 新 Gaussian。
   - prune 低 opacity 或过大 Gaussian。
   - 当前代码会记录 densify 前后点数和 OOM 信息。
7. 到指定 iteration 保存 PLY 和 checkpoint。

## Output Layout

典型输出结构：

```text
output/<experiment>/
  cfg_args
  cameras.json
  input.ply
  chkpnt7000.pth
  point_cloud/
    iteration_7000/
      point_cloud.ply
  train/
    ours_30000/
      renders/
      gt/
  test/
    ours_30000/
      renders/
      gt/
  results.json
  per_view.json
  swanlog/
```

## Large Scene Reconstruction Direction

目标：基于多源数据的大场景重建。

1. 分块训练
   - Coarse GS + 递归划分。
   - 大场景不要直接把单个超大 `block_all` 模型硬塞进单 GPU。
   - 更推荐按空间/相机覆盖划分 block，多 GPU 并行跑多个 block。

2. Depth、光流、语义约束
   - 当前已有 depth regularization 接口。
   - 后续可加入光流约束、语义一致性约束、post-training refinement。

3. 模型压缩
   - training stage 或 after-training stage 均可做。
   - 候选方向：LOD、Gaussian pruning、quantization、某种 learned compression。

## Performance And Memory Notes

- 训练 OOM 的主要来源通常是 Gaussian 数量增长和 Adam 状态，而不是单张 GT image 常驻。
- 默认 SH degree 为 3，每个 Gaussian 仅参数约 236 bytes；训练时加上 grad 和 Adam 状态，核心训练开销约 4 倍。
- densify 时 `torch.cat` 会临时分配连续大张量，可能在 reserved memory 很高时触发 OOM。
- 大场景常用缓解参数：

```bash
--densify_until_iter 9000
--densification_interval 300
--densify_grad_threshold 0.001
--test_iterations -1
--data_device cpu
--camera_load_workers 8
```

- `--data_device cpu` 可以减少相机 alpha/depth 常驻 GPU 显存，但训练时当前视角仍会搬到 GPU。
- `--camera_load_workers` 能缩短启动加载时间，但 worker 过多可能造成磁盘争用或 CPU 内存峰值增加。

## Development Guidelines

- 保持训练主流程在 `train.py`，Gaussian 参数和 densify/prune 逻辑在 `scene/gaussian_model.py`。
- 不要把实验输出、checkpoint、PLY、`output/`、`submodules/` 生成物提交到 Git。
- 新增实验脚本优先放在 `scripts/`，并支持通过环境变量覆盖 GPU、checkpoint、输出目录和关键超参。
- 修改相机加载逻辑时，同时检查 `train.py`、`render.py`、`metrics.py` 的数据访问路径。
- 修改 SwanLab 日志时，避免日志失败遮蔽原始训练异常，尤其是 CUDA OOM。
