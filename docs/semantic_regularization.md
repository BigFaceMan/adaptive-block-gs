# Semantic Regularization Plan For Large-Scene 3DGS

本文档记录在当前 Inria/GraphDECO 3D Gaussian Splatting 代码中加入语义图约束的方案。目标场景是 MatrixCity 等大场景重建：语义图主要用于减少 floaters、约束天空/道路/建筑等区域、屏蔽动态物体，并在需要时学习可渲染的 3D semantic field。

## Current State

当前项目已有 depth regularization：

- `gaussian_renderer.render()` 返回 `render_pkg["depth"]`。
- CUDA rasterizer 底层输出的是 inverse depth，而不是普通 metric depth。
- `train.py` 中已将 `render_pkg["depth"]` 和 `viewpoint_cam.invdepthmap` 做 L1 loss。

语义图还没有接入。当前 `submodules/diff-gaussian-rasterization` 的颜色输出通道固定为：

```cpp
#define NUM_CHANNELS 3
```

因此它天然支持 RGB 和已有 inverse-depth 辅助输出，但不支持直接渲染任意维语义 logits/probabilities。

## What Gsplat Does

`/lfs1/users/spsong/Code/project/gsplat` 的实现方式可以作为参考。它没有为 semantic 单独写一套 rasterizer，而是把语义当作任意维 feature channel：

```text
per-Gaussian semantic feature: [N, E]
RGB feature:                   [N, 3]
combined feature:              [N, 3 + E]
```

然后统一走 alpha compositing：

```text
feature_pixel = sum_i alpha_i * T_i * feature_i + T_final * background_feature
```

`gsplat` 中对应接口是：

```python
render_colors, render_alphas, meta = rasterization(
    means,
    quats,
    scales,
    opacities,
    colors,
    viewmats,
    Ks,
    width,
    height,
    render_mode="RGB+ED",
    extra_signals=semantic_features,
)

render_semantics = meta["render_extra_signals"]
```

它的核心工程点：

1. `extra_signals` 表示 RGB 之外的任意维信号。
2. Python 端把 `colors`、`extra_signals`、可选 depth 拼接成 `proj_features`。
3. CUDA 端用 `CDIM` 模板渲染多通道 feature。
4. 支持的通道数包括 `1,2,3,4,5,8,9,16,17,24,32,...`。
5. 如果实际通道数不在支持列表中，Python 端 pad 到支持通道数。
6. 如果通道数超过 `channel_chunk`，默认 32，则分块多次 rasterize，再 concat。

这和我们计划的“任意维 feature rasterizer + channel chunk”是一致的。

## Design Goals

语义约束分两层：

1. Low-risk semantic supervision
   - 不改 CUDA。
   - 用语义图做 loss mask、区域加权、sky alpha penalty、动态物体屏蔽。
   - 适合先验证大场景收益。

2. Full semantic feature rendering
   - 修改 rasterizer，支持渲染任意维 feature。
   - 每个 Gaussian 携带 semantic logits/probability/embedding。
   - 渲染出 2D semantic map，与语义图做 CE/KL/MSE loss。
   - 适合最终做 semantic 3DGS。

建议先做第 1 层，再做第 2 层。原因是大场景中语义参数和 Adam 状态会显著增加显存，如果语义图本身噪声较大，直接强监督 semantic logits 可能拉坏几何。

## Option A: No-CUDA Semantic Hacks

### Semantic Loss Mask

用语义图控制 RGB/depth loss 的区域权重：

```python
sem = viewpoint_cam.semantic_map.cuda()  # [H, W], int class ids

sky_mask = sem == SKY
road_mask = sem == ROAD
building_mask = sem == BUILDING
dynamic_mask = (sem == CAR) | (sem == PERSON)

rgb_mask = (~dynamic_mask).float()[None]
depth_mask = ((road_mask | building_mask) & valid_depth).float()[None]

loss_rgb = (torch.abs(image - gt_image) * rgb_mask).sum() / rgb_mask.sum().clamp_min(1.0)
loss_depth = (torch.abs(inv_depth - mono_invdepth) * depth_mask).sum() / depth_mask.sum().clamp_min(1.0)
```

用途：

- 屏蔽动态物体，减少 transient artifacts。
- 对道路、建筑等稳定区域加强 depth regularization。
- 对植被、水面等不稳定区域降低监督权重。

### Sky Alpha Penalty

天空区域通常不应出现高 opacity Gaussian。可在 sky mask 上惩罚 rendered alpha。

当前 Inria rasterizer没有直接返回 alpha。短期 hack：

```python
ones = torch.ones_like(gaussians.get_xyz)
alpha_like = render(
    viewpoint_cam,
    gaussians,
    pipe,
    torch.zeros(3, device="cuda"),
    override_color=ones,
)["render"][:1]

loss_sky = alpha_like[sky_mask[None]].mean()
```

这不是严格 alpha，因为当前 RGB render 会叠加三通道颜色，但 `override_color=1` 时可以近似得到 accumulated opacity。更严谨的 alpha 输出需要改 CUDA。

### RGB Palette Semantic Hack

把每个类别映射成 3D RGB palette，然后渲染 semantic color：

```python
sem_prob = torch.softmax(gaussians.get_semantic_logits, dim=-1)  # [N, K]
sem_color = sem_prob @ palette                                  # [N, 3]

sem_render = render(
    viewpoint_cam,
    gaussians,
    pipe,
    torch.zeros(3, device="cuda"),
    override_color=sem_color,
)["render"]

gt_sem_color = palette[sem_gt].permute(2, 0, 1)
loss_sem = torch.abs(sem_render - gt_sem_color).mean()
```

优点：

- 不改 CUDA。
- 只增加一次额外 render。

缺点：

- 只能通过 3D color 间接表达 K 类语义。
- 类别多时 palette 容易混淆。
- 更适合粗类别，例如 `sky/road/building/vegetation/water/dynamic/other`。

### K-Class Chunk Hack

当前 rasterizer 固定 3 通道，可以每次渲染 3 个类别概率，循环 `ceil(K / 3)` 次：

```python
sem_prob = torch.softmax(gaussians.get_semantic_logits, dim=-1)  # [N, K]
chunks = []

for start in range(0, K, 3):
    feat = torch.zeros((sem_prob.shape[0], 3), device="cuda")
    width = min(3, K - start)
    feat[:, :width] = sem_prob[:, start:start + width]

    pred = render(
        viewpoint_cam,
        gaussians,
        pipe,
        torch.zeros(3, device="cuda"),
        override_color=feat,
    )["render"][:width]
    chunks.append(pred)

sem_render = torch.cat(chunks, dim=0)  # [K, H, W]
loss_sem = torch.nn.functional.nll_loss(
    torch.log(sem_render.clamp_min(1e-6)).unsqueeze(0),
    sem_gt.long().unsqueeze(0),
    ignore_index=255,
)
```

优点：

- 不改 CUDA。
- 能直接得到 K 维语义概率图。

缺点：

- `K=19` 需要 7 次额外 render。
- 大场景训练很慢。
- 每次 render 都重复 preprocess/sort/raster。

适合作为功能验证，不适合作为最终训练方案。

## Option B: Full CUDA Feature Rasterizer

长期方案参考 `gsplat`：把当前 RGB-only rasterizer 扩展为任意维 feature rasterizer。

### API

新增 Python API：

```python
render_features(
    viewpoint_camera,
    gaussians,
    pipe,
    features,       # [N, F]
    bg_features,    # [F]
    channel_chunk=32,
)
```

返回：

```python
{
    "features": feature_image,  # [F, H, W]
    "alpha": alpha_image,       # [1, H, W]
    "radii": radii,
    "visibility_filter": visibility_filter,
}
```

其中 `F` 可以是：

- 语义类别数 `K`；
- 低维 semantic embedding；
- RGB + semantic + other signals 的拼接通道。

### Forward Formula

对每个像素：

```text
T = 1
out[c] = 0

for Gaussian i from front to back:
    alpha_i = opacity_i * gaussian_weight_i(pixel)
    out[c] += T * alpha_i * feature_i[c]
    T *= (1 - alpha_i)

out[c] += T * bg_feature[c]
alpha = 1 - T
```

语义概率建议使用：

```python
features = torch.softmax(semantic_logits, dim=-1)
```

如果直接渲染 logits，alpha compositing 后的 logits 不一定有良好的概率意义。第一版更推荐渲染 probability。

### Channel Dispatch

不要写死 `NUM_CHANNELS=K`。建议模仿 `gsplat`：

1. 支持一组模板通道数：

```text
1, 2, 3, 4, 5, 8, 9, 16, 17, 24, 32, 33, 64
```

第一版做到 64 就足够覆盖常见语义类别。

2. Python 端按实际通道数 pad：

```python
supported = [1, 2, 3, 4, 5, 8, 9, 16, 17, 24, 32, 33, 64]
padded_channels = min(c for c in supported if c >= F)
features = pad(features, padded_channels)
bg_features = pad(bg_features, padded_channels)
```

3. CUDA C++ 端按 `channels` dispatch：

```cpp
switch (channels) {
    case 1: launch_feature_rasterizer<1>(...); break;
    case 2: launch_feature_rasterizer<2>(...); break;
    case 3: launch_feature_rasterizer<3>(...); break;
    case 4: launch_feature_rasterizer<4>(...); break;
    case 8: launch_feature_rasterizer<8>(...); break;
    case 16: launch_feature_rasterizer<16>(...); break;
    case 32: launch_feature_rasterizer<32>(...); break;
    case 64: launch_feature_rasterizer<64>(...); break;
    default: throw;
}
```

4. 如果 `F > channel_chunk`，Python 分块：

```python
renders = []
for start in range(0, F, channel_chunk):
    render_chunk = rasterize_features(features[:, start:start + channel_chunk])
    renders.append(render_chunk)
feature_image = torch.cat(renders, dim=0)
```

### Files To Modify

CUDA/C++:

- `submodules/diff-gaussian-rasterization/cuda_rasterizer/config.h`
  - 保留 `NUM_CHANNELS=3` 给原 RGB。
  - 新增 feature channel dispatch 列表或 macro。

- `submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.h`
  - 新增 `render_features(...)` 声明。

- `submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu`
  - 新增 `renderFeaturesCUDA<CDIM>()`。
  - 复用当前 `ranges`、`point_list`、`points_xy_image`、`conic_opacity`。
  - 输出 `[CDIM, H, W]` feature image 和 `[1, H, W]` alpha。

- `submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.h`
  - 新增 `render_features_backward(...)` 声明。

- `submodules/diff-gaussian-rasterization/cuda_rasterizer/backward.cu`
  - 新增 `renderFeaturesBackwardCUDA<CDIM>()`。
  - 计算 `dL_dfeatures`、`dL_dopacity`、`dL_dmean2D`、`dL_dconic`。

- `submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer.h`
  - 新增 `Rasterizer::forward_features` 和 `Rasterizer::backward_features`。

- `submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu`
  - 复用当前 preprocess、duplicateWithKeys、sort、identifyTileRanges。
  - 在 render 阶段调用 feature render kernel。

- `submodules/diff-gaussian-rasterization/rasterize_points.cu`
  - 新增 `RasterizeGaussianFeaturesCUDA` 和 backward。

- `submodules/diff-gaussian-rasterization/ext.cpp`
  - 绑定新接口。

Python:

- `submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py`
  - 新增 `_RasterizeGaussianFeatures(torch.autograd.Function)`。
  - 新增 `GaussianFeatureRasterizer`。

- `gaussian_renderer/__init__.py`
  - 新增 `render_features(...)` wrapper。

- `scene/gaussian_model.py`
  - 新增语义参数，例如 `_semantic_logits`。
  - `training_setup()` 加 optimizer param group。
  - densify clone/split 时复制 semantic logits。
  - prune 时同步 prune semantic logits。
  - save/load semantic logits。

### Backward Notes

当前 RGB backward 已经有类似逻辑：

```text
dL_dcolor += alpha * T * dL_dpixel
dL_dalpha += color-dependent compositing gradient
```

任意维 feature backward 只需要把 RGB 的 3 通道循环扩展为 `CDIM` 通道循环。

需要注意：

- 不同 channel chunk 都会对同一个 Gaussian 的 opacity、mean2D、conic 产生梯度，因此要 `atomicAdd`。
- alpha 输出如果参与 loss，也会对 opacity/geometry 产生额外梯度。
- 如果 semantic features 只想训练语义参数，不想强推几何，可以第一版 detach geometry gradients：

```python
semantic_loss_weight_for_geometry = 0
```

工程实现上可以先只让 semantic loss 回传到 `_semantic_logits`，暂时不回传到 xyz/scale/opacity；确认稳定后再打开几何梯度。

## Semantic Model Parameters

### Direct Class Logits

每个 Gaussian 保存一个 K 维语义 logits：

```python
self._semantic_logits = nn.Parameter(torch.zeros((N, K), device="cuda"))
```

训练时：

```python
semantic_prob = torch.softmax(gaussians.get_semantic_logits, dim=-1)
semantic_render = render_features(..., semantic_prob, bg_semantic)["features"]
```

优点：

- 简单直接。
- 输出就是 K 类概率图。

缺点：

- 参数量大。
- Adam 状态开销大。
- K 大时显存压力明显。

### Low-Dimensional Embedding

每个 Gaussian 保存低维 embedding：

```python
self._semantic_embedding = nn.Parameter(torch.zeros((N, E), device="cuda"))
```

然后用一个小 MLP 或 linear head 得到 class logits：

```python
semantic_logits = semantic_head(rendered_embedding)
```

优点：

- 更省显存。
- 适合大场景和类别较多的语义标签。

缺点：

- 实现复杂一些。
- 语义边界可能不如直接 K 维概率稳定。

第一版建议用 direct class logits，但只使用粗类别。

## Semantic Data Format

建议新增语义图目录：

```text
<scene>/
  images/
    000001.jpg
  semantics/
    000001.png
  sparse/0/
    cameras.bin
    images.bin
    points3D.bin
```

语义图要求：

- 文件名 stem 与 image 对齐。
- 单通道 `uint8` 或 `uint16` class id。
- 使用 `ignore_index=255` 表示无效区域。
- resize 必须使用 nearest neighbor，不能双线性插值。

加载逻辑可参考 depth：

- `arguments/ModelParams` 增加 `--semantics`。
- `scene/dataset_readers.py` 给 `CameraInfo` 增加 `semantic_path`。
- `scene/datasets.py` 读取 semantic PNG 并保存到 camera sample。
- `train.py` 中取 `viewpoint_cam.semantic_map` 参与 loss。

## Loss Design

### Semantic Probability Loss

如果 render 出 `[K, H, W]` probability：

```python
sem_prob = semantic_render.clamp_min(1e-6)
sem_prob = sem_prob / sem_prob.sum(dim=0, keepdim=True).clamp_min(1e-6)

loss_sem = torch.nn.functional.nll_loss(
    torch.log(sem_prob).unsqueeze(0),
    sem_gt.long().unsqueeze(0),
    ignore_index=255,
)
```

### One-Hot MSE Loss

类似 `gsplat/examples/image_fitting_depth_label.py`：

```python
sem_onehot = torch.nn.functional.one_hot(sem_gt, K).permute(2, 0, 1).float()
loss_sem = torch.nn.functional.mse_loss(semantic_render, sem_onehot)
```

MSE 更平滑，适合初期验证；CE 更符合分类任务。

### Region-Aware Weighting

对大场景建议按类别加权：

```python
class_weights = torch.ones(K, device="cuda")
class_weights[SKY] = 0.5
class_weights[ROAD] = 2.0
class_weights[BUILDING] = 1.5
class_weights[DYNAMIC] = 0.0
```

用途：

- 天空主要用于 alpha penalty，而不是强分类。
- 道路和建筑用于加强几何稳定性。
- 动态物体不参与几何监督。

## Recommended Roadmap

### Stage 1: Semantic Masks Only

不改 CUDA。

实现：

1. 加载 semantic PNG。
2. 用 semantic mask 调整 RGB/depth loss。
3. 加 sky alpha penalty hack。
4. 记录 SwanLab 指标：
   - `semantic/valid_ratio`
   - `semantic/sky_alpha`
   - `semantic/dynamic_mask_ratio`

目标：

- 验证语义图质量。
- 看是否减少天空/动态物体 floaters。
- 不引入大显存风险。

### Stage 2: Palette Or K-Chunk Semantic Render

仍不改 CUDA。

实现：

1. 给 Gaussian 加 `_semantic_logits`。
2. 使用 palette hack 或 K-class chunk hack。
3. 每隔 `semantic_interval` 做一次 semantic loss。

目标：

- 验证 semantic field 是否能学起来。
- 测试 CE/MSE/loss weight。
- 估计语义参数的显存和速度成本。

### Stage 3: Full Feature Rasterizer

修改 CUDA。

实现：

1. 新增任意维 feature rasterizer。
2. 支持 channel dispatch 和 channel chunk。
3. 输出 alpha。
4. 接入 `render_features()`。

目标：

- 高效渲染 K 维语义概率。
- 避免 K-class chunk hack 的重复 rasterization。
- 支持后续 semantic、normal、embedding、多模态 feature。

## Large-Scene Notes

1. 语义参数显存不小。

假设 `N=10M, K=20`：

```text
semantic logits: 10M * 20 * 4 bytes = 800 MB
Adam states:     about 2x extra = 1.6 GB
grad:            about 800 MB
```

仅语义参数训练就可能增加数 GB 显存。大场景建议：

- 先用粗类别；
- 或使用低维 embedding；
- 或让 semantic 参数使用较低精度；
- 或分块训练时只训练当前 block semantic。

2. 语义监督不一定要每轮做。

建议：

```text
semantic_interval = 5 / 10 / 20
semantic_resolution = lower than RGB resolution
semantic_start_iter = after coarse geometry stabilizes
```

3. 先不要让语义强推几何。

如果语义图来自 2D segmentation model，边界和远景可能有噪声。第一版可只训练 semantic logits，或给 geometry gradient 很小权重。

4. 天空、动态物体、透明区域需要特殊处理。

建议：

- sky：alpha penalty；
- car/person：mask out RGB/depth geometry loss；
- vegetation：降低 depth loss；
- road/building：增强 depth/semantic consistency。

## First Implementation Recommendation

第一版不要直接改 CUDA。建议按以下顺序：

```text
1. 加 semantic PNG 数据读取。
2. 加 semantic mask loss：
   - dynamic mask
   - road/building depth weighting
   - sky alpha penalty hack
3. 记录指标，跑小 block 验证。
4. 加 palette semantic render 验证 semantic logits。
5. 如果收益明显，再参考 gsplat 实现 full feature rasterizer。
```

最终 CUDA 方案应以 `gsplat` 为参考：

```text
extra_signals / proj_features / CDIM dispatch / channel_chunk / alpha output
```

这样不仅能支持语义图，也能支持后续 normal、feature distillation、CLIP/DINO embedding 等大场景约束。
