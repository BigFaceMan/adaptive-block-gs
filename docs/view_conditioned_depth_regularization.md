# Confidence-Aware Multi-Source Geometry Regularization

本文档记录第二个创新点的设计草案。目标是在 block-wise 3DGS 中更可靠地使用单目 depth 和单目 normal 这类几何先验，同时减少错误几何监督诱导的 floaters 和 merge 后局部伪影。

核心立场：

```text
不要把单目 depth/normal 全图等权当真值。
不要把 coarse 点云 depth/normal 当 dense target。
只在几何先验可信、且当前 GS 几何确实需要修正的区域施加 regularization。
```

## Current Problem

当前仓库已经支持 depth regularization：

- `gaussian_renderer.render()` 返回 `render_pkg["depth"]`，该值是 inverse depth。
- `scene/datasets.py` 读取 Depth Anything 输出，并用 `sparse/0/depth_params.json` 中的 `scale + offset` 对齐到 SfM inverse-depth 尺度。
- `train.py` 中目前使用 masked L1：

```python
Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).sum() / valid_pixels.clamp_min(1.0)
```

当前问题：

1. Depth Anything 是估计深度，局部不一定可信。
2. 单目 normal 也是估计结果，在天空、透明/反光、重复纹理、细结构和遮挡边界处可能不准。
3. 当前 depth mask 基本是 binary mask，mask 内像素被等权监督。
4. 错误 depth/normal loss 会在 densification 阶段影响 Gaussian 生长，可能产生 floaters。
5. block-wise 训练和 merge 后，边界/重叠区域的局部几何误差会被放大。
6. coarse 点云本身也可能有几何误差，不能作为 dense geometry target。

## Design Goal

把 geometry regularization 从：

```text
full/block masked mono-depth L1
```

升级成：

```text
confidence-aware multi-source geometry regularization
```

也就是为每个训练 view 生成 depth 和 normal 各自的 condition/confidence map：

```text
C_depth(p), C_normal(p) in [0, 1]
```

然后只在可信区域使用 robust depth loss 和 normal loss：

```text
L_depth = sum_p C_depth(p) * rho(D_render_inv(p) - D_mono_inv(p)) / sum_p C_depth(p)
L_normal = sum_p C_normal(p) * (1 - <N_render(p), N_mono(p)>) / sum_p C_normal(p)
```

其中 `rho` 应使用 Huber、Charbonnier、Tukey 或 truncated L1，而不是普通 L1。这样少量错误 depth/normal 先验即使漏进 mask，也不会持续产生大梯度。

## Proposed Depth-Normal Pipeline

第二个创新点建议组织成两个互补模块：

```text
Module A: Conditioned Depth-Normal Prior
Module B: Perturbation-based Cross-view Normal Consistency
```

### Module A: Conditioned Depth-Normal Prior

该模块解决“单目估计不一定准”的问题。核心不是直接相信单目 depth/normal，而是为两类先验分别生成 condition map，再进行加权监督。

输入：

```text
RGB image
aligned monocular inverse-depth map
monocular normal map
SfM sparse points
block projection mask
image/depth/normal edges
```

输出：

```text
C_depth: depth confidence map
C_normal: normal confidence map
```

Depth condition map 可以由以下 cue 组合：

```text
C_depth = C_sfm * C_edge * C_depth_smooth * M_block
```

- `C_sfm`：把 SfM sparse points 投影到当前 view，比较 aligned mono depth 和 SfM inverse depth 的局部残差；残差小的区域可信。
- `C_edge`：使用 RGB Canny、depth edge 或二者组合，在物体边界附近降权，避免 depth bleeding。
- `C_depth_smooth`：抑制单目 depth 高频噪声和明显不连续区域。
- `M_block`：当前已有的 block projection mask，只在当前 block 相关像素上监督。

Normal condition map 推荐从以下 cue 生成：

```text
C_normal = C_dn * C_n_edge * C_n_aug * M_block
```

- `C_dn`：depth-normal agreement。由 aligned mono depth 反算 normal `N_depth`，和单目 normal `N_mono` 比较；夹角小则可信。
- `C_n_edge`：在 RGB edge、depth edge、normal discontinuity 附近降权，避免跨物体边界强行平滑。
- `C_n_aug`：normal estimator 的 augmentation consistency。对同一图像做 flip/crop/resize/color jitter 后重新估计 normal，再变换回原图；多次估计一致的区域可信。
- `M_block`：同样使用 block projection mask，避免非当前 block 区域影响训练。

最小实现版本可以先使用：

```text
C_depth = C_sfm * C_edge * M_block
C_normal = C_dn * C_n_edge * M_block
```

对应训练目标：

```text
L_geo =
  lambda_d * C_depth * rho(D_render_inv - D_mono_inv)
+ lambda_n * C_normal * (1 - <N_render, N_mono>)
```

这里的 normal 应统一到同一坐标系下比较，推荐使用 world-space normal 或 camera-space normal，但不能混用。

### Module B: Perturbation-based Cross-view Normal Consistency

该模块解决“不正确的几何会在空中产生 floater”的问题。它不依赖新的 GT view，而是对真实 reference view 做轻微相机扰动生成 pseudo view，再用可见一致区域的 normal consistency 约束局部表面稳定性。

流程：

```text
1. 选择 reference camera C_ref。
2. 对 C_ref 做小幅平移/旋转，得到 pseudo camera C_pseudo。
3. 渲染 ref view 和 pseudo view 的 depth/normal。
4. 对 pseudo view 上的像素 p'，用 rendered depth 反投影到 3D 点 X。
5. 将 X 投影回 ref view，得到像素 p。
6. 用 ref rendered depth 做 visibility/occlusion check。
7. 若 p 和 p' 对应同一可见表面，则比较二者 world-space normal。
```

损失：

```text
L_pseudo_normal =
  sum M_valid * (1 - <N_pseudo_world(p'), stopgrad(N_ref_world(p))>)
  / sum M_valid
```

这里 `stopgrad` 很重要。否则 ref/pseudo 两边会互相追随错误几何，self-consistency 可能强化已有错误。

为了更直接抑制 floater，normal consistency 建议和 depth reprojection check 一起使用：

```text
L_pseudo_depth =
  sum M_valid * rho(D_ref(project(X)) - z_ref(X)) / sum M_valid
```

最终 pseudo-view 几何约束：

```text
L_pseudo =
  lambda_pn * L_pseudo_normal
+ lambda_pd * L_pseudo_depth
```

该项建议在训练中后期启用。早期 geometry 尚未稳定，过早启用 pseudo-view consistency 可能把错误结构固定下来。

## What Coarse Points Should Do

每个 block 可以访问完整 coarse 点云，但 coarse 点云不应该作为 depth target。

合理用途：

- 提供当前 view 的弱几何支撑信息。
- 判断某个区域是否有足够 SfM/coarse support。
- 辅助选择邻近 view 或估计可见区域。
- 作为后处理/可视化诊断工具。

不建议：

```text
rendered depth == coarse depth
```

原因是 coarse depth 本身可能不准。如果把它当 target，会把 coarse reconstruction 的错误继续传播到 block training 中。

因此本文方法里，真正的 depth prior 仍来自 aligned monocular depth；coarse/SfM 只参与 confidence 或 support estimation。

## Related Work And Novelty Boundary

已有方法中有几个相关方向：

| Method | Relevant idea | Risk if copied directly |
| --- | --- | --- |
| Depth-Regularized GS | 单目深度 + SfM scale/offset，对 depth 加 L1 和 edge-aware smooth | 如果只做全图 L1/smooth，创新弱 |
| CDGS | 用 Canny、texture、depth gradient 生成 confidence map | 如果只复现 image-based confidence，会像已有 confidence map |
| ConfidentSplat | 多视角几何一致性估计 confidence | 如果只做 multi-view confidence，缺少 block-wise 差异 |
| In Depth We Trust | 用 depth inconsistency mask 只修 GS 不稳定区域，并加 gradient alignment | 如果只照搬 DIM/GAL，会被认为是同一思想 |
| BlockGaussian | pseudo-view geometry constraint 减少 airspace floaters | 如果直接复用 pseudo-view reprojection loss，容易被认为只是移植 |

需要强调的差异：

```text
We do not propose monocular confidence estimation alone,
and we do not propose pseudo-view consistency alone.
Instead, we propose a conditional geometry regularization framework
that gates monocular depth/normal supervision by both prior reliability
and block-wise geometry instability.
```

中文表达：

```text
本文不是简单估计单目深度置信度，也不是单独加入 normal loss 或伪视角重投影约束；
而是在大场景分块 3DGS 中，将单目 depth/normal 监督条件化：
只有当几何先验自身可信，且当前 GS 几何在该视角下不稳定或有 floater 风险时，
depth/normal 先验才参与优化。
```

这可以和已有工作拉开距离：

- 和 CDGS 不同：confidence 不只来自图像/单目深度先验，还结合 normal agreement 和当前 GS 几何不稳定性。
- 和 In Depth We Trust 不同：不是只找 GS 不稳定区域，还要求 mono depth/normal 可靠，且服务于 block-wise merge/floater 问题。
- 和 BlockGaussian 不同：pseudo-view 不只是额外 photometric loss，也用于构造 cross-view normal consistency 和检测 floater-prone 区域。

## Condition Map

推荐把 confidence 拆成几类 cue：

```text
C = C_mono * C_mv * C_gs * C_edge * C_support
```

实际实现可以从少到多逐步加入。

### 1. Monocular Prior Confidence

这部分判断 Depth Anything 自身在像素 `p` 是否可靠。

可用 cue：

- RGB Canny edge：物体边界附近降权，避免深度 bleeding。
- Laplacian/texture：过低纹理区域谨慎，过强纹理噪声区域也可降权。
- depth gradient：若单目深度剧烈变化但 RGB 没有对应边缘，降权。
- local smoothness：平坦区域中过强 depth 高频噪声降权。

形式：

```text
C_mono = C_edge * C_texture * C_depth_gradient
```

这部分接近 CDGS，但在本文里它只是 condition map 的一个因子，不是完整方法。

### 2. SfM Alignment Confidence

当前 `depth_params.json` 是每张图一个 `scale + offset`。大场景中可以进一步计算局部 alignment residual。

做法：

1. 把 COLMAP sparse points 投影到当前 view。
2. 采样 aligned mono inverse depth。
3. 计算 sparse residual：

```text
r_sfm = |D_mono_inv(u, v) - D_sfm_inv(u, v)|
```

4. 用 sparse residual 生成稀疏 confidence，再扩散/插值到局部区域。

注意：SfM sparse points 只用于校准和筛错，不作为 dense target。

### 3. Multi-View Mono-Depth Consistency

这部分判断单目深度在相邻 view 之间是否自洽。

对 reference view `i` 的像素 `p`：

1. 用 aligned mono depth 反投影到 3D：

```text
X_i(p) = backproject(p, D_i(p), K_i, T_i)
```

2. 投影到邻近 view `j`：

```text
p_j = project(X_i(p), K_j, T_j)
```

3. 采样邻近 view 的 mono depth `D_j(p_j)`。
4. 比较 transformed depth 和 sampled depth：

```text
e_mv = |log D_{i->j}(p) - log D_j(p_j)|
```

5. 多个邻近 view 做 average、median 或 majority vote：

```text
C_mv(p) = mean_j 1[e_mv(i, j, p) < tau_mv]
```

遮挡和越界像素置为 invalid。

这部分用于判断 mono depth 是否具有跨视角稳定性。它不依赖 coarse depth。

### 4. GS Geometry Instability

这部分判断当前 GS 是否在该区域需要 depth 修正。

推荐两种实现。

第一种是 residual trigger：

```text
M_unstable = |D_render_inv - D_mono_inv| > tau_residual
```

但它必须和 `C_mono`/`C_mv` 相乘，否则错误 mono depth 会把稳定几何误判成不稳定。

第二种是 pseudo-view instability，参考 BlockGaussian 的视角扰动思想：

1. 用当前真实 view 渲染 `RGB_i, D_i`。
2. 构造轻微平移的 pseudo view `i'`。
3. 渲染 `RGB_i', D_i'`。
4. 将 pseudo view 根据 depth 重投影回真实 view。
5. 比较重投影 RGB/depth 与真实训练 view 是否一致。

若微小视角变化后重投影误差很大，说明当前几何在该区域不稳定，常见于 floaters、遮挡边界或错误透明叠加。

```text
C_gs = normalize(reprojection_error)
```

注意这里的 pseudo-view 不是直接复制 BlockGaussian 的 loss。本文更推荐把它作为 geometry instability detector，再用于 gate depth/normal supervision。

### 5. Support Confidence

完整 coarse/SfM 点云可以提供弱 support：

```text
C_support = 1 if nearby projected SfM/coarse support exists else low value
```

但不要比较 `D_render` 和 `D_coarse`。这里只判断“这个区域是否完全无几何支撑”，而不是判断深度值是否正确。

## Conditioned Losses

### Robust Depth Loss

主 depth loss：

```text
L_depth = sum C_depth * Huber(D_render_inv - D_mono_inv) / sum C_depth
```

其中：

```text
C_depth = C_mono * C_mv * C_support * M_unstable
```

这里的 `M_unstable` 可以是 soft value，而不是 hard binary mask。

### Confidence-Weighted Normal Loss

单目 normal 作为局部表面方向先验。由于 normal estimator 在边界和天空区域也可能不可靠，必须使用 normal condition map：

```text
L_normal =
  sum C_normal * (1 - <N_render_world, N_mono_world>) / sum C_normal
```

其中：

```text
C_normal = C_dn * C_n_edge * C_n_aug * C_support
```

`C_dn` 来自 depth-derived normal 和 mono normal 的一致性，`C_n_edge` 用于避开 RGB/depth/normal 边界，`C_n_aug` 来自 normal estimator 的增强一致性。

normal loss 不建议全训练阶段等权开启。早期 Gaussian 几何和 normal 渲染都不稳定，建议在 RGB 和 weak depth 已经形成基本几何后再逐步 warm up。

### Gradient Alignment Loss

单目深度绝对值可能不准，但局部结构边缘常有价值。可以加 depth gradient alignment：

```text
L_gal =
  sum C_grad * rho(grad_x(D_render_inv) - grad_x(D_mono_inv))
  + sum C_grad * rho(grad_y(D_render_inv) - grad_y(D_mono_inv))
```

`C_grad` 应该避开强遮挡边界和不可信 mono 区域。

### Edge-Aware Smooth Loss

用于压制道路、屋顶、地面上的小 floaters：

```text
L_smooth =
  exp(-alpha * |grad RGB|) * |grad D_render_inv|
```

不要跨 RGB 边缘强行平滑。

### One-Sided Anti-Floater Loss

低可信区域不应强制贴合 mono depth，但可以阻止明显前景 floater。

对 inverse depth 来说，越靠近相机 inverse depth 越大。可以只惩罚 render depth 比 reference 更靠近相机太多的情况：

```text
L_float = max(0, D_render_inv - D_ref_inv - margin)
```

`D_ref_inv` 可以来自 high-confidence mono depth 或稳定的 EMA render depth，而不是 coarse depth。

### Pseudo-View Geometry Consistency

后期可以加入 pseudo-view normal/depth consistency：

```text
L_pseudo_normal =
  M_valid * (1 - <N_pseudo_world, stopgrad(warp(N_ref_world))>)

L_pseudo_depth =
  M_valid * rho(D_ref(project(X)) - z_ref(X))
```

也可以保留 photometric consistency 作为辅助项，但第二个创新点的主线应放在 normal/depth 几何一致性，而不是 RGB warp。

建议只在训练中后期启用，因为早期 render depth/normal 自身不稳定：

```yaml
pseudo_loss_start: 5000  # or 10000
```

## Training Schedule

推荐 schedule：

```text
0 - 500:
  no geometry prior loss, let RGB initialize geometry

500 - 5000:
  weak robust depth loss only on high-confidence mono pixels

5000 - 15000:
  enable normal loss with warmup
  enable GS instability gating and pseudo-view condition
  avoid letting bad depth/normal dominate densification

15000 - end:
  reduce absolute depth/normal prior weights
  keep gradient/smooth/pseudo-view/anti-floater regularization
```

可以进一步考虑：

- depth/normal loss 不参与 densification statistics，避免错误几何先验梯度驱动 split/clone。
- depth/normal weight 先 warm up 再 decay，而不是一开始最大。
- post-train 阶段只在 boundary/floater-prone 区域启用 stronger geometry cleanup loss。

## Implementation Plan In This Repo

建议分阶段实现，避免一次改太多。

### Phase 1: Weighted Robust Depth-Normal Loss

新增：

```text
utils/depth_confidence.py
utils/normal_confidence.py
```

实现：

- Canny/edge-aware mask。
- depth gradient confidence。
- depth-normal agreement confidence。
- normal edge confidence。
- robust depth loss helper。
- confidence-weighted normal loss helper。

修改：

```text
arguments/__init__.py
utils/config_schema.py
train.py
post_train.py
```

新增配置：

```yaml
optimization:
  depth_reg_confidence_mode: image
  depth_reg_loss_type: huber
  depth_reg_huber_delta: 0.02
  depth_reg_min_confidence: 0.1
  normal_reg_weight_init: 0.1
  normal_reg_weight_final: 0.01
  normal_reg_confidence_mode: depth_normal_edge
```

### Phase 2: Multi-View Mono Consistency

离线预处理更稳：

```text
scripts/prepare_depth_confidence.sh
utils/make_depth_confidence.py
```

输出：

```text
DATASET_ROOT/depths_any_conf/
  00001.npy  # float32 confidence map
```

训练时 `scene/datasets.py` 一并加载 confidence map。

优点：

- 训练时不需要频繁跨 view 读取深度。
- confidence 可视化和 debug 更容易。

### Phase 3: Pseudo-View Normal Consistency

在 `train.py` 中加入可选 pseudo camera render。

需要新增 camera perturbation/helper：

```text
utils/view_warp.py
```

实现：

- build pseudo camera with small lateral perturbation。
- warp pseudo depth/normal render back to reference。
- valid/occlusion mask。
- normal consistency loss。
- depth reprojection error map。

该 map 一方面作为 `L_pseudo_normal/L_pseudo_depth`，另一方面也可以用于 `C_gs`，决定 depth/normal prior loss 的生效区域。

### Phase 4: Post-Train Boundary/Floater Cleanup

`post_train.py` 当前已有 boundary/internal Gaussian mask。可以加入：

```text
boundary: RGB + robust depth + gradient depth + anti-floater
internal: RGB/color/opacity mainly
```

这与 merge 后伪影问题最相关。

## Ablation Plan

建议实验顺序：

| ID | Setting | Purpose |
| --- | --- | --- |
| A0 | RGB-only block GS | baseline |
| A1 | current full/block masked L1 depth | show naive depth problem |
| A2 | robust depth loss only | test outlier suppression |
| A3 | image confidence + robust depth | test mono confidence |
| A4 | depth + normal without confidence | show naive normal risk |
| A5 | depth-normal confidence maps | test prior reliability gating |
| A6 | pseudo-view normal consistency | test floater suppression |
| A7 | pseudo-view gated depth-normal loss | test condition trigger |
| A8 | post-train boundary geometry cleanup | test merge artifact cleanup |

需要记录：

- PSNR/SSIM/LPIPS mean and median。
- worst-view PSNR delta。
- improved view count。
- merged Gaussian count。
- per-block point count growth。
- depth/normal mask coverage and confidence mean。
- floater-prone view visualization。
- boundary/merge artifact close-up。

## Paper Method Name

候选命名：

```text
Confidence-Aware Multi-Source Geometry Regularization
```

或更贴近分块：

```text
Block-wise Conditioned Depth-Normal Regularization
```

一句话贡献：

```text
We condition monocular depth and normal supervision on prior reliability
and block-wise view-dependent geometry instability, enabling geometry
regularization to correct floater-prone regions without corrupting
already stable surfaces.
```

中文：

```text
我们根据单目 depth/normal 先验可靠性和分块 GS 的视角相关几何不稳定性，
有条件地触发几何监督，使几何正则主要修复易产生 floater 的区域，
同时避免破坏已经稳定重建的几何。
```

## Key Takeaway

这个方向的关键不是“用了 confidence map”或“用了 pseudo-view”，而是：

```text
Geometry regularization is conditional.
```

也就是：

```text
可信的单目 depth/normal + 当前 GS 几何不稳定 + block/merge 伪影风险
=> geometry supervision 生效

不可信单目先验 或 已经稳定的几何
=> geometry supervision 降权或关闭
```

这样可以正面回应当前两个核心问题：

1. 单目 depth/normal 不可信。
2. 错误几何约束会导致 floater。
