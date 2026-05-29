# Coarse-GS Importance Guided Recursive Partitioning

本文档记录当前计划实现的分块训练方案。目标是在大场景 3D Gaussian Splatting 中，用 coarse GS 作为场景 importance field，递归生成非均匀 block，并为每个 block 自适应补充训练相机。

## Motivation

一个好的 block 应该满足：

1. block 不会过大。
2. block 边界尽量不切穿完整物体或高重要性结构。
3. 能监督该 block 的 camera 都被合理分配到该 block。
4. block 之间有足够 overlap 用于边界训练，但最终合并时不会产生大量重复 Gaussian。

这里的“block 不会过大”不只表示几何尺寸小，也不只表示点数少，而是表示该 block 的训练负载、表达复杂度和图像贡献不过大。

因此，我们使用 coarse GS 中的 Gaussian 属性定义一个加权场景量，用它衡量 block 的大小和复杂度。

## Relation To Existing Methods

### VastGaussian

VastGS 的分块主要由 camera position 驱动：

1. 将相机中心投影到地面平面。
2. 沿 `x` 方向均分为 `m` 份。
3. 每份内部沿 `z` 方向均分为 `n` 份。
4. 得到 `m x n` 个 block，初始目标是相机数量相对均衡。
5. 对 block bbox 做扩展。
6. 将 block bbox 投影到候选相机图像中，根据投影区域面积占比补充相机。

优点是简单稳定，并且 visibility-based camera selection 能缓解边界监督不足。缺点是 block 划分主要由相机分布决定，不直接感知场景内容复杂度。

### CityGaussian

CityGS 使用 coarse GS，但 block 本身仍是固定网格：

1. 先训练 coarse GS。
2. 用户指定 `aabb` 和 `block_dim=[x, y, z]`。
3. 在归一化空间中将场景均匀切成固定网格。
4. 对每个 block，渲染 full coarse GS 和去掉该 block 后的 coarse GS。
5. 如果 `1 - SSIM(render_full, render_without_block)` 超过阈值，则认为该 camera 与该 block 相关。

优点是 camera selection 使用 coarse GS 的图像贡献，比纯几何可见性更贴近最终渲染质量。缺点是 coarse GS 没有用于决定怎么切 block，block 仍然由人工固定网格决定。

### Ours

我们的方案是：

```text
Coarse GS importance field 同时指导：
1. block 是否继续切分
2. 沿哪个方向切分
3. 切分位置在哪里
4. 如何避开高重要性结构边界
5. 哪些 camera 应该补进该 block
```

一句话概括：

```text
不是让几何面积均匀，也不是让相机数量均匀，而是让每个 leaf block 的 coarse-GS 加权重要性相对均衡，并保证高重要性区域有充分相机监督。
```

## Coarse GS Importance Field

先训练一个低成本 full-scene coarse GS：

```text
full scene images + sparse points -> coarse Gaussian model
```

coarse GS 不要求达到最终质量，它主要用于估计：

- Gaussian 空间分布；
- opacity；
- scale / projected size；
- visibility；
- coarse residual / gradient，可选；
- depth / semantic confidence，可选。

对每个 coarse Gaussian `g` 定义 importance weight：

```text
w_g = f(opacity, visibility, projected_size, residual, gradient, semantic_weight)
```

第一版可使用：

```text
w_g = opacity_g * log(1 + visibility_count_g) * projected_radius_g
```

如果暂时没有 visibility 和 projected radius 统计，可以退化为：

```text
w_g = opacity_g * scale_g
```

或更简单：

```text
w_g = opacity_g
```

后续可加入误差信号：

```text
w_g = opacity_g
    * log(1 + visibility_count_g)
    * projected_radius_g
    * (1 + residual_g)
```

## Block Weighted Size

对任意 block `B`，定义加权大小：

```text
W(B) = sum_{g in B} w_g
```

`W(B)` 是 block 的核心度量。它不等于几何面积，也不等于 raw point count，而是表示该 block 的场景复杂度和图像贡献。

同时可定义 importance density：

```text
D(B) = W(B) / area_or_volume(B)
```

递归切分条件：

```text
split if:
    W(B) > tau_weight
 or D(B) > tau_density
```

停止条件：

```text
stop if:
    W(B) <= tau_weight
 and D(B) <= tau_density
```

还应加入工程保护：

```text
stop if:
    depth(B) >= max_depth
 or num_gaussians(B) <= min_points
 or spatial_size(B) <= min_size
```

注意：当前方案不把显存作为主要监督信号。显存只作为后续工程安全检查，而不是核心算法目标。

## Recursive Binary Split

从 root block 开始递归切分，每次只将当前 block 切成两个子 block，形成 binary tree。

### Split Axis

切分轴使用加权空间方差决定：

```text
axis = argmax weighted_variance(xyz_axis, weight=w_g)
```

即沿 importance 分布最分散的方向切。

### Split Position

切分位置不使用几何中点。候选切分平面 `t` 的评分：

```text
split_score(t) = balance_loss(t) + lambda * boundary_cut_penalty(t)
```

其中：

```text
balance_loss(t) = |W(B_left(t)) - W(B_right(t))|
```

用于让左右子块的重要性相对均衡。

```text
boundary_cut_penalty(t) = sum_{g near split plane} w_g
```

用于避免切分平面穿过高重要性区域，降低“物体被切成两半”的概率。

最终：

```text
t* = argmin_t split_score(t)
```

第一版实现可以在该轴上采样若干候选位置，例如 32 或 64 个分位点。

## Core BBox And Expanded BBox

每个 leaf block 维护两个范围：

```text
core_bbox
expanded_bbox
```

含义：

- `core_bbox`：该 block 的主负责区域，最终合并时只保留这里的 Gaussian。
- `expanded_bbox`：训练时使用的扩展区域，用于提供边界上下文和相机监督。

训练时：

```text
use expanded_bbox
```

合并时：

```text
keep only core_bbox
```

这样可以兼顾边界连续性和最终合并时的重复控制。

## Camera Assignment

相机分配不作为递归切分的主监督信号，而是在 leaf block 生成后执行。

### Geometric Base Cameras

先用几何规则给 block 一个保底相机集合：

```text
camera center inside expanded_bbox
or camera frustum intersects expanded_bbox
or projected bbox coverage > tau_projection
```

这一步用于防止 coarse GS 训练不充分时漏掉关键相机。

### Importance-Aware Camera Completion

对候选相机 `c` 和 block `B`，理想评分是：

```text
S(c, B) = sum_{g in B} w_g * visible(c, g)
```

如果：

```text
S(c, B) > tau_camera
```

则将相机 `c` 分配给 block `B`。

第一版可以使用 CityGS 风格的 render difference 近似：

```text
render_full = render(coarse GS, camera c)
render_without_block = render(coarse GS without block B, camera c)

score(c, B) = 1 - SSIM(render_full, render_without_block)
```

如果：

```text
score(c, B) > tau_camera
```

则加入该相机。

也可以融合 VastGS 的几何 visibility：

```text
candidate cameras = cameras whose projected expanded_bbox coverage > tau_projection
final cameras = candidates with score(c, B) > tau_camera
```

最终相机集合：

```text
C(B) = geometric_cameras(B) union importance_completed_cameras(B)
```

## Robustness To Imperfect Coarse GS

coarse GS 不是真值，只是 soft prior。若 coarse GS 训练不好，importance 和 render difference 都可能不准。

因此相机补充和切分都不应只依赖 coarse GS。第一版至少保留几何兜底：

1. `camera center inside expanded_bbox` 必选。
2. `projected bbox coverage > tau_projection` 作为候选或必选。
3. coarse render difference 用于排序或补充，而不是唯一判断。
4. 每个 block 至少保留 `min_cameras`。
5. 边界 block 可额外加入邻接 block 的部分 cameras。

方法表述：

```text
Coarse GS provides a soft importance prior rather than a hard partition oracle.
```

## Training Each Block

每个 leaf block 独立训练：

```text
training cameras = C(B)
initial points = coarse/sparse points inside expanded_bbox
```

第一版可以所有 block 使用相同训练参数。后续可根据 `W(B)` 做自适应训练预算：

```text
iterations(B) = base_iterations * normalize(W(B))
densify_until_iter(B) = function(W(B))
```

## Merging

所有 block 训练完成后：

```text
for each block B:
    load trained Gaussian
    keep Gaussians inside core_bbox(B)
    discard or downweight Gaussians outside core_bbox(B)

merge all kept Gaussians
```

后续可加入：

- overlap 区域 weighted blending；
- opacity-based pruning；
- boundary consistency refinement；
- LOD/compression。

## Proposed Pipeline

完整流程：

```text
1. Train coarse GS on full scene.
2. Extract coarse Gaussian attributes.
3. Compute Gaussian importance weights.
4. Build recursive binary partition tree.
5. Generate core_bbox and expanded_bbox for each leaf block.
6. Assign geometric base cameras.
7. Complete cameras using coarse-GS importance/render contribution.
8. Export partition metadata.
9. Train each block independently.
10. Merge block models using core_bbox crop.
11. Evaluate render quality and boundary consistency.
```

## Finalized Engineering Design

最终实现按“partition / train / merge”三阶段解耦，原始不分块训练路径保持可用。

核心原则：

1. `train.py` 仍然是唯一训练入口。
2. 不传分块参数时，行为和原始 full-scene 训练一致。
3. 分块阶段只输出可追踪 metadata，不直接产出最终训练 point cloud。
4. block 训练复用原始 `Scene`、`GaussianModel`、render、loss、densify、checkpoint 和 SwanLab 逻辑。
5. merge 阶段只依赖 partition metadata 和各 block 的训练输出。

最终命令形态：

```bash
# 原始 full-scene 训练，保持不变
python train.py -s <data> -m output/full_scene

# 生成递归分块 metadata
python partition.py \
  -s <data> \
  --coarse_model output/coarse/point_cloud/iteration_7000/point_cloud.ply \
  --partition_output output/exp/partitions

# 生成 metadata 并保存原始点云压平后的 top-down block 可视化
python partition.py \
  -s <data> \
  --coarse_model output/coarse/point_cloud/iteration_7000/point_cloud.ply \
  --partition_output output/exp/partitions \
  --visualize_topdown \
  --visualize_topdown_bbox_mode both

# 训练某个 block
python train.py \
  -s <data> \
  -m output/exp/blocks/block_000 \
  --partition_path output/exp/partitions/partition_tree.json \
  --block_id block_000

# 使用 partition_tree.json 中的 coarse_model 作为每个 block 的完整初始 Gaussian，
# 不裁剪初始点云；相机列表仍按 block 过滤
python train.py \
  -s <data> \
  -m output/exp/blocks/block_000 \
  --partition_path output/exp/partitions/partition_tree.json \
  --block_id block_000 \
  --partition_init_mode coarse \
  --image_load_mode cache \
  --max_cache_num 128 \
  --image_cache_workers 8

# 合并所有 block
python merge_blocks.py \
  --partition_path output/exp/partitions/partition_tree.json \
  --blocks_root output/exp/blocks \
  --iteration 30000 \
  --output_path output/exp/merged
```

## Partition Output Contract

分块输出不是一个个点云，而是可追踪的结构化 metadata。推荐目录：

```text
output/<experiment>/partitions/
  partition_config.json
  partition_tree.json
  blocks/
    block_000/
      metadata.json
      train_cameras.txt
      candidate_cameras.txt
    block_001/
      metadata.json
      train_cameras.txt
      candidate_cameras.txt
  traces/
    split_events.jsonl
    camera_assignment.jsonl
    block_stats.json
  visualizations_topdown/
    visualization_report.json
    global_blocks.png
    block_000.png
    block_001.png
```

各文件职责：

- `partition_config.json`：记录本次分块参数、coarse model 路径、数据路径、时间、代码版本。
- `partition_tree.json`：完整树结构和所有 leaf block 的最终信息，是训练和合并的主入口。
- `blocks/<block_id>/metadata.json`：单个 block 的独立 metadata，便于单独训练和排查。
- `train_cameras.txt`：该 block 最终使用的训练相机 image_name 列表。
- `candidate_cameras.txt`：相机补充前的候选集合，方便分析漏选/误选。
- `traces/split_events.jsonl`：每次二叉切分的决策记录。
- `traces/camera_assignment.jsonl`：每个 block 的相机分配打分记录。
- `traces/block_stats.json`：全局统计，例如 block 数量、每块点数、importance、camera 数量分布。
- `visualizations_topdown/`：可选输出，将原始 sparse/init point cloud 沿 `partition_axes` 压到 2D 平面上，画出每个 block 的点云分布和 bbox。

## Metadata Schema

`partition_tree.json` 顶层建议包含：

```json
{
  "schema_version": 1,
  "method": "coarse_gs_importance_recursive_partitioning",
  "source_path": "/path/to/dataset",
  "coarse_model": "/path/to/coarse/point_cloud.ply",
  "partition_axes": ["x", "z"],
  "config": {},
  "root": {},
  "blocks": []
}
```

每个 leaf block 至少包含：

```json
{
  "id": "block_000",
  "parent": "node_003",
  "depth": 4,
  "core_bbox": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
  "expanded_bbox": [-0.1, 0.0, -0.1, 1.1, 1.0, 1.1],
  "importance": 12345.6,
  "importance_density": 456.7,
  "num_coarse_gaussians": 320000,
  "num_sparse_points": 250000,
  "train_cameras": ["000001.png", "000002.png"],
  "camera_scores": {
    "000001.png": {
      "source": "geometric_and_importance",
      "projection_coverage": 0.18,
      "render_difference": 0.06,
      "final_score": 0.24
    }
  },
  "trace": {
    "split_path": ["root", "root_l", "root_l_r"],
    "stop_reason": "importance_below_threshold"
  }
}
```

这样 block 的来源、切分路径、相机来源、训练区域和合并区域都可 trace。

## Code-Level Modification Plan

第一版实现只做必要改动，不重写训练框架。

### New Files

- `partition.py`
  - 递归分块入口。
  - 读取数据相机 metadata 和 coarse GS。
  - 计算 importance。
  - 构建二叉树。
  - 分配 cameras。
  - 写出 partition metadata 和 traces。

- `utils/partition_utils.py`
  - bbox、点云裁剪、camera 名称匹配、metadata load/save、schema 校验。
  - 这里放纯工具函数，避免 `train.py` 和 `scene/__init__.py` 里堆业务逻辑。

- `merge_blocks.py`
  - 读取 `partition_tree.json`。
  - 加载每个 block 的训练结果 PLY。
  - 按 `core_bbox` 裁剪并合并。
  - 输出 merged PLY 和 `merge_report.json`。

- `scripts/matrix_city_aerial_recursive_partition.sh`
  - MatrixCity aerial 的一键分块脚本。
  - 负责调用 `partition.py`，参数通过环境变量覆盖。

- `scripts/matrix_city_aerial_train_blocks.sh`
  - 根据 `partition_tree.json` 启动多个 block 训练。
  - 支持指定 GPU 列表和并发数。

### Modified Files

- `arguments/__init__.py`
  - 在 `ModelParams` 中新增可选参数：

```text
--partition_path
--block_id
--partition_bbox_mode expanded
```

  - 默认值为空；为空时完全不启用分块逻辑。

- `scene/__init__.py`
  - 在读取 `scene_info` 后、构建 camera list 和初始化 point cloud 前应用 block filter。
  - 如果存在 `partition_path + block_id`：
    - 读取 block metadata。
    - 用 `train_cameras` 过滤 `scene_info.train_cameras`。
    - 用 `expanded_bbox` 裁剪初始点云。
    - 将 block metadata 复制/记录到当前 `model_path`。
  - 如果没有分块参数，保持原逻辑。

- `train.py`
  - 尽量不改 training loop。
  - 只在启动日志里打印当前是否为 block mode、block id、camera 数、初始点数。
  - SwanLab config 自动包含 `partition_path` 和 `block_id`。

- `render.py`
  - 第一版不强制改。
  - 后续如需渲染单 block，可复用 `--partition_path --block_id` 过滤 camera。

### Avoided Changes

第一版不改：

- `scene/gaussian_model.py` 的 densify/prune 核心逻辑。
- `gaussian_renderer/`。
- loss 和 optimizer 逻辑。
- 原始 full-scene 脚本的默认训练行为。

这样分块训练是外接能力，不会破坏 baseline。

## Implementation Phases

### Phase 1: Metadata And Block Training Skeleton

目标是先跑通“metadata -> block train”的闭环。

1. 实现 `utils/partition_utils.py`。
2. 手工或脚本生成一个最小 `partition_tree.json`。
3. 修改 `arguments/__init__.py` 和 `scene/__init__.py` 支持 `--partition_path --block_id`。
4. 验证：

```bash
python train.py -s <data> -m output/full_scene_test
python train.py -s <data> -m output/block_test --partition_path <tree> --block_id block_000
```

验收标准：

- full-scene 不受影响。
- block mode 只加载该 block 的 cameras。
- 初始点云被 `expanded_bbox` 裁剪。
- 输出目录保存 block metadata。

### Phase 2: Recursive Partition Generator

目标是自动生成可用 block。

1. 实现 `partition.py`。
2. 从 coarse PLY 读取 `xyz`、`opacity`、`scale`。
3. 第一版 importance：

```text
w_g = opacity_g * mean_scale_g
```

4. 使用 `x/z` ground plane 做二叉递归。
5. split axis 使用 weighted variance。
6. split position 使用 weighted balance + boundary penalty。
7. 输出 tree、block metadata 和 split traces。

验收标准：

- 能稳定生成多个 leaf block。
- 每个 block 有 `core_bbox`、`expanded_bbox`、importance、camera 列表。
- `traces/split_events.jsonl` 能解释每一次切分。

### Phase 3: Camera Assignment

第一版先保证稳，再逐步增强。

1. 几何相机保底：
   - camera center inside expanded bbox；
   - frustum / bbox intersection；
   - projected bbox coverage。
2. 每个 block 保证 `min_cameras`。
3. 加入 CityGS 风格 render difference：

```text
score(c, B) = 1 - SSIM(render_full, render_without_block)
```

4. 相机最终集合：

```text
C(B) = geometric_cameras(B) union importance_completed_cameras(B)
```

验收标准：

- 相机分配 trace 能说明每个 camera 为什么被选中。
- coarse GS 不好时仍有几何规则兜底。
- 不出现 camera 数为 0 的 block。

### Phase 4: Block Runner And Merge

目标是多 block 训练和合并可复现。

1. 实现 `scripts/matrix_city_aerial_train_blocks.sh`。
2. 支持：
   - GPU 列表；
   - 最大并发；
   - block id 过滤；
   - checkpoint/resume；
   - SwanLab experiment name 自动带 block id。
3. 实现 `merge_blocks.py`。
4. 合并时每个 block 只保留 `core_bbox` 内 Gaussian。

验收标准：

- 可以从同一个 `partition_tree.json` 复现实验。
- 单个 block 可重训。
- merge 输出记录每个 block 贡献的 Gaussian 数量。

### Phase 5: Evaluation And Ablation

需要保留以下对照：

1. full-scene `block_all`。
2. dataset 原始 block。
3. VastGS-style camera grid partition。
4. CityGS-style fixed grid + render-difference camera assignment。
5. Ours recursive importance partition。

核心 ablation：

- `opacity` vs `opacity * scale` vs `opacity * visibility`。
- no boundary penalty vs with boundary penalty。
- geometric cameras only vs geometric + render difference。
- fixed grid vs recursive tree。

## Traceability Requirements

任何一次训练都应能回答：

1. 这个 block 是怎么从 root 切出来的？
2. 切分轴和切分位置为什么这么选？
3. block 的训练范围和最终合并范围分别是什么？
4. 每个 camera 是因为几何规则、importance score，还是兜底策略被加入？
5. 该 block 训练用了多少 cameras、多少初始点、最后多少 Gaussian？
6. merge 时每个 block 保留/丢弃了多少 Gaussian？

对应 trace 文件：

```text
split_events.jsonl        answers 1, 2
partition_tree.json       answers 3
camera_assignment.jsonl   answers 4
block_stats.json          answers 5
merge_report.json         answers 6
```

## Default Parameters For First Version

建议第一版先用保守参数：

```text
partition_axes = ["x", "z"]
max_depth = 5
min_points = 50000
min_cameras = 20
expand_ratio = 0.10
num_split_candidates = 64
lambda_boundary = 0.2
tau_projection = 0.02
tau_camera_score = 0.03
importance = opacity * mean_scale
```

这些参数不是最终论文参数，只用于稳定跑通 pipeline。

## Evaluation

需要和以下 baseline 对比：

1. `block_all` 单模型训练。
2. 预定义 dataset block 训练。
3. VastGS 风格 camera-position grid partition。
4. CityGS 风格 fixed grid + coarse-GS camera selection。
5. 本方法 recursive importance-guided partition。

建议指标：

- PSNR / SSIM / LPIPS；
- 每个 block 的 camera 数量；
- 每个 block 的 importance 和 importance density；
- block 边界区域的 PSNR/SSIM；
- 合并后总 Gaussian 数量；
- 训练时间；
- 渲染速度；
- 是否存在明显边界伪影。

## Initial Implementation Plan

第一阶段最小可行版本对应上面的 Phase 1 到 Phase 4，但算法上保持简单：

1. 从 coarse PLY 读取 Gaussian `xyz`、`opacity`、`scale`。
2. 定义：

```text
w_g = opacity_g * mean_scale_g
```

3. 在 ground plane 上做二叉递归划分。
4. split axis 用 weighted variance。
5. split position 用 weighted median + boundary penalty。
6. 使用 expanded bbox 选基础相机。
7. 使用 CityGS 的 `1 - SSIM(full, without_block)` 补相机。
8. 输出 `partition_tree.json` 和每个 block 的相机列表。
9. 先用脚本并行训练每个 block。
10. 合并时只保留 core bbox 内 Gaussian。

第二阶段再加入：

- visibility_count；
- projected_radius；
- residual / gradient；
- greedy coverage camera completion；
- boundary consistency refinement；
- 自适应训练预算。
