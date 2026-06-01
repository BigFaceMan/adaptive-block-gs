# MatrixCity Aerial d5/c32 Depth Regularization Analysis

本文档记录 `mc_aerial_recursive_d5_c32_depth` 相对
`mc_aerial_recursive_d5_c32` 的一次结果分析。分析时间为 2026-06-01。

## Runs

Baseline:

```text
/lfs1/users/spsong/Code/project/gaussian-splatting/output/mc_aerial_recursive_d5_c32
```

Depth regularization:

```text
output/mc_aerial_recursive_d5_c32_depth
```

Depth 版本复用了 baseline 的 partition：

```text
/lfs1/users/spsong/Code/project/gaussian-splatting/output/mc_aerial_recursive_d5_c32/partitions/partition_tree.json
```

关键配置：

```yaml
optimization:
  depth_l1_weight_init: 1.0
  depth_l1_weight_final: 0.01
  depth_reg_mask_mode: block_projection
  depth_reg_mask_bbox_mode: expanded
  depth_reg_mask_dilate_px: 16
  depth_reg_mask_min_pixels: 2048
  depth_reg_mask_max_points: 100000
  depth_reg_mask_cache: true
  depth_reg_mask_cache_max_items: 0
```

## Overall Metrics

Merged model 的 `results.json`：

| Metric | Baseline | Depth reg | Delta |
| --- | ---: | ---: | ---: |
| PSNR | 27.5232 | 27.4471 | -0.0762 |
| SSIM | 0.86934 | 0.87061 | +0.00127 |
| LPIPS | 0.19616 | 0.19023 | -0.00593 |

结论：这不是整体质量崩溃。PSNR 小幅下降，但 SSIM 和 LPIPS 都变好，说明 depth reg
对感知质量有正向作用，问题主要集中在少数 PSNR outlier。

## Training Health

32 个 depth block 日志都完成训练：

```text
Training complete.
```

未发现 OOM、Traceback 或 CUDA RuntimeError。`block_025` 日志里有一次 SwanLab
network error，但训练继续完成，不是质量下降原因。

每个 block 都启用了同一套 depth mask：

```text
[DepthMask] block=block_xxx bbox=expanded points=100000 dilate_px=16 cache=True cache_max_items=0
```

SwanLab 中 block validation PSNR 的均值：

| Run | Mean block validation PSNR |
| --- | ---: |
| Baseline | 26.4014 |
| Depth reg | 26.3699 |

最差 block validation delta 是 `block_016: -0.457dB`。这说明 block 单独训练并没有明显崩，
merged/global rendering 后的问题更明显。

## Per-view Behavior

merged test set 共 741 张图：

| Metric | Mean delta | Median delta | Improved views |
| --- | ---: | ---: | ---: |
| PSNR | -0.0762 | +0.0073 | 377 / 741 |
| SSIM | +0.00127 | +0.00136 | 508 / 741 |
| LPIPS | -0.00593 | -0.00528 | 671 / 741 |

PSNR 中位数是正的，说明均值下降主要由少数严重 outlier 拉低。

最差 PSNR views：

| View | Baseline PSNR | Depth PSNR | Delta |
| --- | ---: | ---: | ---: |
| `00611.png` | 31.802 | 24.785 | -7.017 |
| `00555.png` | 30.149 | 23.611 | -6.538 |
| `00612.png` | 30.734 | 24.851 | -5.883 |
| `00557.png` | 29.737 | 24.030 | -5.707 |
| `00556.png` | 29.250 | 23.578 | -5.672 |
| `00226.png` | 28.190 | 23.125 | -5.065 |
| `00554.png` | 28.424 | 23.403 | -5.021 |
| `00610.png` | 30.611 | 25.706 | -4.906 |

这些 outlier 在 view index 上成簇，说明问题更像局部空间区域或跨 block merge 后的局部重叠问题，
不是所有 view 都被 depth reg 拉低。

按 partition 中 test-camera 分配粗略聚合，PSNR delta 最差的 block 区域：

| Block | Mean PSNR delta | Worst view delta |
| --- | ---: | ---: |
| `block_031` | -0.762 | -7.017 |
| `block_027` | -0.631 | -7.017 |
| `block_030` | -0.505 | -6.538 |
| `block_016` | -0.352 | -2.721 |
| `block_010` | -0.340 | -4.118 |

## Gaussian Count

Merged point count：

| Run | Kept Gaussians |
| --- | ---: |
| Baseline | 25,473,239 |
| Depth reg | 30,375,493 |

Depth reg 多了 4,902,254 个 Gaussian，约 +19.2%。32 个 block 的保存点数全部增加。

点数增加最多的 block：

| Block | Baseline | Depth reg | Delta |
| --- | ---: | ---: | ---: |
| `block_022` | 1,293,602 | 1,608,364 | +24.3% |
| `block_023` | 1,207,787 | 1,476,617 | +22.3% |
| `block_001` | 1,047,931 | 1,312,194 | +25.2% |
| `block_003` | 1,024,261 | 1,268,273 | +23.8% |
| `block_028` | 1,046,643 | 1,281,817 | +22.5% |

这说明 depth reg 明显改变了 densification / opacity / geometry 的演化，而不只是给最终 loss
加了一个很弱的约束。

## Depth Loss And Mask Signals

Depth run 在 step 30000 的 SwanLab 聚合：

| Metric | Mean |
| --- | ---: |
| `train_loss_patches/depth_l1_loss` | 0.000163 |
| `train_loss_patches/depth_l1_loss_pure` | 0.01627 |
| `train_loss_patches/depth_l1_weight` | 0.01 |
| `train_loss_patches/depth_mask_coverage` | 0.4205 |
| `train_loss_patches/depth_applied` | 1.0 |
| `train_loss_patches/depth_reliable` | 1.0 |
| `train_loss_patches/depth_mask_enough` | 1.0 |
| `train_loss_patches/depth_mask_cache_hit` | 1.0 |

Depth loss 占 total loss 的比例随训练变化：

| Step | Mean ratio | Max ratio | Mean weight |
| --- | ---: | ---: | ---: |
| 10 | 0.309 | 0.703 | 0.9985 |
| 100 | 0.202 | 0.586 | 0.9848 |
| 500 | 0.156 | 0.532 | 0.9261 |
| 1000 | 0.122 | 0.299 | 0.8577 |
| 3000 | 0.083 | 0.231 | 0.6310 |
| 7000 | 0.070 | 0.301 | 0.3415 |
| 15000 | 0.030 | 0.118 | 0.1000 |
| 30000 | 0.005 | 0.025 | 0.0100 |

关键点：densification 从 step 500 开始，而此时 depth loss 仍占 total loss 平均 15% 以上。
因此 depth reg 很可能影响了早期 Gaussian 生长、opacity 和局部几何。

Depth mask coverage 偏大。step 30000 时平均 coverage 为 42%，最大接近 89%：

| Block | Mask coverage | Saved point increase |
| --- | ---: | ---: |
| `block_028` | 0.891 | +22.5% |
| `block_000` | 0.832 | +17.5% |
| `block_001` | 0.831 | +25.2% |
| `block_016` | 0.800 | +14.4% |
| `block_029` | 0.784 | +21.1% |

当前 partition 中检查到 `core_bbox == expanded_bbox`，所以这次 `expanded` 不是主要变量。
更可疑的是 `dilate_px=16` 让投影 mask 覆盖过大，以及 `depth_l1_weight_init=1.0`
让早期 depth 约束过强。

## Implementation Notes

Depth block mask 的实现路径：

- `train.py` 中根据 `depth_reg_mask_mode=block_projection` 创建 `BlockDepthMasker`。
- `utils/block_depth_mask.py` 读取 partition tree 中的 coarse model，把当前 block 的 coarse points
  投影到当前 image，得到 depth mask。
- 开启 `depth_reg_mask_cache` 后，每张 image 第一次遇到时生成 CPU mask，后续 cache 命中。
- 每个训练 iter 只把当前 image 的 CPU mask 临时拷到 GPU，并和原始 depth mask 相乘。

相关代码位置：

```text
train.py:198-213
train.py:322-349
utils/block_depth_mask.py:82-112
```

Block 保存和 merge：

- 每个 block 保存时只保留 `core_bbox` 内的 Gaussians。
- `merge_blocks.py` 直接拼接所有 block PLY，不再额外去重或做 opacity 调整。

相关代码位置：

```text
scene/__init__.py:172-190
merge_blocks.py:201-230
```

因此如果 depth reg 让多个 block 在边界附近产生更多、opacity 更强或几何略不一致的 Gaussians，
单 block validation 不一定明显下降，但 merge 后某些 view 会出现 overdraw、ghosting 或局部遮挡错误。

## Diagnosis

当前证据支持以下判断：

1. 训练流程本身正常，32 个 block 都完整训练，没有 OOM 或崩溃。
2. Depth reg 没有全局变差；SSIM、LPIPS、PSNR 中位数都显示大部分区域没有问题。
3. PSNR 均值下降主要来自少数局部 outlier view。
4. Depth reg 让 merged Gaussian 数量增加约 19%，说明它显著改变了 densification。
5. 早期 depth loss 过强，并且发生在 densification 之前和期间，是最可能的根因。
6. Mask coverage 偏大，会让一个 block 在较大图像区域接受 depth 约束，增加跨 block 不一致风险。
7. 问题更像 merge/global rendering 阶段暴露出的局部跨 block 叠加问题，而不是单个 block
   训练质量直接崩溃。

## Suggested Next Experiment

先做一个最小改动 ablation，不要同时改太多变量：

```yaml
optimization:
  depth_l1_weight_init: 0.1
  depth_l1_weight_final: 0.001
  depth_reg_mask_dilate_px: 8
```

如果仍有明显 outlier，再试更小 mask：

```yaml
optimization:
  depth_reg_mask_dilate_px: 4
```

如果点数仍然明显膨胀，可以单独再试更保守的 densification：

```yaml
optimization:
  densify_grad_threshold: 0.0003
```

推荐观察指标：

1. `results.json` 中 PSNR / SSIM / LPIPS。
2. `per_view_test.json` 中 worst PSNR views 是否仍集中在 `00610-00612`、`00554-00558`
   等区域。
3. `merge_report.json` 中 total kept Gaussians 是否仍比 baseline 高 15% 以上。
4. SwanLab 中早期 `depth_l1_loss / total_loss` 比例是否降到 5%-10% 量级。
5. `depth_mask_coverage` 是否明显降低。

