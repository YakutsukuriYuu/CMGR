# ModelNet 本次实验结果对比

本次整理来源：

- 日志：`CMGR/incremental.log`
- 结果：`CMGR/outputs_v3/incremental_kd/results.yaml`
- Base 结果：`CMGR/outputs_v3/base/best_acc.yaml`
- 论文数据：`CMGR/ModelNet论文实验数据.md`

注意：本次运行对应的是 `ModelNet -> ModelNet (M2M)`，即 20 个 base 类 + 4 个增量任务，每个任务 5 个 novel 类。不是 `ModelNet -> ScanObjectNN (M2O)`。

## 本次运行配置摘要

| 项 | 本次运行值 |
|---|---:|
| Base classes | 20 |
| Incremental tasks | 4 |
| Classes per task | 5 |
| Batch size | 22 |
| Num views | 10 |
| Base epochs | 100 |
| Incremental epochs | 30 |
| Base LR | 0.002 |
| Incremental LR | 0.001 -> 0.0001 |
| beta | 0.1 |
| gamma | 2.0 |
| BND threshold | 0.1 |

## Accuracy Curve 对比

单位：`%`。差值为 `本次 - 论文`，负数表示低于论文。

| 阶段 | 已见类别数 | 本次 Acc | 论文 Ours Acc | 差值 |
|---|---:|---:|---:|---:|
| Base | 20 | 73.54 | 95.00 | -21.46 |
| Task 1 | 25 | 54.19 | 84.80 | -30.61 |
| Task 2 | 30 | 41.06 | 81.00 | -39.94 |
| Task 3 | 35 | 38.82 | 72.20 | -33.38 |
| Task 4 | 40 | 34.12 | 65.90 | -31.78 |

## 汇总指标对比

说明：代码输出的 `delta_A=0.1703` 是 0-1 小数；论文表格按百分数展示，因此这里写为 `17.03`。

| 指标 | 本次 | 论文 Ours | 差值 | 趋势 |
|---|---:|---:|---:|---|
| Final Acc | 34.12 | 65.90 | -31.78 | 低 |
| AA | 48.35 | 79.80 | -31.45 | 低 |
| Delta A ↓ | 17.03 | 8.70 | +8.33 | 高，遗忘/退化更重 |

## 每个增量任务的 best epoch

| Task | Best epoch | Best Val Acc |
|---|---:|---:|
| Task 1 | 25 / 30 | 54.19 |
| Task 2 | 27 / 30 | 41.06 |
| Task 3 | 22 / 30 | 38.82 |
| Task 4 | 26 / 30 | 34.12 |

## BND 训练状态

| Task | Base/Novel 训练样本数 | Base logit mean | Novel logit mean | Base route rate | Novel route rate |
|---|---:|---:|---:|---:|---:|
| Task 1 | 20 / 1186 | 1.062 | -1.270 | 0.800 | 0.157 |
| Task 2 | 25 / 818 | 0.671 | -0.778 | 0.600 | 0.145 |
| Task 3 | 30 / 1449 | 0.812 | -0.207 | 0.733 | 0.233 |
| Task 4 | 35 / 1276 | 0.290 | -0.105 | 0.743 | 0.244 |

解读：

- `Base route rate` 越接近 1 越好；本次 Task 2 只有 0.600，说明较多 base 样本没有走 frozen NetB。
- `Novel route rate` 越接近 0 越好；Task 3/4 达到 0.233/0.244，说明约四分之一 novel 样本被错误路由到 base 分支。
- BND 的路由质量仍然偏弱，会拉低增量阶段统一测试集 accuracy。

## 结论

本次结果明显低于论文 M2M：

- Base 阶段已经低论文 `21.46` 个百分点，这是最大信号。即使增量阶段完全正常，也很难追上论文曲线。
- Task 1 后从 `73.54` 降到 `54.19`，后续最终到 `34.12`，累计退化比论文更重。
- 本次 `AA=48.35`，论文 Ours `AA=79.80`，差距 `31.45` 个百分点。
- 本次 `Delta A=17.03`，论文 Ours `8.70`，退化约为论文的 1.96 倍。

## 需要注意的日志问题

日志结尾显示：

```text
(incremental failed)
```

但这不是训练失败。原因是 `run_incremental.sh` 检查的是旧路径：

```text
outputs_v3/incremental/results.yaml
```

实际结果保存到了：

```text
outputs_v3/incremental_kd/results.yaml
```

因此应以后者为准。
