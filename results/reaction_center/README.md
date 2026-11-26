# 反应中心识别与传播对照

本目录报告反应中心识别的 Top-k 准确率，即前 k 个预测位点中包含该反应实际发生变化的键或原子位点的反应比例。目录中提供逐候选分数、原始 Top-10 列表、逐反应排名、分层汇总、选权配置和可用训练曲线。

| 对照 | 数据与种子 | 结果位置 | 主要问题 |
| --- | --- | --- | --- |
| 全谱与直接删除谱传播 | 全量 9,250 测试反应；seed0 | `full_spectral_ablation/` | 当前原子对模型对谱传播路径的依赖 |
| 全谱与局部原子对传播 | 平衡小集 2,400 测试反应；seed0/1/2 | `small_local_propagation/` | 相同原子对框架与参数预算下的传播方式比较 |
| 全谱与普通单图 RGCN | 全量 9,250 测试反应；seed0 | `full_rgcn_comparison/` | 不同模型表示与读出的整体比较 |

三项对照均使用类别未知设置，候选中心包括规范化的无向键和原子。按验证集 Top-1 准确率选择权重，并列时取最早轮次。[完整模型权重](../../experiments/section_4_results/4_1_benchmark/unknown-class/checkpoints/reaction_center.pth)和[消融权重索引](../../experiments/section_4_results/4_2_ablation/models/README.md)可用于核对所选模型。

## 结果文件

`test/` 与 `validation/` 分开保存。`*_predictions.json` 或小集 `*_seedN.json` 含保存的原指标及 `records`；每条记录保留标注候选、候选总数、原 Top10 列表、前十内排名和全部候选分数。小集记录还含样本身份、类别和相似度组。未进入前十的标注排名为 `null`，不删除该反应。

`per_reaction.csv` 给出完整配对反应、原排名、错误类型与结构分层。`stratified_summary.json` 提供类别、相似度、大小和候选数等汇总。直接按保存分数排序的结果标为 diagnostic，独立于原评估器，不用于重新选择权重。

小集 `similarity.json` 给出半径二局部指纹相似度及竞争中心索引。“最相似竞争中心”按指纹度量选取，受根键类型的表征限制。若多个竞争位点的相似度并列第一，只有当模型给实际反应位点的分数高于所有这些竞争位点时，才计为一次胜出。[逐种子测试结果](small_local_propagation/test/per_seed.csv)与[测试汇总](small_local_propagation/test/summary.json)给出每组结果。均值和样本标准差描述三个训练种子在同一组 2,400 条测试反应上的差异。

## 复算

在发布根目录执行：

```bash
python results/reaction_center/reproduce.py --check
```

该命令用 Python 标准库核对文件哈希，从保存的候选列表重算 Top-k 命中数，并复算全量谱删除的排序诊断和小集的分层、最相似竞争中心及三种子统计。输出目录为 `outputs/reproduced_stage1/`。

[全量测试结果](full_spectral_ablation/test/per_reaction.csv) · [小集逐种子结果](small_local_propagation/test/per_seed.csv) · [数据划分](../../datasets/DATASET_CARD.md) · [评估方法](../../docs/guides/evaluation.md)
