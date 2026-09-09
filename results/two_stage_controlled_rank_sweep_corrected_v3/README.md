# corrected_v3 受控 rank 扫描

本目录保存通过完整分片与 provenance 校验的四编码器受控重叠结果。每个 rank 包含 4 个编码器、
10 个构造种子、6 个重叠率和 4 个预算，共 960 个配置。`corrected_v2` 的结果已经作废，不应与
本目录合并。

## 主要结果

| rank | rank-L 相对 rank-only | DPP 相对 rank-only | 平均绝对 log 误差 | 平均理论界 | 区间覆盖 | 已认证步骤 |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | +2.09% | +4.90% | 2.558 | 5.944 | 960/960 | 0/2,400 |
| 20 | +3.81% | +5.48% | 1.877 | 5.505 | 960/960 | 0/2,400 |
| 50 | +4.94% | +5.68% | 0.996 | 4.455 | 960/960 | 0/2,400 |

理论区间在所有选中前缀上均覆盖精确分数，但在 `L <= 50` 时没有认证任何 greedy 决策。它们在
本实验范围内是有效但无实际决策力的界。DPP 总体至少同样有竞争力。当前比较记录了每种方法的
实际通信量，但不是等字节匹配；不能据此主张 rank-L 的成本优势。

通信计费勘误：本目录已签名 manifest 中的 legacy `collapse_sketch` 只计入 rank-L 因子，漏计
同时传输的边际有效秩和核质量两个 `float64` 标量，因此每池少计 16 bytes。修正版代码按
$8(Ld+2)$ bytes/pool 计费。为保持 provenance，本目录不回写既有 manifest；选择顺序、准确率及
上表结论均不受该计费勘误影响。

## 文件

每个 `L*/` 目录包含合并后的逐配置结果、方法配对、决策摘要、区间诊断、重叠检测指标及聚合
manifest。原始特征和逐编码器 shard 体积较大，不在匿名仓库中分发；manifest 保留其内容哈希。

在具有相同标签盲特征缓存的环境中，可通过以下命令复算：

```bash
PROJECT_ROOT="$PWD" \
FEATURE_DIR=/path/to/unlabeled/source/features \
RESULT_ROOT=/path/to/output \
bash scripts/run_two_stage_controlled_rank_sweep.sh
```
