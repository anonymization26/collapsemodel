# corrected_v3 Full-Gram 精确对照

本目录保存预先声明的 $L=20$ 小规模精确参照实验。实验覆盖 4 个编码器、3 个构造种子、6 个
重叠率和 4 个预算，共 288 个配置；每个配置包含 Random-100、rank-only、rank-$L$、DPP、
Facility Location、凝聚式 medoid、legacy Collapse、lineage-aware oracle 及 Full-Gram greedy。
Full-Gram greedy 只是逐步精确评分参照，不是组合全局最优 oracle。

## 主要结果

| 方法 | 相对 rank-only 平均增益 | 相对 Full-Gram greedy | 对 rank-only 胜/平/负 |
|---|---:|---:|---:|
| rank-$L$ Gram | +3.65% | -2.03% | 200/0/88 |
| DPP-subspace | +5.37% | -0.41% | 285/0/3 |
| Full-Gram greedy | +5.81% | 0.00% | 288/0/0 |

rank-$L$ 的 288/288 个选中前缀区间覆盖精确分数，但 720 个 greedy 步骤均未获得区间分离
证书，且 288 个预算前缀无一与 Full-Gram greedy 的有序前缀完全相同。rank-$L$、DPP 和
Full-Gram greedy 在 288/288 个配置中都高于 Random-100 的均值；这只是受控谱保持指标，不能
外推为下游效用。结合 DPP 的表现，结果不支持 rank-$L$ 优于经典基线的主张。

通信计费勘误：已签名分片 manifest 中的 legacy `collapse_sketch` 漏计边际有效秩和核质量两个
`float64` 标量，每池少计 16 bytes。修正版代码按 $8(Ld+2)$ bytes/pool 计费。为保持 provenance，
本目录不回写既有 manifest；选择结果与上述指标不受影响。本实验也不是等字节比较。

## 文件与完整性

- `aggregate/summary_exact_results.csv`：31,104 行逐配置结果。
- `aggregate/summary_exact_*.csv`：方法汇总、配对比较、重叠指标与 rank-$L$ 诊断。
- `aggregate/summary_exact_manifest.json`：聚合文件和四个来源 manifest 的 SHA-256。
- `manifests/`：四个编码器的签名运行清单；原始分片 CSV 和特征缓存不在匿名仓库中分发。
- `logs/`：四个编码器及总运行日志。

在具有相同标签盲特征缓存的环境中，可通过以下命令复算：

```bash
PROJECT_ROOT="$PWD" \
FEATURE_DIR=/path/to/unlabeled/source/features \
RESULT_ROOT=/path/to/output \
bash scripts/run_two_stage_controlled_exact_pilot.sh
```
