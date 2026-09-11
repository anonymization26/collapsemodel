# E2 事后 Oracle Headroom 诊断

> 状态：四个数据集-编码器运行均已完成并通过完整性校验。本实验是打开 `target-test` 后的探索性诊断，不能更新、替代或挽救原 H2/H2b 门控结论。

## 实验契约

- 运行器版本：`35a3222863a38c81bf632da63f906673b9f50290`
- 父评估运行器版本：`1d3ce3885c6f7da2e165f6c7635adb654e54be78`
- 数据集：PACS、Office-Home
- 编码器：ResNet-50、DINOv2 ViT-B/14
- 候选块：每个目标任务 12 个
- 预算：`K in {1, 3, 5}`，分别穷举 12、220 和 792 个组合
- 主预算与指标：`K=3`、多分类 Brier score
- 规模：16 个编码器-目标域任务、16,384 个唯一组合、1,536 条父方法选择诊断

每个组合使用与父 E2 完全相同的特征归一化、ridge 正则、类别词表、目标测试集和指标实现。
运行器要求每条父评估指标均由穷举表在 `rtol=1e-8, atol=1e-10` 内复现，否则拒绝写入结果。
Target A-opt 组合级排序使用 64 个固定共同 Rademacher 探针估计；目标测试标签只用于定义事后
Brier oracle，不进入任何可部署选择器。

## 主预算结果

统计时先在每个数据集-目标域内平均两个编码器，再跨 8 个任务单位汇总。

| 指标 | 结果 |
| --- | ---: |
| 最强父目标无关基线 | DPP Subspace |
| DPP Brier | 0.829162 |
| Target A-opt Brier | 0.826769 |
| 真实 oracle Brier | 0.823794 |
| Oracle 相对 DPP headroom | 0.781% |
| Oracle headroom 95% bootstrap CI | [0.162%, 1.617%] |
| Target A-opt 相对 oracle 归一化遗憾 | 0.436% |
| Target A-opt 遗憾 95% bootstrap CI | [0.100%, 0.878%] |
| Target A-opt 真实组合平均排名 | 32.56 / 220 |
| Target A-opt top-10 命中率 | 62.5% |
| Target A-opt 捕获 oracle 改善比例 | 31.6% |
| Oracle 相对父方法事后 envelope 的剩余 headroom | 0.087% |
| A-opt 代理与 Brier 的组合级 Spearman | 0.826 |
| 两编码器 Brier 组合排序 Spearman | 0.803 |

Oracle 相对最强目标无关基线的平均 headroom 及其置信区间上界均低于父协议的 2% 实质门槛。
因此，首要诊断是当前候选任务整体缺少足够大的可利用效应，而不是仅因某个选择器没有找到一个
普遍存在的大幅改进组合。

## 预算与搜索诊断

下表按 16 个编码器-目标域任务汇总。机器字段 `top_q_recall` 表示一个所选组合是否落入真实
top-`Q` 的跨任务命中率，不是 H3 所定义的 shortlist 集合召回率。

| 预算 | 组合数 | Target A-opt 遗憾 | 平均真实排名 | top-10 命中率 | 代理-Brier Spearman | Oracle 相对父 envelope 剩余 headroom |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12 | 0.274% | 1.94 | 100.0% | 0.858 | 0.008% |
| 3 | 220 | 0.443% | 32.56 | 62.5% | 0.826 | 0.087% |
| 5 | 792 | 0.458% | 134.88 | 31.25% | 0.688 | 0.136% |

`K=1` 的 top-10 覆盖了 12 个组合中的 10 个，不能据其 100% 命中率宣称强预筛选能力。随着组合
空间扩大，真实排名、top-10 命中率和秩相关均恶化。穷举最小化 Hutchinson A-opt 代理的组合在
`K=3` 下平均遗憾为 0.466%，并不优于原 Target A-opt 贪心选择的 0.443%；在当前代理估计精度下，
没有证据表明贪心搜索是主要瓶颈。

## 异质性与限制

- PACS `photo` 是唯一平均 headroom 超过 2% 的任务单位，oracle 相对 DPP 改善 3.376%。其余
  7 个单位均不超过 1.378%，Office-Home 的 4 个单位均不超过 0.115%。
- Target A-opt 在 Office-Home `Art/Product/Real World` 接近 oracle，但在 Office-Home `Clipart`
  和 PACS `art_painting/cartoon` 未捕获 DPP 到 oracle 的改善，后两者甚至比 DPP 更差。
- 64 探针两半估计在主预算上的平均相对差异为 3.82%，最大为 25.00%；所有预算中的最差项是
  PACS/ResNet-50/`sketch`/`K=1` 的 25.94%。组合级 A-opt 排序是稳定性受限的估计，不是精确证书。
- 父方法事后 envelope 距 oracle 平均只剩 0.087%，说明不同启发式的并集已覆盖多数可见 headroom，
  但这属于使用测试结果挑方法的事后上界，不能作为部署性能。

综合来看，实验支持“无标签二阶目标几何包含一定组合排序信号”，但不支持“当前 Target A-opt 在该
候选构造下能获得具有投稿主张强度的实质风险优势”。下一轮若继续，应优先增加候选任务的真实差异、
改进条件机制信息或引入严格独立的新测试集，而不是在当前测试集上继续调选择器。

## 运行时间与产物

| 数据集 | 编码器 | 组合数 | 运行时间 |
| --- | --- | ---: | ---: |
| PACS | DINOv2 ViT-B/14 | 4,096 | 346.68 秒 |
| PACS | ResNet-50 | 4,096 | 1,683.96 秒 |
| Office-Home | DINOv2 ViT-B/14 | 4,096 | 653.00 秒 |
| Office-Home | ResNet-50 | 4,096 | 2,392.53 秒 |

四组记录内总运行时间为 5,076.16 秒，均使用 CPU；并行执行的实际墙钟时间更短。

- 权威汇总：[`summary/summary.json`](summary/summary.json)
- 方法与 oracle 比较：[`summary/method_oracle_summary.csv`](summary/method_oracle_summary.csv)
- 逐任务诊断：[`summary/task_units.csv`](summary/task_units.csv)
- 编码器排序相关：[`summary/correlations.csv`](summary/correlations.csv)
- 主预算逐单位 headroom：[`summary/primary_headroom_units.csv`](summary/primary_headroom_units.csv)
- 四个运行目录中的 `combinations.csv`、`selection_diagnostics.csv`、`task_summary.csv` 和
  `manifest.json` 保存无损组合结果、父结果复现诊断、访问披露与文件哈希。

服务器结果归档与本地下载的 SHA-256 均为
`f6475e4af1487fb6851b614ad35526abca6e84f80a2bcf7ca4c54d37a4f5481f`。本地重新运行汇总器得到与
服务器汇总目录逐字节相同的文件。
