# 自然多源候选池预筛选实验

## 实验状态

本目录记录 2026-09-05 完成的跨域混合自然池 pilot。它覆盖一个由 21 个源数据集组成的候选集合、
7 个独立目标数据集和 4 个冻结编码器，但尚未覆盖 `PLAN.md` 要求的四组独立自然候选集合，因而
不能标记为完整 P4。

- 源池：21 个数据集，每个训练 split 最多分层采样 5,000 个样本。
- 目标：Beans、DTD、EuroSAT、Flowers102、Food101、GTSRB、Oxford Pets。
- 编码器：ResNet-50、ViT-B/16、CLIP ViT-B/32、DINOv2 ViT-B/14。
- Stage-1：每池无标签抽取 1,000 个特征，`top-k=20`，短名单预算为 3、5、10。
- Stage-2 效用测量：每个源池分层抽取 500 个样本，宽度 256，训练 600 步，5 个 adapter seeds。
- 目标选择：目标 train split 内部按类别划分 80%/20% 训练/验证，最终只在官方 test split 上报告。
- 每个编码器穷举 `21 sources x 5 seeds x 7 targets = 735` 行结果，不使用 test 指标选择源池。

Stage-1 不读取源标签、目标特征或目标标签。源标签仅用于 Stage-2 的固定预算分层抽样，目标标签
仅用于 Stage-2 探针训练和验证。CIFAR-100 与 CIFAR-100-Coarse 共享原始图像，不能在后续统计中
视为独立 source family。

## 运行与完整性

- 服务器结果目录：`/data/Paper06/results/two_stage_natural_shortlist/`。
- 源特征缓存：4 个编码器各 21 个，共 84 个 `.npz`；目标缓存为 4 x 7 x 2，共 56 个 `.npz`。
- 四个 `adaptation_results.csv` 均为 736 行（含表头）。
- 最终筛选脚本 SHA-256：`09ea5fe8645a88401e5c895b0a39ebffc37995bca33a63c4de3d9b47b992d153`。
- DINOv2、CLIP、ResNet-50 重建清单前后的 `selections` 哈希完全一致。
- 对 CIFAR-10、PathMNIST、Rendered-SST2 执行四编码器复跑，12/12 个特征文件逐字节一致。
- NPU 4 被其他任务占用且未触碰；ViT 剩余特征在 NPU 5–7 分片完成，Stage-2 使用 NPU 5。

ViT-B/16 的 `aten::_transform_bias_rescale_qkv` 在当前 CANN/PyTorch-NPU 环境回退到 CPU，约
29–40 images/s。因此这里的时间是混合 CPU/NPU 墙钟，不应表述为纯 NPU 吞吐。

## 主要结果

下表对 4 个编码器、28 个 encoder-target 对取平均。`通过编码器` 表示该编码器同时满足最佳候选
召回率至少 90% 且移除至少 50% 候选。

| L | 移除比例 | DPP-subspace Recall / 通过编码器 | rank-only Recall / 通过编码器 | Full-rank Recall / 通过编码器 | Random-100 Recall |
|---:|---:|---:|---:|---:|---:|
| 3 | 85.7% | 50.0% / 0/4 | 64.3% / 0/4 | 50.0% / 0/4 | 14.5% |
| 5 | 76.2% | 57.1% / 1/4 | 75.0% / 1/4 | 75.0% / 1/4 | 26.1% |
| 10 | 52.4% | 82.1% / 2/4 | 82.1% / 2/4 | 78.6% / 1/4 | 48.8% |

DPP-subspace 相对 rank-only 的 28 个 encoder-target 配对结果：

| L | 最佳候选召回差 | 验证 regret 差 | 召回胜/平/负 | regret 胜/平/负 |
|---:|---:|---:|---:|---:|
| 3 | -14.29 个百分点 | +0.000149 | 1/22/5 | 3/20/5 |
| 5 | -17.86 个百分点 | +0.001096 | 1/21/6 | 2/19/7 |
| 10 | 0.00 个百分点 | 0.000000 | 0/28/0 | 0/28/0 |

差值定义为 DPP 减 rank-only；召回越高越好，regret 越低越好。L=10 时 DPP 仅在 DINOv2 和
CLIP 上通过门槛，ResNet-50 与 ViT-B/16 的召回分别为 57.1% 和 71.4%。rank-only 在相同两个
编码器上也通过，且总体指标与 DPP 相同。

`test_regret_vs_exhaustive_validation_selection` 允许为负：oracle 按验证准确率定义，不是按测试集
定义；短名单内部按验证集选出的候选偶尔会有更高测试准确率，这不表示使用测试集获得了改进。

## 当前结论

该 pilot 显著优于随机选择，但未达到 P4 的 90% 最优候选召回门槛，也未显示 alignment 信息
相对简单 effective-rank 排序的稳定增益。按预注册停止规则，不应基于这组结果扩大 P5 adapter
消融或启动 E6 LoRA，也不能声称该方法能够安全地移除一半候选池。

两个额外目标内部划分复跑了全部 8 个 encoder-split 组合。三个划分的 DPP Recall@10 分别为
82.1%、82.1%、75.0%，rank-only 分别为 82.1%、82.1%、78.6%；负面结论保持不变。更重要的
是，只有 5/28 个 encoder-target 在三个划分上共享 tie-aware 最优源池，说明当前 Stage-2 oracle
本身不稳定。完整诊断见相邻的 `results/two_stage_split_sensitivity/README.md`。

尚需完成的证据包括：四组各至少 20 个候选的独立自然集合、第 8 个目标、source-family-aware
LODO/cluster bootstrap、等字节通信扫描，以及固定总预算与固定每池预算协议。若后续仅使用现有
21 池，应将结果定位为跨域混合 pilot 和负面结果，而不是完整自然池主实验。

## 文件

- `cross_encoder_summary.csv`：跨编码器、目标和 Random-100 replicate 的聚合结果。
- `dpp_vs_rank_paired.csv`：每个 encoder-target-budget 的配对差值。
- `dpp_vs_rank_summary.csv`：配对差值的预算级汇总。
- `aggregation_manifest.json`：输入文件和聚合脚本哈希。
- 各编码器子目录：筛选清单、资源清单、穷举适配结果、明细与汇总、运行日志。
