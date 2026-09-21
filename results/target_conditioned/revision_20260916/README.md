# 2026-09-16 补实验结果

本轮完成 R1 隔离输入重建复现和 R2 逐样本条件稳健性诊断。原实验结果、论文和原成功门槛未修改。
这些是已知旧测试结果后的补充分析，不是独立新数据集的确认性证据。

## 输入与资源

- 18:54 UTC 检查时，3 张 NVIDIA vGPU-32GB 均为 0% 利用率，显存各占用 1 MiB，无计算任务。
- 容器配额为 36 CPU 核、270 GiB 内存，数据盘剩余约 127 GiB。
- [准备验收](preparation_report.json)为 `ready`：46,080 张 DomainNet 图像逐张字节数和 SHA-256 匹配，三份 CUDA 冻结特征和六个角色数据包已验收。
- 原候选成员、角色划分和全部参数保持不变，配置 SHA-256 为 `5d405f4e3d9036d056f9c0f32d693c019e93f949f1c885fa3dcac694ffee1ed5`。
- R1/R2 基线为 `566679a0ee59ace27f6da7ed44fbd18ebe6cd5ad`，新增源码另有实际 SHA-256。准备任务原基线仍为 `4bb2f502...`，未回写。
- 实验使用冻结特征，两个单线程 BLAS CPU 进程并行，不重复执行 GPU 特征提取。

## R1：重建结果可复现

19:02:17 至 19:05:33 UTC，约 3 分 16 秒。每编码器：screen 约 8 秒，validation 约 166 秒，test-audit 约 21 秒。
共 12 个编码器/目标域任务，每任务 455 个固定训练样本成本组合。所有方法独立评估自己的验证 shortlist。

| 主指标，shortlist=227 | Target-A | second-moment MMD |
| --- | ---: | ---: |
| 经验 top-10 recall | 98.333% | 98.333% |
| 平均所选组合 normalized regret | 0.001320% | 0.001320% |
| 组合验证次数减少比例 | 50.110% | 50.110% |

两方法在 12/12 任务中仍选择相同组合。因此，这次复现没有提供 Target-A 优于同信息 MMD 基线的新证据。
验证次数减少比例不等于完整端到端运行时间加速比例。

与旧 NPU 结果比较：

- 12/12 任务的测试 top-10 集合完全一致；全组合测试排序 Spearman 最低为 0.999992。
- 最大逐组合 Brier 差异为 `6.46e-6`；最大准确率差异为 0.390625 个百分点。
- 1,620 条验证选择记录中有 7 条变化，全部列于 [选择差异](reconstruction_comparison/selection_changes.csv)。Target-A 的验证选择记录没有变化。
- 因此可说主要结果稳定，不能说 CUDA/NPU 特征或所有预测逐位一致。

完整 [R1 汇总](revision_results/e2b_isolated_cuda_v1/summary/README.md)、
[逐任务重建对照](reconstruction_comparison/per_task.csv)和输入/输出哈希均已取回。
自动 R1 汇总沿用原 H3 报告模板；其中原有 “independent dataset” 指 DomainNet 相对于旧 E2 的地位，
不表示本次重复使用同一 DomainNet 划分构成新的独立验证。

## R2：排序稳定性与指标依赖

19:09:20 至 19:09:55 UTC，含配对核验约 35 秒。
6 个目标域各固定 2,000 次测试样本 bootstrap；每域内的全部组合、两个编码器、四种指标共用相同权重。
固定训练集和 ridge 读出，不在重采样中重新训练，不改 shortlist 或超参数。
逐样本均值对 R1 聚合结果的最大误差为 `1.78e-15`；[配对输入核验](revision_results/e2b_loss_diagnostics_v1/pairing_verification.json)通过。

下表为 12 个任务的等权描述性平均；两个编码器不被当作额外独立目标域，未据此计算跨域显著性。
“集合稳定性”是重采样 top-10 与原经验 top-10 的交集比例，不是 shortlist recall。

| 指标 | top-10 集合稳定性均值 | 各任务稳定性均值范围 | Target-A bootstrap recall，S=227 | MMD bootstrap recall，S=227 |
| --- | ---: | ---: | ---: | ---: |
| Brier | 85.154% | 77.630%-91.890% | 97.420% | 97.848% |
| 原始平方损失 | 83.276% | 71.820%-95.275% | 99.628% | 99.561% |
| NLL | 85.085% | 75.465%-95.155% | 97.432% | 97.754% |
| 错误率 | 76.189% | 59.890%-91.815% | 93.860% | 93.170% |

经验指标之间的平均排序 Spearman / top-10 重叠：

| 指标对 | Spearman | top-10 重叠 |
| --- | ---: | ---: |
| Brier / NLL | 0.9972 | 90.833% |
| Brier / 原始平方损失 | 0.8173 | 30.833% |
| Brier / 错误率 | 0.8571 | 29.167% |

解释：

1. top-10 成员具有有限测试集不确定性，但较宽 shortlist 的召回总体较稳定。这是有用的鲁棒性信息，不是重新定义 H3 门槛。
2. Brier 与 NLL 高度一致，但不能把 Brier top-10 等同于准确率或原始平方损失 top-10。论文的理论目标和评价指标之间仍需明确桥接。
3. Target-A 没有显示稳定压过 MMD 的优势；各指标上的小幅差异不能据此宣称统计显著。
4. bootstrap 只覆盖固定模型下的有限测试样本不确定性，不覆盖训练不确定性、跨数据集泛化或额外独立测试。错误率平局采用组合名词典序；CSV 同时报告边界平局数量。

## 产物与验证

本地 R2 目录为 `revision_results/e2b_loss_diagnostics_v1/<encoder>/`：

- `topq_frequency.csv`：每组合的经验指标、经验排名和重采样 top-10 入选频率。
- `topq_stability.csv`：逐域集合稳定性及条件分布的 2.5%/97.5% 分位数。
- `shortlist_recall.csv`：每方法、随机重复、shortlist 大小及指标的召回分布。
- `metric_agreement.csv`：四指标之间的排序相关和 top-10 重叠。
- `manifest.json`：输入、配置、源码、CSV 和逐样本数组的 SHA-256、样本顺序哈希及实际运行时间。

大型 `*_losses.npz` 和 `*_bootstrap.npz` 保留在计算服务器
`/root/autodl-tmp/collapsemodel_revision_20260916/revision_results/e2b_loss_diagnostics_v1/`，未下载入代码库。
本地 manifest 仍保留它们的文件名和 SHA-256。完整配对验证须在含这些数组的服务器目录运行。

[测试日志](logs/e2b-revision-tests.log)：22 项通过，包括原流程、角色隔离、准备流水线、逐样本指标一致性、
ridge 一致性、配对重采样/平局规则、上游缺失或篡改时禁止提前读取测试特征，以及合成端到端诊断。
新旧比较脚本另在全部真实输出上执行通过。

仍未执行：真正独立数据集、新读出器、异质性干预、A-opt/MMD 受控机制挑战、完整端到端成本比较、
跨投影种子稳定性。R1/R2 完成不代表论文所有补实验已经完成。
