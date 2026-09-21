# R3/R4：读出模型与投影种子审计

本轮由用户要求检查服务器环境并补实验后启动。时间均为 UTC；启动为 2026-09-17 20:55:30。
原始 DomainNet 结果及 R1/R2 已知，因此本轮是探索性稳健性实验，不是独立数据集确认。
2026-09-19 已将本轮结果整合到 `paper/iclr2027/` 中英文稿及摘要，单列为探索性补实验；
没有更改旧实验的门槛或替换原主实验表。

## 固定协议

- 数据、候选成员、角色划分和源样本成本与原 E2b 相同；每目标 455 个组合，每组合 1,536 张源图像。
- 两编码器：ResNet-50、DINOv2-B/14；六目标域。
- 原投影种子 20260911；新种子 20260917、20260918、20260919、20260920，均为 128 维。
- 两读出：原始 ridge-softmax；C=1、无截距、LBFGS 的多项逻辑回归，不使用测试结果调参。
- 保留全部八个方法族、20 次随机排名和五种 shortlist 大小。
- 每指标分别使用同名验证损失选择；Brier 为主，NLL 和错误率为辅助终点。
- 10 个任务、54,600 次拟合及对应的冻结模型测试评估。

完整配置见 [readout_projection_v1.json](../../../code/configs/target_conditioned_e2b/readout_projection_v1.json)，
设计与停止规则见 [补实验计划](../../../docs/ICLR_SUPPLEMENT_PLAN.md)。

## 访问与成本边界

筛选、验证和测试分别读取隔离数据包。验证保存模型及选择后，测试入口验证源码、配置、输入
谱系、模型及选择的哈希，再读取测试特征；测试阶段不重新训练或选择超参数。
各方法在此审计中复用同一组合的验证计算。结果是穷举数据上的 shortlist 反事实审计，
不能把本轮运行时间解释为真实部署每方法独立执行的端到端成本。

## 完成与验收

全部 10/10 任务已完成，结束时间为 2026-09-17 21:13:37 UTC，实际耗时 18 分 7 秒。
54,600 次拟合和对应的测试评估均完成；逻辑回归最大迭代数为 24，没有收敛失败。
36 项相关测试在实际服务器环境通过，包含 9 项新入口测试和 5 项汇总统计测试。
原投影种子的两个编码器均通过与 R1 的核验：
5,460 个测试组合最大损失差异 1.78e-15；1,620 条 Brier 验证选择完全一致。
原种子的 ridge 属于实现对照，不计作新增独立证据。

本轮原管理 PID：30997（任务已结束）。服务器输出目录：
`/root/autodl-tmp/collapsemodel_revision_20260916/revision_results/e2b_readout_projection_v1/`。
源码使用隔离快照 `supplement-code-20260917-7yowpv/`；基线提交及实际新增源码哈希均存入 `status.json`。

环境：3 张 NVIDIA vGPU-32GB，容器 36 核 CPU、270 GiB 内存；启动前剩余磁盘约 127 GiB。
Python 3.12.3、NumPy 1.26.4、SciPy 1.16.1、scikit-learn 1.7.1、PyTorch 2.5.1+cu124。
使用三个单线程 CPU worker，不重抽特征，不占 GPU 计算资源。

## 四个新种子的结果

以下为新增四个种子、两编码器的域内平均，再对六个目标域取平均；每行涉及 48 个重复任务，
不是 48 个独立数据集。shortlist 固定为 227/455；数值为对应测试损失 top-10 的召回率。

| 读出 | 指标 | A-opt | 二阶 MMD | 随机排名 |
| --- | --- | ---: | ---: | ---: |
| ridge | Brier | 98.33% | 99.38% | 50.12% |
| 逻辑回归 | Brier | 98.33% | 98.96% | 48.97% |
| ridge | NLL | 98.54% | 99.38% | 49.86% |
| 逻辑回归 | NLL | 98.54% | 99.79% | 48.61% |
| ridge | 错误率 | 96.25% | 97.50% | 48.63% |
| 逻辑回归 | 错误率 | 97.29% | 97.92% | 49.04% |

八个方法族、全部 shortlist 大小和逐种子结果均在完整 CSV/JSON 中，不限于上表三种方法。
作为主指标补充，Brier 下的其他方法召回率为：

| 方法 | ridge | 逻辑回归 |
| --- | ---: | ---: |
| Bayesian D-opt | 82.29% | 86.46% |
| 合并有效秩 | 80.42% | 85.42% |
| 子空间 DPP | 28.75% | 35.42% |
| 域平衡 | 26.46% | 31.67% |
| 目标能量 | 41.04% | 33.96% |

原种子单独对照：ridge 的 A-opt/MMD 均为 98.33%；逻辑回归为 98.33%/99.17%。

## 对论文的意义

1. 高候选召回不只出现在单个投影或 ridge-softmax 读出上；本轮也观察到了错误率与 NLL 的高召回。
2. 新种子上的平均相对 Brier 候选跨度为 ridge 的 0.248%、逻辑回归的 14.652%。
   后者仍有高召回，因此现象不局限于原来损失差异很小的设置；这不等于已解释校准或优化的因果机制。
3. A-opt 与 MMD 的平均验证所选 Brier regret 相同：ridge 为 oracle Brier 的 0.001006%，
   逻辑回归为 0.102830%；两个方法的 Brier shortlist omission 在全部新种子任务上均为零。
   实际性能增益仍需按任务容忍差距与成本衡量，不能只凭召回或 regret 比例认定值得部署。
4. 这些结果不支持 A-opt 相对 MMD 的优势。MMD 在上述三类指标的平均召回均略高，未做新的显著性宣称。

## 本地结果索引

- [最终汇总](revision_results/e2b_readout_projection_v1/summary/summary.json)
- [四个新种子汇总表](revision_results/e2b_readout_projection_v1/summary/new_seeds.csv)
- [逐种子结果](revision_results/e2b_readout_projection_v1/summary/per_seed.csv)
- [域内重复任务明细](revision_results/e2b_readout_projection_v1/summary/unit_rows.csv)
- [环境记录](revision_results/e2b_readout_projection_v1/environment_check.json)
- [运行状态和源码哈希](revision_results/e2b_readout_projection_v1/status.json)
- [36 项测试日志](logs/readout-projection-regression-tests-final.log)
- [本地完整性校验](local_verification.json)

结果包 SHA-256 为 `7e4d73b357792b29c7d0d752c1a325fec718680d13e30dfbeec9d4f61492b3f6`。
本地已核验 410 个阶段数值文件、汇总输入与运行源码；120 个模型 NPZ 仍保留在服务器，
未随数值包重复下载。服务器测试入口在使用这些模型前已完成其哈希校验。

## 解释限制

种子与编码器是同一目标域内的重复测量，不是独立数据集；随机排名重复先平均，随后按目标域汇总。
原种子单列，新增种子不择优。相对 regret 分母及候选跨度为零时记为不可定义。
本轮不能补齐独立数据集、候选异质性因果干预、验证预算或匹配端到端成本的证据缺口。
