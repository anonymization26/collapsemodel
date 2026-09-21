# E2b 修订实验准备

本文件记录数据、环境和执行代码的准备，不修改已有冻结结果。
早期记录见 [准备快照](../results/target_conditioned/preparation_20260916/README.md)，保留其取回时状态，不回写为完成。
2026-09-16 15:32 UTC 准备已验收完成；随后按用户指示执行的 R1/R2 补实验及最终验收记录见
[本轮结果](../results/target_conditioned/revision_20260916/README.md)和[补实验计划](ICLR_SUPPLEMENT_PLAN.md)。

## 当前状态

- [x] 新建专用虚拟环境，复用 PyTorch 2.5.1+cu124 / torchvision 0.20.1+cu124。
- [x] 在虚拟环境中隔离安装 NumPy 1.26.4、open_clip_torch 3.3.0、timm 1.0.9，并通过兼容性检查。
- [x] 恢复 PACS、Office-Home 的原始压缩包、manifest 和四份特征缓存。
- [x] 恢复 CLIP 与 DINOv2 权重及本地 DINOv2 源码。
- [x] CUDA/CPU/NPU 共用一个特征提取实现，保留旧 `--npu` 入口。
- [x] 实现筛选、验证和测试分别读取独立数据包的执行入口。
- [x] 16 项测试通过：原有 6 项 E2b 测试、7 项阶段隔离测试和 3 项准备流水线测试，包含合成端到端对照。
- [x] 三个模型的权重、模型状态、预处理指纹和 CUDA 前向检查全部通过。
- [x] 可断线运行的准备流水线完成下载、特征提取和验收。
- [x] 按冻结清单恢复并逐张验证 46,080 张 DomainNet 图像的字节数及 SHA-256。
- [x] 重建三份 CUDA 特征缓存，并生成六个角色隔离数据包。
- [x] 完成准备状态总检查，保存 [ready 验收报告](../results/target_conditioned/revision_20260916/preparation_report.json)。

## 目录与环境

计算端工作目录为 `/root/autodl-tmp/collapsemodel_revision_20260916`。基础 Git 版本为
`4bb2f5027212bf6921d409da2d1d9560822f8cc3`；服务器任务启动时，准备源码包含未提交改动，
运行报告另记录实际文件 SHA-256。后续代码提交不回写既有任务的基线记录；不要把基础 Git
版本误写成已包含本轮改动的提交。

目录约定：

- `.venv/`：实验专用环境。
- `dataset/`：按原始样本清单恢复的图像。
- `downloads/`：固定版本 Parquet 分片及断点续传记录。
- `weights/`、`dinov2_repository/`、`torch_cache/`：冻结模型资源。
- `legacy_e2/`：PACS、Office-Home 的历史数据和缓存，不作为新独立验证集。
- `reconstructed_features/`：本轮 CUDA 缓存，不覆盖仓库中的历史元数据。
- `stage_bundles/<encoder>/<stage>/`：隔离后的工作输入。
- `logs/`、`environment_check.json`、`preparation_report.json`：日志和核验记录。
- `preparation_status.json`：后台流水线的当前阶段、PID、完成步骤与错误记录；`ready` 才表示全部验收通过。

## 数据恢复与可复现性

原始 NPU 缓存未在现有服务器找到。恢复过程保持原 `samples.csv`、类别子集、角色划分和
候选样本 ID 不变，不根据旧测试结果重新采样。Parquet 来源由仓库配置固定版本、大小和内容标识；
最终每张图像都必须与冻结清单的字节数和 SHA-256 一致。

CUDA 特征要求权重文件、加载后模型状态和预处理指纹匹配原元数据。禁用 CUDA TF32、xFormers，
指定 SDPA math 后端，不使用自动混合精度。跨设备浮点差异仍然可能存在，因此新缓存使用新哈希；不能声称与旧 NPU
缓存逐位一致。后续需要单独报告重建前后的排序和结果差异。
环境报告还保存实际包版本、模型结构描述和 DINOv2 源码指纹；由于旧记录没有覆盖所有这些信息，
通过权重/状态校验不等于已经证明跨框架版本的完整功能等价。

基础镜像存在 NumPy 混装：`numpy/core` 与 `numpy/_core` 下的二进制扩展给出不同的
`ndarray` 类型，导致 PyTorch 转换失败。准备环境需在虚拟环境中安装干净的 NumPy，
而不是仅依赖基础环境报告的版本号。环境检查首先执行 NumPy/PyTorch 往返转换。
OpenCLIP 2.26.1 不符合原预处理结构；3.3.0 的
[官方预处理实现](https://github.com/mlfoundations/open_clip/blob/v3.3.0/src/open_clip/transform.py)
包含原记录中的 `MaybeConvertMode` 和 `MaybeToTensor`，仍须通过实际指纹检查后才能用于重建。

候选构造保持原 `candidates.json` 的成员列表，不因 CUDA 上 CLIP 特征的数值差异重新计算分层。
新计算的 CLIP 缓存用于构造敏感性等后续实验，不冒充原候选构造时使用的缓存。

## 阶段访问边界

`prepare_target_conditioned_e2b_stage_bundles.py` 是可读取完整数据的离线准备步骤。
实验入口 `run_target_conditioned_e2b_isolated.py` 不接收原始完整缓存、原始 manifest 或 anchor 路径。

| 阶段 | 数据包内容 |
| --- | --- |
| screen | 候选源特征与 target-selection 特征，无标签数组，无类别名或图像路径 |
| validate | 候选源特征及标签、target-validation 特征及标签，无 target-test 样本 |
| test-audit | 候选源特征及标签、target-test 特征及标签；先验证上游冻结产物，再加载特征 |

这是一项经过测试的文件访问约束，不是操作系统权限沙箱。旧入口为归档数值对照保留，
其 `input_isolation` 明确标记为 `legacy-full-cache`；新实验必须使用隔离入口。

## 准备命令

在具有上述 CUDA 基础环境的新工作目录中创建专用环境；单独安装 NumPy 以避开基础镜像混装：

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --ignore-installed --no-deps numpy==1.26.4
.venv/bin/python -m pip install -r code/requirements-e2b-cuda.txt
```

在工作目录内执行，不在原历史结果目录中生成新输出：

```bash
export SOURCE_GIT_REVISION=4bb2f5027212bf6921d409da2d1d9560822f8cc3
bash scripts/prepare_target_conditioned_e2b_cuda.sh check
bash scripts/prepare_target_conditioned_e2b_cuda.sh restore
bash scripts/prepare_target_conditioned_e2b_cuda.sh features
bash scripts/prepare_target_conditioned_e2b_cuda.sh bundles
.venv/bin/python scripts/check_target_conditioned_e2b_preparation.py \
  --work-root "$PWD" --output preparation_report.json
```

`restore` 可复用相同分块配置的下载断点；不要在断点存在时更改 `chunk-mib`。
`features` 将 ResNet、DINOv2、CLIP 分配到 GPU 0、1、2。完整特征缓存存在时先校验再复用；
不完整缓存和已有阶段数据包不会被静默覆盖。

也可串行运行准备流水线；内部 `features` 阶段使用三张 GPU 并行。它持有进程锁，任何步骤失败
立即停止并记录错误，不会自动重启反复消耗算力，也不会调用真实数据上的筛选或下游测试：

```bash
export SOURCE_GIT_REVISION=4bb2f5027212bf6921d409da2d1d9560822f8cc3
.venv/bin/python -u scripts/run_target_conditioned_e2b_preparation.py --work-root "$PWD"
```

服务器使用 `nohup` 启动的上述流程已完成，日志为 `logs/preparation-pipeline.log`；
已通过核验的缓存和数据包直接复用，不重复启动 `restore` 或 `features`。
`environment_check.json` 通过仅代表环境可用，不能代替最后的 `preparation_report.json`。

正式实验前还需冻结新增基线、读出器、独立数据集和统计方案。本轮准备完成不代表这些科学问题
已经验证，也不代表原先失败的假设已转为成功。
