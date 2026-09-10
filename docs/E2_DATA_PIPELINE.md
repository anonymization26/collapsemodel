# E2 可核验数据与冻结特征流水线

## 目标

E2 只接受能从官方发布物追溯到逐样本冻结表示的真实数据。旧实验中的单域
DomainNet 特征不满足本协议：它不包含完整多域任务、逐样本 ID 和统一的编码器预处理，
因此只能作为格式探索，不能作为 E2 证据。

首轮门控数据集为 PACS 和 Office-Home，编码器为 ImageNet 监督预训练 ResNet-50
与 DINOv2 ViT-B/14。所有原始数据、发布包和大体积特征只能写入服务器的
`/data` 分区，不得写入根分区或提交到 Git。

## 权威来源

- PACS：作者项目页
  <https://dali-dl.github.io/project_iccv2017.html>，并由 SketchX 官方下载页链接；
- Office-Home：作者数据页
  <https://www.hemanthdv.org/officeHomeDataset.html>；
- Office-Home 页面给出非商业研究和教育用途的 fair-use notice；
- PACS 页面未给出明确软件式许可证，实验收据必须如实记录这一点，不得自行补写许可证。

官方页面没有发布文件 SHA-256。下载后必须先保留原始发布包，再由收据生成器测量文件
大小和 SHA-256。不能只保留解压目录，也不能用第三方镜像名称替代官方来源。

## 目录

```text
/data/Paper06/e2/
  downloads/                 # 未修改的官方发布包
  datasets/                  # 解压后的 domain/class/image 树
  receipts/                  # 来源、用途、引用和发布包哈希
  manifests/                 # manifest.json、samples.csv、assignments.csv
  features/                  # dataset/encoder/features.npz + metadata.json
  logs/
```

版本库只保存脚本、协议以及最终的小体积 manifest/metadata 副本。图像和
`features.npz` 不进入 Git。

## 数据收据

`create_target_conditioned_e2_receipt.py` 从原始发布包直接计算大小和 SHA-256。
收据至少绑定：

- 数据集名称与版本描述；
- 官方项目页和实际下载 URL；
- 每个原始发布包的文件名、字节数与 SHA-256；
- 预期域、类别数和样本数；
- 官方用途说明或“未提供明确许可证”的事实；
- 论文 BibTeX。

示例调用中的值必须由实际发布包和官方页面填写：

```bash
python3 scripts/create_target_conditioned_e2_receipt.py \
  --dataset DATASET \
  --dataset-version OFFICIAL_RELEASE \
  --official-page-url HTTPS_URL \
  --download-url HTTPS_DOWNLOAD_URL \
  --source-artifact /data/Paper06/e2/downloads/ARCHIVE \
  --expected-domain DOMAIN_A \
  --expected-domain DOMAIN_B \
  --expected-class-count CLASS_COUNT \
  --expected-sample-count SAMPLE_COUNT \
  --usage-summary "VERBATIMLY_CHECKED_SUMMARY" \
  --usage-terms-url HTTPS_TERMS_URL \
  --citation-key CITATION_KEY \
  --citation-bibtex-file /data/Paper06/e2/receipts/citation.bib \
  --output /data/Paper06/e2/receipts/DATASET.json
```

## Manifest 与无标签划分

数据树必须是 `domain/class/image`。构建器逐图验证可解码性，记录相对路径、字节数、
内容 SHA-256、稳定样本 ID、域和审计用类别。任何绝对路径都禁止进入产物。

构建器另生成并绑定 `content_duplicates.csv`，明确报告域内、跨域和跨类别的完全相同内容。
为防止直接泄漏，当某域作为目标时，其他源域中与该目标域字节完全相同的样本统一标为
`excluded_target_duplicate`，不得进入候选块；目标域内重复内容仍共享同一目标划分。
跨类别重复保留原始标注并单独报告，不能任意保留一个标签后声称仍是官方任务。目标域按内容哈希分为：

- `target_selection`：20%，U0 可见，只估计无标签目标二阶矩；
- `target_calibration`：10%，只供 U2；
- `target_test`：70%，选择冻结后才供评价。

每个非目标源域按内容哈希分为 4 个近似等大小候选块。划分函数不接收类别、损失、
预测或特征；同一域内内容完全相同的文件共享划分，避免重复内容跨目标子集泄漏。

```bash
bash scripts/run_target_conditioned_e2_manifest_server.sh \
  /data/Paper06/e2/datasets/DATASET \
  /data/Paper06/e2/receipts/DATASET.json \
  /data/Paper06/e2/manifests/DATASET \
  /data/Paper06/e2/downloads/ARCHIVE
```

输出中的 `manifest_id` 同时绑定数据树、收据、样本表、划分表和协议参数。修改 CSV
后重新计算单个文件哈希仍无法绕过协议重算。

## 冻结特征

每个编码器只按 `samples.csv` 顺序抽取一次原始池化表示：

```bash
bash scripts/run_target_conditioned_e2_features_npu.sh \
  NPU_ID \
  /data/Paper06/e2/datasets/DATASET \
  /data/Paper06/e2/manifests/DATASET \
  /data/Paper06/e2/features/DATASET \
  resnet50 dinov2_b14
```

`features.npz` 固定包含：

- `H`：`float32 [n, d]` 原始池化表示；
- `y`：`int64 [n]`，只供拟合和最终评价；
- `sample_ids`：与 manifest 完全同序的固定宽度字符串。

`metadata.json` 绑定 manifest、样本表、划分表、特征文件、模型最终状态、检查点、
预处理、抽取脚本、编码器加载脚本和 Git revision。元数据不得记录服务器绝对路径。
已有缓存只有通过完整校验才会复用；残缺或失配缓存默认停止，不会静默覆盖。

## 访问边界

一次性抽取全部表示只是确定性预处理，不代表选择算法可以读取全部缓存。后续 E2 运行器
必须通过 `assignments.csv` 创建角色视图：

- U0 选择阶段只能读取源候选的 `H` 和 `target_selection.H`；
- U0 不得读取源或目标标签；
- U2 才能读取 `target_calibration.y`；
- 选择 manifest 冻结后，评价器才能读取 `target_test.H/y`。

后续 E2 选择与评价代码需把实际访问过的 sample ID 哈希写入运行 manifest，以便证明
`target_test` 未参与选择。

## 当前完成条件

“E2 数据准备完成”必须同时满足：

1. 两个官方发布包及收据均通过哈希验证；
2. PACS 和 Office-Home manifest 均通过数据树复核；
3. 两个数据集的 ResNet-50 与 DINOv2 缓存全部通过逐样本校验；
4. 小体积 manifest 和 metadata 已复制回版本库并复核无身份、主机地址和绝对路径；
5. 才能开始 E2 方法比较，且不能把数据准备本身写成 H2 的经验结果。

## 完成记录

截至 2026-09-11，上述五项准备条件均已满足：PACS 与 Office-Home 的 manifest 在服务器和
版本库副本上均验证为 `valid`，四份冻结特征缓存通过了样本顺序、形状、dtype、manifest 绑定
和文件哈希复核。版本库副本与服务器原件逐文件 SHA-256 一致，且未包含身份、主机地址或绝对路径。

可提交产物位于 `results/target_conditioned/manifests/` 和
`results/target_conditioned/e2_dataset_selection/frozen_features/`；机器可读汇总见
`results/target_conditioned/e2_dataset_selection/artifact_index.json`。此记录仅解除方法实验的
数据门禁，不代表 E2 或 H2 已经通过。
