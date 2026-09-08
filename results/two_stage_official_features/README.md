# 五数据集四编码器正式特征缓存

## 范围

本目录保存服务器正式特征缓存的元数据和原始日志，不提交体积较大的 `.npz` 特征文件。服务器缓存位于：

`/data/Paper06/features/two_stage_official_n5000`

每个编码器使用自身 checkpoint 对应的官方或发布方评估预处理，固定采样种子为 `20260902`。
采样按类别比例确定性执行；Stage-1 只读取 `H` 和原始样本索引，不读取保存的标签。

| 数据集 | 使用部分 | 原始样本 | 保存样本/编码器 | 说明 |
|---|---|---:|---:|---|
| CIFAR-10 | 官方 train | 50,000 | 5,000 | 分层采样 |
| CIFAR-100 | 官方 train | 50,000 | 5,000 | 分层采样 |
| DTD | 官方 train partition 1 | 1,880 | 1,880 | 保留全部 |
| EuroSAT | full | 27,000 | 5,000 | 数据集无官方 train/test split，固定分层采样 |
| SVHN | 官方 train | 73,257 | 5,000 | 从本地 Hugging Face Arrow 读取 |

每个编码器共 21,880 个向量，四个编码器合计 87,520 个向量。特征维度分别为 ResNet-50
2,048、ViT-B/16 768、CLIP ViT-B/32 512、DINOv2 ViT-B/14 768。

## 提取时间

| 编码器 | 五数据集纯推理总时间 | 备注 |
|---|---:|---|
| ResNet-50 | 39.87 s | 官方 ImageNet V2 预处理 |
| DINOv2 ViT-B/14 | 50.18 s | 发布方 ImageNet eval 预处理 |
| CLIP ViT-B/32 | 122.91 s | OpenCLIP 官方预处理 |
| ViT-B/16 | 639.51 s | attention 算子回退 CPU |

这些时间包含逐数据集 DataLoader 和推理，不包含一次性的模型加载。各数据集的精确时间、HBM、
checkpoint、preprocess hash 和 feature-file hash 位于 `feature_extraction_summary.csv`。

## 审计信息

- 提取脚本 SHA-256：`1a355698c9749780d56737370eff87df608c0b3d5bf732803a578498b1760bf0`
- 编码器加载脚本 SHA-256：`53cd670916952e78ea29c2f19270bd0cf315e1d44a2b15f19dd982985b1796a4`
- `metadata/`：20 个逐缓存 JSON，含原始路径、shape、预处理文本和文件哈希。
- `logs/`：四个编码器的完整提取日志和后端警告。

旧缓存 `/data/Paper06/features/mv_npu` 对不同编码器统一使用 ImageNet normalization，不能作为
CLIP 等模型的正式输入；后续 E2/E3 仅使用本目录记录的新缓存。
