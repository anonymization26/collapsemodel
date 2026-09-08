# 四编码器特征确定性复跑

2026-09-05 在独立输出目录中重新提取 CIFAR-10、PathMNIST 和 Rendered-SST2 的训练特征，
每个数据集最多 5,000 个样本，覆盖 ResNet-50、ViT-B/16、CLIP ViT-B/32 和 DINOv2
ViT-B/14。正式缓存未被覆盖。

服务器上使用 `cmp` 将 12 个复跑 `.npz` 与
`/data/Paper06/features/two_stage_source_splits_n5000/` 中的正式缓存逐文件比较，结果为：

- 匹配：12/12。
- 不匹配：0/12。
- 样本索引、标签、特征数组和 NPZ 容器均逐字节一致。

NPU 6 负责 ViT-B/16，NPU 7 依次负责其余三个编码器。ViT-B/16 因注意力算子 CPU fallback，
三个数据集分别耗时约 140 秒；其余编码器的单数据集耗时约 8–30 秒。`logs/` 保存模型加载、
特征形状、实际吞吐和每个输出文件的 SHA-256。
