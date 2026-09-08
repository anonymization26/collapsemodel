# 四编码器 Ascend NPU 基准

## 协议

- 数据：CIFAR-10 官方训练集中的固定 10,000 样本。
- 硬件：单张 Ascend 910B4，batch size 64，4 个 DataLoader workers。
- 精度：FP32 inference，不保存特征。
- 预处理：每个 checkpoint 对应的官方或发布方评估预处理。
- 指标：模型加载时间、纯推理时间、吞吐和 NPU 峰值内存。

## 结果

| 编码器 | 特征维度 | 10k 推理时间 | 吞吐 | 峰值分配/保留 HBM |
|---|---:|---:|---:|---:|
| ResNet-50 supervised | 2,048 | 12.668 s | 789.40 img/s | 845/1,016 MB |
| ViT-B/16 supervised | 768 | 304.556 s | 32.83 img/s | 882/1,192 MB |
| CLIP ViT-B/32 | 512 | 55.180 s | 181.23 img/s | 745/824 MB |
| DINOv2 ViT-B/14 | 768 | 15.136 s | 660.68 img/s | 946/1,122 MB |

四个作业的脚本 SHA-256 均为
`53cd670916952e78ea29c2f19270bd0cf315e1d44a2b15f19dd982985b1796a4`。

## 解释限制

torchvision ViT-B/16 和 OpenCLIP 在当前 CANN/torch_npu 组合上会将
`aten::_transform_bias_rescale_qkv` 回退到 CPU，因此其吞吐不是纯 NPU kernel 性能。
该结果可用于本服务器的实际 wall-clock 排期，但不能宣称是 ViT 的 Ascend 原生性能。

OpenCLIP 在构造空模型时会先记录 `No pretrained weights loaded`；脚本随后从本地官方
`ViT-B-32.pt` 加载 state dict，并严格检查 missing/unexpected keys。因此该日志不表示最终模型
使用随机权重。DINOv2 未启用 xFormers，结果同样按当前实际软件栈报告。

每个 JSON 保存 checkpoint、设备、软件版本、精确时间和脚本哈希；对应 `.log` 保留原始警告。
