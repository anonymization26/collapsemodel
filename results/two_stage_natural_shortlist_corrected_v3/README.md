# corrected_v3 自然池标签盲筛选

本目录保存 21 个自然来源池在 ResNet-50、ViT-B/16、CLIP ViT-B/32 和 DINOv2-B/14 下的冻结
筛选 manifest。第一阶段每个来源使用标签无关随机样本和等来源权重，manifest 绑定模型状态、
预处理、来源文件、采样索引、依赖脚本与参数的 SHA-256。

四个编码器的前 10 个 rank-L 决策均无认证步骤，40/40 个选中前缀区间覆盖精确分数。rank-L 与
Full-Gram greedy 的有序前缀在大小 3、5、10 时均不完全一致；该指标比较顺序，不能据此断言对应
集合一定不同。各编码器的平均绝对 log 误差/平均理论界分别为：CLIP `1.839/4.714`、DINOv2
`2.077/5.756`、ResNet-50 `2.671/6.339`、ViT-B/16 `2.049/5.540`。

本目录只包含第一阶段 manifest。严格 repeated-CV、冻结选择和 held-out 结果位于
`../two_stage_repeated_cv_utility_corrected_v3/`。

```bash
PROJECT_ROOT="$PWD" \
SOURCE_DIR=/path/to/unlabeled/source/features \
RESULT_ROOT=/path/to/output \
bash scripts/run_two_stage_corrected_screen.sh resnet50 vit_b16 clip_b32 dinov2_b14
```
