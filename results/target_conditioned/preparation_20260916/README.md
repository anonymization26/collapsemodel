# E2b 准备核验快照

这些文件是 2026-09-16 从计算服务器取回的准备阶段记录，不是新一轮真实数据对比结果。

- `environment_check.json`：三个编码器的权重、模型状态和预处理指纹匹配旧记录，CUDA 合成前向通过；记录包版本、数值设置和 DINOv2 源码哈希。
- `e2b-tests.log`：16 项测试通过；其中端到端结果来自合成小型 fixture，不是 DomainNet 实验。
- `preparation_status.json`：取回时的状态快照，`running/restore` 表示数据恢复尚未完成，不是实时状态。

截至该快照，PACS、Office-Home 历史缓存已恢复且四份文件 SHA-256 匹配；DomainNet 源分片
累计约 6.73 GB 已完成下载分块，全部约 18.52 GB。图像恢复、三份新特征缓存和六个阶段数据包
均须由后台流水线继续完成。最终以服务器 `preparation_report.json` 的全量核验结果为准。

快照对应的基础 Git 版本为 `4bb2f5027212bf6921d409da2d1d9560822f8cc3`，部署时准备代码含未提交改动，
不能把该版本号误认为全部准备源码的提交标识；后续提交不改变历史运行记录。完整操作与限制见
[准备说明](../../../docs/E2B_CUDA_PREPARATION.md)。
