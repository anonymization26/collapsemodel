# E2b DomainNet 固定成本 shortlist

- 主方法：`target_a`
- 主 shortlist：`227/455`
- 平均真实 top-10 recall：`0.983`
- 平均验证选择测试遗憾：`0.001%`
- 平均组合缩减：`50.110%`
- H3：`PASS`

## 同信息基线核查

- Second-moment MMD 的平均真实 top-10 recall：`0.983`
- Second-moment MMD 的平均验证选择测试遗憾：`0.001%`
- 两者验证选中组合相同：`12/12` 个编码器-目标域任务
- Second-moment MMD 的 H3：`PASS`

该结果以六个目标域为统计单位并在域内平均两个评价编码器。它是独立 DomainNet 上的探索性 E2b，不改变 PACS/Office-Home E2 的 H2/H2b 失败结论。H3 是主方法的绝对可用性门槛，不是相对优势检验；同信息 MMD 得到相同的主指标和最终选择，因此当前结果支持目标条件二阶几何预筛选这一方法类别，不支持 Target A 的独特优势。
