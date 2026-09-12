# E2b：固定样本成本的独立 shortlist 实验

> 状态：协议已在读取 DomainNet 结果前冻结，完整实验已于 2026-09-12 完成。E2b 是 E2 失败后的新探索性假设，不回写或替代 E2 的 H2/H2b 结论。

## 1. 要回答的问题

E2a 的 12 个哈希块只形成较弱的候选差异，而且所谓 top-10 指标实际是“单个选择是否命中 top-10”，并不是真正的 shortlist 召回。E2b 改为回答更窄且可证伪的问题：

1. 在每个组合训练样本数完全相同的条件下，无标签几何分数能否排除至少一半组合？
2. 被保留的 shortlist 是否覆盖独立测试集上的真实 top-10 组合？
3. 只在 shortlist 中用目标验证标签选出的组合，是否在独立测试集上接近穷举 oracle？

## 2. 独立数据与固定成本

- 数据集：DomainNet 2019 官方推荐的 cleaned 版本，六个域 `clipart/infograph/painting/quickdraw/real/sketch`。Clipart/Painting 使用 `groundtruth` 归档，六域 train/test 划分均固定为 `domainnet/txt`，下载和 manifest 构建均强制核对数据仓库公布的 MD5，并在 manifest 中另记 SHA-256；这一来源修正在特征提取及任何结果读取前完成，不改变样本配额、候选构造或成功门槛。
- E2a 的 PACS 和 Office-Home 只视为开发数据，不参与 E2b 调参或统计。
- 从 345 类中按类名的固定 SHA-256 顺序取 100 类；规则不读取样本数量、特征或结果。
- 每个域从官方 train 划分中固定抽取 1,536 个 `target-selection`、1,024 个 `target-validation` 和 3,072 个 `anchor-pool` 样本，三者不重叠。
- 每个域从官方 test 划分中固定抽取 2,048 个 `target-test` 样本。该集合只在 screen 和 validation 产物冻结后打开。
- 每个候选块恰有 512 个样本，每个组合恰选 3 块，因此每次 ridge 拟合恰用 1,536 个源样本。

样本按路径哈希确定。无效图像或完全相同内容按配置中的固定顺序跳过并补足；若任一域无法满足配额，实验中止，不事后降低配额。

## 3. 扩大候选异质性

候选构造使用不参与最终评价的 CLIP ViT-B/32 作为锚编码器：

1. 对每个域的 3,072 个 anchor 样本做行归一化和域内中心化。
2. 计算协方差最大特征向量，并固定其符号。
3. 按该方向投影稳定排序，等分为 low、middle、high 三层。
4. 每层再按独立固定哈希取 512 个样本，形成三个等大小块。

当某域作为目标时，其余五域各提供三个块，共 15 个候选块。固定预算 `K=3` 共有 `C(15,3)=455` 个组合。锚模型不读取类别标签、目标域信息或测试结果，且不作为评价编码器。

## 4. 三阶段访问边界

### Screen：U0

仅可读取源候选特征和 `target-selection` 特征。ResNet-50 与 DINOv2 表示先行归一化，再通过固定 Rademacher JL 矩阵投影到 128 维并再次归一化。所有 455 个组合都由下列分数产生完整排序：

- Target A-opt；
- 二阶矩 MMD；
- Target Energy；
- Bayesian D-opt；
- 合并有效秩；
- DPP Subspace；
- 域平衡；
- 20 次随机排序。

shortlist 大小固定为 `10/25/50/100/227`。`227` 是主设置，刚好排除超过一半的 455 个组合。Collapse-4S 是顺序式两块合并预测器，没有定义 455 个三块集合的全序；E2b 不为它临时发明有利的 set score，而用其实际几何目标“合并有效秩”作为可穷举基线。

### Validate：U2

验证阶段先校验 screen 文件及配置哈希，然后才读取源标签和 `target-validation` 标签。每个方法（以及每个随机重复）独立评估自己排序中的 top-227，较小的 `10/25/50/100` shortlist 只复用该排序内已经计算的嵌套前缀；不同方法之间不共享验证结果。对每个 shortlist 内的组合使用同一个正则强度 `lambda=1` 拟合闭式 ridge，并按目标验证 Brier score 选择一个组合。因此主方法 Target A 在每个目标域物理上恰评估 227 个而非接近全部 455 个组合。验证阶段不得加载官方 test 特征或标签。

### Test-audit：冻结后审计

测试阶段必须校验 screen 与 validation 产物哈希。随后对全部 455 个组合在官方 test 上计算同一 ridge 指标，以得到真实 oracle 和真实 top-10。它报告：

- 真正的集合召回：`|shortlist ∩ test top-10| / 10`；
- shortlist 内验证选中组合的测试归一化遗憾；
- shortlist 缩减比例；
- 完整 455 组合排序相关和任务级原始值。

测试集上的穷举只用于事后审计，不能再改变 shortlist 或验证选择。

## 5. 预注册判定

以目标域为统计单位，先在两个评价编码器内求均值，再跨六个目标域汇总。Target A-opt 是预注册主方法，其余方法用于基线与探索比较。主设置 `S=227` 同时满足以下三项才视为 H3 通过：

- shortlist 缩减至少 50%；
- 平均真实 top-10 召回至少 90%；
- 验证选中组合的平均测试归一化遗憾不超过 5%。

随机方法是方差基线，不以 20 次重复伪装成独立统计单位。即使 E2b 通过，单个 DomainNet 也只能支持本协议下冻结线性读出的等成本预筛选，不能宣称跨数据集通用性。

## 6. 运行产物

```text
results/target_conditioned/e2b_fixed_cost_shortlist/domainnet_v1/
  data_manifest/
  anchor_clip_b32/
  candidates/
  features/{resnet50,dinov2_b14}/
  screen/{resnet50,dinov2_b14}/
  validation/{resnet50,dinov2_b14}/
  test_audit/{resnet50,dinov2_b14}/
  summary/
  logs/
```

所有阶段记录配置、样本 ID、输入文件 SHA-256、manifest 构建脚本哈希、源代码 Git revision、运行时间和拒绝覆盖已有冻结产物的检查。

## 7. 完成结果与判读

数据 manifest 共冻结 46,080 个样本、100 个类和六个目标域；每个域严格满足 1,536/1,024/3,072/2,048
四项角色配额，无跳过或重复补位。18 个 CLIP 锚点候选块均为 512 个样本；对任一目标域，候选池均为
其余五域的 15 个块和全部 455 个三块组合。ResNet-50 与 DINOv2-B/14 的 screen、独立 validation
选择和官方 test 穷举审计均通过产物哈希及访问边界校验。

主设置 `S=227` 下，Target A-opt 的平均组合缩减为 `50.110%`，平均真实 top-10 recall 为
`98.333%`，六个目标域 bootstrap 95% 区间为 `[96.667%, 100%]`；验证选中组合的平均测试归一化
Brier 遗憾为 `0.001311%`，95% 区间为 `[0.000228%, 0.003031%]`。三项预注册门槛均通过，故 H3
按协议判定为 `PASS`。

该通过结论需要同时保留以下限制：

- 同信息 Second-moment MMD 在 `S=227` 也达到 `98.333%` recall 和 `0.001311%` 遗憾，并在全部
  12 个编码器-目标域任务中选出与 Target A-opt 相同的最终组合；在 `S=10/25/50/100` 时，其 recall
  还分别高于 Target A-opt。因此结果不支持 Target A-opt 相对简单同信息基线的独特优势。
- 事后检查发现，每个任务 455 个组合的完整相对 Brier 跨度只有 `0.0280%` 至 `0.4242%`，远低于
  预注册的 `5%` 遗憾门槛。H3 的 recall 证据具有区分力，但遗憾门槛在该 100 类 Brier 标度上过宽，
  不能单独证明验证选择具有实际效应。
- 该实验只有一个独立数据集和六个目标域统计单位。合理结论是：在本协议的固定成本冻结线性读出中，
  目标条件二阶几何可以把候选池减半并保留绝大多数测试 top-10；不能据此宣称跨数据集通用性，也不能
  修改 E2 的 H2/H2b 失败结论。

权威结果、逐方法表和完整限制见
`results/target_conditioned/e2b_fixed_cost_shortlist/domainnet_v1/README.md`。
