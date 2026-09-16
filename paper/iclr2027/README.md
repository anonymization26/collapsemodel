# ICLR 2027 工作初稿

题目：**When Does Feature Geometry Help Data-Pool Screening?**

本版基于 `65a6435` 已提交实验基线重新组织论文。它是供作者审阅的完整工作初稿，尚未提交会议。
不覆盖旧 NeurIPS 稿，不启动新实验，不改变已冻结的实验门控。

## 文件

- `main.tex` / `main.pdf`：英文稿入口。
- `main_zh.tex` / `main_zh.pdf`：中文对照入口，正文、证明与限制同步。
- `body_en.tex` / `body_zh.tex`：六节主文。
- `appendix_en.tex` / `appendix_zh.tex`：数学推导、实现审计和补充失败诊断。
- `abstract_en.txt` / `abstract_zh.txt`：唯一摘要文本源。修改后运行构建脚本，自动同步 `abstract.md` 与 PDF 输入。
- `build_assets.py`：从已提交结果生成主表、两张图、数字宏及摘要。
- `generated/asset_manifest.json`：上述图表的输入 SHA-256 及描述性诊断。
- `references.bib`：本版实际引用；第三方论文作者信息不属于本稿署名。
- `iclr2027-style/`：用户提供的完整正式模板，原样保留。
- `.latexmkrc`：优先加载模板附带的 `fancyhdr.sty`、`natbib.sty`，避免系统包版本改变审稿页眉。

## 构建

在本目录执行：

```bash
python3 build_assets.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error main_zh.tex
python3 verify_draft.py
```

必须保留 `.latexmkrc` 与完整的 `iclr2027-style/` 目录，并从本目录执行上述 `latexmk` 命令。
两个入口都直接引用模板目录中的 `.sty` 和 `.bst`；本目录原有的同名副本保留，但不再作为主样式入口。
不要使用未配置依赖搜索路径的裸 `pdflatex` / `xelatex` 命令：它们不读取 `.latexmkrc`，可能重新加载
系统版 `fancyhdr` 并丢失页眉。必须直接调用引擎时，显式设置 `TEXINPUTS=./iclr2027-style//:`，
保留结尾的冒号以继续搜索 TeX 系统目录。首次迁移到完整模板可给 `latexmk` 加 `-g` 强制重建。

绘图需要 NumPy、Matplotlib。验证使用标准库、NumPy、Poppler 的 `pdftotext` 和 `pdfinfo`。
英文使用 pdfLaTeX/Times；中文使用 XeLaTeX、TeX Gyre 西文字体与 macOS 的 Songti SC、Heiti SC。
中文 `ctex` 仅负责汉字支持与断行，不接管字号、行距或标题格式；两版正文均沿用正式模板的
10pt 字号、11pt 基线行距、零首行缩进和 5.5 × 9 英寸版心。中文保留图表和参考文献名称的翻译。
其他操作系统需要显式替换为已安装的中文字体，不能假设默认回退能正确显示汉字。
PDF 视觉检查使用 Poppler 渲染，不以成功编译代替检查。

2026-09-13 正式模板修订：修复审稿页眉缺失，取消两版表格的 `\small`，三个声明使用模板示例的
无编号二级标题。英文正文结束于第 8 页（共 13 页），中文正文结束于第 7 页（共 12 页）；
中文页数减少来自恢复模板字号和行距，没有删减论文内容。系统宋体仍发出 OpenType Script 元数据提示，
但没有缺字。作者科学复审和新实验仍待完成。

`verify_draft.py` 在已有摘要、数据与引用检查之外，还核验正式依赖文件的 SHA-256、实际加载路径、
正文及版心参数、US Letter 页面、逐页审稿页眉与页码、匿名标题区、正文图表页限及表格字号命令。
这些检查不能代替 PDF 视觉检查或作者科学核验。

## 论证边界

1. 完整 Gram 可决定谱；边际谱与主角度不足。局部双方向公式有显式相容条件。
2. 目标加权 A-opt 风险属于标准共享贝叶斯线性模型，不是 softmax Brier 的普适定理。
3. PACS/Office-Home 的 H2/H2b 仍失败；事后 oracle 改善空间不能挽救原门控。
4. DomainNet H3 支持限定协议下的等成本预筛选，但 MMD 主设置完全持平，小短名单均值更好。
5. 小 Brier 遗憾不等于高准确率；候选完整跨度小于门槛使遗憾子门控失去区分力。
6. 只减少组合评价次数，不声称端到端加速、实际通信压缩、全局最优或标签隐私。

后续证据缺口和作者核验任务见 [改写 checklist](../../docs/ICLR_REWRITE_CHECKLIST.md)。

## 模板来源

`iclr2027-style/` 中的正式模板与依赖未作任何修改；该模板的发布来源为
[ICLR 2027 官方模板](https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip)，未改版式。
依据 [作者指南](https://iclr.cc/Conferences/2027/AuthorGuidelines)，投稿正文最多 9 页；参考文献、附录及
规定的声明不计入正文页限。本版使用匿名样式，页眉由官方模板生成，不表示已经实际投稿。
已按 [AI 使用政策](https://iclr.cc/Conferences/2027/AIPolicyForAuthors) 写入辅助工作范围，但未代替人类作者
声明“所有 AI 工作已人工核验”；该核验保留为提交前待办。
