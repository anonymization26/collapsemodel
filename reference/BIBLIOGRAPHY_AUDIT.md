# 参考文献下载与书目信息核对

核对日期：2026-09-26。

## 范围与结论

- 下载核对以当时英文稿、中文稿共同使用的 `paper/iclr2027/references.bib` 中14个条目为起点；正文和附录的引用均在范围内。随后按作者要求纳入6篇补充文献，当前书目共20个条目，两版均实际引用。
- 原14篇全部下载到本目录，下载核对时新增13篇，复用已下载的 `yin2025`。连同前次检索的另外6篇，目录共20篇不同论文。
- 不包含 `paper/archive/` 中旧稿的独有参考文献，也不包含正式模板自带的示例书目。
- 逐条比对原书目、出版机构或作者提供的记录，以及下载 PDF 的标题和作者信息。未发现论文身份、标题或会议／期刊归属的实质性错配。
- 发现3处作者中间名省略及 PACS 条目缺页码，已按核验记录补全，详见下表。这些差异不是引用了错误论文。
- Audenaert 论文保存的是作者预印本，标题与正式期刊版不同；arXiv 记录明确给出同一正式论文的 DOI 和期刊卷页。不能将该文件标为出版社正式 PDF。
- DINOv2 保存的是 arXiv v2，首页明确写有 `Published in Transactions on Machine Learning Research (01/2024)`。它是带正式发表信息的作者公开版本；未声称与 OpenReview 文件逐字节相同。
- 初次下载核对未修改论文；随后引用整合已同步修改中英文相关工作、引言和方法中的相关引用及共享书目，并重编译 PDF。没有新增实验比较，也没有改变实验数据、数学公式或结论。

## 已补全的信息

| 条目 | 修改前书目 | PDF 或正式记录 | 已应用的修正 |
| --- | --- | --- | --- |
| `oquab2024` | `Vo, Huy` | PDF 首页为 `Huy V. Vo` | 已补为 `Vo, Huy V.`；作者顺序不变。 |
| `li2017` | `Hospedales, Timothy` | PDF 首页为 `Timothy M. Hospedales` | 已补为 `Hospedales, Timothy M.`。 |
| `yin2025` | `Rush, Alexander` | PDF 首页为 `Alexander M. Rush` | 已补为 `Rush, Alexander M.`；会议网页也使用不带中间名的写法，不是作者身份错误。 |
| `li2017` | 无 `pages` 字段 | CVF PDF 首尾印刷页码为5542和5550 | 已补充 `pages={5542--5550}`。 |

大小写、重音符号和姓名显示习惯的差异未一律判为错误。例如 RankMe 的 PMLR 网页显示 `Lecun`，论文 PDF 显示 `LeCun`，当前书目的 `LeCun` 无需修改。

## 原14篇逐条核对

### 1. roy2007

- 标题：The Effective Rank: A Measure of Effective Dimensionality。
- 作者及顺序：Olivier Roy; Martin Vetterli。
- 发表：15th European Signal Processing Conference (EUSIPCO), 2007, pp. 606-610。
- 依据：[会议论文存档](https://zenodo.org/records/40328)及下载 PDF 的 EUSIPCO 2007 页眉、作者栏和印刷页码。
- [本地 PDF](2007_EUSIPCO_The_Effective_Rank.pdf)，5页。标题、作者和会议信息匹配。

### 2. garrido2023rankme

- 标题：RankMe: Assessing the Downstream Performance of Pretrained Self-Supervised Representations by Their Rank。
- 作者及顺序：Quentin Garrido; Randall Balestriero; Laurent Najman; Yann LeCun。
- 发表：Proceedings of the 40th International Conference on Machine Learning (ICML), PMLR 202, 2023, pp. 10929-10974。
- 依据：[PMLR 正式记录](https://proceedings.mlr.press/v202/garrido23a.html)及 PDF 首页。
- [本地 PDF](2023_ICML_RankMe.pdf)，46页。匹配；作者姓氏大小写按 PDF 保留 `LeCun`。

### 3. fontaine2021

- 标题：Online A-Optimal Design and Active Linear Regression。
- 作者及顺序：Xavier Fontaine; Pierre Perrault; Michal Valko; Vianney Perchet。
- 发表：Proceedings of the 38th International Conference on Machine Learning (ICML), PMLR 139, 2021, pp. 3374-3383。
- 依据：[PMLR 正式记录](https://proceedings.mlr.press/v139/fontaine21a.html)及 PDF 首页。
- [本地 PDF](2021_ICML_Online_A_Optimal_Design_and_Active_Linear_Regression.pdf)，10页。匹配。

### 4. yu2006

- 标题：Active Learning via Transductive Experimental Design。
- 作者及顺序：Kai Yu; Jinbo Bi; Volker Tresp。
- 发表：Proceedings of the 23rd International Conference on Machine Learning (ICML), 2006。当前书目所列 DOI 为 `10.1145/1143844.1143980`。
- 依据：[共同作者 Jinbo Bi 的大学主页 PDF](https://jinbo-bi.uconn.edu/wp-content/uploads/sites/2638/2018/12/experimentdesign_icml.pdf)，首页注明 ICML 2006、Pittsburgh。
- [本地 PDF](2006_ICML_Active_Learning_via_Transductive_Experimental_Design.pdf)，8页。标题、作者、会议和年份匹配。下载的是作者公开副本，不声称与 ACM 文件逐字节相同；本轮未独立验证 ACM 页码元数据。

### 5. gretton2012

- 标题：A Kernel Two-Sample Test。
- 作者及顺序：Arthur Gretton; Karsten M. Borgwardt; Malte J. Rasch; Bernhard Schölkopf; Alexander Smola。
- 发表：Journal of Machine Learning Research, 13(25), 2012, pp. 723-773。
- 依据：[JMLR 正式记录](https://www.jmlr.org/papers/v13/gretton12a.html)及期刊 PDF。
- [本地 PDF](2012_JMLR_A_Kernel_Two_Sample_Test.pdf)，51页。匹配。

### 6. kulesza2012

- 标题：Determinantal Point Processes for Machine Learning。
- 作者及顺序：Alex Kulesza; Ben Taskar。
- 发表：Foundations and Trends in Machine Learning, 5(2-3), 2012, pp. 123-286；DOI `10.1561/2200000044`。
- 依据：[Alex Kulesza 作者主页](https://www.alexkulesza.com/)及其提供的期刊排版 PDF，封面列明卷、期、页码和 DOI。
- [本地 PDF](2012_FnTML_Determinantal_Point_Processes_for_Machine_Learning.pdf)，166页，含前置目录页。匹配；不是会议论文，也没有误用作者博士论文。

### 7. yin2025

- 标题：Compute-Constrained Data Selection。
- 作者及顺序：Junjie Oscar Yin; Alexander M. Rush。
- 发表：International Conference on Learning Representations (ICLR), 2025。
- 依据：[ICLR 正式记录](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5acb720a361eecb34ee62d356859d246-Abstract-Conference.html)及 PDF 首页。
- [本地 PDF](2025_ICLR_Compute_Constrained_Data_Selection.pdf)，22页。标题、作者身份和会议匹配；已补全第二作者中间名 `M.`。复用前次下载文件，没有重复保存。

### 8. peng2019

- 标题：Moment Matching for Multi-Source Domain Adaptation。
- 作者及顺序：Xingchao Peng; Qinxun Bai; Xide Xia; Zijun Huang; Kate Saenko; Bo Wang。
- 发表：Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2019, pp. 1406-1415。
- 依据：[CVF 正式记录](https://openaccess.thecvf.com/content_ICCV_2019/html/Peng_Moment_Matching_for_Multi-Source_Domain_Adaptation_ICCV_2019_paper.html)及 PDF 首页。
- [本地 PDF](2019_ICCV_Moment_Matching_for_Multi_Source_Domain_Adaptation.pdf)，10页。匹配。

### 9. li2017

- 标题：Deeper, Broader and Artier Domain Generalization。
- 作者及顺序：Da Li; Yongxin Yang; Yi-Zhe Song; Timothy M. Hospedales。
- 发表：IEEE International Conference on Computer Vision (ICCV), 2017, pp. 5542-5550。
- 依据：[CVF 会议论文 PDF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Li_Deeper_Broader_and_ICCV_2017_paper.pdf)的作者栏、首页及末页印刷页码。
- [本地 PDF](2017_ICCV_Deeper_Broader_and_Artier_Domain_Generalization.pdf)，9页。标题、作者身份和会议匹配；已补全 `Timothy M.` 和页码。

### 10. venkateswara2017

- 标题：Deep Hashing Network for Unsupervised Domain Adaptation。
- 作者及顺序：Hemanth Venkateswara; Jose Eusebio; Shayok Chakraborty; Sethuraman Panchanathan。
- 发表：Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 5018-5027。
- 依据：[CVF 正式记录](https://openaccess.thecvf.com/content_cvpr_2017/html/Venkateswara_Deep_Hashing_Network_CVPR_2017_paper.html)及 PDF 首页。
- [本地 PDF](2017_CVPR_Deep_Hashing_Network_for_Unsupervised_Domain_Adaptation.pdf)，10页。匹配。

### 11. radford2021

- 标题：Learning Transferable Visual Models From Natural Language Supervision。
- 作者及顺序：Alec Radford; Jong Wook Kim; Chris Hallacy; Aditya Ramesh; Gabriel Goh; Sandhini Agarwal; Girish Sastry; Amanda Askell; Pamela Mishkin; Jack Clark; Gretchen Krueger; Ilya Sutskever。
- 发表：Proceedings of the 38th International Conference on Machine Learning (ICML), PMLR 139, 2021, pp. 8748-8763。
- 依据：[PMLR 正式记录](https://proceedings.mlr.press/v139/radford21a.html)及 PDF 首页。
- [本地 PDF](2021_ICML_CLIP_Learning_Transferable_Visual_Models.pdf)，16页。12位作者及其顺序、标题和会议信息匹配。

### 12. he2016

- 标题：Deep Residual Learning for Image Recognition。
- 作者及顺序：Kaiming He; Xiangyu Zhang; Shaoqing Ren; Jian Sun。
- 发表：Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778。
- 依据：[CVF 正式记录](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)及 PDF 首页。
- [本地 PDF](2016_CVPR_Deep_Residual_Learning_for_Image_Recognition.pdf)，9页。匹配。

### 13. oquab2024

- 标题：DINOv2: Learning Robust Visual Features without Supervision。
- 作者及顺序：Maxime Oquab; Timothée Darcet; Théo Moutakanni; Huy V. Vo; Marc Szafraniec; Vasil Khalidov; Pierre Fernandez; Daniel Haziza; Francisco Massa; Alaaeldin El-Nouby; Mahmoud Assran; Nicolas Ballas; Wojciech Galuba; Russell Howes; Po-Yao Huang; Shang-Wen Li; Ishan Misra; Michael Rabbat; Vasu Sharma; Gabriel Synnaeve; Hu Xu; Hervé Jégou; Julien Mairal; Patrick Labatut; Armand Joulin; Piotr Bojanowski。
- 发表：Transactions on Machine Learning Research (TMLR), January 2024。
- 依据：[作者 arXiv 记录](https://arxiv.org/abs/2304.07193)和 v2 PDF 首页。首页给出 TMLR 01/2024 发表信息及 [OpenReview 评审链接](https://openreview.net/forum?id=a68SUt6zFt)。本次直接访问 OpenReview 遇到浏览器验证，未绕过验证。
- [本地 PDF](2024_TMLR_DINOv2_arxiv_v2.pdf)，32页。26位作者及其顺序与当前条目对应；已补全第四作者为 `Huy V. Vo`。不同平台的姓名别称和重音显示不能据此视为缺失作者。
- 初始 arXiv 年份为2023，正式期刊年份为2024；当前条目使用 TMLR 2024 正确，不应改成 ICLR 或 CVPR。

### 14. audenaert2007

- 正式标题：A sharp continuity estimate for the von Neumann entropy。
- 作者：Koenraad M. R. Audenaert。
- 正式发表：Journal of Physics A: Mathematical and Theoretical, 40(28), 2007, pp. 8127-8136；DOI `10.1088/1751-8113/40/28/S18`。
- 正式书目信息依据：[作者所在机构的研究记录](https://pure.royalholloway.ac.uk/en/publications/a-sharp-continuity-estimate-for-the-von-neumann-entropy/)。当前书目与该记录匹配。
- 下载版本：[arXiv:quant-ph/0610146v1](https://arxiv.org/abs/quant-ph/0610146)，首次提交于2006-10-18，标题为 *A Sharp Fannes-type Inequality for the von Neumann Entropy*。arXiv 记录明确列出上述期刊卷页和 DOI，因此可以确认是所引论文的预印本，而非另一篇同主题论文。
- [本地 PDF](2007_JPhysA_Audenaert_Entropy_Continuity_arxiv_preprint.pdf)，5页；正式期刊文章为10个印刷页，两种排版不能用页数直接比较。文件名中的2007按所引期刊年份命名，`arxiv_preprint` 标明实际版本。
- 直接下载 IOP 官方 PDF 未成功，保留公开预印本。PDF 中出现的 `Dated: November 6, 2018` 不用于确定发表年；发表年份以期刊记录和 DOI 为准。

## 前次检索的6篇补充文献

以下文献已下载并加入当前论文的 `references.bib`，均在中英文相关工作中引用。标题、作者和会议信息均与正式会议 PDF 对应；引用键和各自的叙述位置见 [README.md](README.md)。其中 Transductive Active Learning 同时用于引言和方法的理论定位。引用不表示这些方法已纳入本文实验基线。

| 标题 | 作者及顺序 | 会议 | 依据 |
| --- | --- | --- | --- |
| Sketchy Moment Matching: Toward Fast and Provable Data Selection for Finetuning | Yijun Dong; Hoang Phan; Xiang Pan; Qi Lei | NeurIPS 2024 | [正式会议记录](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4c79c359b3c5f077c0b955f93cb0f53e-Abstract-Conference.html) |
| Transductive Active Learning: Theory and Applications | Jonas Hübotter; Bhavya Sukhija; Lenart Treven; Yarden As; Andreas Krause | NeurIPS 2024 | [正式会议记录](https://papers.nips.cc/paper_files/paper/2024/hash/e17fe6fe9990fffb637b42c98c005515-Abstract-Conference.html) |
| TSDS: Data Selection for Task-Specific Model Finetuning | Zifan Liu; Amin Karbasi; Theodoros Rekatsinas | NeurIPS 2024 | [正式会议记录](https://proceedings.neurips.cc/paper_files/paper/2024/hash/13848b5893119ff772b69812c95914fa-Abstract-Conference.html) |
| Efficiently Learning at Test-Time: Active Fine-Tuning of LLMs | Jonas Hübotter; Sascha Bongni; Ido Hakimi; Andreas Krause | ICLR 2025 | [正式会议记录](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ba942323c447c9bbb9d4b638eadefab9-Abstract-Conference.html) |
| TAROT: Targeted Data Selection via Optimal Transport | Lan Feng; Fan Nie; Yuejiang Liu; Alexandre Alahi | ICML 2025, PMLR 267:16837-16852 | [PMLR 正式记录](https://proceedings.mlr.press/v267/feng25l.html) |
| High-dimensional Analysis of Synthetic Data Selection | Parham Rezaei; Filip Kovačević; Francesco Locatello; Marco Mondelli | ICLR 2026 | [正式会议记录](https://proceedings.iclr.cc/paper_files/paper/2026/hash/97c5b2707228e7e3fb67e4ecc2e0e607-Abstract-Conference.html) |

## 文件完整性

- 下载来源和本地链接见 [README.md](README.md)。
- 20个文件均可由 `pdfinfo` 解析；标题及作者栏通过 `pdftotext` 提取核对。
- [SHA256SUMS](SHA256SUMS) 记录本地文件的 SHA-256，可检查后续文件变化；它不是出版社提供的独立签名或官方校验值。
- 本次不对各论文的定理、实验或科学结论作新评估。
