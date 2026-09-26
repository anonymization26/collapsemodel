# Reference Library

Downloaded, checked, and integrated on 2026-09-26. This directory contains all
20 papers cited in the current English/Chinese manuscript's
[`references.bib`](../paper/iclr2027/references.bib): the original 14 entries and
six papers added from the recent-literature search. Compute-Constrained Data
Selection is in both search lists and is stored only once. Archived manuscripts
and the style template's sample bibliography are outside this audit's scope.

See [BIBLIOGRAPHY_AUDIT.md](BIBLIOGRAPHY_AUDIT.md) for the title, full author list,
venue, verification sources, and discrepancies for every paper. After the initial
download/audit, all six additional papers were cited in both manuscripts, and
the four recorded metadata corrections were applied. The additions provide
related-work context, not new experimental comparisons.

PDFs are saved without modification. Most are proceedings or journal-layout
copies. The following source/version distinctions are important:

- `yu2006`: conference paper hosted on coauthor Jinbo Bi's university website.
- `kulesza2012`: journal-layout copy hosted on author Alex Kulesza's website.
- `oquab2024`: author-hosted arXiv v2; the PDF states publication in TMLR (01/2024).
- `audenaert2007`: arXiv preprint, titled *A Sharp Fannes-type Inequality for the
  von Neumann Entropy*. The published journal title is different. The arXiv
  record explicitly links the cited journal article and DOI. The publisher PDF
  could not be retrieved; this file is not presented as the publisher version.

Page counts below refer to the downloaded PDF, not necessarily the proceedings
page span. Separate supplementary files are not included. All PDFs were checked
with `pdfinfo` and `pdftotext`; title/author blocks were inspected. Digests for all
20 files are in [SHA256SUMS](SHA256SUMS). From this directory, verify with
`shasum -a 256 -c SHA256SUMS`.

## Original Manuscript References: 14 Papers

| BibTeX key | Paper / local PDF | Venue | PDF pages | Download source |
| --- | --- | --- | ---: | --- |
| `roy2007` | [The Effective Rank: A Measure of Effective Dimensionality](2007_EUSIPCO_The_Effective_Rank.pdf) | EUSIPCO 2007 | 5 | [Proceedings deposit](https://zenodo.org/records/40328/files/a5p-h05.pdf?download=1) |
| `garrido2023rankme` | [RankMe: Assessing the Downstream Performance of Pretrained Self-Supervised Representations by Their Rank](2023_ICML_RankMe.pdf) | ICML 2023 | 46 | [PMLR](https://proceedings.mlr.press/v202/garrido23a/garrido23a.pdf) |
| `fontaine2021` | [Online A-Optimal Design and Active Linear Regression](2021_ICML_Online_A_Optimal_Design_and_Active_Linear_Regression.pdf) | ICML 2021 | 10 | [PMLR](https://proceedings.mlr.press/v139/fontaine21a/fontaine21a.pdf) |
| `yu2006` | [Active Learning via Transductive Experimental Design](2006_ICML_Active_Learning_via_Transductive_Experimental_Design.pdf) | ICML 2006 | 8 | [Author / UConn](https://jinbo-bi.uconn.edu/wp-content/uploads/sites/2638/2018/12/experimentdesign_icml.pdf) |
| `gretton2012` | [A Kernel Two-Sample Test](2012_JMLR_A_Kernel_Two_Sample_Test.pdf) | JMLR 2012 | 51 | [JMLR](https://jmlr.org/papers/volume13/gretton12a/gretton12a.pdf) |
| `kulesza2012` | [Determinantal Point Processes for Machine Learning](2012_FnTML_Determinantal_Point_Processes_for_Machine_Learning.pdf) | Foundations and Trends in Machine Learning, 2012 | 166 | [Author](https://www.alexkulesza.com/pubs/dpps_fnt12.pdf) |
| `yin2025` | [Compute-Constrained Data Selection](2025_ICLR_Compute_Constrained_Data_Selection.pdf) | ICLR 2025 | 22 | [ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/file/5acb720a361eecb34ee62d356859d246-Paper-Conference.pdf) |
| `peng2019` | [Moment Matching for Multi-Source Domain Adaptation](2019_ICCV_Moment_Matching_for_Multi_Source_Domain_Adaptation.pdf) | ICCV 2019 | 10 | [CVF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Peng_Moment_Matching_for_Multi-Source_Domain_Adaptation_ICCV_2019_paper.pdf) |
| `li2017` | [Deeper, Broader and Artier Domain Generalization](2017_ICCV_Deeper_Broader_and_Artier_Domain_Generalization.pdf) | ICCV 2017 | 9 | [CVF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Li_Deeper_Broader_and_ICCV_2017_paper.pdf) |
| `venkateswara2017` | [Deep Hashing Network for Unsupervised Domain Adaptation](2017_CVPR_Deep_Hashing_Network_for_Unsupervised_Domain_Adaptation.pdf) | CVPR 2017 | 10 | [CVF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Venkateswara_Deep_Hashing_Network_CVPR_2017_paper.pdf) |
| `radford2021` | [Learning Transferable Visual Models From Natural Language Supervision](2021_ICML_CLIP_Learning_Transferable_Visual_Models.pdf) | ICML 2021 | 16 | [PMLR](https://proceedings.mlr.press/v139/radford21a/radford21a.pdf) |
| `he2016` | [Deep Residual Learning for Image Recognition](2016_CVPR_Deep_Residual_Learning_for_Image_Recognition.pdf) | CVPR 2016 | 9 | [CVF](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) |
| `oquab2024` | [DINOv2: Learning Robust Visual Features without Supervision](2024_TMLR_DINOv2_arxiv_v2.pdf) | TMLR 2024 | 32 | [Author arXiv v2](https://arxiv.org/pdf/2304.07193v2) |
| `audenaert2007` | [A sharp continuity estimate for the von Neumann entropy (preprint under an earlier title)](2007_JPhysA_Audenaert_Entropy_Continuity_arxiv_preprint.pdf) | Journal of Physics A: Mathematical and Theoretical, 2007 | 5 | [Author arXiv preprint](https://arxiv.org/pdf/quant-ph/0610146) |

## Newly Integrated References: 6 Papers

All six are now entries in `references.bib` and cited in both versions of Related
Work. Transductive Active Learning is also cited in the Introduction and the
Target A-opt derivation to clarify the score's connection to prior work.

The expanded Related Work explains the selection mechanisms rather than only
listing the papers. Appendix D, "Connections to Recent Data-Selection Methods,"
discusses their representations, selection units, information access, and
requirements for future controlled comparisons. The English and Chinese
manuscripts cover the same distinctions. These discussions do not report new
baseline runs or transfer the cited papers' guarantees to our classification
experiments.

| BibTeX key | Context |
| --- | --- |
| `dong2024skmm` | Gradient sketching and moment matching for fine-tuning |
| `hubotter2024transductive` | Target entropy/variance and transductive experimental design |
| `liu2024tsds` | Target-distribution alignment with diversity regularization |
| `hubotter2025sift` | Uncertainty-based selection for test-time fine-tuning |
| `feng2025tarot` | Targeted optimal transport in loss-gradient features |
| `rezaei2026synthetic` | Covariance matching in specified linear synthetic-data regimes |

| Paper | Venue | Pages | Local PDF | Official PDF source |
| --- | --- | ---: | --- | --- |
| Sketchy Moment Matching: Toward Fast and Provable Data Selection for Finetuning | NeurIPS 2024 | 36 | [SkMM](2024_NeurIPS_SkMM_Sketchy_Moment_Matching.pdf) | [Download](https://proceedings.neurips.cc/paper_files/paper/2024/file/4c79c359b3c5f077c0b955f93cb0f53e-Paper-Conference.pdf) |
| Transductive Active Learning: Theory and Applications | NeurIPS 2024 | 70 | [Transductive Active Learning](2024_NeurIPS_Transductive_Active_Learning.pdf) | [Download](https://papers.nips.cc/paper_files/paper/2024/file/e17fe6fe9990fffb637b42c98c005515-Paper-Conference.pdf) |
| TSDS: Data Selection for Task-Specific Model Finetuning | NeurIPS 2024 | 31 | [TSDS](2024_NeurIPS_TSDS_Task_Specific_Data_Selection.pdf) | [Download](https://proceedings.neurips.cc/paper_files/paper/2024/file/13848b5893119ff772b69812c95914fa-Paper-Conference.pdf) |
| Efficiently Learning at Test-Time: Active Fine-Tuning of LLMs | ICLR 2025 | 58 | [SIFT](2025_ICLR_SIFT_Efficiently_Learning_at_Test_Time.pdf) | [Download](https://proceedings.iclr.cc/paper_files/paper/2025/file/ba942323c447c9bbb9d4b638eadefab9-Paper-Conference.pdf) |
| TAROT: Targeted Data Selection via Optimal Transport | ICML 2025 | 16 | [TAROT](2025_ICML_TAROT_Targeted_Data_Selection_via_Optimal_Transport.pdf) | [Download](https://raw.githubusercontent.com/mlresearch/v267/main/assets/feng25l/feng25l.pdf) |
| High-dimensional Analysis of Synthetic Data Selection | ICLR 2026 | 52 | [Synthetic Data Selection](2026_ICLR_High_Dimensional_Analysis_of_Synthetic_Data_Selection.pdf) | [Download](https://proceedings.iclr.cc/paper_files/paper/2026/file/97c5b2707228e7e3fb67e4ecc2e0e607-Paper-Conference.pdf) |
