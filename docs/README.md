# ARC HTS 磁体技术经济论文 V8

当前稿件工作根为 v8，冻结数据身份为 `scientific_results_v7_splice_equivalent_20260906`。

- 维护规则（含润色原则与数字精度规则）：[AGENTS.md](AGENTS.md)
- 目录分区（代码/数据、绘图、写作、公开包、记录）：[V8_DIRECTORY_MAP.md](V8_DIRECTORY_MAP.md)
- 修改与验证顺序：[WORKFLOW_V8.md](WORKFLOW_V8.md)
- 后处理定义、分母与稿件数字含义：[POSTPROCESSING_RESULTS_V8.md](POSTPROCESSING_RESULTS_V8.md)
- 脚本分类索引：[pipeline/README.md](pipeline/README.md)
- 绘图入口：[publication_figures/README_V8.md](publication_figures/README_V8.md)

稿件文件：

- 英文 Word：[main_EN_full_WORD_v8.docx](word_version/main_EN_full_WORD_v8.docx)（导出 PDF 同目录）
- 双语 TeX：[English](current/submission/main_EN_full.tex) / [中文](current/submission/main_CN_full.tex)
- 投稿信：[cover letter-v8.docx](word_version/cover%20letter-v8.docx)
- 作者 Zotero 待办：[ZOTERO_INSERTIONS_V8.md](word_version/ZOTERO_INSERTIONS_V8.md)

公开包（未上传）：[HTSmag_v8_postprocessing_20260911](public_release/HTSmag_v8_postprocessing_20260911/README.md)，含四组计算与 V8 派生分析的代码、ledger、参考结果和[稿件数字溯源表](public_release/HTSmag_v8_postprocessing_20260911/NUMBER_PROVENANCE.md)。

GitHub/Zenodo 双包发布骨架（未构建 RC、未上传）：[发布源规范](public_release/V8_GITHUB_ZENODO_RELEASE_SPEC.json)、[逐文件源清单](public_release/V8_RELEASE_SOURCE_INVENTORY.csv)和[计划报告](public_release/V8_RELEASE_PLAN_REPORT.json)。构建入口为 `pipeline/build_v8_github_zenodo_rc.py`；当前因 9 个绝对路径便携化事项保持 STOP。

最近一轮（Step 199）：结果精度统一（3.6→37.1 nΩ 等）、去统计报告化、主图数学字体统一、Pan/Ries 引用位置调整、Word 行内 MathType 全部改为原生格式、稿件包整理与公开包重建。报告：[docs/revision_log_v8/V8_PRECISION_DESTAT_REVISION_20260911.md](docs/revision_log_v8/V8_PRECISION_DESTAT_REVISION_20260911.md)。历次报告见 `docs/revision_log_v8/`。
