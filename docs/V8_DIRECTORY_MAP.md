# V8 稿件包目录

稿件工作根：`17-提交版本-SA/1-正文-V6/v8`。活动路径保持不变（脚本以相对路径互相引用），以下按用途分区。

## 一、代码与数据

| 路径 | 内容 | 规则 |
|---|---|---|
| `model_code/` | 上游物理模型：耦合电路、T–A、低温、年度运行与经济模型，参数与测试 | 模型变化才重算；本轮未重算 |
| `scientific_results_v7_splice_equivalent_20260906/` | 冻结科学结果。`stage_*` 为上游各阶段；`figure_panel_data/` 为面板数据 | 目录名 v7 是数据身份，不改名 |
| `…/experiments/full_realization_robustness_matrix_v7/realization_ledgers/` | 四份完整五参数 realization ledger（5,904,800 行，S1–S3 的可用率、净售电、LCOE、制冷占比） | 全部后处理的共同输入 |
| `…/experiments/<分析>/` | 四组原始计算与 V8 派生分析的输出（CSV/JSON/Parquet） | 定义与分母见 `POSTPROCESSING_RESULTS_V8.md` |
| `pipeline/` | 分析、图件晋升、稿件编辑、质检脚本 | 分类索引见 `pipeline/README.md` |
| `publication_data/` | 绘图用冻结数据发布层 | `ACTIVE_RELEASE.json` 指向活动数据 |

## 二、绘图

| 路径 | 内容 | 规则 |
|---|---|---|
| `publication_figures/figures/<资产>/plot/` | Python 绘图源；主图入口见 `publication_figures/README_V8.md` | 只改源程序再导出 |
| `publication_figures/figures/<资产>/source/` | PPT/矢量可编辑源与导出；Fig0 为作者 PNG 及逐次编辑记录 | PPT 源由作者修改；PNG 编辑保留原图与脚本 |
| `publication_figures/build_v8/EN|CN/<资产>/` | 本地构建输出（PDF/PNG/SVG/Word 用轮廓 SVG） | 中间产物 |
| `current/formal_figures/`、`current/formal_figures_cn/` | **唯一稿件取图目录**（英文/中文） | TeX 与 Word 都从这里取图 |

资产编号：显示 Fig.1=Fig0.png，Fig.2–5=Fig1–Fig4，SI Fig.S1–S6=FigS1–FigS6。主图 mathtext 统一为图内正文字体（英文 Arial，中文 Times New Roman）。

## 三、写作

| 路径 | 内容 | 规则 |
|---|---|---|
| `current/submission/main_EN_full.tex`、`main_CN_full.tex` | 中英文单文件稿（正文+SI，内嵌文献表） | 与 Word 同步；文献改动后运行 `pipeline/rebuild_inline_bibliography.py` |
| `word_version/main_EN_full_WORD_v8.docx`（及 `.pdf`） | 英文 Word 正式稿 | 仅英文；Zotero 由作者管理；显示公式 MathType，行内变量用原生格式 |
| `word_version/cover letter-v8.docx` | 投稿信 | 与稿件数字同步 |
| `word_version/ZOTERO_INSERTIONS_V8.md` | 待作者插入的 Zotero 引用 | 每轮更新 |
| `current/submission/refs.bib`、`scicite.sty`、`*.bst` | 文献与样式依赖 | |

## 四、公开包

| 路径 | 内容 |
|---|---|
| `public_release/HTSmag_v8_postprocessing_20260911/` | 当前后处理公开包：镜像目录层级，四组原始计算＋V8 派生分析、ledger、参考结果、`NUMBER_PROVENANCE.md`、`MANIFEST.json`；包内 `run_analysis.py` 可复算 |
| `public_release/V8_GITHUB_ZENODO_RELEASE_SPEC.json` | 新 GitHub/Zenodo 双包的源选择、权利状态与发布闸门规范 |
| `public_release/V8_RELEASE_SOURCE_INVENTORY.csv` | 由构建骨架展开的逐文件来源、目标、大小与 SHA-256 清单 |
| `public_release/V8_RELEASE_PLAN_REPORT.json` | 计划状态、文件规模及构建前便携性 STOP 项 |
| `public_release/HTSmag_v8_bundle_templates/` | 公开包 README、复算入口、数字溯源表模板 |
| `public_release/postprocessing_v8_20260911/` | 旧四组模块（已被取代；其 MANIFEST 仍为 ledger 哈希基准） |

## 五、记录与历史

| 路径 | 内容 |
|---|---|
| `AGENTS.md`、`ACTIVE_VERSION_POLICY.md`、`WORKFLOW_V8.md` | 维护规则与流程 |
| `POSTPROCESSING_RESULTS_V8.md` | 四组与派生分析定义、分母、稿件数字含义 |
| `docs/revision_log_v8/` | V8 逐轮修订报告（最新：`V8_PRECISION_DESTAT_REVISION_20260911.md`） |
| `docs/history_v7/` | V7 目录图与流程 |
| `audits/<日期_主题>/` | 每轮候选稿、公式供体、导出 PDF、验收 JSON |
| `archive/<日期_before_主题>/` | 每轮修改前备份 |
| `run_release.ps1`、`run_scientific_upstream_v6_7.ps1` | 旧整包发布/上游入口（历史，不用于局部改稿） |
