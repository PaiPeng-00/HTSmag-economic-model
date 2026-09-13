# V8 工作流程

1. 查阅 `AGENTS.md`、`POSTPROCESSING_RESULTS_V8.md` 和项目工作文档，确认本轮修改范围与统计口径；先在工作文档记录本轮决定。
2. 备份将修改的 Word、投稿信、TeX、图源与正式图到 `archive/<日期>_before_<主题>/`；确认 Word 未打开（无 `~$` 锁文件）且自上次备份后未被作者改动。
3. 需要新数字时，只用冻结数据运行后处理脚本并写入新输出目录；不跑物理模型。
4. 图件：改 Python 源 → `pipeline/render_local_revision_figures_v8.py --part main`（或单图入口）→ 核对字体与像素 → 晋升到 `current/formal_figures(_cn)`（参考 `pipeline/promote_precision_figures_v8.py`）。PNG-only 图写确定性编辑脚本并保留原图。
5. TeX：中英同步修改；引用集合变化后运行 `pipeline/rebuild_inline_bibliography.py`；pdflatex（英）与 xelatex（中）各 3 遍，检查 Overfull、未定义引用与页数。
6. Word：raw OOXML 候选稿（参考 `pipeline/revise_precision_word_v8.py`）；显示公式经 `word-mathtype-docx` 技能生成 MathType 供体；行内变量用原生格式；Zotero 变更用黄色作者年份标记；隔离只读 Word 实例导出 PDF，目视核验受影响页后受保护晋升。
7. 更新 `word_version/ZOTERO_INSERTIONS_V8.md`、`POSTPROCESSING_RESULTS_V8.md`（如口径变化）、修订报告（`docs/revision_log_v8/`）和工作文档结果补记。
8. 公开包：`pipeline/build_public_bundle_v8.py` 生成新目录，在包内用 `run_analysis.py` 抽样复算并与参考结果逐字节比对。本地整理不等于上传。

常用命令（v8 根目录；本机 Python 需经包装器）：

```powershell
# 主图重建（中英）
python -S publication_figures/programs/run_python_v7.py pipeline/render_local_revision_figures_v8.py --part main
# 文献重排
python -S -X utf8 pipeline/rebuild_inline_bibliography.py
# 公开包
python -S publication_figures/programs/run_python_v7.py pipeline/build_public_bundle_v8.py
```

TeX 编译：`current/submission/build_submission.ps1` 在 Windows PowerShell 5.1 下会被 MiKTeX 的 stderr 更新提示中断，可在 cmd 中逐一运行 `pdflatex -interaction=nonstopmode main_EN_full.tex`、`xelatex -interaction=nonstopmode main_CN_full.tex` 各 3 遍后检查日志。
