# V8 现行版本治理规则

当前编辑根为 v8。英文Word和中英文TeX的同步维护规则见 AGENTS.md。V7目录为历史基线，不接收V8编辑。

冻结科学结果为 scientific_results_v7_splice_equivalent_20260906/。沿用绝对plant availability、80%判据、7%计划外不可用度和等效单带承流拼接模型。冻结结果身份及上游审计保持不变；原始数据版本号不随稿件版号改写。

当前四组完整realization后处理及摘要口径见 POSTPROCESSING_RESULTS_V8.md。publication_data/ACTIVE_RELEASE.json指向v8本地冻结绘图数据副本；ACTIVE_POSTPROCESSING_V8.json指向四组后处理及导出模块。

唯一活动路径、显示图号与文件名映射见 V8_DIRECTORY_MAP.md。局部修改只重建关联图和稿件。模型变化才触发新的隔离科学重算；后处理定义变化应生成独立输出目录并验证。

每轮修改先保留目标文件备份，最后核验Word/TeX同步、引用待办及受影响页。公共导出使用显式清单，排除历史、私人稿件和缓存；本地准备不等于平台上传。
