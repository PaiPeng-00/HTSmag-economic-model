# V8 冻结数据后处理：四组计算与稿件派生分析

原始科学身份为 `scientific_results_v7_splice_equivalent_20260906`（目录名中的 v7 是数据版本身份，不随稿件版号改写）。全部计算使用同一冻结 B1-anchor 经济模型，未重跑物理模型。S1/S2/S3 情景特定最低 LCOE 为 1122.1472612313785、321.97916120106885、78.82074592882216 USD/MWh。

稿件报告精度（Step 199）：派生结果 ≥1 保留一位小数，<1 保留两位有效数字；需要说明阈值穿越时例外。下表保留完整精度，稿件数字与数据文件的逐项对应见公开包 `NUMBER_PROVENANCE.md`。

## 一、四组原始计算

| 组 | 定义及分母 | 核心结果（完整精度） | 代码 | experiments 下数据 |
|---|---|---|---|---|
| 完整 realization | 五变量全部保留；200×61×121×4=5,904,800；三情景交集 | A≥0.80：4,904,614（83.06%）；10% 经济保持 4,068,226，条件保持率 82.9469%；1/5/20%：4.6727/57.8758/91.1160% | `pipeline/run_full_realization_robustness_matrix_v7.py`；先前 5% 审计 `run_full_realization_plant_robustness_v7.py` | `full_realization_robustness_matrix_v7/`（含四份 realization ledger）；`full_realization_plant_robustness_v7/` |
| YT 配对温度 | 同一 Npw/rho_turn/Rj，He，4.2 K↔20 K；YT0=1,476,200；YT1=1,197,416；YT2=1,168,762 | 配对制冷比率中位降幅 S1/S2/S3=77.95/77.90/77.88%；YT2 中 10% 保持率 59.11%→97.17% | `pipeline/run_paired_temperature_effect_v7.py` | `paired_temperature_effect_v7/` |
| YR 基础状态容限 | b=(Npw,rho_turn)，12,200；从 1 nΩ 起连续通过；共同有效 9,896 | 10% 配对容限中位 12.115→≥100 nΩ；比值中位保守下界 8.254 | `pipeline/run_joint_tolerance_temperature_ratio_v7.py` | `joint_tolerance_temperature_ratio_v7/` |
| YJ 高电流容限（原生节点） | Npw=200，He；固定 rho_turn=10,000 或逐温度可选 rho | 末通过节点 3.548→36.869 nΩ | `pipeline/run_high_current_joint_resistance_yj_matrix_v7.py` | `high_current_joint_resistance_yj_matrix_v7/` |

YT、YR 为支撑分析，当前稿件不引用其数字。YJ 的原生节点端点已被下面的插值容限取代，仅作历史夹逼证据。

## 二、当前稿件使用的派生分析

| 分析 | 稿件位置 | 代码 | 数据目录 |
|---|---|---|---|
| 经济保持率 1/5/10/20%、10% 余量、10% 前后制冷分布 | 摘要、Results3、Fig.4、Table S4、Note S6 | `pipeline/run_economic_retention_v8.py` | `economic_retention_v8/` |
| Fig.3D 共同可用率九组制冷统计 | Results2、Fig.3D | `pipeline/run_fig3d_common_availability_stats_v8.py` | `fig3d_common_availability_v8/` |
| S2 各温度最低 LCOE、价格敏感性 | Results4、Fig.5C | `pipeline/audit_results4_v8.py` | `results4_temperature_price_v8/` |
| Table S4 总体与基准母集合核验 | Methods、Note S6 | `pipeline/audit_methods_populations_v8.py` | `methods_populations_v8/`；价格复核 `methods_price_recheck_v8/` |
| 温度平衡权重敏感性 | Note S6 | `pipeline/audit_temperature_balance_v8.py` | `temperature_balance_v8/` |
| 逐电阻率原生边界、全电阻率色图 | Fig.5 溯源 | `pipeline/audit_fig5_rho_invariance_v8.py`、`materialize_fig5_conservative_v8.py` | `fig5_rho_invariance_v8/`、`fig5_conservative_v8/` |
| 插值保守接头容限 | 摘要、Fig.1、Results4、Fig.5A/B、Note S7 | `pipeline/interpolate_fig5_tolerance_v8.py` | `fig5_interpolated_v8/` |

## 三、稿件数字的准确含义

- **82.9%**：分母是三情景均满足 80% availability 的全部完整 realizations（4,904,614）；经济通过要求各情景 E_net>0、有限正 LCOE、LCOE≤1.10×情景最低 LCOE。经济无效状态保留在分母中。
- **0.6%–121.5%**（中位 2.0%，P10–P90 0.94%–8.7%）：上述 availability 总体在三个情景下的全部制冷占比记录（真实极值 0.60128353%/121.53727551%）。制冷占比 >68% 时净售电量非正，这类磁体仍属 availability 总体但经济无效。最小值全稿统一写 0.6%。
- **3.6→37.1 nΩ**（10.3 倍）：Npw=200、He；对三个情景、61 个匝间接触电阻率分别沿 Rj 做首次跨越 10% 阈值的线性插值，再取最小值（完整精度 3.601519/14.555657/37.101546 nΩ，均由 S1、rho_turn=10 μΩ·cm² 控制）。Npw=50 时为 74.4、≥100、≥100 nΩ。20 K 下 Npw<48 时低匝间电阻率磁体不满足 availability，保守容限无定义，因此 Fig.5B 从 50 起画。
- **阈值敏感性（Npw=200、He、同一全电阻率保守定义）**：1% 时 4.2 K 与 20 K 均无从 1 nΩ 开始的合格区间；5% 时 4.2 K 为 1.1852 nΩ、20 K 无合格区间；20% 时 4.2/20 K 为 7.8298/98.0738 nΩ，比值 12.5。正式来源为 `fig5_interpolated_v8/all_margins_conservative_boundaries.csv`，不得以固定 `rho_turn=10,000 μΩ·cm²` 的旧示例替代。
- **79.1%**：完整网格上预先赋予 4.2 K He、10 K He、20 K He、20 K H2 相对权重 1、1、0.5、0.5（各温度总权重 1/3）后的 10% 条件保持率；筛选后不再重分配。
- 旧 70.9%、以 5% 作为主判据、3.55→36.87 nΩ、2.4→22.3 nΩ 均为已废弃口径，只保留在历史数据目录，不得冒充当前主结果；5% 仅保留为使用当前全电阻率保守定义的阈值敏感性。

## 四、复现与公开包

- 活动目录内复现：各脚本带 `--output`/`--output-dir` 参数，输出目录必须尚不存在；不跑物理模型。Python 需使用 `python -S publication_figures/programs/run_python_v7.py SCRIPT ARGS` 包装器（本机 site 编码问题）。
- 公开包：`python -S publication_figures/programs/run_python_v7.py pipeline/build_public_bundle_v8.py` 生成 `public_release/HTSmag_v8_postprocessing_20260911/`。包内目录镜像 v8 工作树，脚本原样可运行；包含四组与派生分析代码、四份 realization ledger、全部参考结果、`NUMBER_PROVENANCE.md`、`MANIFEST.json`（SHA-256）。包内复算入口 `python -B run_analysis.py --group {full,YT,YR,YJ,retention,results4,populations,temperature-balance,fig5} --output NEW_DIR`。
- 早先的 `public_release/postprocessing_v8_20260911/` 只含四组原始计算，已被新包取代；其 `MANIFEST.json` 仍被 V8 派生脚本用作 ledger 哈希基准，因此保留。
- 本地整理不等于上传；许可证、DOI 与平台由作者决定。
