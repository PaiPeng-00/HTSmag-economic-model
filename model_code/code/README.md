# TF Magnet 技术—经济分析（fusion-tem）

## 当前权威代码成本边界：`plant_v4_core_pcs_coolant_2025usd_direct_hts`

主分析唯一指标仍为 `LCOE_plant_USD_per_MWh`：

```text
LCOE_plant = [CRF × (CAPEX_mag_installed + C0_background)
              + OPEX_core_VOM + OPEX_PCS_VOM
              + OPEX_PCS_FOM + OPEX_coolant_VOM] / E_net_year
```

本次将 HTS conductor price 从历史价格 CPI 换算改为直接的 constant-2025-US$ 情景假设，不改变上述方程。权威换算表为 `configs/2025_USD_conversion_table.csv`；每项保存 source value、source price year、CPI factor、conversion method 和 2025-US$ value。HTS 三档均使用 source_price_year=2025、CPI factor=1.0、conversion_method=direct_scenario_assumption_2025usd；文献只支撑情景范围。

S1/S2/S3 的 C0 为 6.200979/4.133986/2.066993 billion 2025 US$；core VOM 为 6.296385/3.777831/1.259277 2025 US$/MWh(th)；PCS capital reference 为 961.571163 2025 US$/kW(e)，PCS VOM 为 2.191142 2025 US$/MWh(e)。PCS FOM 2.5%/yr 与制冷剂补充率 25%/yr 不变。S1/S2/S3 与对应的 S4/S5/S6 HTS 价格均直接定义为 100/50/10 constant-2025-US$/(kA·m)。S4-S6 完整继承 S2 的全部非价格假设。

完整重算：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/8_economic/run_full_grid_2025usd_direct_hts.ps1
powershell -ExecutionPolicy Bypass -File scripts/8_economic/harmonize_2025usd_direct_hts_outputs.ps1
```

第一条命令执行 fresh rerun；第二条只在差异为有限浮点且不超过 4 ULP 时，将共同非货币字段恢复为plant-v3 基线的精确非货币字符串，并保留 raw-fresh 备份。正式输出为 `outputs/arc_16pancake_nuc600/tables/scan_full_grid_tidy_plant_opex_2025usd_direct_hts.csv`、`scan_full_grid_plant_opex_2025usd_direct_hts.xlsx` 和 `outputs/arc_16pancake_nuc600/manifest_plant_opex_2025usd_direct_hts.json`。plant-v3 文件不会被覆盖。

经济归一化、非货币列不变性和 exact robust-frontier 审计：

```powershell
python scripts/8_economic/audit_2025usd_direct_hts.py --baseline outputs/arc_16pancake_nuc600/tables/scan_full_grid_tidy_plant_opex_2025usd.csv --new outputs/arc_16pancake_nuc600/tables/scan_full_grid_tidy_plant_opex_2025usd_direct_hts.csv --output-dir outputs/arc_16pancake_nuc600/tables
```

Fig. 5/Fig. 6 继续写入独立候选目录，不覆盖冻结图：

```powershell
$env:SCAN_INPUT_CSV = 'outputs/arc_16pancake_nuc600/tables/scan_full_grid_tidy_plant_opex_2025usd_direct_hts.csv'
$env:SCAN_FIGURE_OUTPUT_ROOT = 'outputs/arc_16pancake_nuc600/figures/plant_v4_2025usd_direct_hts_review'
$env:SUBMISSION_FIGURE_OUTPUT_ROOT = 'outputs/arc_16pancake_nuc600/figures/plant_v4_2025usd_direct_hts_submission_candidate'
python scripts/8_economic/8.1_plot_from_scan_full_grid.py
python scripts/9_figures/9.0_stitch_econimic_svgs.py
```
本仓库扫描 HTS TF 磁体设计参数，计算电磁 → 热负荷 → 制冷 → 经济（LCOE）指标，并基于统一结果表生成论文图件。基线器件为 ARC（`FUSION_DEVICE=arc_16pancake_nuc600`）。

## 安装

代码以可安装包 `fusion-tem` 组织（`src/` 布局）。在 `code/` 目录下：

```bash
pip install -e .            # 注册 fusion_tem / tfmag 两个包（可从任意目录 import）
# 可选额外依赖：
pip install -e .[accel]     # numba（utils 的加速路径）
pip install -e .[svg]       # cairosvg（svg→pdf 导出）
```

安装后即可 `from fusion_tem import device`、`from tfmag.paths import get_paths`，无需 `sys.path` hack。

## 目录结构

```
code/
├─ pyproject.toml          # 打包配置（fusion_tem + tfmag）
├─ configs/                # YAML 扫描配置（scan_full_grid.yaml）
├─ src/
│  ├─ fusion_tem/          # 领域库
│  │   ├─ device.py        # 全局参数/器件配置（脚本中 `import ... as cfg`）
│  │   ├─ materials.py     # 材料物性
│  │   ├─ utils.py         # 电感/D 形几何等公共函数（原根目录 utils.py）
│  │   ├─ em/              # 电磁：cp1–cp6、场校核、磁化损耗接口
│  │   ├─ cryo/heat_load.py# 热负荷与制冷电功率模型
│  │   ├─ economic/lcoe.py # 经济模型（compute_case / define_parameters）
│  │   └─ plotting/library.py (+ library2.py)  # 绘图与等高线（原 plot_library.py）
│  └─ tfmag/               # 基础设施：paths.py（repo 根/IO）、manifest.py
├─ scripts/                # 管线脚本，按阶段分子目录（保留编号）
│  ├─ 1_inductance/  2_charging/  3_comsol/  4_heatload/
│  ├─ 5_cooldown/   6_thermal_processing/   7_operation/
│  ├─ 8_economic/   （scan_full_grid.py 在此）   9_figures/
├─ data/  raw/ · processed/     # 原始/标准化输入（大文件不入 git）
├─ outputs/  figures/ · tables/ · logs/   # 生成产物（git 忽略，可重算）
└─ results/                     # 扫描原始结果（大文件不入 git）
```

## 管线阶段（scripts/）

| 阶段目录 | 内容 | 关键脚本 |
|---|---|---|
| `1_inductance/` | TF 电感矩阵计算与可视化 | `1.0_calculate_inductance.py` |
| `2_charging/` | 充电仿真、99.9% 充电时间、可用度 | `2.5_charge_time999_TF_system_all_Npw=1-200.py`、`2.9_af_time999_3&3.py` |
| `3_comsol/` | 生成 COMSOL 输入 | `3.0_generate_comsol_files.py` |
| `4_heatload/` | 热负荷分解与拼图 | `4.10_heatload_2bars.py`、`4.11_stitch_relative_heatload.py` |
| `5_cooldown/` | 降温分析 | `5.0_analyze_cooldown.py` |
| `6_thermal_processing/` | 热工况数据后处理 | `6.1/6.2/6.3_*` |
| `7_operation/` | 运行/失超分析 | `7.2_analyze_operation.py`、`7.3_analyze_quench.py` |
| `8_economic/` | **全网格扫描 + 经济分析**（所有图的数据源） | `scan_full_grid.py`、`8.1_plot_from_scan_full_grid.py` |
| `9_figures/` | 经济图拼接 | `9.0_stitch_econimic_svgs.py` |

## 快速开始

1) 将输入数据放入 `data/raw/`（见下文）。
2) 编辑扫描参数 `configs/scan_full_grid.yaml`。
3) 从仓库根目录运行统一入口（默认论文基线 `arc_16pancake_nuc600`）：

```bash
python scripts/run_scan.py
python scripts/run_scan.py --steps scan,figures
```

输出保存到 `outputs/arc_16pancake_nuc600/tables/scan_full_grid_tidy_plant_opex_2025usd_direct_hts.csv` 和同一器件目录下的图件与清单。

提交前运行快速回归测试：

```bash
cd code
python -m unittest discover -s tests -v
```

> **路径映射（本 README 下文历史小节沿用旧名）**：`config.py` → `src/fusion_tem/device.py`；`plot_library.py` → `src/fusion_tem/plotting/library.py`；根目录 `utils.py` → `src/fusion_tem/utils.py`；根目录编号脚本 `N.x_*.py` → `scripts/<阶段>/N.x_*.py`。下文引用的行号为历史值，仅供定位函数名。

## 关键输入（data/raw）

请按现有数据组织以下内容：

- 电感矩阵等相关数据（如 `data/raw/inductance/`）
- 不同温度的电磁损耗数据：
  - `data/raw/heat_input/data_s2_radial_loss.xlsx` (Data S2)
  - `data/raw/heat_input/data_s3_magnetization_loss.xlsx` (Data S3)
- 热工况数据：
  - `data/raw/charge/`, `data/raw/operation/`, `data/raw/cooldown/`, `data/raw/quench/`

## 扫描配置

编辑 `configs/scan_full_grid.yaml`，设置：

- `temp_coolant_pairs`
- `npw_values`
- `rho_turn_uohm_cm2_values`
- `r_joint_nohm_values`
- `scenarios`

扫描会直接使用这些参数；`scan_full_grid.py` 中的扫描逻辑保持不变。

## 核心数据生成：`scan_full_grid.py`

`scan_full_grid.py` 是**所有绘图的基础数据源**，它执行完整的参数网格扫描，生成包含所有设计参数组合及其计算结果的统一数据表。

### 扫描维度

脚本扫描以下参数空间的所有组合：

- **温度-制冷剂组合** (`temp_coolant_pairs`)：
  - (4.2 K, He)
  - (10 K, He)
  - (20 K, He)
  - (20 K, H₂)
- **并绕根数** (`npw_values`)：38个值，即1–20的每个整数及30、40、…、200
- **匝间电阻率** (`rho_turn_uOhm_cm2_values`)：21个值，即10–100（步长10）、200–1000（步长100）、5000和10000 μΩ·cm²
- **线圈间接头电阻** (`r_joint_nohm_values`)：1–100 nΩ的31个对数均匀值
- **技术情景** (`scenarios`)：S1–S6

### 输出数据

脚本默认生成CSV和清单；仅在设置 `WRITE_SCAN_XLSX=1` 时额外生成Excel：

1. **`outputs/<device>/tables/scan_full_grid_tidy_plant_opex_2025usd_direct_hts.csv`**（主要输出）
   - 长表格式（tidy format）：每一行代表一个参数组合及其所有计算结果
   - 包含所有计算指标（见下文）
   - 所有后续绘图脚本的**唯一数据源**

2. **`outputs/<device>/tables/scan_full_grid_plant_opex_2025usd_direct_hts.xlsx`**（可选）
   - 仅供内部便览；投稿正式数据仍以CSV为准
   - 通过环境变量 `WRITE_SCAN_XLSX=1` 显式启用

### 计算指标

对于每个参数组合，脚本计算以下指标：

#### 1. 充电时间指标
- `Charging_time_999_h` / `t_charge_999_h`：充电到99.9%电流的时间（小时）

#### 2. 电磁损失指标
- `Mag_loss_at_charge_W`：充电结束时刻的磁化损耗功率（W）
- `Radial_loss_at_charge_W`：充电结束时刻的径向损耗功率（W）
- `Mag_loss_energy_MWh`：充电过程中的磁化损耗总能量（MWh）
- `Radial_loss_energy_MWh`：充电过程中的径向损耗总能量（MWh）

#### 3. 热学指标
- `Total_heat_Tc_W`：运行温度（Top）下的总热负荷（W）
- `Total_heat_77K_W`：77K温区的总热负荷（W）
- `P_cryo_electric_W`：制冷系统电功率（W）
- `COP_Tc` / `COP_77`：运行温度/77K的制冷系数
- 各热源详细功率：`Coil_internal_joint_W`, `Pancake_joint_W`, `Nuclear_heat_W`, `Radiation_W` 等

#### 4. 电磁指标
- `R_radial_per_pancake_Ohm`：单个饼状线圈的径向电阻（Ω）
- `R_radial_per_TF_Ohm`：单个TF磁体的径向电阻（Ω）
- `R_radial_system_Ohm`：TF系统的径向电阻（Ω）

#### 5. 经济学指标
- `LCOE_plant_USD_per_MWh`：当前论文唯一主 LCOE，采用 `plant_v4_core_pcs_coolant_2025usd_direct_hts` 成本边界（USD/MWh）
- `LCOE_magnet_only_USD_per_MWh`：仅含年化磁体 CAPEX 与制冷剂补充的 legacy diagnostic
- `LCOE_fullplant_legacy_USD_per_MWh` / `LCOE_fullplant_USD_per_MWh`：升级前 C0 + magnet-only 公式，仅用于前后诊断
- `C0_background_USD`、`CAPEX_mag_direct_USD`、`CAPEX_mag_installed_USD`：标准化资本成本边界
- `Annualized_CAPEX_mag_USD_per_year`、`Annualized_C0_USD_per_year`、`Annualized_CAPEX_total_modeled_USD_per_year`：年化资本成本
- `OPEX_core_VOM_USD_per_year`、`OPEX_PCS_VOM_USD_per_year`、`OPEX_PCS_FOM_USD_per_year`、`OPEX_coolant_VOM_USD_per_year`：四项年度 OPEX
- `OPEX_total_modeled_USD_per_year`：四项 OPEX 之和
- `Tape_cost_USD`：带材成本（USD）
- `Coolant_fill_cost_USD`：制冷剂填充成本（USD）
- `Power_supply_cost_USD`：单一串联 TF 电源的电流容量成本代理（USD；不解析电压和储能相关设备成本）
- `AF`：可用因子（Availability Factor）= 年度生产小时数 / 年度总小时数
- `r_cryo_re`：年度低温耗电量占年度毛发电量的比例（%）；毛电功率采用 708 MW(th) × 0.40 = 283.2 MW(e)
- 制冷功率分解：`P_cryo_prod_W`, `P_cryo_dwell_W`, `P_cryo_static_W`, `P_cryo_coolwarm_W`, `P_cryo_excdec_W`
- 年度能量：`E_fusion_th_year_MWh`、`E_gross_year_MWh`、`E_cryo_year_MWh`、`E_other_year_MWh`、`E_net_year_MWh`

#### 6. 衍生指标
- `AF_max_scenario`：同一技术情景全部成功求解设计中的最大绝对可用因子；S4-S6 显式沿用 S2
- `AF_ref_system`：系统级相对可用因子 = `AF / AF_max_scenario`；兼容字段 `AF_ref` 与其完全一致
- `feasible_availability_090/095/099`：三个可用度阈值的敏感性标志；主分析采用 `099`
- `net_export_fraction`：`E_net_year_MWh / E_gross_year_MWh`，仅在毛发电量为正且数值有限时定义
- `feasible_net_export`：旧净输出比例诊断，不参与正式联合可行性
- `feasible_joint`：`AF_ref_system ≥ 0.99`、`r_cryo_re_fraction ≤ 0.50`、主 LCOE 有限且无模型错误；兼容字段 `feasible` 与其完全一致
- `charging_time_warning_120h`：充电时间超过 120 h 的组件级诊断，不参与联合可行性
- `feasible_charge_time`、`feasible_r_cryo_re`：仅保留为 legacy/diagnostic 字段
- `LCOE_reference_joint_feasible`：同一情景联合可行设计中的最低 `LCOE_plant_USD_per_MWh`
- `delta_LCOE_joint_feasible_USD_per_MWh`：相对于上述联合可行基准的增量；兼容字段 `delta_LCOE_min_USD_per_MWh` 与其一致
- `feasibility_flags`：availability、net export、model error 与 120 h warning 的文本诊断

### 数据使用

**当前经济分析与绘图脚本必须从 `scan_full_grid_tidy_plant_opex_2025usd_direct_hts.csv` 读取数据；`scan_full_grid_tidy_equal_charge_discharge.csv`、`scan_full_grid_tidy_AFref099.csv` 和更早文件仅作版本追溯**，包括：

- `8.1_plot_from_scan_full_grid.py`：从扫描表生成各种热力图
- `8.0_run_economic_analysis_unified.py`：经济分析图（实际上也依赖扫描表）
- `9.0_stitch_econimic_svgs.py`：拼接经济分析图

### 配置

扫描参数通过 `configs/scan_full_grid.yaml` 配置文件设置，包括：

- **`temp_coolant_pairs`**：温度-制冷剂组合列表
  - 默认值（`config.py`）：`[(4.2, "He"), (10.0, "He"), (20.0, "He"), (20.0, "H2")]`
  - 共4个组合：4.2K He、10K He、20K He、20K H₂

- **`npw_values`**：并绕根数数组
  - 当前投稿配置：`[1, 2, ..., 20, 30, 40, ..., 200]`（共38个值）
  - 取值由 `configs/scan_full_grid.yaml` 显式定义
  - 取值范围：通常为 1-200

- **`rho_turn_uOhm_cm2_values`**：匝间电阻率数组（支持对数刻度配置）
  - 当前投稿配置：`[10, 20, ..., 100, 200, 300, ..., 1000, 5000, 10000]`（共21个值）
  - YAML配置示例（显式列表）：`[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 5000, 10000]`
  - 支持对数刻度配置（YAML格式）：
    ```yaml
    rho_turn_uOhm_cm2_values:
      logspace:
        start: 10
        stop: 10000
        num: 31
        include: [5000]  # 可选：额外包含的值
    ```
  - 取值范围：通常为 10-10000 μΩ·cm²
  - 注意：如果配置中不包含 5000.0，脚本会自动添加

- **`r_joint_nohm_values`**：线圈间接头电阻数组（对数刻度）
  - 默认值（`config.py`）：31个对数均匀分布的值，范围 1-100 nΩ
    ```
    [1.000, 1.166, 1.359, 1.585, 1.848, 2.154, 2.512, 2.929, 3.415, 3.981,
     4.642, 5.412, 6.310, 7.356, 8.577, 10.000, 11.659, 13.594, 15.849, 18.478,
     21.544, 25.119, 29.286, 34.145, 39.811, 46.416, 54.117, 63.096, 73.564, 85.770,
     100.000]
    ```
  - 取值范围：1-100 nΩ（对数均匀分布，共31个点）

- **`scenarios`**：技术情景列表
  - 默认值（`config.py`）：`['S1', 'S2', 'S3', 'S4', 'S5', 'S6']`（所有可用情景）
  - YAML配置示例：`['S1', 'S2', 'S3']`（通常只使用前三个情景）
  - 可用情景：
    - **S1, S2, S3**：正文主情景（运行与经济参数随技术成熟度变化）
    - **S4**：HTS带材价格敏感性（100 constant-2025-US$ kA⁻¹ m⁻¹；其余参数完全同S2）
    - **S5**：HTS带材价格敏感性（50 constant-2025-US$ kA⁻¹ m⁻¹；其余参数完全同S2）
    - **S6**：HTS带材价格敏感性（10 constant-2025-US$ kA⁻¹ m⁻¹；其余参数完全同S2）

如果配置文件不存在，脚本将使用 `config.py` 中的默认值。

### Resume 功能

脚本支持断点续传：
- 如果输出文件已存在，脚本会自动检测已计算的组合并跳过
- 只计算新的参数组合，大大节省时间

### 性能优化

- 使用 `@lru_cache` 缓存重复计算（充电时间、电阻、热负荷等）
- 预加载充电时间DataFrame和电磁损失数据（每个温度只读一次）
- 批量写入CSV（每500行flush一次）

## 绘图流程

### 基于扫描表的绘图

绘图应优先使用 tidy 扫描表：

- `8.1_plot_from_scan_full_grid.py` 读取 `outputs/<device>/tables/scan_full_grid_tidy_plant_opex_2025usd_direct_hts.csv`
- 经济分析图：`8.0_run_economic_analysis_unified.py`
- 拼接图：`9.0_stitch_delta_lcoe_min_svgs.py`

### 电感矩阵热力图

电感矩阵热力图用于可视化TF系统的电感矩阵，展示不同线圈之间的互感关系。

**绘制脚本：**
- `1.2_plot_inductance_TF_system.py`：绘制TF系统电感矩阵热力图（支持不同温度和Npw）

**输入数据：**
- 电感矩阵数据文件：`data/raw/inductance/TF_system_L_matrix.xlsx`
- 需要先运行 `1.0_calculate_inductance.py` 生成电感矩阵数据

**输出位置：**
- `outputs/figures/inductance/`

**输出文件命名：**
- `Npw={npw}_Top={temp}K_heatmap.svg`
- 例如：`Npw=1_Top=4.2K_heatmap.svg`, `Npw=20_Top=10K_heatmap.svg`, `Npw=100_Top=20K_heatmap.svg`

**扫描参数：**
- **温度**：从 `config.py` 的 `Ip_list` 获取，默认 `[4.2, 10.0, 20.0]` K
- **并绕根数 (Npw)**：默认 `[1, 20, 100]`（可在脚本中修改 `NPW_LIST`）

**特点：**
- 支持不同温度下的电感矩阵可视化（虽然电感矩阵本身不随温度变化，但不同温度下使用的带材根数不同，影响缩放系数）
- 使用对数刻度显示电感值（lg(Inductance [H])），提高可读性
- 热力图中标注显示原始电感值（最多4位有效数字，不使用科学计数法，避免重叠）
- 使用YlGnBu配色方案

**拼接功能：**
脚本运行完成后会自动调用 `stitch_inductance_heatmaps()` 函数，将选定的子图拼接成一个大图：

- **布局**：2列3行（竖着拼接）
- **左边列**：Npw=1的三个温度（从上到下：4.2K, 10K, 20K）- 编号 **(a), (b), (c)**
- **右边列**：Npw=20的三个温度（从上到下：4.2K, 10K, 20K）- 编号 **(d), (e), (f)**
- **输出文件**：`outputs/figures/inductance/inductance_matrix_grid_Npw1&20.svg`
- **标签位置**：每个子图左上角显示编号标签（a-f）

**相关脚本：**
- `1.0_calculate_inductance.py`：计算电感矩阵（需要先运行）
- `1.1_plot_inductance_single_TF.py`：绘制单个TF的电感矩阵热力图

### 热工况后处理与绘图

- `5.0_analyze_cooldown.py`
- `6.1_process_charge_data.py`
- `6.2_process_operation_data.py`
- `6.3_process_all_thermal_data.py`
- `7.2_analyze_operation.py`
- `7.3_analyze_quench.py`

图像格式由 `config.py` 统一设置为 `svg`。


## 等高线计算说明

本文档详细说明热力图中等高线的计算流程和相关函数的使用方法。等高线计算主要用于经济分析热力图（如 ΔLCOE 热力图、寄生功耗热力图等），确保等高线在视觉空间中均匀分布且数值"整齐"（如整数、1-2-5×10^k 等）。

### 计算流程概述

等高线计算采用**视觉空间均匀分布 + 数值对齐**的两阶段策略：

1. **第一阶段：视觉空间均匀分布**

   - 将数据值转换到归一化空间（norm空间，即视觉空间）
   - 在视觉空间中均匀选择目标位置
   - 逆变换回数据空间得到初始等高线级别
2. **第二阶段：数值对齐**

   - 从"整齐候选集"中选择最接近目标位置的数值
   - 确保等高线数值易于阅读（如整数、1-2-5×10^k 等）
   - 保持视觉空间中的均匀分布

### 核心函数详解

#### 1. `auto_contour_levels()` - 自动生成等高线级别

**位置：** `plot_library.py` (第596行)

**功能：** 为单个子图自动生成4-6条等高线，在视觉空间（norm空间）中均匀分布。

**参数说明：**

- `Z_sub`: 子图数据数组（2D numpy数组）
- `norm`: Matplotlib归一化对象（必须支持 `inverse`方法，如 `SymLogNorm`、`LogNorm`）
- `n_min`: 最小等高线数量（默认4）
- `n_max`: 最大等高线数量（默认6）
- `q`: 分位数范围，用于避免极值挤压（默认0.03，即使用3%-97%分位数）
- `margin`: 视觉空间中的边距比例（默认0.03，即3%）
- `include_zero`: 如果数据跨0，是否将最接近0的等高线替换为0.0（默认True）
- `verbose`: 是否输出详细信息（默认False）

**计算步骤：**

1. 展平数据并去除NaN/inf，获取有效数据范围（z_min, z_max）
2. 将有效数据转换到norm空间（视觉空间），使用分位数范围（3%-97%）避免极值挤压
3. 根据视觉空间范围自适应决定等高线数量（n_min到n_max之间）
4. 在视觉空间中留边距并取等距点（不包含端点）
5. 逆变换得到原始等高线级别（raw levels）
6. 调用 `snap_levels_to_nice()`将原始级别对齐到整齐数值

**返回值：** 等高线级别数组（已排序，不包含zmin/zmax）

**使用示例：**

```python
contour_levels = auto_contour_levels(
    Z_sub=Z,  # 2D数据数组
    norm=norm,  # SymLogNorm或LogNorm对象
    n_min=4,
    n_max=6,
    q=0.03,
    margin=0.03,
    include_zero=True,
    verbose=False
)
```

---

#### 2. `snap_levels_to_nice()` - 将等高线对齐到整齐数值

**位置：** `plot_library.py` (第1189行)

**功能：** 从"整齐候选集"中选择最接近目标位置的数值，确保等高线数值易于阅读。

**参数说明：**

- `levels`: 原始等高线级别数组（来自 `auto_contour_levels`的raw levels）
- `z_min`: 数据最小值
- `z_max`: 数据最大值
- `n_min`: 最小等高线数量（默认4）
- `prefer_integers`: 是否优先选择整数（默认True）
- `norm`: Matplotlib归一化对象（可选，用于视觉空间映射）
- `verbose`: 是否输出详细信息（默认False）

**整齐候选集生成策略：**

1. **Local Grid（局部网格）**：根据数据范围选择适当的步长（20, 10, 5, 2, 1, 0.5, 0.2），生成整数或整倍数网格
2. **Coarse 1-2-5**：生成 `{1, 2, 5} × 10^k` 的整齐刻度，覆盖多个数量级
3. **扩展候选集**（如果不够）：
   - 加入 `{1, 1.5, 2, 3, 5, 7} × 10^k` 的密集候选值
   - 最后将原始levels也加入候选集（fallback）

**选择算法：**

- 使用**回溯算法（backtracking）**从候选集中选择n条等高线
- 确保在视觉空间中均匀分布（通过 `norm`映射）
- 防止等高线过于拥挤（最小间隔：`t_gap_min = t_span / (n * 4.0)`）
- **上端锚定（Upper Anchor）**：尽量将最后一条等高线推到接近z_max的整齐值

**返回值：** 对齐后的等高线级别数组（已排序，数值"整齐"）

**使用示例：**

```python
nice_levels = snap_levels_to_nice(
    levels=raw_levels,  # 来自auto_contour_levels的原始级别
    z_min=z_min,
    z_max=z_max,
    n_min=4,
    prefer_integers=True,
    norm=norm,
    verbose=False
)
```

---

#### 3. `filter_contour_levels_to_target_count()` - 筛选等高线到目标数量

**位置：** `plot_library.py` (第1407行)

**功能：** 筛选等高线到目标数量（4-5条），用于多子图场景下统一等高线数量。

**参数说明：**

- `levels`: 当前等高线级别数组（已排序）
- `z_min`: 数据最小值
- `z_max`: 数据最大值
- `master_levels`: 可选，主等高线数组，用于补充等高线
- `target_min_count`: 目标最小等高线数量（默认4条）
- `target_max_count`: 目标最大等高线数量（默认5条）
- `verbose`: 是否输出详细信息

**筛选策略：**

1. **如果超过目标最大数量**：

   - 保留最大值和最小值
   - 从中间部分均匀选择剩余的等高线
   - 例如：从10条筛选到5条，保留min和max，从中间8条中选择3条
2. **如果少于目标最小数量**：

   - 尝试从 `master_levels`中补充等高线
   - 优先选择中间位置的等高线

**返回值：** 筛选后的等高线级别数组

**使用示例：**

```python
filtered_levels = filter_contour_levels_to_target_count(
    levels=contour_levels,
    z_min=z_min,
    z_max=z_max,
    master_levels=global_levels,  # 可选
    target_min_count=4,
    target_max_count=5,
    verbose=False
)
```

---

#### 4. `determine_contour_color_by_background()` - 确定单条等高线颜色

**位置：** `plot_library.py` (第1546行)

**功能：** 根据等高线路径上背景色的亮度，动态确定等高线颜色（黑色或白色）。

**参数说明：**

- `contour_path`: 等高线路径点数组，形状为 `(N, 2)`，每行为 `(x, y)`坐标
- `X`: 数据网格的X坐标，形状为 `(ny, nx)`
- `Y`: 数据网格的Y坐标，形状为 `(ny, nx)`
- `Z`: 数据值数组，形状为 `(ny, nx)`，用于确定背景色
- `cmap`: 颜色映射名称（字符串）
- `norm`: Matplotlib归一化对象，用于将Z值映射到颜色
- `sample_points`: 在等高线路径上采样的点数（默认20）
- `luminance_threshold`: 亮度阈值（0-1），超过此值使用黑色，否则使用白色（默认0.5）

**计算步骤：**

1. 在等高线路径上均匀采样点（默认20个点）
2. 对每个采样点：
   - 使用双线性插值获取该点的Z值（考虑对数轴的特殊处理）
   - 将Z值通过 `norm`归一化到[0,1]范围
   - 通过 `cmap`获取颜色（RGBA格式）
   - 计算亮度：`L = 0.299*R + 0.587*G + 0.114*B`（标准RGB到亮度转换公式）
3. 计算所有采样点的平均亮度
4. 根据平均亮度决定颜色：
   - 如果 `avg_luminance >= luminance_threshold`：返回 `'black'`（背景色浅，使用黑色等高线）
   - 否则：返回 `'white'`（背景色深，使用白色等高线）

**返回值：** 颜色字符串（`'black'`或 `'white'`）

**使用示例：**

```python
color = determine_contour_color_by_background(
    contour_path=path_points,  # numpy数组，形状(N, 2)
    X=X, Y=Y, Z=Z,
    cmap='viridis',
    norm=norm,
    sample_points=20,
    luminance_threshold=0.5
)
```

---

#### 5. `get_contour_colors_by_background()` - 批量确定等高线颜色

**位置：** `plot_library.py` (第1688行)

**功能：** 为等高线集合中的每条等高线确定颜色（基于背景色亮度）。

**参数说明：**

- `contour_set`: matplotlib contour对象（`QuadContourSet`）
- `X`: 数据网格的X坐标
- `Y`: 数据网格的Y坐标
- `Z`: 数据值数组
- `cmap`: 颜色映射名称（字符串）
- `norm`: Matplotlib归一化对象
- `sample_points`: 每条等高线路径上采样的点数（默认20）
- `luminance_threshold`: 亮度阈值（默认0.5）

**计算步骤：**

1. 遍历 `contour_set`中的所有等高线级别
2. 对每条等高线：
   - 收集该等高线的所有路径段（`contour_set.allsegs[i]`）
   - 合并所有路径段为一条完整路径
   - 调用 `determine_contour_color_by_background()`确定颜色
3. 返回字典：`{level: color}`

**返回值：** 字典，`{level: color}`，其中level是等高线值，color是 `'black'`或 `'white'`

**使用示例：**

```python
# 先临时绘制等高线以获取路径信息
temp_CS = ax.contour(X, Y, Z, levels=levels, colors="black", linewidths=0.01)
# 获取每条等高线的颜色
level_colors = get_contour_colors_by_background(temp_CS, X, Y, Z, cmap, norm)
# 清除临时等高线
for collection in temp_CS.collections:
    collection.remove()

# 使用颜色逐条绘制等高线
for level in levels:
    color = level_colors.get(level, 'black')
    CS_level = ax.contour(X, Y, Z, levels=[level], colors=color, linewidths=2)
    ax.clabel(CS_level, inline=True, fontsize=12, fmt="%.1f", colors=color)
```

---

#### 6. `format_contour_label()` - 格式化等高线标签

**位置：** `plot_library.py` (第2208行，在 `plot_delta_lcoe_heatmap_single`函数内部)

**功能：** 格式化等高线标签，根据数值范围自动调整精度，确保不重复，并去掉末尾的0。

**格式化策略：**

1. **根据数值范围确定精度**：

   - `max_val >= 10`：使用整数或1位小数（如 `10`或 `10.5`）
   - `max_val >= 1`：使用1-2位小数（如 `1.5`或 `1.25`）
   - `max_val >= 0.1`：使用2-3位小数（如 `0.15`或 `0.125`）
   - `max_val < 0.1`：使用3-4位小数（如 `0.015`或 `0.0125`）
2. **去掉末尾的0**：

   - `1.0` → `1`
   - `1.50` → `1.5`
   - `1.5` → `1.5`
3. **去重处理**：

   - 如果格式化后出现重复标签，保留更"整齐"的值（更接近整数）
   - 例如：如果 `1.0`和 `1.00`都格式化为 `1`，保留更接近整数的值

**使用示例：**

```python
def format_contour_label(x, all_levels):
    """格式化等高线标签"""
    # ... 根据数值范围确定精度 ...
    formatted = f"{x:.1f}"  # 或其他精度
    return remove_trailing_zeros(formatted)  # 去掉末尾的0

# 在clabel中使用
CS.clabel(inline=True, fmt=format_contour_label_final, ...)
```

---

### 完整计算流程示例

以下是在 `plot_delta_lcoe_heatmap_single()`函数中的完整等高线计算流程：

```python
# 1. 自动生成等高线级别（如果启用）
if draw_contours and auto_generate_levels and contour_levels is None:
    contour_levels = auto_contour_levels(
        Z_sub=Z,
        norm=norm,
        n_min=4,
        n_max=6,
        q=0.03,
        margin=0.03,
        include_zero=True,
        verbose=False
    )

# 2. 筛选等高线到目标数量（可选）
if len(contour_levels) > 5:
    contour_levels = filter_contour_levels_to_target_count(
        levels=contour_levels,
        z_min=z_min,
        z_max=z_max,
        target_min_count=4,
        target_max_count=5
    )

# 3. 去重和格式化标签
# ... format_contour_label逻辑 ...

# 4. 确定等高线颜色（基于背景色亮度）
temp_CS = ax.contour(X, Y, Z, levels=filtered_levels, colors="black", linewidths=0.01)
level_colors = get_contour_colors_by_background(temp_CS, X, Y, Z, cmap, norm)
# 清除临时等高线
for collection in temp_CS.collections:
    collection.remove()

# 5. 按颜色逐条绘制等高线
for level in filtered_levels:
    color = level_colors.get(level, 'black')
    CS_level = ax.contour(X, Y, Z, levels=[level], colors=color, linewidths=2)
    labels_level = ax.clabel(CS_level, inline=True, fontsize=12, 
                             fmt=format_contour_label_final, colors=color)
```

---

### 关键设计理念

1. **视觉空间均匀分布**：等高线在归一化空间（视觉空间）中均匀分布，确保在不同数据范围下都能获得良好的视觉效果。
2. **数值对齐**：通过 `snap_levels_to_nice()`将等高线对齐到"整齐"数值（整数、1-2-5×10^k等），提高可读性。
3. **动态颜色**：根据背景色亮度动态选择等高线颜色（黑色或白色），确保等高线在任何背景下都清晰可见。
4. **智能标签格式化**：根据数值范围自动调整精度，去掉末尾的0，避免重复标签。
5. **容错处理**：使用分位数范围避免极值挤压，使用回溯算法确保总能找到合适的等高线级别。

---

### 相关文件

- **主要实现文件**：`plot_library.py`
- **调用示例**：
  - `plot_delta_lcoe_heatmap_single()` - ΔLCOE热力图
  - `plot_parasitic_heatmap_single()` - 寄生功耗热力图
- **配置参数**：`config.py`中的 `LCOE_CONTOUR_METHOD`、`CONTOUR_MIN_SPACING_USD`等

## Manifest

每次完整扫描会写入 `outputs/manifest.json`，内容包括：

- 代码版本（如存在 git hash）
- 配置文件 hash
- 输入数据 hash
- 输出文件 hash

如需手动刷新，重新运行 `scan_full_grid.py` 即可。

## Python 依赖

常用依赖包括：

- Python 3.9+
- numpy, pandas, matplotlib, scipy
- openpyxl（Excel 读写）
- pyyaml（读取扫描配置）

## 备注

- 所有路径均相对仓库根目录。
- 表格输出到 `outputs/tables/`，图像输出到 `outputs/figures/`。
