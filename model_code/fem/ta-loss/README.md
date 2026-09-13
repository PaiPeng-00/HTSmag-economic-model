# ARC TF 线圈 T-A 磁化损耗模型 — 使用说明

> **给后续对话的 Claude**：读完即可理解并执行这个 T-A 磁化损耗模型。
> 上游=磁场模型（`../1-field-FEM/README.md` 给出 Ip(T)/Nt(T)）；下游=CP5导出→CP6经济。
> 操作手册：`skill comsol-magnet-livelink`。

---

## 0. 这个模型干什么

2D 轴对称 **T-A formulation**，解 96h 充电过程中 HTS 带材内的电流/场分布，算**磁化（迟滞）损耗**[W/coil]。
T-A = 把 REBCO 带材当无厚度薄壳上的 T-formulation（电流矢势）+ 全域 A-formulation（磁矢势）耦合，降维省算量。

**关键物理结论**：96h 慢充 → dB/dt 极小 → **磁化损耗小（~2W 量级）**；径向（匝间接触）损耗占主导。这是论文论点之一。

---

## 1. ⭐ 两个核心概念（务必分清：等效匝 ≠ 并绕）

### 1.1 等效匝 equ_turn —— 计算均匀化（Wang 2020, High Volt.）
**只是减少计算匝数/网格，不改物理线圈。**
- 把**中间**的 `nt` 个物理匝合并成 **1 个等效匝**：`Ic_eq = nt·Ic`，`Iop_eq = nt·Iop`（论文式7）。
- **内、外两侧的匝不合并**——那里径向场/电流剧烈变化（边缘效应、屏蔽电流大），要保留实匝+细网格保精度。
- **中间匝场平坦** → 合并成等效匝、粗网格。
- `nt` 典型 2-10（可更大，随匝数增加）。模型里体现为 Ic 表达式 `*equ_turn`。
- 参考：Wang et al. (2020), T-A formulation paper

### 1.2 并绕 Npw（论文里 p）—— 物理设计参数（Fu 2023, SUST）
**改变线圈电磁特性的真实设计选择。**
- 每匝**并绕 Npw 根带材** → **匝数/饼 = Nt/Npw**。
- 每匝电流 = **Npw·Ip**（工作电流 ×Npw）。
- **电感 ∝ 匝数² → ×(1/Npw²)**（匝数少了 Npw 倍）。NI 线圈并绕的好处：电感小→充电延迟小、ramp 快。
- **Npw 不改 Nt 总带材数**，只改 Nt 如何分配给各匝。模型里体现为 `par2` Npw。
- 参考：Fu et al. (2023), parallel-wound no-insulation paper

### 1.3 一句话区分
> **equ_turn 是为了算得快（把若干实匝当一匝算）；Npw 是真实并绕（决定匝数、电感、每匝电流）。两者独立。**

---

## 2. 关键守恒关系

**安匝固定**（由环向场要求，与温度无关）：
```
Nt · Nc · Ip = amp_turns = 8.4 MA      → 每饼 Nt·Ip = 8.4MA/Nc = 8.4e6/12 = 700 kA
```
**温度只改 Nt 与 Ip 的分配，不改乘积 Nt·Ip**（低温 Jc↑ → Ip↑、Nt↓）。来自磁场模型的权威分配：

| T(K) | Ip(A) | Nt(/饼) | Nc | 带材/线圈 |
|---|---|---|---|---|
| 4.2 | 700 | 1000 | 12 | 12000 |
| 10  | 583 | 1200 | 12 | 14400 |
| 20  | 400 | 1750 | 12 | 21000 |

> Nt 取整后载流比 0.704/0.699/0.696≈0.70；三段式 Nt3=Nt−2·30−2·70 全被 nt3=50 整除
> (800/1000/1550 → 16/20/31 等效匝)；模型等效匝 90/94/105 (vs 实匝 1000/1200/1750)。

---

## 3. 参数映射（T-A 模型 ↔ 物理量）

| 模型参数 | 物理含义 | ARC 取值 |
|---|---|---|
| `Nt` | **物理带材数/饼**（须=磁场模型 Nt(T)）| 1000/1200/1750 |
| `Nc`(或Np) | **饼数**（须=磁场模型 Nc）| 12 |
| `par3` 三段式 | nt1=1/nt2=10/nt3=50, Nt1=30/Nt2=70/Nt3=Nt−200 | Nt3 全整除50 |
| `par2` Npw | 并绕根数（匝数=Nt/Npw）| 扫描 5/10/20/100/200 |
| `par6` Ip | 单带电流 | 400/583/700 |
| `par6` T | 操作温度 | 4.2/10/20 K |
| `par7` rho_turn | 匝间接触电阻率 | 5000 µΩ·cm² |
| `par5` M11-M18 | 互感行（背景场耦合，来自 CP4）| ⚠️ 见下 |
| `Rin,R2,wid` | 等效圆线圈半径、**WP径向厚度**、带宽 | 4.14m/**0.64m**/12mm |

> ⭐ **R2 一致性**：R2 = WP 径向厚度，**必须 = Mf FEM 的 dr_tf = 0.64m**，且在
> Python(CP4 电感, R2_A)、Mf FEM(dr_tf)、TA 模型(R2) 三处统一为 **0.64m**。
> （曾不一致：TA=0.40, CP4=0.55, FEM=0.64 → 已全部改 0.64。CP4 L_loop scale 2.40，ARC M_ii=1.399e-5。）
| 输出 `int4` | **磁化损耗**[W/coil]（IntLine 数值节点）| |
| 输出 `gev9` | **径向损耗 total**[W/coil] = `Ir1²·Npw²·Rcoil·Np`（用对称 Ir1，自动适配 Np/Npw）| |

> ⚠️ 径向损耗用 **gev9**，不是 gev3/gev4（那俩用 Ntape 逐个 Ir 求和，已废）。**Ntape 不再需要**。

---

## 3b. 物理接口 + ge 电路（★ T-A 代表 1 个 TF，不是 18 个）

| 接口 | 类型 | 作用 |
|---|---|---|
| `gb` | GeneralFormBoundaryPDE | **T-formulation**（超导电流矢势 T，A/m）；源=−dB⊥/dt，含 E-J `Ephi` |
| `mf` | InductionCurrents | **A-formulation**（磁场）|
| `mf2` | InductionCurrents | 第二磁场接口，**⚠️ 当前未激活**（不参与求解，读/改模型须注意）|
| `ge` | GlobalEquations | **TF 系统电路**：18 个 Ia=18 个 TF 线圈环向电流 |

**后处理输出**：磁化损耗=`int4`(IntLine, Jphi·Ephi·thHTS·2)；**径向损耗 total=`gev9`**=`Ir1²·Npw²·Rcoil·Np`
（不是 gev3）。两者都因对称性只需 1 个代表量（Ir1）。**Ntape 已废弃不用**。

**ge 的物理意义**：T-A 只解 1 个 TF 线圈，但它处在 18 个 TF 环形系统里。ge 注入 **18×18 circulant
TF 互感矩阵**（自感+17 互感）的耦合。
- **18 个 TF 线圈电流完全相等**（circulant + 相同电流源驱动，𝟙 是特征向量；数值验证差异~1e-12A）。
- 故 ge 加权和 = Ia·**L_eff = Σ(互感行) = 1.63e-5H**（SPARC 例）；自感占 35.8%，**17 互感占 64.2%**。
- → T-A 用 1 个 TF 的 L_eff（已含 18 TF 耦合）。详见 `0-思路/2.7.2续-TF电路与TA模型ge接口.md`。
- **两级电感别混淆**：TF 系统级 18×18(`TF_system_L_matrix.xlsx`) vs 饼级 16×16(`inductance_matrix_Np16_*`)。

**ARC 重构待办**：ge 的 18 系数=SPARC 几何 TF 互感行 → 须用 ARC 几何重算 18×18 circulant 第一行替换。
Npw 经 `(Nt/Npw·Np)²` 和 `rho_eff` 已含依赖；M 矩阵用对应 Npw 版本。

---

## 4. ⚠️ 坑 / 注意

1. **提取磁化损耗用数值结果节点，不能用 mphglobal**：
   ```matlab
   m.result.table.create('tmag','Table');
   m.result.numerical('int4').set('table','tmag'); m.result.numerical('int4').set('data','dset1');
   m.result.numerical('int4').setResult;  D=mphtable(m,'tmag').data;  magpk=max(D(:,2));
   ```
   （`mphglobal(m,'int4')` 会返回 NaN——int4 是 Global Evaluation 节点名，不是变量。）
2. **几何匝数不能乱改**：直接改 Nt/Np 会让 geom 的 OpArray 崩。改结构要配合 equ_turn 谨慎做。
   实际根因=fullsize 非整除（Np/2、Nt3/nt3 必须整数）；整除则改 Np/Nt 不崩。
2b. **★ge8 有 36 个方程，不是 18**：18 个 Ia 方程(含互感和 `...-Ia_i`) + 18 个 Ir 方程
   (`current(t) - Ia_i - Ir_i`)。改 ge8 互感时**只能原地改 Ia 方程、必须保全 18 个 Ir 方程**；
   若只 set 18 个 → Ir1-Ir18 变空方程(void equations)→ 奇异矩阵、求解失败。变量顺序=Ia1-18,Ir1-18。
2c. **LiveLink 坑**：`getStringMatrix(...)(1)` 直接索引非法，先存临时变量；`m.param('parX').evaluate` 不存在，用全局 `m.param.evaluate`。
3. **wid 与 Ip 经 Ic 关联**：wid 变小→Ic 变小，若 Ip 不变会过临界→E-J 幂律发散（曾 wid=4mm 炸 1e10W）。
4. 单点求解慢（~18-28min/点），用 `run_in_background:true`，并**保存解后的 .mph**（别丢解）。

---

## 4b. ⚡ 求解器加速（72min→11min/点，损耗不变）

默认 `rtol=5E-6`（极紧）+ `initialstepbdf=1e-13` 导致单点 72min。优化（损耗与基准 0.1% 内）：
| 设置 | 默认 | **优化** | 作用 |
|---|---|---|---|
| `sol1/t1` rtol | 5E-6 | **1e-4** | 主提速 |
| `t1` maxstepbdf | 关 | **开, 4000s** (`maxstepbdfactive=true`) | ⭐关键：限步防 E-J 刚性伪峰 |
| `t1` initialstepbdfactive | true(1e-13) | **false**(自动) | 放开初始步 |
| `dis1` numelem (带材段) | 30 | **20** | 网格 37k→30k |
| `size2` hmax (带材) | 0.003 | **0.005** | |

→ **11.4 min/点**，磁化 18.14W(基准18.13)、径向 346.6W 全准。
**坑**：单放松 rtol(1e-3) 会让磁化损耗(n=28 幂律刚性)出伪峰(4593W)；**必须配 maxstep**。径向损耗对 rtol 鲁棒。
脚本 `run_arc_sweep15.m` 已内置这套设置。

## 5. 当前状态 / 待办

- 当前 `ARC_TA_loss.mph` 仍是 **SPARC 结构（Np=16, Nt=800）+ ARC 尺寸**——**结构未对上 ARC**。
- **待重构**：Nc=12、Nt=Nt(T)、equ_turn 均匀化（边缘留实匝）。等确认 equ_turn 取值 + Npw/M 矩阵缩放后做。
- 冒烟测试已证 Ip=400 在当前结构下**不发散**（关键验证过）。
- 重构后：15 点扫描（Npw×T，各温度用 Ip(T)）→ CP5 导出 → CP6。

---

## 6. 如何执行（MATLAB LiveLink）

服务器单 license，每次 matlab -batch 前重启服务器并配对（见 `../1-field-FEM/README.md` 第5节相同模式）。
脚本开头：`addpath('${COMSOL_ROOT}\mli'); mphstart(2036); import com.comsol.model.util.*;`

**现有脚本**：
- `build_arc_ta.m` — 从 SPARC 模板建 ARC 模型（改尺寸+物理，保留 SPARC 匝结构）
- `run_arc_sweep.m` — 15 点扫描（Npw×T），断点续跑
- `cp5_resolve_extract.m` — 单点重解+正确提取磁化损耗+存解
- `cp5_extract*.m` — 提取脚本

---

## 7. 参考文献
- Wang et al. 2020, *High Voltage* 5(2):218-226 — 等效匝 T-A 模型（`...9924K4LJ\...`）
- Fu et al. 2023, *Supercond. Sci. Technol.* — 并绕 NI 线圈电流分布（`...8LTUR2DC\...`）
