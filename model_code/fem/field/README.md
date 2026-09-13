# ARC TF 线圈 磁场 FEM 模型 — 使用说明

> **给后续对话的 Claude**：读完本文件即可直接理解并通过 MATLAB LiveLink 执行这个磁场模型，
> 无需重新摸索。配套操作手册：`skill comsol-magnet-livelink`；完整推导流程：`../../docs/FEM_field_workflow.md`。

---

## 0. 这个模型是干什么的

用 3D 磁场 FEM（COMSOL `mf` 模块）复现 ARC 聚变装置 TF 线圈的磁场，输出：
- **B0**（等离子体中心轴上场，应≈9.2T）、**Bmax**（导体峰值场，≈23-25T）
- **B⊥**（垂直场）、**载流比**（Iop/Ic）
- 据载流比≤0.7 反推**单带电流 Ip** 和**带材匝数 Nt**
- 多温度下的 **Ip(T)/Nt(T)**（喂给 T-A 磁化损耗模型 + 经济模型）

**几何简化思想**：1/(4·Ntf)=**1/72 扇区**对称（PMC 完美磁导体 + SymmetryPlane），只建**一个**均匀化 D 形导体，
`ICoil=1` 单位电流求解，**真实场 = mf.normB × Ip**（后处理缩放）。

---

## 1. 文件清单（只保留这 4 个 + 本说明）

| 文件 | 作用 |
|---|---|
| **`ARC_field_mangiarotti.mph`** | ⭐ 主模型（已解，含 Mangiarotti Jc、载流比绘图组、Top 温度参数）。所有操作 load 它 |
| `ARC_field_mangiarotti.m` | 模型的 .m 文本导出（读它可了解全部 geom/mesh/physics/后处理结构；也可重建模型）|
| `lock_ip400.m` | **改 Ip → 重解 → 确认 B0/Bmax/载流比** 的模板脚本 |
| `ipT_resolve.m` | **各温度算 Ip(T)/Nt(T)** 的脚本（载流比0.7）|

结果 CSV/日志写到 `../../results/`。

---

## 2. 当前锁定结果（2026-06-26）

**几何**：R10=0.70m(WP内边), dr_TF=0.64m(=Nt·thk), R1=0.30m, L1=7.24m, Nc=12, Ntf=18, wid=12mm, dist=3mm。
R0=R10+R1/2+L1/4+Nt·thk=3.30m。

**安匝**：`amp_turns = Nt·Nc·Ip = 8.4MA`（= B0·R0·2π/(μ0·Ntf)，**固定**，与 Ip/Nt 怎么分配无关）。

**锁定工作点（20K）**：Ip=400A, Nt=1750 → B0=9.16T ✓, Bmax=24.9T ✓, **载流比 0.695<0.7** ✓。

**Ip(T) 表**（载流比0.7，场固定，只 Jc 随温度变）：

| T(K) | min Jc(A/mm²) | Ic_tape(A) | **Ip** | **Nt** | 带材/线圈 |
|---|---|---|---|---|---|
| 4.2 | 828 | 994 | 700 | 1000 | 12000 |
| 10 | 695 | 834 | 583 | 1200 | 14400 |
| 20 | 479 | 575 | 400 | 1750 | 21000 |

→ **温度越低 Jc 越高 → Ip 越大 → 带材越少**（论文的 techno-economic 权衡核心）。

---

## 3. 关键变量 / 物理（写在 `var1`）

```
Br      = 投影到 D 形法向的垂直场分量（内侧直段 Br=-mf.Bx）  [单位电流]
Bperp_T = Ip*abs(Br)/1[T]/1[A]            % 真实垂直场[T]数值
Bmag_T  = Ip*mf.normB/1[T]/1[A]           % 真实总场[T]数值
Bpara_T = sqrt(max(Bmag_T^2-Bperp_T^2,0)) % 平行场
thetad  = atan2(Bpara_T,Bperp_T+1e-9)*180/pi
fperp=0.6781-0.0107*Bmag_T; fpara=0.7808-0.0105*Bmag_T; th0m=11.99-0.069*Bmag_T
Jcperp = 3268.2*Bmag_T^(-0.6442)*exp(-(Top-4.2)*log(1/fperp)/17.8)
Jcpara = (4086-71.859*Bmag_T)*exp(-(Top-4.2)*log(1/fpara)/17.8)
Jc_mang = Jcperp+(Jcpara-Jcperp)*exp(-(90-thetad)/th0m)   % [A/mm²] = REBCO带材Jc(Mangiarotti)
```
- **Mangiarotti = 带材 Jc**，分母=整根带材 **1.2mm²**(12mm×0.1mm)，含REBCO+铜+Hastelloy。对 ARC Table3 误差≤8%，**无需标定**。
- **单带 Ic = Jc_mang × 1.2** [A]；**载流比 = `Ip/(Jc_mang*wid/10[mm])`**（注意括号在分母！wid/10mm=1.2 数值上=带材面积）。
- **`Top`** = 操作温度[K]，纯数参数；改它即可换温度。

---

## 4. ⚠️ 必须知道的坑

1. **安匝 = Nt·Nc·Ip（含 z 对称）**，不是 N·Ip（coil 的 N=Nt·Nc/2）。曾因此场偏高 2×。
2. **参数快照 bug**：COMSOL 在已解 dataset 上求值，用的是**求解时的参数快照**。改 `Top` 后 `mphmin(Jc_mang)` 不变！
   → 换温度必须 **`m.sol('sol1').runAll` 重解一次**（场不变，只刷新快照），再取值。`ipT_resolve.m` 已这样做。
3. **网格按尺度**：导体 hmax≈dr_TF/8，空气 hmax=50cm/hmin=1cm。**别用全局 hauto**（曾致 1500万单元）。
4. **几何不重叠扇区**：`Heig/2 < R10·tan(360/Ntf/2)`；Nc 太大会捅穿对称面发散 → 减小 Nc。
5. **载流比与 Ip 线性**（场固定）：lf=Ip/Ic_tape；Ip_max≈408A@lf=0.70。

---

## 5. 如何执行（MATLAB LiveLink）

**服务器单 license，每次 matlab -batch 前必须重启服务器并配对**（Bash 工具，PowerShell 起服务器）：

```bash
SRV='${COMSOL_ROOT}\bin\win64\comsolmphserver.exe'
powershell -Command "Get-Process comsolmphserver -EA SilentlyContinue | Stop-Process -Force -EA SilentlyContinue; \
  Start-Sleep 2; Start-Process -FilePath '$SRV' -ArgumentList '-silent' -WindowStyle Hidden"
# 等端口 2036 监听
for i in $(seq 1 30); do powershell -Command "(Get-NetTCPConnection -LocalPort 2036 -State Listen -EA SilentlyContinue) -ne \$null" | grep -q True && break; sleep 2; done
sleep 5
# 跑脚本（MATLAB 慢，用 run_in_background:true）
timeout 400 "D:/matlab/bin/matlab.exe" -batch "cd('<本文件夹>'); ipT_resolve()"
```
脚本内开头固定：`addpath('${COMSOL_ROOT}\mli'); mphstart(2036); import com.comsol.model.util.*;`
场求解 ~20s；中文乱码是终端编码问题，不影响数值。

**两个常用操作**：
- 改单带电流：编辑 `lock_ip400.m` 里 `Ip/Nt/thk`（保持 Nt·thk=0.64、Nt·Nc·Ip=8.4e6）→ 运行 → 看 B0/Bmax/载流比。
- 取某温度 Ip：运行 `ipT_resolve.m`（默认 4.2/10/20K）→ 读 `../../results/ip_vs_T_comsol.csv`。

---

## 6. 下游耦合

`mf 场模型 → 各温度 Ip(T)/Nt(T) → T-A 磁化损耗模型(../2-TA-loss) → CP5导出 → CP6 经济模型`。
**每换一个温度，先用本模型定 Ip(T)/Nt(T)，再进 T-A**（载流比≈0.7）。
