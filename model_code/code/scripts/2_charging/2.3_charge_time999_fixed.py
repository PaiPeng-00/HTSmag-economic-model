import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from fusion_tem import device as cfg
from matplotlib.colors import LogNorm, PowerNorm
import matplotlib.colors as colors

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'  # 普通字体
plt.rcParams['mathtext.it'] = 'Arial:italic'  # 斜体
plt.rcParams['mathtext.bf'] = 'Arial:bold'  # 粗体
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['savefig.transparent'] = True

# --- 1. 定义要绘制的固定组合参数 ---
# 请在这里指定您希望绘制的 Npw 值列表
# 例如: fixed_Npw_values = [1, 21, 41]
fixed_Npw_values = np.concatenate((np.arange(1, 10, 1), np.arange(10, 50, 5), np.arange(50, 110, 10)))

# 请在这里指定您希望绘制的匝间电阻率 (μΩ·cm²) 值列表
# 例如: fixed_rho_turn_values_uOhm_cm2 = [10, 215, 4641]
fixed_rho_turn_values_uOhm_cm2 = np.concatenate((np.arange(10, 100, 10), np.arange(100, 1100, 100), np.arange(2000, 11000, 2000))) # 匝间电阻率，从 10 到 10^4 微欧姆·厘米² (取值: 10, ~46, ~215, ~1000, ~4641, 10000)



# --- 2. 设置文件路径 ---
script_dir = Path(__file__).parent.resolve()
output_dir = script_dir / cfg.CHARGING_SIM_OUTPUT_DIR 
input_excel_path = output_dir / cfg.CHARGING_SIM_OUTPUT_FILE
figure_dir = Path(cfg.OUTPUTS_FIGURES_DIR) / "charging"
figure_dir.mkdir(parents=True, exist_ok=True)
output_plot_path = figure_dir / cfg.CHARGING_SIM_OUTPUT_PLOT_FILE

# --- 3. 从 Excel 文件读取全部数据 ---
print(f"正在从 Excel 文件读取数据: {input_excel_path}")
df_full_data = None
try:
    df_full_data = pd.read_excel(input_excel_path, index_col=0)
    # 确保列名和索引名与数值类型匹配
    df_full_data.columns = df_full_data.columns.astype(int)
    df_full_data.index = df_full_data.index.astype(float)
    print("✅ 数据成功读取。")
except FileNotFoundError:
    print(f"❌ 错误: 未找到文件 {input_excel_path}。请确保文件存在。")
    exit()
except Exception as e:
    print(f"❌ 读取文件时发生错误: {e}")
    exit()

# --- 4. 抽取固定的组合数据 ---
print("正在抽取固定组合的数据...")
# 筛选行 (匝间电阻率)
df_filtered_rows = df_full_data.loc[df_full_data.index.isin(fixed_rho_turn_values_uOhm_cm2)]
# 筛选列 (Npw)
df_fixed_combinations = df_filtered_rows[fixed_Npw_values]

# 将 DataFrame 转换为 NumPy 数组用于绘图
charging_time_matrix_for_plot = df_fixed_combinations.values
plot_Npw_values = df_fixed_combinations.columns.values
plot_rho_turn_values_uOhm_cm2 = df_fixed_combinations.index.values

print("✅ 数据抽取完成。")

# --- 5. 绘制热力图 ---
plt.figure(figsize=(12, 10))
norm = LogNorm(vmin=np.nanmin(charging_time_matrix_for_plot[charging_time_matrix_for_plot > 0]),
                 vmax=np.nanmax(charging_time_matrix_for_plot))
norm = colors.PowerNorm(gamma=0.2, vmin=np.nanmin(charging_time_matrix_for_plot),
                 vmax=np.nanmax(charging_time_matrix_for_plot))
print(np.nanmin(charging_time_matrix_for_plot), np.nanmax(charging_time_matrix_for_plot))
# 绘制热力图，移除单元格分隔
ax = sns.heatmap(
    charging_time_matrix_for_plot,
    norm=norm,
    annot=False, # 不在单元格内显示数值，等值线会显示
    fmt=".2f",
    cmap="cividis", # 使用 viridis 色图
    cbar_kws={'label': '99.9% charging time (hours)'},
    xticklabels=[str(int(x)) for x in plot_Npw_values],
    yticklabels=[f'{x:.0f}' for x in plot_rho_turn_values_uOhm_cm2]
)
# 取消 seaborn 默认的纵轴翻转（从下往上递增）
ax.invert_yaxis()  # 再次调用会反转回正常方向
# 获取 colorbar 对象
colorbar = ax.collections[0].colorbar

# 自定义刻度位置（colorbar 是按数据值显示的）
tick_values = [96, 100,120,200,500,1000,5000,10000]  # 可自由修改
colorbar.set_ticks(tick_values)
colorbar.set_ticklabels([f"{v:.0f}" for v in tick_values])
# 添加等值线
X_plot, Y_plot = np.meshgrid(np.arange(len(plot_Npw_values)), np.arange(len(plot_rho_turn_values_uOhm_cm2)))

# 定义等值线级别，确保等值线级别在数据的有效范围内
levels = [96,96.5,97,98,99,100, 110, 120, 130, 140, 150,  200, 300,500,1000,5000,10000]  # 自定义等值线值


if levels is not None:
    # 绘制等值线，并添加标签
    contour_lines = ax.contour(X_plot + 0.5, Y_plot + 0.5, charging_time_matrix_for_plot, 
                               levels=levels, colors='white', linestyles='dashed', linewidths=1.5)
    ax.clabel(contour_lines, inline=True, fontsize=14, fmt='%.2f')


plt.xlabel('Number of parallel wound ($N_{\mathrm{pw}}$)')
plt.ylabel('Turn resistivity ($\\rho_{\\mathrm{turn}}$, μΩ·cm²)')

#plt.title('99.9% charging time heatmap of fixed combinations (with contour lines)', fontsize=16)

plt.tight_layout() # 自动调整子图参数，使之填充整个图像区域
plt.savefig(output_plot_path, dpi=300) # 保存为高分辨率图片
plt.close()

print(f"\n✅ 热力图已保存为: {output_plot_path}")
