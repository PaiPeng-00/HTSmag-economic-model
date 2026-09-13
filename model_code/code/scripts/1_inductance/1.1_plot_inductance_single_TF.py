# utility_plot_inductance.py
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
# 导入全局配置，以便我们知道要处理哪些文件
from fusion_tem import device as cfg
# 从我们的库中导入绘图函数
from fusion_tem.utils import plot_inductance_heatmap
# =============================================================================
# 全局绘图配置
# =============================================================================
# 1. 首先设定一个基础样式
plt.style.use('default')
plt.rcParams['font.family'] = 'Arial' 
plt.rcParams['axes.labelsize'] =20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20

# [新增] 设置刻度线方向朝内
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['savefig.transparent'] = True
# =============================================================================
NPW_LIST = [1,20]
def main():
    """
    主执行函数：
    读取已计算好的电感矩阵.xlsx文件，并重新生成对应的热力图。
    """
    print("--- 开始执行电感矩阵可视化脚本 ---")
    
    script_dir = Path(__file__).parent.resolve()
    # 定义电感矩阵数据所在的输入目录
    input_dir = script_dir / cfg.INDUCTANCE_OUTPUT_DIR

    if not input_dir.exists():
        print(f"错误: 找不到输入目录 '{input_dir}'。")
        print("请先运行 '01_calculate_inductance.py' 来生成电感矩阵数据。")
        return

    # 从全局配置读取要遍历的Npw列表
    for npw in NPW_LIST:
        # 1. 根据当前的 Npw，调用函数获取动态的 ng 值
        ng_current = cfg.get_ng_for_npw(npw)
        
        print(f"\n正在处理 Npw = {npw} (ng = {ng_current}) 的情况...")

        # 2. 构建输入和输出文件的路径
        filename_prefix = f"inductance_matrix_Np{cfg.NP}_Npw{npw}_ng{ng_current}"
        excel_path = input_dir / f"{filename_prefix}.xlsx"
        plot_dir = Path(cfg.OUTPUTS_FIGURES_DIR) / 'inductance'
        plot_dir.mkdir(parents=True, exist_ok=True)
        heatmap_path = plot_dir / f"{filename_prefix}_heatmap.svg"

        # 3. 检查Excel文件是否存在
        if not excel_path.exists():
            print(f"  -> 警告: 未找到对应的Excel文件 '{excel_path.name}'，跳过。")
            continue

        # 4. 读取数据并绘图
        try:
            print(f"  -> 正在读取: {excel_path.name}")
            # 从Excel读取数据，并转换为Numpy矩阵
            m_pancakes = pd.read_excel(excel_path, header=None).values
            
            print(f"  -> 正在生成图像: {heatmap_path.name}")
            plot_inductance_heatmap(
                m_pancakes,
                title=f'Inductance Matrix (Np={cfg.NP}, Npw={npw})',
                save_path=str(heatmap_path)
            )
            print("  -> ✅ 处理完成。")

        except Exception as e:
            print(f"  -> ❌ 处理失败: {e}")

    print("\n--- 所有可视化任务完成 ---")

if __name__ == "__main__":
    main()