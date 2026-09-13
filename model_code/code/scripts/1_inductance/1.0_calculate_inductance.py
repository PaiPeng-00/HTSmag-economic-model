import numpy as np
import pandas as pd
import time
from pathlib import Path

# [修改] 导入全局配置文件，并使用别名 cfg
from fusion_tem import device as cfg
from fusion_tem.utils import (
    build_multicoil_Dshape_equiv,
    compute_group_matrix_gpu,
    aggregate_matrix_to_pancakes,
    plot_inductance_heatmap
)

def main():
    """主执行函数，计算不同Npw配置下的电感矩阵。"""
    start_time_total = time.time()
    
    script_dir = Path(__file__).parent.resolve()
    # [修改] 从全局配置读取输出文件夹名
    output_dir = script_dir / cfg.INDUCTANCE_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"电感矩阵将保存至: {output_dir}")

    # [修改] 从全局配置读取要遍历的Npw列表
    for Npw in cfg.NPW_LIST_EXP1:
        # 1. 根据当前的 Npw，调用函数获取动态的 ng 值
        ng_current = cfg.get_ng_for_npw(Npw)
        print(f"\n开始计算 Npw = {Npw} 的情况 (动态 ng = {ng_current})...")
        print(f"\n开始计算 Npw = {Npw} 的情况...")

        start_time_case = time.time()
        
        # 计算物理参数
        Nt_physical_per_pancake = cfg.N_TOTAL_TAPE // Npw
        pitch = cfg.R2 / Nt_physical_per_pancake if Nt_physical_per_pancake > 0 else 0

        # [修改] 所有参数均来自 cfg 对象
        pts_list, wts_list, info_list = build_multicoil_Dshape_equiv(
            Np=cfg.NP, Nt=Nt_physical_per_pancake, Nm=1, ng=ng_current,
            L1=cfg.L1, R1_base=cfg.R1, dR=pitch,
            thickness=cfg.WID, dist=cfg.DIST,
            points_per_turn=cfg.POINTS_PER_TURN
        )

        M_groups = compute_group_matrix_gpu(pts_list, wts_list)
        M_pancakes = aggregate_matrix_to_pancakes(M_groups, info_list, cfg.NP)

        filename_prefix = f"inductance_matrix_Np{cfg.NP}_Npw{Npw}_ng{ng_current}"
        excel_path = output_dir / f"{filename_prefix}.xlsx"
        pd.DataFrame(M_pancakes).to_excel(excel_path, index=False, header=False, engine="openpyxl")
        print(f"矩阵已保存至: {excel_path}")

        heatmap_path = output_dir / f"{filename_prefix}_heatmap.svg"
        plot_inductance_heatmap(
            M_pancakes,
            title=f'Inductance Matrix (Np={cfg.NP}, Npw={Npw})',
            save_path=str(heatmap_path)
        )
        
        end_time_case = time.time()
        print(f"✅ Npw = {Npw} 的计算完成，用时: {end_time_case - start_time_case:.2f} 秒")

    end_time_total = time.time()
    print(f"\n✅ 全部电感计算任务完成，总用时: {end_time_total - start_time_total:.2f} 秒")

if __name__ == "__main__":
    main()