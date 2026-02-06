import numpy as np
import pandas as pd
import time
from pathlib import Path

import config as cfg
from utils import (
    build_multicoil_Dshape_equiv,
    compute_group_matrix_gpu,
    aggregate_matrix_to_pancakes,
    plot_inductance_heatmap
)

def main():
    """Compute inductance matrices for each Npw in config."""
    start_time_total = time.time()
    script_dir = Path(__file__).parent.resolve()
    output_dir = script_dir / cfg.INDUCTANCE_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Inductance matrices will be saved to: {output_dir}")
    for Npw in cfg.NPW_LIST_EXP1:
        ng_current = cfg.get_ng_for_npw(Npw)
        print(f"\nComputing Npw = {Npw} (ng = {ng_current})...")
        start_time_case = time.time()
        Nt_physical_per_pancake = cfg.N_TOTAL_TAPE // Npw
        pitch = cfg.R2 / Nt_physical_per_pancake if Nt_physical_per_pancake > 0 else 0
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
        print(f"Matrix saved: {excel_path}")

        heatmap_path = output_dir / f"{filename_prefix}_heatmap.svg"
        plot_inductance_heatmap(
            M_pancakes,
            title=f'Inductance Matrix (Np={cfg.NP}, Npw={Npw})',
            save_path=str(heatmap_path)
        )
        
        end_time_case = time.time()
        print(f"Npw = {Npw} done in {end_time_case - start_time_case:.2f} s")
    end_time_total = time.time()
    print(f"\nAll inductance calculations done in {end_time_total - start_time_total:.2f} s")

if __name__ == "__main__":
    main()