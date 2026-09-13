import numpy as np
import pandas as pd
from pathlib import Path

# 导入全局配置
from fusion_tem import device as cfg
# 从工具文件中导入电阻计算函数
from fusion_tem.utils import calculate_radial_resistance

def generate_comsol_ode_file(L_matrix: np.ndarray, output_path: Path):
    """
    根据给定的电感矩阵和径向电阻，生成 COMSOL 可读的 ODE txt 文件。
    格式示例：
    Ia1	current(t)-(Nt/Npw*Np)^2/(Rcoil*Np)*(Σ L_ij*Ia_jt)-Ia1	0	0	azimuthal transport current of coil1
    Ir1	current(t)-Ia1-Ir1	0	0	radial current of coil1
    """
    n = L_matrix.shape[0]
    lines = []

    # --- 生成 Ia 方程 ---
    for i in range(n):
        # 电感耦合项：Σ(L_ij[H]*Ia_jt)
        sum_L_dIadt = " + ".join([f"{L_matrix[i, j]:.6g}[H]*Ia{j+1}t" for j in range(n)])
        # 构造完整表达式
        expression = (
            f"current(t) - (Nt/Npw*Np)^2/(Rcoil*Np)*({sum_L_dIadt}) - Ia{i+1}"
        )
        description = f"azimuthal transport current of coil{i+1}"
        line = f"Ia{i+1}\t{expression}\t0\t0\t{description}"
        lines.append(line)



    # --- 生成 Ir 方程 ---
    for i in range(n):
        expression = f"current(t) - Ia{i+1} - Ir{i+1}"
        description = f"radial current of coil{i+1}"
        line = f"Ir{i+1}\t{expression}\t0\t0\t{description}"
        lines.append(line)

    # 写入文件
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"✅ 已生成 COMSOL ODE 文件: {output_path.name}")


def main():
    """主函数：为不同配置生成对应的COMSOL文件"""
    script_dir = Path(__file__).parent.resolve()
    inductance_M = script_dir / cfg.INDUCTANCE_OUTPUT_DIR /cfg.TF_SYSTEM_MATRIX

    output_dir_base = script_dir / cfg.COMSOL_ODE_OUTPUT_DIR
    output_dir_base.mkdir(exist_ok=True)
    
    print(f"COMSOL ODE 文件将被保存至: {output_dir_base}")
  
    # 1. 确定输入和输出文件路径
    
    output_file = output_dir_base /'1.txt'
        
    if not inductance_M.exists():
        print(f"警告: 找不到电感文件 {inductance_M.name}，跳过。")
        return None

    # 2. 加载电感矩阵
    L_matrix = pd.read_excel(inductance_M, header=None).values
        


    # 4. 生成文件
    generate_comsol_ode_file(L_matrix,  output_file)



if __name__ == "__main__":
    main()