# utility_preprocess_data.py
import pandas as pd
from pathlib import Path
from fusion_tem import device as cfg

def process_and_save(input_path: Path, output_path: Path):
    """
    读取长格式的原始COMSOL数据文件，处理后保存为多Sheet的Excel文件。
    
    输出格式:
    - 每个变量值 (如 Npw=100) 对应一个Sheet。
    - 每个Sheet包含两列：时间(s) 和 功率(W)。
    - 没有表头。
    """
    if not input_path.exists():
        print(f"警告: 找不到输入文件 '{input_path}'，跳过处理。")
        return

    print(f"--- 正在处理文件: {input_path.name} ---")
    
    # 1. 加载原始数据 (跳过前4行，使用第5行做表头)
    try:
        df = pd.read_excel(input_path, skiprows=4, header=0)
        columns = df.columns
        variable_col_name, time_col_name, value_col_name = columns[0], columns[1], columns[2]
    except Exception as e:
        print(f"  [✘] 错误: 读取或解析文件失败，请检查格式。错误: {e}")
        return

    # 2. 获取所有独立的变量值
    variable_list = df[variable_col_name].unique()
    print(f"  -> 在文件中发现变量 '{variable_col_name}' 的工况: {list(variable_list)}")

    # 3. 创建Excel写入器，准备写入多个Sheet
    with pd.ExcelWriter(output_path) as writer:
        # 4. 循环处理每个变量值
        for var_value in variable_list:
            # 筛选出当前变量值对应的所有行
            subset_df = df[df[variable_col_name] == var_value]
            
            # 只保留时间和功率这两列
            output_df = subset_df[[time_col_name, value_col_name]]
            
            # 将这个子DataFrame写入一个新的Sheet中
            # Sheet名就是变量的值
            # index=False, header=False 确保输出的Excel没有行索引和表头
            output_df.to_excel(writer, sheet_name=str(var_value), index=False, header=False)
    
    print(f"  [✔] 处理完成，已保存至: {output_path.name}")


def main():
    """主函数，处理所有实验文件夹下的数据。"""
    base_path = Path(__file__).parent.resolve()
    heat_data_root = base_path / cfg.HEAT_DATA_DIR
    
    # 获取所有实验文件夹 (例如 Npw=var_rhot=50, Npw=5_rhot=var)
    experiment_folders = [d for d in heat_data_root.iterdir() if d.is_dir()]

    for folder in experiment_folders:
        print(f"\n{'='*25}\n正在扫描文件夹: {folder.name}\n{'='*25}")
        
        # 处理磁化损耗
        mag_input = folder / 'mag_loss.xlsx'
        mag_output = folder / 'mag_loss_processed.xlsx'
        process_and_save(mag_input, mag_output)
        
        # 处理径向损耗
        rad_input = folder / 'radial_loss.xlsx'
        rad_output = folder / 'radial_loss_processed.xlsx'
        process_and_save(rad_input, rad_output)

if __name__ == "__main__":
    main()