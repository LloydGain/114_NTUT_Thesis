import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def process_logs(log_files, target_col='best_cost', target_len=None):
    all_data = []
    max_len = 0
    
    for file in log_files:
        try:
            df = pd.read_excel(file)
            if target_col in df.columns:
                data = df[target_col].values
                all_data.append(data)
                if len(data) > max_len:
                    max_len = len(data)
        except Exception as e:
            print(f"讀取 {file} 時發生錯誤: {e}")
            
    if not all_data:
        return None
        
    if target_len is not None:
        max_len = max(max_len, target_len)
        
    padded_data = []
    for data in all_data:
        if len(data) < max_len:
            pad_width = max_len - len(data)
            padded = np.pad(data, (0, pad_width), 'edge')
        else:
            padded = data
        padded_data.append(padded)
        
    padded_data = np.array(padded_data)
    mean_curve = np.mean(padded_data, axis=0)
    std_curve = np.std(padded_data, axis=0)
    
    return mean_curve, std_curve, max_len

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, "..", "output"))
    
    output_dir = os.path.join(base_dir, "convergence")
    os.makedirs(output_dir, exist_ok=True)
    
    log_types = {
        'CCGA': 'store_extraction_log.xlsx',
        'HACO': 'store_allocation_log.xlsx',
        'MACS': 'store_support_line_log.xlsx',
        'SingleStageGA': 'pure_ga_log.xlsx'
    }
    
    target_cols = {
        'best_cost': 'Best Cost',
        'iter_avg_cost': 'Iteration Average Cost'
    }
    
    # Scale representative instances
    instances = {
        'All_Instances': '',
        'D20221223': '20221223',
        'D20221227': '20221227',
        'D20260106': '20260106'
    }
    
    all_logs = glob.glob(os.path.join(base_dir, '**', 'logs', '*.xlsx'), recursive=True)
    
    for scale_name, instance_date in instances.items():
        print(f"\n=== Processing {scale_name.replace('_', ' ')} ({instance_date}) ===")
        
        for method_name, filename in log_types.items():
            # Filter logs by exact method and instance date
            method_files = [f for f in all_logs if filename in os.path.basename(f) and instance_date in f]
            
            if method_name != 'SingleStageGA':
                method_files = [f for f in method_files if 'alb' not in f.split(os.sep)]
            
            if not method_files:
                continue
                
            print(f"  找到 {len(method_files)} 個 {method_name} 的紀錄檔")
                
            if method_name == 'SingleStageGA':
                expected_len = 1000
            else:
                expected_len = 200 if method_name in ['CCGA', 'HACO'] else 100
            
            for col, col_label in target_cols.items():
                result = process_logs(method_files, target_col=col, target_len=expected_len)
                if result is None:
                    continue
                    
                mean_curve, std_curve, max_len = result
                
                # 強制截斷到 expected_len
                if max_len > expected_len:
                    max_len = expected_len
                    mean_curve = mean_curve[:expected_len]
                    std_curve = std_curve[:expected_len]
                    
                # --- Plotting ---
                plt.figure(figsize=(10, 6))
                
                x_axis = np.arange(1, max_len + 1)
                plt.xlabel("Iteration", fontsize=12)
                    
                display_label = col_label
                if method_name == 'CCGA':
                    display_label = display_label.replace("Cost", "Fitness")
                    
                plt.plot(x_axis, mean_curve, label=f'Average of {display_label}', color='#156082', linewidth=2)
                plt.fill_between(x_axis, mean_curve - std_curve, mean_curve + std_curve, color='#156082', alpha=0.3, label='±1 Std Dev')
                plt.xlim(0, max_len)
                
                # 動態調整 X 軸刻度
                if max_len > 500:
                    step = 200
                elif max_len > 200:
                    step = 50
                else:
                    step = 20
                plt.xticks(np.arange(0, max_len + 1, step))
                
                if method_name == 'CCGA':
                    y_label = "Fitness"
                else:
                    y_label = "Cost"
                plt.ylabel(y_label, fontsize=12)
                
                # Title formatting
                plt.title(f"Convergence Curve of {method_name} - {scale_name}", fontsize=14)
                
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(fontsize=12)
                
                output_path = os.path.join(output_dir, f"{method_name}_{col}_{scale_name}_convergence.png")
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # --- Generating Excel Table (Only for Best Cost) ---
                if col == 'best_cost':
                    table_data = []
                    generations = list(range(1, max_len + 1))
                    
                    for g in generations:
                        idx = g - 1
                        table_data.append({
                            'Generation': g,
                            'Best Cost Mean': round(mean_curve[idx], 2) if method_name != 'CCGA' else round(mean_curve[idx], 4),
                            'Std. Dev.': round(std_curve[idx], 2) if method_name != 'CCGA' else round(std_curve[idx], 4)
                        })
                    
                    df_table = pd.DataFrame(table_data)
                    table_path = os.path.join(output_dir, f"{method_name}_{scale_name}_Table.xlsx")
                    df_table.to_excel(table_path, index=False)
                    print(f"  已儲存 {method_name} {scale_name} 收斂表至: {table_path}")

if __name__ == "__main__":
    main()
