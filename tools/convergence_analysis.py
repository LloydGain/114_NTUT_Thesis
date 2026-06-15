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
    
    return mean_curve, max_len

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, "..", "output"))
    
    output_dir = os.path.join(base_dir, "convergence")
    os.makedirs(output_dir, exist_ok=True)
    
    log_types = {
        'CCGA': 'store_extraction_log.xlsx',
        'HACO': 'store_allocation_log.xlsx',
        'MACS': 'store_support_line_log.xlsx'
    }
    
    target_cols = {
        'best_cost': 'Best Cost',
        'iter_avg_cost': 'Iteration Average Cost'
    }
    
    all_logs = glob.glob(os.path.join(base_dir, '*', '*', 'logs', '*.xlsx'))
    
    for method_name, filename in log_types.items():
        method_files = [f for f in all_logs if filename in os.path.basename(f) and 'alb' not in f.split(os.sep)]
        
        print(f"找到 {len(method_files)} 個 {method_name} 的紀錄檔")
        
        expected_len = 200 if method_name in ['CCGA', 'HACO'] else 100
        
        for col, col_label in target_cols.items():
            result = process_logs(method_files, target_col=col, target_len=expected_len)
            if result is None:
                print(f"沒有找到 {method_name} ({col}) 的有效資料，跳過。")
                continue
                
            mean_curve, max_len = result
            
            plt.figure(figsize=(10, 6))
            
            x_axis = np.arange(1, max_len + 1)
            # 依要求統一 CCGA 與 HACO 標籤為 Iteration
            if method_name == 'CCGA':
                plt.xlabel("Iteration", fontsize=12)
            else:
                plt.xlabel("Iteration", fontsize=12)
                
            plt.plot(x_axis, mean_curve, label=f'Average of {col_label}', color='blue', linewidth=2)
            plt.xlim(0, max_len)
            plt.xticks(np.arange(0, max_len + 1, 20))
            plt.ylabel("Cost", fontsize=12)
            plt.title(f"{method_name} Convergence Curve ({col_label})", fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=12)
            
            output_path = os.path.join(output_dir, f"{method_name}_{col}_convergence.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"已儲存 {method_name} ({col}) 收斂曲線至: {output_path}")

if __name__ == "__main__":
    main()
