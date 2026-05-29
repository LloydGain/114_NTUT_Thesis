import pandas as pd
import numpy as np
import os
import glob

def process_exp_results(input_dir, manual_path, output_path):
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found at {input_dir}")
        return

    all_data = []
    
    # 讀取資料夾中所有的 Excel 檔案
    for file_path in glob.glob(os.path.join(input_dir, "*.xlsx")):
        # 檔名即為日期，例如：20221203.xlsx
        filename = os.path.basename(file_path)
        date = filename.split('.')[0]
        
        try:
            df = pd.read_excel(file_path)
            
            # 若有 status 欄位，可以過濾出成功的執行結果 (可依需求調整)
            if 'status' in df.columns:
                df = df[df['status'] == 'ok']
                
            if df.empty:
                continue
                
            # 計算該日期的所有 seed 的平均值
            avg_metrics = {
                'date': str(date),
                'vehicle num': df['vehicle_num'].mean(),
                'total_dist(km)': df['total_dist(km)'].mean(),
                'total_time(hr)': df['total_time(hr)'].mean(),
                'avg_load_rate': df['avg_load_rate'].mean(),
                'on_time_rate': df['on_time_rate'].mean(),
            }
            if 'running_time(s)' in df.columns:
                avg_metrics['avg_running_time'] = df['running_time(s)'].mean()
            all_data.append(avg_metrics)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if not all_data:
        print("No valid data found in the directory.")
        return

    df_program = pd.DataFrame(all_data)
    
    if os.path.exists(manual_path):
        df_manual = pd.read_excel(manual_path)
        if 'date' in df_manual.columns:
            df_manual['date'] = df_manual['date'].astype(str)
    else:
        print(f"Warning: Manual file not found at {manual_path}")
        df_manual = pd.DataFrame()
    
    # 格式化 DataFrame 的函式
    def format_df(df):
        if df.empty:
            return df
        
        numeric_cols = [c for c in df.columns if c != 'date']
        
        # 確保數值型態並四捨五入到小數點後兩位
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').round(2)
            
        df = df.sort_values('date')

        # 計算所有日期的平均值
        avg_values = df[numeric_cols].mean().round(2)
        
        # 建立平均值列
        avg_row_data = {'date': 'Average'}
        for col in numeric_cols:
            avg_row_data[col] = avg_values[col]
            
        avg_row = pd.DataFrame([avg_row_data])
        
        # 將平均值列附加到最後
        df = pd.concat([df, avg_row], ignore_index=True)
        return df

    df_program = format_df(df_program)
    df_manual = format_df(df_manual)

    # 輸出到新的 Excel 檔案
    with pd.ExcelWriter(output_path) as writer:
        df_manual.to_excel(writer, sheet_name='手動編排', index=False)
        df_program.to_excel(writer, sheet_name='程式編排', index=False)

    print(f"Success! Organized Excel saved to: {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    input_directory = os.path.join(base_dir, "output", "exp")
    manual_excel_path = os.path.join(base_dir, "output", "all_manual_summaries.xlsx")
    output_excel_path = os.path.join(base_dir, "docs", "路線最佳化比較結果.xlsx")
    
    process_exp_results(input_directory, manual_excel_path, output_excel_path)
