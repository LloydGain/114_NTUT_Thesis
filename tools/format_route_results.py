import pandas as pd
import numpy as np
import os
import glob

def process_exp_results(input_dir, manual_path, output_path):
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found at {input_dir}")
        return

    # 讀取並聚合指定目錄下的所有 Excel 檔案
    def read_and_aggregate_dir(directory):
        all_data = []
        for file_path in glob.glob(os.path.join(directory, "*.xlsx")):
            filename = os.path.basename(file_path)
            date = filename.split('.')[0]
            try:
                df = pd.read_excel(file_path)
                if 'status' in df.columns:
                    df = df[df['status'] == 'ok']
                if df.empty:
                    continue
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
        return pd.DataFrame(all_data)

    # 讀取完整程式編排結果
    df_program = read_and_aggregate_dir(input_dir)
    
    # 讀取 ALB (消融實驗) 結果
    alb_dfs = {}
    alb_dir = os.path.join(input_dir, "alb")
    if os.path.exists(alb_dir):
        for alb_name in os.listdir(alb_dir):
            alb_sub_dir = os.path.join(alb_dir, alb_name)
            if os.path.isdir(alb_sub_dir):
                df_alb = read_and_aggregate_dir(alb_sub_dir)
                if not df_alb.empty:
                    alb_dfs[alb_name] = df_alb

    # 讀取手動編排結果
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
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').round(2)
        df = df.sort_values('date')
        avg_values = df[numeric_cols].mean().round(2)
        avg_row_data = {'date': 'Average'}
        for col in numeric_cols:
            avg_row_data[col] = avg_values[col]
        avg_row = pd.DataFrame([avg_row_data])
        df = pd.concat([df, avg_row], ignore_index=True)
        return df

    # 計算 gap 的函式: (target - base) / base * 100
    def calculate_gap(df_target, df_base):
        df_g = pd.DataFrame()
        if not df_target.empty and not df_base.empty:
            common_cols = [c for c in df_target.columns if c in df_base.columns and c != 'date']
            tgt_tmp = df_target.set_index('date')[common_cols]
            base_tmp = df_base.set_index('date')[common_cols]
            
            common_dates = tgt_tmp.index.intersection(base_tmp.index)
            tgt_tmp = tgt_tmp.loc[common_dates]
            base_tmp = base_tmp.loc[common_dates]
            
            gap_tmp = ((tgt_tmp - base_tmp) / base_tmp.replace(0, np.nan) * 100).round(2)
            gap_tmp = gap_tmp.fillna("")
            gap_tmp.columns = [f"{col}(%)" for col in common_cols]
            df_g = gap_tmp.reset_index()
            
            if 'Average' in df_g['date'].values:
                avg_row = df_g[df_g['date'] == 'Average']
                df_g = df_g[df_g['date'] != 'Average']
                df_g = pd.concat([df_g, avg_row], ignore_index=True)
        return df_g

    # 套用格式化
    df_program = format_df(df_program)
    df_manual = format_df(df_manual)
    for alb_name in alb_dfs:
        alb_dfs[alb_name] = format_df(alb_dfs[alb_name])

    # 計算程式與手動的 Gap
    df_gap_manual = calculate_gap(df_program, df_manual)

    # 定義輸出的排序順序
    desired_order = ['extract', 'allocate', 'support']
    sorted_alb_names = sorted(
        alb_dfs.keys(), 
        key=lambda x: desired_order.index(x) if x in desired_order else len(desired_order)
    )

    # 輸出到新的 Excel 檔案
    with pd.ExcelWriter(output_path) as writer:
        if not df_manual.empty:
            df_manual.to_excel(writer, sheet_name='手動編排', index=False)
        if not df_program.empty:
            df_program.to_excel(writer, sheet_name='程式編排', index=False)
            
        # 先輸出所有的 ALB 程式編排
        for alb_name in sorted_alb_names:
            df_alb = alb_dfs[alb_name]
            sheet_name = f'程式(alb_{alb_name})'
            df_alb.to_excel(writer, sheet_name=sheet_name[:31], index=False)
            
        # 再輸出所有的 Gap (放到最後面)
        if not df_gap_manual.empty:
            # 確保 sheet 名稱不超過 31 字元
            df_gap_manual.to_excel(writer, sheet_name='Gap(程式-手動)'[:31], index=False)
            
        for alb_name in sorted_alb_names:
            df_alb = alb_dfs[alb_name]
            # 計算 ALB 與完整程式的 Gap
            df_gap_alb = calculate_gap(df_alb, df_program)
            if not df_gap_alb.empty:
                gap_sheet_name = f'Gap(alb_{alb_name}-程式)'
                df_gap_alb.to_excel(writer, sheet_name=gap_sheet_name[:31], index=False)

    print(f"Success! Organized Excel saved to: {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    input_directory = os.path.join(base_dir, "output", "exp")
    manual_excel_path = os.path.join(base_dir, "output", "all_manual_summaries.xlsx")
    output_excel_path = os.path.join(base_dir, "docs", "路線最佳化比較結果.xlsx")
    
    process_exp_results(input_directory, manual_excel_path, output_excel_path)
