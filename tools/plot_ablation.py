import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    excel_path = os.path.join(base_dir, "result", "路線最佳化比較結果.xlsx")
    output_dir = os.path.join(base_dir, "output", "plots")
    os.makedirs(output_dir, exist_ok=True)

    methods_map = {
        '程式編排': 'Full Method',
        '程式(alb_extract)': 'Random Extraction',
        '程式(alb_allocate)': 'HACO Without VND',
        '程式(alb_support)': 'MACS Without VND',
        '程式(alb_single_stage_ga)': 'Single-Stage GA'
    }

    metrics = {
        'NV': 'vehicle num',
        'TD': 'total_dist(km)',
        'U': 'avg_load_rate',
        'O': 'on_time_rate'
    }

    y_labels = {
        'NV': 'Number of Vehicles',
        'TD': 'Total Distance (km)',
        'U': 'Utilization Rate (%)',
        'O': 'On-Time Rate (%)'
    }
    
    titles = {
        'NV': 'Number of Vehicles',
        'TD': 'Total Distance',
        'U': 'Utilization Rate',
        'O': 'On-Time Rate'
    }

    try:
        xls = pd.ExcelFile(excel_path)
    except FileNotFoundError:
        print(f"Error: {excel_path} not found.")
        return

    # Load data for all methods
    df_dict = {}
    for sheet, display_name in methods_map.items():
        if sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet)
            if 'date' in df.columns:
                # 排除 Average 列
                df = df[df['date'] != 'Average']
                df['date'] = df['date'].astype(str)
                # 統一欄位名稱
                if 'vehicle num' not in df.columns and 'vehicle_num' in df.columns:
                    df['vehicle num'] = df['vehicle_num']
                df_dict[display_name] = df.set_index('date')

    if not df_dict:
        print("No valid data found.")
        return
        
    # 取所有方法都有的日期交集
    common_dates = None
    for df in df_dict.values():
        if common_dates is None:
            common_dates = df.index
        else:
            common_dates = common_dates.intersection(df.index)
            
    if len(common_dates) == 0:
        print("No common dates found across methods.")
        return
        
    common_dates = sorted(common_dates)
    x_labels = [f"D{d}" for d in common_dates]

    # 設定線條顏色與標記符號 (依據您的要求，完整方法為 #156082)
    color_map = {
        'Full Method': '#156082',
        'Random Extraction': '#D9534F', # 紅色系
        'HACO Without VND': '#5CB85C',  # 綠色系
        'MACS Without VND': '#F0AD4E',  # 橘色系
        'Single-Stage GA': '#9467bd'    # 紫色系
    }

    # Setup Excel writer for combined table
    combined_excel_path = os.path.join(output_dir, 'Ablation_Instances_All_Metrics.xlsx')
    excel_writer = pd.ExcelWriter(combined_excel_path, engine='openpyxl')

    # Plot
    for metric_key, col_name in metrics.items():
        plt.figure(figsize=(14, 6))
        
        for display_name, df in df_dict.items():
            if col_name not in df.columns:
                continue
                
            y_values = df.loc[common_dates, col_name]
            
            # Full Method 置頂 (zorder=10)，確保重疊時至少看得到藍線
            is_full = (display_name == 'Full Method')
            zorder = 10 if is_full else 3
            
            # 畫折線圖：全部使用細實線('-')與實心圓形('o')
            plt.plot(x_labels, y_values, 
                     label=display_name,
                     color=color_map.get(display_name, '#000000'),
                     marker='o',
                     linestyle='-',
                     linewidth=1.5,
                     markersize=5,
                     alpha=0.8,
                     zorder=zorder)
        
        plt.title(titles[metric_key], fontsize=16, fontweight='bold', pad=15)
        plt.ylabel(y_labels[metric_key], fontsize=12, fontweight='bold')
        
        # 網格與刻度
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        
        # 調整圖例位置與 Y 軸範圍避免重疊
        if metric_key in ['U', 'O']:
            # 圖例放左下
            plt.legend(loc='lower left', fontsize=11, framealpha=0.9, edgecolor='black')
            # 為了避免圖例跟線條重疊，把 Y 軸最小值再往下延伸 25% 的空間
            ymin, ymax = plt.ylim()
            plt.ylim(ymin - (ymax - ymin) * 0.25, ymax)
        else:
            # NV, TD 維持左上
            plt.legend(loc='upper left', fontsize=11, framealpha=0.9, edgecolor='black')
            
        # 移除頂部與右側框線
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        out_file = os.path.join(output_dir, f'Ablation_Instances_{metric_key}.png')
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_file}")

        # Export corresponding excel table to the combined writer
        table_df = pd.DataFrame()
        for display_name, m_df in df_dict.items():
            if col_name in m_df.columns:
                table_df[display_name] = m_df.loc[common_dates, col_name]
        table_df.index = x_labels
        table_df.reset_index(names=['Instance']).to_excel(excel_writer, sheet_name=metric_key, index=False)

    # Save the combined Excel file
    excel_writer.close()
    print(f"Saved combined Excel: {combined_excel_path}")

if __name__ == "__main__":
    main()
