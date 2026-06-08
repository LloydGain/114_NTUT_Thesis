import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare
import os
import warnings

warnings.filterwarnings("ignore")

def format_pval(pval):
    if pd.isna(pval):
        return "NaN"
    s = f"{pval:.20f}".rstrip('0')
    if s.endswith('.'):
        s += '0'
    return s

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    excel_path = os.path.join(base_dir, "output", "路線最佳化比較結果.xlsx")
    
    if not os.path.exists(excel_path):
        print(f"Error: 找不到檔案 {excel_path}")
        return

    xls = pd.ExcelFile(excel_path)
    sheet_names = xls.sheet_names
    
    data_dict = {}
    for sheet in sheet_names:
        if sheet.startswith('Gap') or '比較' in sheet:
            continue
        df = pd.read_excel(excel_path, sheet_name=sheet)
        df = df[df['date'] != 'Average'].set_index('date')
        
        if 'vehicle num' in df.columns and 'total_dist(km)' in df.columns:
            v_num = pd.to_numeric(df['vehicle num'], errors='coerce')
            dist = pd.to_numeric(df['total_dist(km)'], errors='coerce')
            df['階層式目標(車+距)'] = v_num * 2000 + dist
            
        if '手動' in sheet:
            df['running_time(s)'] = 5400
            df['avg_running_time'] = 5400
            
        if 'support_num' not in df.columns:
            df['support_num'] = 0
        
        if len(df) >= 26:
            data_dict[sheet] = df
        else:
            print(f"警告: '{sheet}' 只有 {len(df)} 筆資料 (未達 26 筆)，將不參與此次比較。")

    if '程式編排' not in data_dict:
        print("錯誤：找不到 '程式編排' 這個工作表，或是其資料筆數未達 26 筆！")
        return
        
    df_prog = data_dict['程式編排']
    
    metrics = ['階層式目標(車+距)', 'vehicle num', 'vehicle_num', 'support_num', 'total_dist(km)', 'avg_load_rate', 'on_time_rate', 'running_time(s)', 'avg_running_time']
    
    # ==========================================
    # 1. Wilcoxon Signed-Rank Test
    # ==========================================
    wilcoxon_results = []
    targets = [s for s in data_dict.keys() if s != '程式編排']
    
    print("=" * 60)
    print("1. Wilcoxon Signed-Rank Test (Pairwise vs 程式編排)")
    print("=" * 60)
    
    metric_name_map = {
        '階層式目標(車+距)': 'Hierarchical Objective',
        'vehicle num': 'NV',
        'vehicle_num': 'NV',
        'support_num': 'SV',
        'total_dist(km)': 'TD',
        'avg_load_rate': 'U',
        'on_time_rate': 'O',
        'avg_running_time': 'RT',
        'running_time(s)': 'RT'
    }
    
    for target in targets:
        df_target = data_dict[target]
        common_dates = df_prog.index.intersection(df_target.index)
        if len(common_dates) == 0:
            continue
            
        prog_tmp = df_prog.loc[common_dates]
        tgt_tmp = df_target.loc[common_dates]
        
        print(f"\n【Wilcoxon】程式編排 vs {target} (樣本數: {len(common_dates)})")
        
        for col in metrics:
            if col not in tgt_tmp.columns or col not in prog_tmp.columns:
                continue
            x = pd.to_numeric(prog_tmp[col], errors='coerce')
            y = pd.to_numeric(tgt_tmp[col], errors='coerce')
            
            valid_mask = ~x.isna() & ~y.isna()
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            
            if len(x_valid) < 2:
                w_stat, pval = np.nan, np.nan
            else:
                diff = x_valid - y_valid
                if np.all(diff == 0):
                    w_stat, pval = np.nan, 1.0
                else:
                    try:
                        w_stat, pval = wilcoxon(x_valid, y_valid)
                    except ValueError:
                        w_stat, pval = np.nan, np.nan
                        
            pval_str = format_pval(pval)
            w_str = f"{w_stat:.1f}" if pd.notna(w_stat) else "NaN"
            print(f"  - {col:18s} : W = {w_str:6s} | p-value = {pval_str}")
            
            wilcoxon_results.append({
                'Metric': metric_name_map.get(col, col),
                'Comparison': f'程式編排 vs {target}',
                'W': w_stat,
                'p-value': pval_str
            })
        
    # ==========================================
    # 2. Friedman Test
    # ==========================================
    print("\n" + "=" * 60)
    print("2. Friedman Test (多組樣本的無母數檢定與平均排名)")
    print("=" * 60)
    
    all_methods = list(data_dict.keys())
    
    # True = 越大越好 (Descending Rank, 值越大排名數字越小即越靠近第一名)
    # False = 越小越好 (Ascending Rank, 值越小排名數字越小即越靠近第一名)
    higher_is_better = {
        '階層式目標(車+距)': False,
        'avg_load_rate': True,
        'on_time_rate': True,
        'vehicle num': False,
        'vehicle_num': False,
        'support_num': False,
        'total_dist(km)': False,
        'total_time(hr)': False,
        'avg_running_time': False,
        'running_time(s)': False
    }
    
    def run_friedman(method_names, group_name):
        common_dates_f = data_dict[method_names[0]].index
        for m in method_names[1:]:
            common_dates_f = common_dates_f.intersection(data_dict[m].index)
            
        print(f"\n【Friedman】{group_name}")
        print(f"參與比較: {', '.join(method_names)}")
        print(f"有效樣本數: {len(common_dates_f)}")
        
        results_f = []
        if len(common_dates_f) >= 2:
            for col in metrics:
                valid_metric = True
                for m in method_names:
                    if col not in data_dict[m].columns:
                        valid_metric = False
                        break
                
                if not valid_metric:
                    continue
                    
                df_metric = pd.DataFrame({m: data_dict[m].loc[common_dates_f, col] for m in method_names})
                df_metric = df_metric.apply(pd.to_numeric, errors='coerce').dropna()
                
                if len(df_metric) < 2:
                    continue
                    
                is_higher_better = higher_is_better.get(col, False)
                df_ranks = df_metric.rank(axis=1, ascending=not is_higher_better, method='average')
                avg_ranks = df_ranks.mean().to_dict()
                
                is_all_same = True
                for i in range(1, len(method_names)):
                    if not np.allclose(df_metric[method_names[0]], df_metric[method_names[i]]):
                        is_all_same = False
                        break
                        
                if is_all_same:
                    stat, pval = np.nan, 1.0
                else:
                    try:
                        samples = [df_metric[m].values for m in method_names]
                        stat, pval = friedmanchisquare(*samples)
                    except Exception:
                        stat, pval = np.nan, np.nan
                        
                row_res = {
                    'Group': group_name,
                    'Metric': col,
                    'N': len(df_metric),
                    'Friedman Statistic': stat,
                    'p-value': pval
                }
                for m in method_names:
                    row_res[f'Rank_{m}'] = avg_ranks[m]
                    
                results_f.append(row_res)
                
                pval_str = format_pval(pval)
                stat_str = f"{stat:7.2f}" if pd.notna(stat) else "NaN"
                print(f"  - {col:18s} | Stat: {stat_str} | p-value: {pval_str} | Ranks: ", end="")
                ranks_str = ", ".join([f"{m}: {avg_ranks[m]:.2f}" for m in method_names])
                print(ranks_str)
                
        else:
            print("樣本數不足，無法檢定。")
            
        return results_f

    friedman_results_all = []
    
    f_res1 = run_friedman(all_methods, '所有方法 (包含手動)')
    friedman_results_all.extend(f_res1)
    
    algo_methods = [m for m in all_methods if '手動' not in m]
    if len(algo_methods) > 1:
        f_res2 = run_friedman(algo_methods, '僅比較所有程式演算法 (排除手動)')
        friedman_results_all.extend(f_res2)

    pivot_sheets = {}
    
    for group in pd.DataFrame(friedman_results_all)['Group'].unique():
        group_rows = [r for r in friedman_results_all if r['Group'] == group]
        
        method_keys = [k for k in group_rows[0].keys() if k.startswith('Rank_')]
        method_names = [k.replace('Rank_', '') for k in method_keys]
        
        pivot_data = []
        for method in method_names:
            row = {'Method': method}
            for grp_row in group_rows:
                metric_name = grp_row['Metric']
                if metric_name in ('vehicle num', 'vehicle_num'):
                    col_name = 'NV Rank'
                elif metric_name == 'support_num':
                    col_name = 'SV Rank'
                elif metric_name == 'total_dist(km)':
                    col_name = 'TD Rank'
                elif metric_name == '階層式目標(車+距)':
                    col_name = 'Object Rank'
                elif metric_name == 'avg_load_rate':
                    col_name = 'U Rank'
                elif metric_name == 'on_time_rate':
                    col_name = 'O Rank'
                elif metric_name in ('running_time(s)', 'avg_running_time'):
                    col_name = 'RT Rank'
                else:
                    col_name = f"{metric_name} Rank"
                    
                row[col_name] = round(grp_row[f'Rank_{method}'], 2)
            pivot_data.append(row)
            
        df_pivot = pd.DataFrame(pivot_data)
        
        stat_row = {'Method': 'Friedman Statistic'}
        pval_row = {'Method': 'p-value'}
        for grp_row in group_rows:
            metric_name = grp_row['Metric']
            if metric_name in ('vehicle num', 'vehicle_num'):
                col_name = 'NV Rank'
            elif metric_name == 'support_num':
                col_name = 'SV Rank'
            elif metric_name == 'total_dist(km)':
                col_name = 'TD Rank'
            elif metric_name == '階層式目標(車+距)':
                col_name = 'Object Rank'
            elif metric_name == 'avg_load_rate':
                col_name = 'U Rank'
            elif metric_name == 'on_time_rate':
                col_name = 'O Rank'
            elif metric_name in ('running_time(s)', 'avg_running_time'):
                col_name = 'RT Rank'
            else:
                col_name = f"{metric_name} Rank"
                
            if pd.notna(grp_row['Friedman Statistic']):
                stat_row[col_name] = round(grp_row['Friedman Statistic'], 2)
            else:
                stat_row[col_name] = np.nan
            
            pval_row[col_name] = format_pval(grp_row['p-value'])
                
        df_pivot = pd.concat([df_pivot, pd.DataFrame([stat_row, pval_row])], ignore_index=True)
        
        if '排除手動' in group:
            sheet_name = 'Friedman排名表(僅演算法)'
        else:
            sheet_name = 'Friedman排名表(含手動)'
            
        pivot_sheets[sheet_name] = df_pivot

    # ==========================================
    # 輸出到 Excel
    # ==========================================
    print("\n" + "=" * 60)
    print("註記：*** p<0.01, ** p<0.05, * p<0.1")
    
    out_excel = os.path.join(base_dir, "output", "統計檢定結果.xlsx")
    with pd.ExcelWriter(out_excel) as writer:
        pd.DataFrame(wilcoxon_results).to_excel(writer, sheet_name='Wilcoxon', index=False)
        
        for sheet_name, df_p in pivot_sheets.items():
            df_p.to_excel(writer, sheet_name=sheet_name, index=False)
            
    print(f"\n已將包含 Wilcoxon 與 Friedman 檢定(含轉置表格)的結果儲存至:\n-> {out_excel}")

if __name__ == "__main__":
    main()
