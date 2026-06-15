import os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, norm
try:
    from baycomp import SignedRankTest
    HAS_BAYCOMP = True
except ImportError:
    HAS_BAYCOMP = False

# 使用相對路徑取得專案根目錄
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(base_dir, "docs", "Baseline比較結果.xlsx")

try:
    df = pd.read_excel(file_path, sheet_name=0, header=[0, 1])
    
    methods = []
    current_method = None
    for col in df.columns:
        method = col[0]
        if not method.startswith('Unnamed'):
            current_method = method
        methods.append((current_method, col[1]))
    
    df.columns = pd.MultiIndex.from_tuples(methods)
    
    # 過濾掉 Dataset 欄位中包含 Average, Avg, Gap 等字眼的統計列
    dataset_col = df.columns[0]
    df = df[~df[dataset_col].astype(str).str.contains('Average|Avg|gap|Gap|Total', case=False, na=False)]
    
    # We specifically want to compare "我的方法(new)" against "baseline敘述"
    baseline_method = 'baseline敘述'
    other_method = '我的方法(new)'
    
    # Handle the fact that names might have mojibake or slightly different chars in pandas
    # Let's find the exact keys
    unique_methods = []
    for m in [m for m, _ in methods if pd.notna(m)]:
        if m not in unique_methods and m != 'Dataset' and not 'Gap' in m:
            unique_methods.append(m)
            
    # Assuming baseline敘述 is the second one, and 我的方法(new) is the fourth one
    # from previous output: ['baseline實際', 'baseline敘述', '我的方法', '我的方法(new)', ...]
    # wait, the exact names from previous output:
    # 'baseline', 'baselineԭz', 'ڪk', 'ڪk(new)'
    
    # Let's map them based on order
    b_actual = unique_methods[0]
    b_desc = unique_methods[1]
    my_method = unique_methods[2]
    my_method_new = unique_methods[3]
    
    baseline_key = b_desc
    other_key = my_method_new
    
    results = []
    metrics = ['NV', 'Distance']
    
    for metric in metrics:
        try:
            base_vals = pd.to_numeric(df[(baseline_key, metric)], errors='coerce').dropna()
            other_vals = pd.to_numeric(df[(other_key, metric)], errors='coerce').dropna()
            
            common_idx = base_vals.index.intersection(other_vals.index)
            base_data = base_vals.loc[common_idx]
            other_data = other_vals.loc[common_idx]
            
            if len(base_data) < 2:
                continue
                
            diff = base_data - other_data
            if np.all(diff == 0):
                w_stat, pval = np.nan, 1.0
            else:
                w_stat, pval = wilcoxon(base_data, other_data)
            
            r_effect = np.nan
            if pd.notna(pval) and pval != 1.0 and len(base_data) > 0:
                z_score = norm.ppf(1 - pval/2)
                r_effect = z_score / np.sqrt(2 * len(base_data))
                
            n_win = np.sum(diff > 0) # Base > Other
            n_lose = np.sum(diff < 0)
            n_tie = np.sum(diff == 0)
            
            p_win, p_tie_prob, p_lose = np.nan, np.nan, np.nan
            if HAS_BAYCOMP and len(base_data) > 0:
                # baycomp 回傳的 probs 順序為: (P(x > y), P(x == y), P(x < y))
                # 這裡 x = other_data(我的方法), y = base_data(baseline)
                # 因為數值越小越好，所以 win 是 P(x < y)，對應 p_right
                p_left, p_rope, p_right = SignedRankTest(other_data.values, base_data.values, rope=0.01).probs()
                p_win, p_tie_prob, p_lose = p_right, p_rope, p_left
            else:
                # 若沒有安裝 baycomp，採用與原本calculate_statistical_tests相同的Dirichlet計算方式做為 fallback
                n_total = len(diff)
                p_win = (1 + n_win) / (3 + n_total)
                p_tie_prob = (1 + n_tie) / (3 + n_total)
                p_lose = (1 + n_lose) / (3 + n_total)
            
            results.append({
                'Metric': metric,
                'Comparison': f'我的方法(new) vs baseline敘述',
                'N': len(base_data),
                'W': float(w_stat) if pd.notna(w_stat) else None,
                'p-value': float(pval) if pd.notna(pval) else None,
                'r': float(r_effect) if pd.notna(r_effect) else None,
                'Win (Other Better)': int(n_win),
                'Tie': int(n_tie),
                'Lose (Base Better)': int(n_lose),
                'P(Win)': float(p_win) if pd.notna(p_win) else None,
                'P(Tie)': float(p_tie_prob) if pd.notna(p_tie_prob) else None,
                'P(Lose)': float(p_lose) if pd.notna(p_lose) else None
            })
        except Exception as e:
            print(f"Error on {metric}: {e}")

    # 輸出成 Excel 檔案，存放在 output 資料夾
    out_excel = os.path.join(base_dir, "output", "Baseline_Wilcoxon分析結果.xlsx")
    df_out = pd.DataFrame(results)
    
    with pd.ExcelWriter(out_excel) as writer:
        df_out.to_excel(writer, sheet_name="Wilcoxon_vs_baseline敘述", index=False)
        
    print(f"Analysis complete. Results written to {out_excel}")

except Exception as e:
    print(f"Error: {e}")

