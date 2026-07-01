import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import openpyxl
from openpyxl.drawing.image import Image

# Setup sys.path to find src/
from setup import *
from data.store_data import StoreData
from data.origin_data import ODataManager

def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    output_dir = root / "result" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dist_file = str(data_dir / "osrm" / "store_distance_matrix.json")
    time_file = str(data_dir / "osrm" / "store_time_matrix.json")
    store_info_file = str(data_dir / "store_info.xlsx")
    route_network_file = str(data_dir / "route_network_and_dwell_times.xlsx")
    
    print("Loading Store Data and Matrices...")
    s_data = StoreData(store_info_file)
    distance_matrix, time_matrix = s_data.load_matrices_from_file(dist_file, time_file)
    
    daily_stats = []
    
    # Store raw data for plotting
    raw_stores = []
    raw_load_rates = []
    date_labels = []
    
    # Find all date folders (e.g. 20221203)
    date_folders = sorted(glob.glob(str(data_dir / "20*")))
    
    for folder in date_folders:
        date_str = os.path.basename(folder)
        route_file = os.path.join(folder, f"{date_str}route.xlsx")
        
        if not os.path.exists(route_file):
            continue
            
        print(f"Processing {date_str}...")
        
        try:
            o_data = ODataManager([route_file, route_network_file, store_info_file], distance_matrix, time_matrix)
            routes_info = o_data.routes_info
            
            total_routes = len(routes_info)
            if total_routes == 0:
                continue
                
            total_stores = 0
            store_counts = []
            load_rates = []
            overload_routes = 0
            overload_rates = []
            
            for route_id, route in routes_info.items():
                s_count = len(route['stores'])
                total_stores += s_count
                store_counts.append(s_count)
                
                lr = route['dc']['load_rate']
                load_rates.append(lr)
                if lr > 1.0:
                    overload_routes += 1
                    overload_rates.append(lr)
            
            # Save raw arrays for this date to draw box plot later
            raw_stores.append(store_counts)
            raw_load_rates.append(load_rates)
            date_labels.append(f"D{date_str}")
                    
            daily_stats.append({
                "Date": f"D{date_str}",
                "Total Routes": total_routes,
                "Total Stores": total_stores,
                "Stores Mean": np.mean(store_counts),
                "Stores Min": np.min(store_counts),
                "Stores 25%": np.percentile(store_counts, 25),
                "Stores Median": np.median(store_counts),
                "Stores 75%": np.percentile(store_counts, 75),
                "Stores Max": np.max(store_counts),
                "LoadRate Mean": np.mean(load_rates),
                "LoadRate Min": np.min(load_rates),
                "LoadRate 25%": np.percentile(load_rates, 25),
                "LoadRate Median": np.median(load_rates),
                "LoadRate 75%": np.percentile(load_rates, 75),
                "LoadRate Max": np.max(load_rates),
                "Overload Route Ratio": overload_routes / total_routes if total_routes else 0,
                "Avg Overload Rate": np.mean(overload_rates) if overload_rates else 0.0
            })
            
        except Exception as e:
            print(f"Error processing {date_str}: {e}")
            
    if not daily_stats:
        print("No daily stats generated.")
        return
        
    df_daily = pd.DataFrame(daily_stats)
    
    # Generate Overall Summary
    summary = []
    numeric_cols = ["Total Routes", "Total Stores", 
                   "Stores Mean", "Stores Min", "Stores 25%", "Stores Median", "Stores 75%", "Stores Max",
                   "LoadRate Mean", "LoadRate Min", "LoadRate 25%", "LoadRate Median", "LoadRate 75%", "LoadRate Max",
                   "Overload Route Ratio", "Avg Overload Rate"]
                   
    for col in numeric_cols:
        summary.append({
            "Metric": col,
            "Mean": df_daily[col].mean(),
            "Std": df_daily[col].std(ddof=1) if len(df_daily) > 1 else 0,
            "Min": df_daily[col].min(),
            "Max": df_daily[col].max()
        })
        
    df_summary = pd.DataFrame(summary)
    
    # Save to Excel
    out_file = output_dir.parent / "dataset_analysis_report.xlsx"
    
    # Prepare Raw Load Rates DataFrame (in percentage) - Long Format for Excel Boxplot
    long_format_data = []
    for label, day_rates in zip(date_labels, raw_load_rates):
        for rate in day_rates:
            long_format_data.append({"Date": label, "Load Rate (%)": rate * 100})
    df_raw = pd.DataFrame(long_format_data)

    with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
        df_daily.to_excel(writer, sheet_name='Daily Statistics', index=False)
        df_summary.to_excel(writer, sheet_name='Overall Summary', index=False)
        df_raw.to_excel(writer, sheet_name='Raw Load Rates', index=False)
        
    print(f"\nAnalysis complete! Report saved to: {out_file}")
    
    # Draw and insert Box Plots
    print("Generating Box Plots...")
    
    # Set Chinese font for matplotlib
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create figures (stores boxplot removed as requested)
    
    raw_load_rates_pct = [[val * 100 for val in day_rates] for day_rates in raw_load_rates]
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.boxplot(raw_load_rates_pct, labels=date_labels)
    ax.axhline(100.0, color='r', linestyle='--', label='100% 容量限制')
    ax.set_title("原始路線之裝載率分佈")
    ax.set_ylabel("裝載率 (%)")
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    plt.tight_layout()
    loadrate_img_path = output_dir / "loadrate_boxplot.png"
    plt.savefig(loadrate_img_path)
    plt.close()
    
    # Daily Total Stores Line Chart
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(date_labels, df_daily["Total Stores"], color='#156082', marker='o', linestyle='-', linewidth=2, markersize=6)
    ax.set_title("每日總店鋪數量波動趨勢")
    ax.set_ylabel("總店鋪數量")
    ax.set_ylim(df_daily["Total Stores"].min() * 0.95, df_daily["Total Stores"].max() * 1.05)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    total_stores_img = output_dir / "daily_total_stores.png"
    plt.savefig(total_stores_img)
    plt.close()
    
    # Daily Total Routes Line Chart
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(date_labels, df_daily["Total Routes"], color='#156082', marker='o', linestyle='-', linewidth=2, markersize=6)
    ax.set_title("每日總路線數量波動趨勢")
    ax.set_ylabel("總路線數量")
    ax.set_ylim(df_daily["Total Routes"].min() * 0.95, df_daily["Total Routes"].max() * 1.05)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    total_routes_img = output_dir / "daily_total_routes.png"
    plt.savefig(total_routes_img)
    plt.close()
    
    # Insert images into Excel
    wb = openpyxl.load_workbook(out_file)
    ws = wb.create_sheet(title="Charts")
    
    img_load = Image(loadrate_img_path)
    ws.add_image(img_load, "A2")
    
    img_tot_stores = Image(total_stores_img)
    ws.add_image(img_tot_stores, "A30")
    
    img_tot_routes = Image(total_routes_img)
    ws.add_image(img_tot_routes, "A58")
    
    wb.save(out_file)
    print("All charts inserted into Excel sheet 'Charts'.")

if __name__ == "__main__":
    main()
