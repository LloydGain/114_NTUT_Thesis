import os
import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict

def count_extracted_stores(base_dir: str, end_date: str = None):
    """
    Traverse the base directory to find all 'extract_routes_info.xlsx' files
    and count the number of stores in the second sheet (Sheet2), up to a specified end_date.
    """
    print(f"Scanning directory: {base_dir}...\n")
    results = defaultdict(list)
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Error: Directory '{base_dir}' does not exist.")
        return

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == 'extract_routes_info.xlsx':
                rel_path = os.path.relpath(root, base_dir)
                rel_parts = rel_path.split(os.sep)
                # Ensure the path is exactly base_dir/YYYYMMDD/timestamp
                if len(rel_parts) == 2 and rel_parts[0].isdigit() and len(rel_parts[0]) == 8:
                    date_folder = rel_parts[0]
                    if end_date and date_folder > end_date:
                        continue
                    timestamp_folder = rel_parts[1]
                    file_path = os.path.join(root, file)
                    
                    try:
                        # Use engine='openpyxl' just to be safe with modern Excel files
                        xl = pd.ExcelFile(file_path, engine='openpyxl')
                        # We expect at least two sheets, the second one contains the extracted stores
                        if len(xl.sheet_names) > 1:
                            df = xl.parse(xl.sheet_names[1])
                            results[date_folder].append(len(df))
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

    if not results:
        print("No extract_routes_info.xlsx files found with a valid second sheet.")
        return

    print("每個日期的 extract_routes_info.xlsx 中 Sheet2 的平均抽取店鋪數量：")
    print("-" * 50)
    total_daily_averages = 0
    for date_folder in sorted(results.keys()):
        counts = results[date_folder]
        daily_avg = sum(counts) / len(counts)
        print(f"{date_folder} : 平均 {daily_avg:g} 筆")
        total_daily_averages += daily_avg
    print("-" * 50)
    overall_avg = total_daily_averages / len(results)
    print(f"每日平均加總: {total_daily_averages:g} 筆 (共 {len(results)} 天)")
    print(f"總平均每天抽取: {overall_avg:.2f} 筆")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count extracted stores from extract_routes_info.xlsx files.")
    
    # Default to the output directory relative to the project root
    current_dir = Path(__file__).resolve().parent
    default_output_dir = current_dir.parent / '100'
    
    parser.add_argument(
        "-d", "--output_dir", 
        type=str, 
        default=str(default_output_dir),
        help="Path to the output directory containing the date folders."
    )
    
    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="Specify an end date (e.g. 20221231) to only calculate up to that day (inclusive)."
    )
    
    args = parser.parse_args()
    count_extracted_stores(args.output_dir, args.end_date)
