import os
import sys
import time
import argparse
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data.store_data import StoreData
from data.manual_data import MDataManager
from models.route_manager import RouteManager
from services.osrm import OSRM

def parse_args():
    parser = argparse.ArgumentParser(description="Run manual data logic for all dates or specific ones.")
    parser.add_argument("--exclude", nargs='*', default=[], help="List of dates to exclude (e.g. --exclude 20221203 20221205)")
    parser.add_argument("--include", nargs='*', default=[], help="List of dates to include (e.g. --include 20221203 20221205)")
    parser.add_argument("--google", action="store_true", help="Update routes via Google Maps API")
    return parser.parse_args()

def process_date(file_date, google, distance_matrix, time_matrix):
    print(f"\n{'='*50}\nProcessing manual data for {file_date}\n{'='*50}")
    manual_file = f'../data/{file_date}/{file_date}manual.xlsx'
    route_network_file = '../data/route_network_and_dwell_times.xlsx'
    store_info_file = '../data/store_info.xlsx'

    if not os.path.exists(manual_file):
        print(f"[WARNING] Manual file {manual_file} not found. Skipping {file_date}.")
        return None

    out_base = f'../output/{file_date}'
    manual_routes_file = f'{out_base}/manual_routes_info.json'
    manual_routes_dir = f'{out_base}/manual_routes'
    manual_routes_excel_file = f'{manual_routes_dir}/manual_routes_info.xlsx'
    summary_file = f'{manual_routes_dir}/manual_summary.xlsx'

    os.makedirs(out_base, exist_ok=True)
    os.makedirs(manual_routes_dir, exist_ok=True)

    print(f"Loading manual route data for {file_date}...")
    m_data = MDataManager([manual_file, route_network_file, store_info_file], distance_matrix, time_matrix)
    
    m_route_manager = RouteManager(m_data.routes_info, distance_matrix, time_matrix)
    if google:
        print("Updating manual routes via Google Maps API...")
        m_route_manager.update_all_routes_distance_and_duration_with_google_api()
        m_data.routes_info = m_route_manager.routes_info
    
    print(f"Saving manual routes JSON to {manual_routes_file}...")
    m_data.save_routes_to_json(manual_routes_file)

    print(f"Exporting manual routes Excel to {manual_routes_excel_file}...")
    m_route_manager.export_excel_file(manual_routes_excel_file)

    print("Exporting manual results to excel...")
    manual_r_info = m_data.routes_info
    total_vehicles = len(manual_r_info)
    support_vehicles = sum(1 for r in manual_r_info.values() if str(r['dc']['route_id']).isdigit())
    total_distance = sum(r['dc']['distance'] for r in manual_r_info.values())
    total_duration = sum(r['dc']['duration'] for r in manual_r_info.values()) / 3600
    total_stores = sum(len(r['stores']) for r in manual_r_info.values())
    avg_load_rate = sum(r['dc']['load_rate'] for r in manual_r_info.values()) / total_vehicles if total_vehicles else 0
    on_time = sum(1 for r in manual_r_info.values() for s in r['stores'] if s['earliest_time'] <= s['pred_time'] <= s['latest_time'])
    on_time_rate = on_time / total_stores if total_stores else 0

    summary_data = {
        "date": file_date,
        "vehicle num": total_vehicles,
        "support_num": support_vehicles,
        "total_dist(km)": round(total_distance, 2),
        "total_time(hr)": round(total_duration, 2),
        "avg_load_rate": round(avg_load_rate, 4),
        "on_time_rate": round(on_time_rate, 4)
    }
    
    df = pd.DataFrame([summary_data])
    df.to_excel(summary_file, index=False)
    print(f"Manual summary exported to {summary_file}")
    return summary_data

def main(exclude_dates, include_dates, google=False):
    try:
        OSRM().check_osrm()
    except Exception as e:
        print(f"[ERROR] OSRM service check failed: {e}")
        return

    # Find all date folders in ../data/
    data_dir = '../data'
    all_dates = []
    if os.path.exists(data_dir):
        for d in os.listdir(data_dir):
            if os.path.isdir(os.path.join(data_dir, d)) and d.isdigit() and len(d) == 8:
                all_dates.append(d)
            
    # Filter dates
    if exclude_dates is None:
        exclude_dates = []
    
    if include_dates:
        dates_to_run = [d for d in all_dates if d in include_dates and d not in exclude_dates]
    else:
        dates_to_run = [d for d in all_dates if d not in exclude_dates]
    dates_to_run.sort()
    
    print(f"Found {len(all_dates)} dates. Including: {include_dates}. Excluding: {exclude_dates}. Running {len(dates_to_run)} dates.")

    if not dates_to_run:
        print("No dates to run.")
        return

    dist_file = '../data/osrm/store_distance_matrix.json'
    time_file = '../data/osrm/store_time_matrix.json'
    store_info_file = '../data/store_info.xlsx'

    start_time = time.time()
    print("Calculating store distance & time (only once for all dates)...")
    s_data = StoreData(store_info_file)
    distance_matrix, time_matrix = s_data.load_matrices_from_file(dist_file, time_file)

    all_summaries = []
    combined_summary_file = '../output/all_manual_summaries.xlsx'
    for file_date in dates_to_run:
        summary = process_date(file_date, google, distance_matrix, time_matrix)
        if summary:
            all_summaries.append(summary)
            combined_df = pd.DataFrame(all_summaries)
            combined_df.to_excel(combined_summary_file, index=False)
            print(f"Incremental summary updated in {combined_summary_file}")

    end_time = time.time()
    print(f"\nAll-dates manual summary exported to {combined_summary_file}")
    print(f"Total batch execution finished in {round(end_time - start_time, 2)} seconds.")

if __name__ == "__main__":
    args = parse_args()
    main(args.exclude, args.include, args.google)
