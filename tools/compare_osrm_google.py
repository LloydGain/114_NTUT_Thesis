import os
import sys
import json
import random
import statistics
import requests
import pandas as pd
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))
from config import config

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    store_cache_file = os.path.join(base_dir, "data", "google", "google_maps_store_cache.json")

    if not os.path.exists(store_cache_file):
        print(f"Error: Cache file not found at {store_cache_file}")
        sys.exit(1)

    with open(store_cache_file, 'r', encoding='utf-8') as f:
        google_data = json.load(f)

    print(f"Loaded {len(google_data)} cached routes from Google Maps.")

    # We will sample to avoid overloading OSRM, even if it's local
    sample_size = min(5000, len(google_data))
    sampled_keys = random.sample(list(google_data.keys()), sample_size)

    osrm_url = config.OSRM_HOST

    # Check OSRM
    try:
        requests.get(f"{osrm_url}/health", timeout=2)
    except requests.exceptions.RequestException:
        print("[ERROR] OSRM not running. Please start it first.")
        sys.exit(1)

    print(f"Querying OSRM for {sample_size} routes...")

    distance_ratios = []
    duration_ratios = []
    binned_duration_ratios = {i: [] for i in range(101)}
    
    valid_pairs = 0

    for key in sampled_keys:
        # Key format: "DRIVE:lat1,lon1->lat2,lon2"
        try:
            mode, coords = key.split(':')
            p1_str, p2_str = coords.split('->')
            lat1, lon1 = p1_str.split(',')
            lat2, lon2 = p2_str.split(',')
            
            g_dist_km, g_dur_sec = google_data[key]
            
            if g_dist_km <= 0 or g_dur_sec <= 0:
                continue

            # OSRM expects lon,lat
            url = f"{osrm_url}/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data['code'] == 'Ok':
                    osrm_dist_m = data['routes'][0]['distance']
                    osrm_dur_sec = data['routes'][0]['duration']
                    
                    osrm_dist_km = osrm_dist_m / 1000.0
                    
                    if osrm_dist_km > 0 and osrm_dur_sec > 0:
                        dist_ratio = g_dist_km / osrm_dist_km
                        dur_ratio = g_dur_sec / osrm_dur_sec
                        
                        distance_ratios.append(dist_ratio)
                        duration_ratios.append(dur_ratio)
                        
                        dist_bin = int(g_dist_km)
                        if dist_bin <= 100:
                            binned_duration_ratios[dist_bin].append(dur_ratio)
                            
                        valid_pairs += 1
                        
        except Exception as e:
            continue
            
    print(f"\nSuccessfully compared {valid_pairs} valid routes.")
    
    if valid_pairs == 0:
        print("No valid comparisons could be made.")
        sys.exit(1)

    print("\n--- Statistical Comparison (Google / OSRM Ratio) ---")
    
    print("\n[Distance Ratio] (Google Distance / OSRM Distance)")
    print(f"Mean:   {statistics.mean(distance_ratios):.4f}")
    print(f"Median: {statistics.median(distance_ratios):.4f}")
    print(f"Min:    {min(distance_ratios):.4f}")
    print(f"Max:    {max(distance_ratios):.4f}")
    print(f"Stdev:  {statistics.stdev(distance_ratios):.4f}")

    print("\n[Duration Ratio] (Google Duration / OSRM Duration)")
    print(f"Mean:   {statistics.mean(duration_ratios):.4f}")
    print(f"Median: {statistics.median(duration_ratios):.4f}")
    print(f"Min:    {min(duration_ratios):.4f}")
    print(f"Max:    {max(duration_ratios):.4f}")
    print(f"Stdev:  {statistics.stdev(duration_ratios):.4f}")
    
    print("\nSuggestion:")
    print(f"To approximate Google Maps results using OSRM, you can multiply OSRM distances by ~{statistics.median(distance_ratios):.2f}")
    print(f"and OSRM durations by ~{statistics.median(duration_ratios):.2f}.")
    print("Note: In your current src/services/osrm.py, duration is multiplied by 1.75.")

    print("\n--- Binned Duration Ratio (by 1km Google Distance) ---")
    print(f"{'Dist(km)':<10} | {'Count':<6} | {'Mean':<6} | {'Median':<6} | {'Std':<6} | {'Min':<6} | {'Max':<6}")
    print("-" * 65)
    
    excel_data = []
    
    for i in range(0, 99):
        vals = binned_duration_ratios[i]
        bin_label = f"{i:02d}-{i+1:02d}"
        if len(vals) > 0:
            avg = statistics.mean(vals)
            med = statistics.median(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            min_val = min(vals)
            max_val = max(vals)
            print(f"{bin_label}    | {len(vals):<6} | {avg:<6.2f} | {med:<6.2f} | {std:<6.2f} | {min_val:<6.2f} | {max_val:<6.2f}")
            excel_data.append({
                "Dist(km)": bin_label,
                "Count": len(vals),
                "Mean": avg,
                "Median": med,
                "Std": std,
                "Min": min_val,
                "Max": max_val
            })
        else:
            print(f"{bin_label}    | {0:<6} | nan    | nan    | nan    | nan    | nan")
            excel_data.append({
                "Dist(km)": bin_label,
                "Count": 0,
                "Mean": None,
                "Median": None,
                "Std": None,
                "Min": None,
                "Max": None
            })
            
    # Export to Excel
    output_excel_path = os.path.join(base_dir, "data", "binned_duration_ratios.xlsx")
    df = pd.DataFrame(excel_data)
    df.to_excel(output_excel_path, index=False)
    print(f"\nSaved binned statistics to Excel: {output_excel_path}")

if __name__ == "__main__":
    main()
