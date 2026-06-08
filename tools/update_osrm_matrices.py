import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from data.store_data import StoreData

def main():
    store_file = os.path.join(PROJECT_ROOT, "data", "store_info.xlsx")
    dist_file = os.path.join(PROJECT_ROOT, "data", "osrm", "store_distance_matrix.json")
    time_file = os.path.join(PROJECT_ROOT, "data", "osrm", "store_time_matrix.json")
    
    print(f"Loading stores from {store_file}...")
    store_data = StoreData(store_file)
    
    print(f"Loaded {len(store_data.stores)} valid stores.")
    print("Computing distance and time matrices via OSRM...")
    
    store_data.get_cost_matrices(dist_file, time_file)
    
    print("Done! OSRM matrices have been updated successfully.")

if __name__ == "__main__":
    main()
