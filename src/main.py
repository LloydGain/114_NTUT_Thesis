import copy
import time
from data import DataManager
from route import RouteManager
from extract_ga import StoreExtractionGA
from allocate_aco import StoreAllocationACO
from support_line_aco import SupportLinePlanningACO


def main():
    route_file = '../data/1203route.xlsx'
    route_network_file = '../data/route_network_and_dwell_times.xlsx'
    original_route = '../output/original_routes_info.json'
    optimized_route_file = '../output/optimized_routes_info.json'

    start_time = time.time()

    print("Loading route data...")
    manager = DataManager([route_file, route_network_file])
    # manager.save_routes_to_json(original_route)
    routes = copy.deepcopy(manager.routes_info)
    distance_matrix, time_matrix = manager.distance_matrix, manager.time_matrix

    end_time = time.time()
    print(f"資料讀取執行時間: {end_time - start_time:.2f} 秒")

# -----------------------------------------------------------------------------------

    start_time = time.time()

    print("Starting Store Extraction using GA...")
    store_extractor = StoreExtractionGA(routes, distance_matrix, time_matrix)
    main_routes, extracted_stores = store_extractor.run()

    end_time = time.time()
    print(f"店鋪抽取執行時間: {end_time - start_time:.2f} 秒")

# -----------------------------------------------------------------------------------

    start_time = time.time()

    print("Starting Store Allocation using ACO...")
    store_allocator = StoreAllocationACO(main_routes, extracted_stores, distance_matrix, time_matrix)
    best_cost, best_solution = store_allocator.run()

    end_time = time.time()
    print(f"店鋪再分配執行時間: {end_time - start_time:.2f} 秒")

# -----------------------------------------------------------------------------------

    total_stores = 0
    for vehicle_id, vehicle in best_solution.items():
        total_stores += len(vehicle['stores'])
        print(f"Vehicle {vehicle_id}: {len(vehicle['stores'])} stores -> load_rate: {vehicle['dc']['load_rate']}")
    print(f"Total Cost: {best_cost}")
    print(f"Total Vehicle Num: {len(best_solution)}")
    print(f"Total Store: {total_stores}")

    route_manager = RouteManager(best_solution)
    route_manager._export_routes_info(optimized_route_file)


if __name__ == "__main__":
    main()