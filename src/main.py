import copy
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

    print("Loading route data...")
    manager = DataManager([route_file, route_network_file])
    # manager.save_routes_to_json(original_route)
    routes = copy.deepcopy(manager.routes_info)

    print("Starting Store Extraction using GA...")
    store_extractor = StoreExtractionGA(routes)
    main_routes, extracted_stores = store_extractor.run()

    print("Starting Store Allocation using ACO...")
    store_allocator = StoreAllocationACO(main_routes, extracted_stores)
    best_cost, best_solution = store_allocator.run()

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