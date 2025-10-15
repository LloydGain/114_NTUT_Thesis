import copy
from data import DataManager
from extract_ga import StoreExtractionGA
from allocate_aco import StoreAllocationACO
from support_line_aco import SupportLinePlanningACO


def main():
    route_file = '../data/1203route.xlsx'
    route_network_file = '../data/route_network_and_dwell_times.xlsx'
    original_route = '../output/original_routes_info.json'

    print("Loading route data...")
    manager = DataManager([route_file, route_network_file])
    # manager.save_routes_to_json(original_route)
    routes = copy.deepcopy(manager.routes_info)

    print("Starting Store Extraction using GA...")
    extractor = StoreExtractionGA(routes)
    extracted_stores = extractor.run()
    for store in extracted_stores:
        print(store)

    print("\nStarting Store Allocation using ACO...")
    support_line = SupportLinePlanningACO(extracted_stores)
    best_cost, best_solution = support_line.run()
    print("Best Cost:", best_cost)
    for vehicle_id, stores in best_solution.items():
        store_ids = [store['store_id'] for store in stores]
        print(f"Vehicle {vehicle_id}:\n Stores {store_ids}\nVolume: {sum(store['volume'] for store in stores)}\n\n")

if __name__ == "__main__":
    main()