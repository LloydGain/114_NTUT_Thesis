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
    store_extractor = StoreExtractionGA(routes)
    main_routes, extracted_stores = store_extractor.run()
    # num = 0
    # for store in extracted_stores:
    #     num += 1
    #     print(store)
    # print(f"Total extracted stores: {num}")

    # for route_id in main_routes:
    #     if main_routes[route_id]['dc']['load_rate'] > 1:
    #         print(f"Route {route_id} exceeds capacity! Load Rate: {main_routes[route_id]['dc']['load_rate']}")
    #     else:
    #         print(f"Route {route_id} is within capacity. Load Rate: {main_routes[route_id]['dc']['load_rate']}")

    print("Starting Store Allocation using ACO...")
    store_allocator = StoreAllocationACO(main_routes, extracted_stores)
    store_allocator.run()

    # print("\nStarting Support Line Planning using ACO...")
    # support_line = SupportLinePlanningACO(extracted_stores)
    # best_cost, best_solution = support_line.run()
    # print("Best Cost:", best_cost)
    # num = 0
    # for vehicle_id, stores in best_solution.items():
    #     store_ids = [store['store_id'] for store in stores]
    #     num += len(store_ids)
    #     print(f"Vehicle {vehicle_id}:\n Stores {store_ids}\nVolume: {sum(store['volume'] for store in stores)}\n\n")
    # print(f"Total allocated stores: {num}")

if __name__ == "__main__":
    main()