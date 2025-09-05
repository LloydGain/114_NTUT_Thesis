import copy
from data import DataManager
from extract_aco import StoreExtractionACO
from allocate_aco import StoreAllocationACO
from support_line_aco import SupportLinePlanningACO


def print_extracted_stores(extracted_stores):
    volume = sum(store['volume'] for store in extracted_stores)
    for store in extracted_stores:
        print(store)
    print(len(extracted_stores))
    print(f"Total extracted volume: {volume / 7.2}")


def main():
    route_file = '../data/1203route.xlsx'
    route_network_file = '../data/route_network_and_dwell_times.xlsx'
    original_route = '../output/original_routes_info.json'

    manager = DataManager([route_file, route_network_file])
    manager.save_routes_to_json(original_route)
    routes = copy.deepcopy(manager.routes_info)

    extractor = StoreExtractionACO(routes)
    remaining_stores = extractor.run()
    # print_extracted_stores(remaining_stores)
    # for i in manager.routes_info:
    #     print(f"Route {i} now has {manager.routes_info[i]['dc']['load_rate']} volume.")

    allocator = StoreAllocationACO(routes, remaining_stores)
    solution = allocator.run()
    print(solution)

if __name__ == "__main__":
    main()