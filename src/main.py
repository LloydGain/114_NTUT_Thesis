from data_manager import DataManager
from extract_aco import StoreExtractionACO

def main():
    route_file = '../data/1203route.xlsx'
    route_network_file = '../data/route_network_and_dwell_times.xlsx'
    original_route = '../output/original_routes_info.json'

    manager = DataManager([route_file, route_network_file])
    manager.save_routes_to_json(original_route)
    routes = manager.routes_info

    extractor = StoreExtractionACO(routes)
    extracted_stores = extractor.run()

    for store in extracted_stores:
        print(store)
    print(len(extracted_stores))


if __name__ == "__main__":
    main()