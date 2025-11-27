import copy
import time
from data.origin_data import ODataManager
from data.manual_data import MDataManager
from route.route import RouteManager
from route.extract_ga import StoreExtractionGA
from route.allocate_aco import StoreAllocationACO
from route.support_line_aco import SupportLinePlanningACO
from eval.display_routes import DisplayRoutes


def main():
    file_date = '1203'
    route_file = f'../data/{file_date}route.xlsx'
    manual_file = f'../data/{file_date}manual.xlsx'
    route_network_file = '../data/route_network_and_dwell_times.xlsx'
    original_route = '../output/original_routes_info.json'
    manual_routes_file = '../output/manual_routes_info.json'
    optimized_routes_file = '../output/optimized_routes_info.json'

# -----------------------------------------------------------------------------------

    start_time = time.time()

    print("Loading route data...")
    o_data = ODataManager([route_file, route_network_file])
    o_data.save_routes_to_json(original_route)
    routes = copy.deepcopy(o_data.routes_info)
    distance_matrix, time_matrix = o_data.distance_matrix, o_data.time_matrix

    end_time = time.time()
    print(f"資料讀取執行時間: {end_time - start_time:.2f} 秒")

# -----------------------------------------------------------------------------------

    start_time = time.time()

    print("Starting Store Extraction using GA...")
    store_extract = StoreExtractionGA(routes, distance_matrix, time_matrix, population_size=10, generations=1)
    main_routes, extracted_stores = store_extract.run()

    end_time = time.time()
    print(f"店鋪抽取執行時間: {end_time - start_time:.2f} 秒")

# -----------------------------------------------------------------------------------

    start_time = time.time()

    print("Starting Store Allocation using ACO...")
    store_allocate = StoreAllocationACO(main_routes, extracted_stores, distance_matrix, time_matrix, num_ants=10, iterations=1)
    allocate_cost, main_routes, remaining_stores = store_allocate.run()

    end_time = time.time()
    print(f"店鋪再分配執行時間: {end_time - start_time:.2f} 秒")

# -----------------------------------------------------------------------------------

    start_time = time.time()

    print("Starting Support Line Planning using ACO...")
    support = SupportLinePlanningACO(remaining_stores, distance_matrix, time_matrix, num_ants=10, iterations=1)
    support_cost, support_routes = support.run()

    end_time = time.time()
    print(f"支援線規劃執行時間: {end_time - start_time:.2f} 秒")

# -----------------------------------------------------------------------------------

    optimized_routes = {**support_routes, **main_routes}
    total_optimized_cost = support_cost + allocate_cost
    print(f"Total Optimized Cost (OSRM): {total_optimized_cost}")

# -----------------------------------------------------------------------------------

    start_time = time.time()

    print("Exporting optimized route data...")
    route_manager = RouteManager(optimized_routes)
    route_manager.export_routes_info(optimized_routes_file)

    end_time = time.time()
    print(f"最佳化路線匯出執行時間: {end_time - start_time:.2f} 秒")

# -----------------------------------------------------------------------------------

    start_time = time.time()

    print("Loading manual route data...")
    m_data = MDataManager([manual_file, route_network_file], distance_matrix, time_matrix)
    m_data.save_routes_to_json(manual_routes_file)

    end_time = time.time()
    print(f"手動編排資料讀取執行時間: {end_time - start_time:.2f} 秒")

# -----------------------------------------------------------------------------------

    start_time = time.time()

    print("Evaluating and comparing routes...")
    eval = EvaluateRoutes(optimized_routes_file, manual_routes_file)
    eval.compare_routes()
    # dis_routes = DisplayRoutes(optimized_route_file)
    # dis_routes.show_optimized_result()
    # dis_routes.plot_routes()

    end_time = time.time()
    print(f"最佳化路線與手動編排路線比較執行時間: {end_time - start_time:.2f} 秒")

# -----------------------------------------------------------------------------------


if __name__ == "__main__":
    main()