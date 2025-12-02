import copy
import time
from datetime import datetime
from data.origin_data import ODataManager
from data.manual_data import MDataManager
from data.program_data import PDataManager
from log.log import Log
from route.route import RouteManager
from route.extract_ga import StoreExtractionGA
from route.allocate_aco import StoreAllocationACO
from route.support_line_aco import SupportLinePlanningACO
from eval.eval_routes import EvalRoutes
from eval.display_routes import DisplayRoutes


def main():
    file_date = '1203'
    dt_folder = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f'../output/{file_date}/{dt_folder}'
    route_file = f'../data/{file_date}route.xlsx'
    manual_file = f'../data/{file_date}manual.xlsx'
    program_file = f'../data/{file_date}program.xlsx'
    route_network_file = '../data/route_network_and_dwell_times.xlsx'
    original_route = f'../output/{file_date}/original_routes_info.json'
    manual_routes_file = f'../output/{file_date}/manual_routes_info.json'
    program_routes_file = f'../output/{file_date}/program_routes_info.json'
    optimized_routes_file = f'../output/{file_date}/{dt_folder}/optimized_routes_info.json'
    route_comparison_file = f'../output/{file_date}/{dt_folder}/routes_comparison.xlsx'

    # manual_routes_img = f'../output/{file_date}/manual_routes'
    # program_routes_img = f'../output/{file_date}/program_routes'
    optimized_routes_img = f'../output/{file_date}/{dt_folder}/optimized_routes'

# -----------------------------------------------------------------------------------

    params = {
        'store_extraction_ga': {
            'population_size': 100,
            'generations': 1000
        },
        'store_allocation_aco': {
            'num_ants': 50,
            'iterations': 100
        },
        'support_line_aco': {
            'num_ants': 50,
            'iterations': 100
        }
    }

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
    store_extract = StoreExtractionGA(routes, distance_matrix, time_matrix, population_size=params['store_extraction_ga']['populatin_size'], generations=params['store_extraction_ga']['generations'])
    main_routes, extracted_stores = store_extract.run()

    end_time = time.time()
    print(f"店鋪抽取執行時間: {end_time - start_time:.2f} 秒")

# -----------------------------------------------------------------------------------

    start_time = time.time()

    print("Starting Store Allocation using ACO...")
    store_allocate = StoreAllocationACO(main_routes, extracted_stores, distance_matrix, time_matrix, num_ants=params['store_allocation_aco']['num_ants'], iterations=params['store_allocation_aco']['iterations'])
    _, main_routes, remaining_stores = store_allocate.run()

    end_time = time.time()
    print(f"店鋪再分配執行時間: {end_time - start_time:.2f} 秒")

# -----------------------------------------------------------------------------------

    start_time = time.time()

    print("Starting Support Line Planning using ACO...")
    support = SupportLinePlanningACO(remaining_stores, distance_matrix, time_matrix, num_ants=params['support_line_aco']['num_ants'], iterations=params['support_line_aco']['iterations'])
    _, support_routes = support.run()

    end_time = time.time()
    print(f"支援線規劃執行時間: {end_time - start_time:.2f} 秒")

# -----------------------------------------------------------------------------------

    optimized_routes = {**support_routes, **main_routes}
    print(f'Extracted Store Count: {len(extracted_stores)}')
    print(f'Allocate Store Count: {len(extracted_stores) - len(remaining_stores)}')
    print(f'Support Line Store Count: {len(remaining_stores)}')

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

    print("Loading program route data...")
    m_data = PDataManager([program_file, route_network_file], distance_matrix, time_matrix)
    m_data.save_routes_to_json(program_routes_file)

    end_time = time.time()
    print(f"學長編排資料讀取執行時間: {end_time - start_time:.2f} 秒")

# -----------------------------------------------------------------------------------

    start_time = time.time()

    print("Evaluating and comparing routes...")
    eval = EvalRoutes(manual_routes_file, program_routes_file, optimized_routes_file)
    eval.export_to_excel(route_comparison_file)

    end_time = time.time()
    print(f"最佳化路線與手動編排路線比較執行時間: {end_time - start_time:.2f} 秒")

# -----------------------------------------------------------------------------------

    start_time = time.time()

    print("Displaying route visualizations...")
    # manu_routes = DisplayRoutes(manual_routes_file, manual_routes_img)
    # manu_routes.plot_routes()

    # prog_routes = DisplayRoutes(program_routes_file, program_routes_img)
    # prog_routes.plot_routes()

    opt_routes = DisplayRoutes(optimized_routes_file, optimized_routes_img)
    opt_routes.plot_routes()

    end_time = time.time()
    print(f"路線視覺化執行時間: {end_time - start_time:.2f} 秒")

# -----------------------------------------------------------------------------------

    start_time = time.time()

    print("Logging ...")
    logger = Log(log_dir, params)
    logger.log_parameters()
    
    end_time = time.time()
    print(f"記錄實驗參數和結果執行時間: {end_time - start_time:.2f} 秒")

# ---------------------------------------------------------------------------


if __name__ == "__main__":
    main()