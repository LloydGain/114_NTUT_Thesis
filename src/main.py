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
    log_dir = f'../output/{file_date}/{dt_folder}/logs'
    route_file = f'../data/{file_date}route.xlsx'
    manual_file = f'../data/{file_date}manual.xlsx'
    program_file = f'../data/{file_date}program.xlsx'
    route_network_file = '../data/route_network_and_dwell_times.xlsx'
    original_route = f'../output/{file_date}/original_routes_info.json'
    manual_routes_file = f'../output/{file_date}/manual_routes_info.json'
    program_routes_file = f'../output/{file_date}/program_routes_info.json'
    optimized_routes_file = f'../output/{file_date}/{dt_folder}/optimized_routes_info.json'
    route_comparison_file = f'../output/{file_date}/{dt_folder}/routes_comparison.xlsx'

    manual_routes_dir = f'../output/{file_date}/manual_routes'
    program_routes_dir = f'../output/{file_date}/program_routes'
    optimized_routes_dir = f'../output/{file_date}/{dt_folder}/optimized_routes'
    manual_routes_img = f'{manual_routes_dir}/img'
    program_routes_img = f'{program_routes_dir}/img'
    optimized_routes_img = f'{optimized_routes_dir}/img'
    manual_routes_html = f'{manual_routes_dir}/routes.html'
    program_route_html = f'{program_routes_dir}/routes.html'
    optimized_routes_html = f'{optimized_routes_dir}/routes.html'

    store_extract_log_file = 'store_extraction_log.xlsx'
    store_allocate_log_file = 'store_allocation_log.xlsx'
    support_line_log_file = 'support_line_log.xlsx'

# -----------------------------------------------------------------------------------

    params = {
        'store_extraction_ga': {
            'p_size': 200,
            'generations': 2000
        },
        'store_allocation_aco': {
            'num_ants': 100,
            'iterations': 200
        },
        'support_line_aco': {
            'num_ants': 100,
            'iterations': 200
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
    store_extract = StoreExtractionGA(routes, distance_matrix, time_matrix, population_size=params['store_extraction_ga']['p_size'], generations=params['store_extraction_ga']['generations'])
    main_routes, extracted_stores = store_extract.run()
    store_extract_log_data = store_extract.log

    end_time = time.time()
    print(f"店鋪抽取執行時間: {end_time - start_time:.2f} 秒")

# -----------------------------------------------------------------------------------

    start_time = time.time()

    print("Starting Store Allocation using ACO...")
    store_allocate = StoreAllocationACO(main_routes, extracted_stores, distance_matrix, time_matrix, num_ants=params['store_allocation_aco']['num_ants'], iterations=params['store_allocation_aco']['iterations'])
    _, main_routes, remaining_stores = store_allocate.run()
    store_allocate_log_data = store_allocate.log

    end_time = time.time()
    print(f"店鋪再分配執行時間: {end_time - start_time:.2f} 秒")

# -----------------------------------------------------------------------------------

    start_time = time.time()

    print("Starting Support Line Planning using ACO...")
    support = SupportLinePlanningACO(remaining_stores, distance_matrix, time_matrix, num_ants=params['support_line_aco']['num_ants'], iterations=params['support_line_aco']['iterations'])
    _, support_routes = support.run()
    support_line_log_data = support.log

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
    manu_routes = DisplayRoutes(manual_routes_file)
    manu_routes.plot_routes_png(manual_routes_img)
    manu_routes.plot_routes_html(manual_routes_html)

    prog_routes = DisplayRoutes(program_routes_file)
    prog_routes.plot_routes_png(program_routes_img)
    prog_routes.plot_routes_html(program_route_html)

    opt_routes = DisplayRoutes(optimized_routes_file)
    opt_routes.plot_routes_png(optimized_routes_img)
    opt_routes.plot_routes_html(optimized_routes_html)

    end_time = time.time()
    print(f"路線視覺化執行時間: {end_time - start_time:.2f} 秒")

# -----------------------------------------------------------------------------------

    start_time = time.time()

    print("Logging ...")
    logger = Log(log_dir, params)
    logger.log_parameters()
    logger.log_execution(store_extract_log_file, store_extract_log_data)
    logger.log_execution(store_allocate_log_file, store_allocate_log_data)
    logger.log_execution(support_line_log_file, support_line_log_data)
    
    end_time = time.time()
    print(f"記錄實驗參數和結果執行時間: {end_time - start_time:.2f} 秒")

# ---------------------------------------------------------------------------


if __name__ == "__main__":
    main()