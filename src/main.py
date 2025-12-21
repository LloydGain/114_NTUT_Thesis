import os
import copy
import time
import random
import numpy as np
import argparse
from dotenv import load_dotenv
from datetime import datetime
from data.store_data import StoreData
from data.origin_data import ODataManager
from data.manual_data import MDataManager
from data.program_data import PDataManager
from log.log import Log
from route.route import RouteManager
from route.extract_ga import StoreExtractionGA
from route.allocate_aco import StoreAllocationACO
from route.support_line_aco import SupportLinePlanningACO
from route.local_search import LocalSearch
from eval.eval_routes import EvalRoutes
from eval.display_routes import DisplayRoutes

load_dotenv()
r_seed = os.getenv("RANDOM_SEED")
r_seed = int(r_seed)
random.seed(r_seed)
np.random.seed(r_seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_date", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional). If not set, use env or random behavior.")
    parser.add_argument("--test", action="store_true", help="Run in test mode with reduced parameters")
    parser.add_argument("--google", action="store_true", help="Update routes via Google Maps API")
    return parser.parse_args()


def main(file_date, random_seed=None, test_mode=False, google=False):
    dt_folder = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f'../output/{file_date}/{dt_folder}/logs'
    route_file = f'../data/{file_date}/{file_date}route.xlsx'
    manual_file = f'../data/{file_date}/{file_date}manual.xlsx'
    program_file = f'../data/{file_date}/{file_date}program.xlsx'
    route_network_file = '../data/route_network_and_dwell_times.xlsx'
    store_info_file = '../data/store_info.xlsx'
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

    store_count_log_file = 'store_count_log.xlsx'
    store_extract_log_file = 'store_extraction_log.xlsx'
    store_allocate_log_file = 'store_allocation_log.xlsx'
    support_line_log_file = 'support_line_log.xlsx'

# -----------------------------------------------------------------------------------

    if random_seed is None:
        env_seed = os.getenv("RANDOM_SEED")
        random_seed = int(env_seed)
    
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

# -----------------------------------------------------------------------------------

    times = {}
    comment = "With local search. after optimization.（Google)"

    production_params = {
        'store_extraction_ga': {
            'population_size': 50,
            'elite_size': 10,
            'generations': 10000,
            'cross_rate': 0.9,
            'mutation_rate': 0.2,
            'early_stop_patience': 1000
        },
        'store_allocation_aco': {
            'num_ants': 50,
            'iterations': 500,
            'alpha': 1,
            'beta': 1,
            'rho': 0.1, 
            'tau_ratio': 50,
            'q': 100,
            'early_stop_patience': 10
        },
        'support_line_aco': {
            'num_ants': 50,
            'iterations': 500,
            'alpha': 1,
            'beta': 1,
            'gamma': 1,
            'local_rho': 0.1,
            'global_rho': 0.1,
            'tau_ratio': 50,
            'q': 100,
            'early_stop_patience': 50,
            'support_capacity': 7.2
        },
        'comment': comment,
        'date': file_date,
        'google': google,
        'random.seed': random_seed
    }

    test_params = {
        'store_extraction_ga': {
            'population_size': 2,
            'elite_size': 2,
            'generations': 2,
            'cross_rate': 0.8,
            'mutation_rate': 0.2,
            'early_stop_patience': 1
        },
        'store_allocation_aco': {
            'num_ants': 1,
            'iterations': 1,
            'alpha': 1,
            'beta': 1,
            'rho': 0.1, 
            'tau_ratio': 50,
            'q': 1,
            'early_stop_patience': 1
        },
        'support_line_aco': {
            'num_ants': 1,
            'iterations': 1,
            'alpha': 1,
            'beta': 1,
            'gamma': 1,
            'local_rho': 0.1,
            'global_rho': 0.1,
            'tau_ratio': 50,
            'q': 1,
            'q0': 0.9,
            'early_stop_patience': 1,
            'support_capacity': 7.2
        },
        'Test': True,
        'comment': comment
    }

    if test_mode:
        params = test_params
    else:
        params = production_params

# -----------------------------------------------------------------------------------

    start_time = time.time()

    print("Calculating store distance & time...")
    s_data = StoreData(store_info_file)
    distance_matrix, time_matrix = s_data.distance_matrix, s_data.time_matrix

    end_time = time.time()
    time_consume = round(end_time - start_time, 2)
    print(f"計算店鋪矩陣執行時間: {time_consume} 秒")

    times['Calculating store distance & time'] = time_consume

# -----------------------------------------------------------------------------------

    start_time = time.time()

    print("Loading route data...")
    o_data = ODataManager([route_file, route_network_file, store_info_file], distance_matrix, time_matrix)
    o_data.save_routes_to_json(original_route)
    routes = copy.deepcopy(o_data.routes_info)

    end_time = time.time()
    time_consume = round(end_time - start_time, 2)
    print(f"資料讀取執行時間: {time_consume} 秒")

    times['Loading route data...'] = time_consume

# -----------------------------------------------------------------------------------

    start_time = time.time()

    print("Starting Store Extraction using GA...")
    ga_params = params['store_extraction_ga']
    store_extract = StoreExtractionGA(routes, distance_matrix, time_matrix, **ga_params)
    main_routes, extracted_stores = store_extract.run()
    store_extract_log_data = store_extract.log

    end_time = time.time()
    time_consume = round(end_time - start_time, 2)
    print(f"店鋪抽取執行時間: {time_consume} 秒")

    times['Starting Store Extraction using GA...'] = time_consume

# -----------------------------------------------------------------------------------

    start_time = time.time()

    print("Starting Store Allocation using ACO...")
    allocate_params = params['store_allocation_aco']
    store_allocate = StoreAllocationACO(main_routes, extracted_stores, distance_matrix, time_matrix, **allocate_params)
    _, main_routes, remaining_stores = store_allocate.run()
    store_allocate_log_data = store_allocate.log

    end_time = time.time()
    time_consume = round(end_time - start_time, 2)
    print(f"店鋪再分配執行時間: {time_consume} 秒")

    times['Starting Store Allocation using ACO...'] = time_consume

# -----------------------------------------------------------------------------------

    start_time = time.time()

    print("Starting Support Line Planning using ACO...")
    support_params = params['support_line_aco']
    support = SupportLinePlanningACO(remaining_stores, distance_matrix, time_matrix, **support_params)
    _, support_routes = support.run()
    support_line_log_data = support.log

    end_time = time.time()
    time_consume = round(end_time - start_time, 2)
    print(f"支援線規劃執行時間: {time_consume} 秒")

    times['Starting Support Line Planning using ACO...'] = time_consume

# -----------------------------------------------------------------------------------

    optimized_routes = {**support_routes, **main_routes}
    # optimized_routes = {**main_routes, **support_routes}
    print(f'Extracted Store Count: {len(extracted_stores)}')
    print(f'Allocate Store Count: {len(extracted_stores) - len(remaining_stores)}')
    print(f'Support Line Store Count: {len(remaining_stores)}')

# -----------------------------------------------------------------------------------

    start_time = time.time()

    print("Starting Local Search...")
    optimized_cost = sum(route['dc']['distance'] for route in optimized_routes.values())
    ls = LocalSearch(distance_matrix, time_matrix)
    optimized_routes, optimized_cost = ls.optimize_inter_route(optimized_routes, optimized_cost)
    optimized_routes, optimized_cost = ls.optimize_intra_route(optimized_routes, optimized_cost)
    
    end_time = time.time()
    time_consume = round(end_time - start_time, 2)
    print(f"Local Search 執行時間: {time_consume} 秒")

    times['Starting Local Search...'] = time_consume

# -----------------------------------------------------------------------------------

    start_time = time.time()

    print("Exporting optimized route data...")
    route_manager = RouteManager(optimized_routes)
    if google:
        route_manager.update_all_routes_distance_and_duration_with_GoogleAPI()
    route_manager.export_routes_info(optimized_routes_file)

    end_time = time.time()
    time_consume = round(end_time - start_time, 2)
    print(f"最佳化路線匯出執行時間: {time_consume} 秒")

    times['Exporting optimized route data...'] = time_consume

# -----------------------------------------------------------------------------------

    start_time = time.time()

    print("Loading manual route data...")
    m_data = MDataManager([manual_file, route_network_file, store_info_file], distance_matrix, time_matrix)
    if google:
        m_route_manager = RouteManager(m_data.routes_info, distance_matrix, time_matrix)
        m_route_manager.update_all_routes_distance_and_duration_with_GoogleAPI()
    m_data.save_routes_to_json(manual_routes_file)

    end_time = time.time()
    time_consume = round(end_time - start_time, 2)
    print(f"手動編排資料讀取執行時間: {time_consume} 秒")

    times['Loading manual route data...'] = time_consume

# -----------------------------------------------------------------------------------

    # start_time = time.time()

    # print("Loading program route data...")
    # p_data = PDataManager([program_file, route_network_file, store_info_file], distance_matrix, time_matrix)
    # p_data.save_routes_to_json(program_routes_file)

    # end_time = time.time()
    # print(f"學長編排資料讀取執行時間: {end_time - start_time:.2f} 秒")

# -----------------------------------------------------------------------------------

    start_time = time.time()

    print("Evaluating and comparing routes...")
    # eval = EvalRoutes(manual_routes_file, optimized_routes_file, program_routes_file) # 1203
    eval = EvalRoutes(manual_routes_file, optimized_routes_file) # 1205 # 1207
    eval.export_to_excel(route_comparison_file)

    end_time = time.time()
    time_consume = round(end_time - start_time, 2)
    print(f"最佳化路線與手動編排路線比較執行時間: {time_consume} 秒")
    
    times['Evaluating and comparing routes...'] = time_consume

# -----------------------------------------------------------------------------------

    start_time = time.time()

    print("Logging ...")
    logger = Log(log_dir, params, times)
    logger.log_parameters()
    logger.log_times()
    logger.log_route(store_count_log_file, o_data.routes_info, m_data.routes_info, optimized_routes)
    logger.log_execution(store_extract_log_file, store_extract_log_data)
    logger.log_execution(store_allocate_log_file, store_allocate_log_data)
    logger.log_execution(support_line_log_file, support_line_log_data)
    
    end_time = time.time()
    print(f"記錄實驗參數和結果執行時間: {end_time - start_time:.2f} 秒")

# -----------------------------------------------------------------------------------

    start_time = time.time()

    print("Displaying route visualizations...")
    manu_routes = DisplayRoutes(manual_routes_file)
    manu_routes.plot_routes_png(manual_routes_img)
    manu_routes.plot_routes_html(manual_routes_html)

    # prog_routes = DisplayRoutes(program_routes_file)
    # prog_routes.plot_routes_png(program_routes_img)
    # prog_routes.plot_routes_html(program_route_html)

    opt_routes = DisplayRoutes(optimized_routes_file)
    opt_routes.plot_routes_png(optimized_routes_img)
    opt_routes.plot_routes_html(optimized_routes_html)

    end_time = time.time()
    print(f"路線視覺化執行時間: {end_time - start_time:.2f} 秒")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    main(args.file_date, args.seed, args.test, args.google)