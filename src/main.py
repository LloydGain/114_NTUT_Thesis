import os
import copy
import time
import random
import numpy as np
import argparse
from config import config
from datetime import datetime
from data.store_data import StoreData
from data.origin_data import ODataManager
from data.manual_data import MDataManager
from data.program_data import PDataManager
from utils.logger import Log
from models.route_manager import RouteManager
from solvers.extract_ga import StoreExtractionGA

from solvers.allocate_aco import StoreAllocationACO
from solvers.support_line_macs import SupportLinePlanningMACS
from eval.eval_routes import EvalRoutes
from eval.display_routes import DisplayRoutes
from services.osrm import OSRM

def parse_args():
    """
    Notes:
        Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_date", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional). If not set, use env or random behavior.")
    parser.add_argument("--test", action="store_true", help="Run in test mode with reduced parameters")
    parser.add_argument("--google", action="store_true", help="Update routes via Google Maps API")
    parser.add_argument("--comment", type=str, default=None, help="Comment for the run (optional)")
    parser.add_argument("--skip_compare", action="store_true", help="Skip comparison with manual and program routes")
    parser.add_argument("--alb", type=str, nargs='+', choices=['extract', 'allocate', 'support', 'vnd'], default=[], help="Ablation options: extract, allocate, support, vnd")
    return parser.parse_args()


def main(file_date, random_seed=None, test_mode=False, google=False, comment=None, skip_compare=False, hyper_params=None, alb=None):
    if alb is None:
        alb = []
    """
    Notes:
        Main function for running the program.
    """
    
    try:
        OSRM().check_osrm()
    except Exception as e:
        print(f"[ERROR] OSRM service check failed: {e}")
        return

    dt_folder = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f'../output/{file_date}/{dt_folder}/logs'
    route_file = f'../data/{file_date}/{file_date}route.xlsx'
    manual_file = f'../data/{file_date}/{file_date}manual.xlsx'
    program_file = f'../data/{file_date}/{file_date}program.xlsx'
    dist_file = '../data/osrm/store_distance_matrix.json'
    time_file = '../data/osrm/store_time_matrix.json'
    route_network_file = '../data/route_network_and_dwell_times.xlsx'
    store_info_file = '../data/store_info.xlsx'
    original_routes_file = f'../output/{file_date}/original_routes_info.json'
    manual_routes_file = f'../output/{file_date}/manual_routes_info.json'
    program_routes_file = f'../output/{file_date}/program_routes_info.json'
    optimized_routes_file = f'../output/{file_date}/{dt_folder}/optimized_routes_info.json'
    optimized_routes_excel_file = f'../output/{file_date}/{dt_folder}/optimized_routes_info.xlsx'
    route_comparison_file = f'../output/{file_date}/{dt_folder}/routes_comparison.xlsx'
    route_comparison_simple_file = f'../output/{file_date}/{dt_folder}/routes_comparison_simple.xlsx'

    original_routes_dir = f'../output/{file_date}/original_routes'
    manual_routes_dir = f'../output/{file_date}/manual_routes'
    program_routes_dir = f'../output/{file_date}/program_routes'
    optimized_routes_dir = f'../output/{file_date}/{dt_folder}/optimized_routes'
    original_routes_img = f'{original_routes_dir}/img'
    manual_routes_img = f'{manual_routes_dir}/img'
    program_routes_img = f'{program_routes_dir}/img'
    # optimized_routes_img = f'{optimized_routes_dir}/img'
    original_routes_osrm_html = f'{original_routes_dir}/osrm_routes.html'
    manual_routes_osrm_html = f'{manual_routes_dir}/osrm_routes.html'
    program_route_osrm_html = f'{program_routes_dir}/osrm_routes.html'
    optimized_routes_osrm_html = f'{optimized_routes_dir}/osrm_routes.html'
    original_routes_html = f'{original_routes_dir}/routes.html'
    manual_routes_html = f'{manual_routes_dir}/routes.html'
    program_route_html = f'{program_routes_dir}/routes.html'
    optimized_routes_html = f'{optimized_routes_dir}/routes.html'

    store_count_log_file = 'log_count.xlsx'
    store_extract_log_file = 'store_extraction_log.xlsx'
    store_allocate_log_file = 'store_allocation_log.xlsx'
    support_line_log_file = 'store_support_line_log.xlsx'

# -----------------------------------------------------------------------------------

    if random_seed is None:
        random_seed = config.RANDOM_SEED

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

# -----------------------------------------------------------------------------------

    times = {}

    production_params = {
        'store_extraction_ga': {
            'population_size': 20,
            'elite_rate': 0.1,
            'generations': 200,
            'cross_rate': 0.8,
            'mutation_rate': 0.1,
            'early_stop_patience': 20
        },
        'store_allocation_aco': {
            'num_ants': 50,
            'iterations': 200,
            'alpha': 1,
            'beta': 1,
            'q0': 0.8,
            'rho': 0.7,
            'early_stop_patience': 20
        },
        'support_line_macs': {
            'time_limit': 100,
            'num_ants': 50,
            'alpha': 1,
            'beta': 7,
            'rho': 0.5,
            'q0': 0.8,
            'early_stop_patience': 20,
            'support_capacity': 7.2,
            'vehicle_cost': 2000,
        },
        'comment': comment,
        'date': file_date,
        'google': google,
        'random.seed': random_seed
    }

    test_params = {
        'store_extraction_ga': {
            'population_size': 10,
            'elite_rate': 0.1,
            'generations': 2,
            'cross_rate': 0.8,
            'mutation_rate': 0.2,
            'early_stop_patience': 1
        },
        'store_allocation_aco': {
            'num_ants': 5,
            'iterations': 5,
            'alpha': 1,
            'beta': 5,
            'q0': 0.8,
            'rho': 0.2,
            'early_stop_patience': 3,
        },
        'support_line_macs': {
            'time_limit': 30,
            'num_ants': 25,
            # 'alpha': 1,
            'beta': 1,
            'rho': 0.1,
            # 'q': 1,
            'q0': 0.8,
            'early_stop_patience': 10,
            'support_capacity': 7.2,
            'vehicle_cost': 2000,
        },
        'Test': True,
        'comment': comment
    }

    if test_mode:
        params = test_params
    else:
        params = production_params

    if hyper_params:
        if 'ex_iter' in hyper_params:
            params['store_extraction_ga']['generations'] = hyper_params['ex_iter']
        if 'ex_pop' in hyper_params:
            params['store_extraction_ga']['population_size'] = hyper_params['ex_pop']
        if 'ex_cx' in hyper_params:
            params['store_extraction_ga']['cross_rate'] = hyper_params['ex_cx']
        if 'ex_mut' in hyper_params:
            params['store_extraction_ga']['mutation_rate'] = hyper_params['ex_mut']

        if 'al_iters' in hyper_params:
            params['store_allocation_aco']['iterations'] = hyper_params['al_iters']
        if 'al_ants' in hyper_params:
            params['store_allocation_aco']['num_ants'] = hyper_params['al_ants']
        if 'al_alpha' in hyper_params:
            params['store_allocation_aco']['alpha'] = hyper_params['al_alpha']
        if 'al_beta' in hyper_params:
            params['store_allocation_aco']['beta'] = hyper_params['al_beta']
        if 'al_rho' in hyper_params:
            params['store_allocation_aco']['rho'] = hyper_params['al_rho']

        if 'time_limit' in hyper_params:
            params['support_line_macs']['time_limit'] = hyper_params['time_limit']
        if 'beta' in hyper_params:
            params['support_line_macs']['beta'] = hyper_params['beta']
        if 'rho' in hyper_params:
            params['support_line_macs']['rho'] = hyper_params['rho']
        if 'q0' in hyper_params:
            params['support_line_macs']['q0'] = hyper_params['q0']
        if 'ants' in hyper_params:
            params['support_line_macs']['num_ants'] = hyper_params['ants']

    if 'vnd' in alb:
        params['support_line_macs']['vnd_strategy'] = 'none'

# -----------------------------------------------------------------------------------

    start_time = time.time()

    print("Calculating store distance & time...")
    s_data = StoreData(store_info_file)
    # distance_matrix, time_matrix = s_data.get_cost_matrices(dist_file, time_file)
    distance_matrix, time_matrix = s_data.load_matrices_from_file(dist_file, time_file)

    end_time = time.time()
    time_consume = round(end_time - start_time, 2)
    print(f"計算店鋪矩陣執行時間: {time_consume} 秒")

    times['Calculating store distance & time'] = time_consume

# -----------------------------------------------------------------------------------

    start_time = time.time()

    print("Loading route data...")
    o_data = ODataManager([route_file, route_network_file, store_info_file], distance_matrix, time_matrix)
    o_data.save_routes_to_json(original_routes_file)
    routes = copy.deepcopy(o_data.routes_info)

    end_time = time.time()
    time_consume = round(end_time - start_time, 2)
    print(f"資料讀取執行時間: {time_consume} 秒")

    times['Loading route data...'] = time_consume

# -----------------------------------------------------------------------------------

    start_time = time.time()

    mode_extract = 'random' if 'extract' in alb else 'ccga'
    print(f"Starting Store Extraction using GA (Mode: {mode_extract})...")
    ga_params = params['store_extraction_ga']
    store_extract = StoreExtractionGA(routes, distance_matrix, time_matrix, mode=mode_extract, **ga_params)
    main_routes, extracted_stores = store_extract.run()
    store_extract_log_data = store_extract.log

    end_time = time.time()
    time_consume = round(end_time - start_time, 2)
    print(f"店鋪抽取執行時間: {time_consume} 秒")

    times['Starting Store Extraction using GA...'] = time_consume

# -----------------------------------------------------------------------------------

    start_time = time.time()

    mode_allocate = 'aco' if 'allocate' in alb else 'aco_vnd'
    print(f"Starting Store Allocation using ACO (Mode: {mode_allocate})...")
    heatmap_dir = f'../output/{file_date}/{dt_folder}/heatmaps'
    allocate_params = params['store_allocation_aco']
    store_allocate = StoreAllocationACO(main_routes, extracted_stores, distance_matrix, time_matrix, mode=mode_allocate, output_dir=heatmap_dir, **allocate_params)
    _, main_routes, remaining_stores, _ = store_allocate.run()
    store_allocate_log_data = store_allocate.log

    end_time = time.time()
    time_consume = round(end_time - start_time, 2)
    print(f"店鋪再分配執行時間: {time_consume} 秒")

    times['Starting Store Allocation using ACO...'] = time_consume

# -----------------------------------------------------------------------------------

    start_time = time.time()

    mode_support = 'macs' if 'support' in alb else 'macs_vnd'
    print(f"Starting Support Line Planning using MACS (Mode: {mode_support})...")
    support_params = params['support_line_macs']
    support = SupportLinePlanningMACS(remaining_stores, distance_matrix, time_matrix, mode=mode_support, **support_params)
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

    print("Exporting optimized route data...")
    route_manager = RouteManager(optimized_routes, distance_matrix, time_matrix)
    if google:
        route_manager.update_all_routes_distance_and_duration_with_google_api()
    else:
        route_manager.update_all_routes_info()
    route_manager.export_routes_info(optimized_routes_file)
    route_manager.export_excel_file(optimized_routes_excel_file)

    end_time = time.time()
    time_consume = round(end_time - start_time, 2)
    print(f"最佳化路線匯出執行時間: {time_consume} 秒")

    times['Exporting optimized route data...'] = time_consume

# -----------------------------------------------------------------------------------

    m_data = None
    if not skip_compare:
        start_time = time.time()

        print("Loading manual route data...")
        m_data = MDataManager([manual_file, route_network_file, store_info_file], distance_matrix, time_matrix)
        if google:
            m_route_manager = RouteManager(m_data.routes_info, distance_matrix, time_matrix)
            m_route_manager.update_all_routes_distance_and_duration_with_google_api()
        m_data.save_routes_to_json(manual_routes_file)

        end_time = time.time()
        time_consume = round(end_time - start_time, 2)
        print(f"手動編排資料讀取執行時間: {time_consume} 秒")

        times['Loading manual route data...'] = time_consume

# -----------------------------------------------------------------------------------

        if os.path.exists(program_file):
            start_time = time.time()

            print("Loading program route data...")
            p_data = PDataManager([program_file, route_network_file, store_info_file], distance_matrix, time_matrix)
            if google:
                p_route_manager = RouteManager(p_data.routes_info, distance_matrix, time_matrix)
                p_route_manager.update_all_routes_distance_and_duration_with_google_api()
            p_data.save_routes_to_json(program_routes_file)

            end_time = time.time()
            time_consume = round(end_time - start_time, 2)
            print(f"學長編排資料讀取執行時間: {time_consume} 秒")

            times['Loading program route data...'] = time_consume

# -----------------------------------------------------------------------------------

        start_time = time.time()

        print("Evaluating and comparing routes...")
        # eval = EvalRoutes(manual_routes_file, optimized_routes_file, program_routes_file) # 1203
        eval_routes = EvalRoutes(manual_routes_file, optimized_routes_file) # 1205 # 1207
        eval_routes.export_to_excel(route_comparison_file)
        eval_routes.export_to_excel(route_comparison_simple_file, simple=True)

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
    m_routes_info = m_data.routes_info if m_data else {}
    logger.log_route(store_count_log_file, o_data.routes_info, m_routes_info, optimized_routes)
    logger.log_execution(store_extract_log_file, store_extract_log_data)
    logger.log_execution(store_allocate_log_file, store_allocate_log_data)
    logger.log_execution(support_line_log_file, support_line_log_data)

    end_time = time.time()
    print(f"記錄實驗參數和結果執行時間: {end_time - start_time:.2f} 秒")

# -----------------------------------------------------------------------------------

    if not test_mode:
        start_time = time.time()

        print("Displaying route visualizations...")
        if not os.path.exists(original_routes_img) or not os.path.exists(original_routes_html):
            original_routes = DisplayRoutes(original_routes_file)
            original_routes.make_dir(original_routes_dir)
            # original_routes.plot_routes_png(original_routes_img)
            original_routes.plot_routes_html_in_osrm(original_routes_osrm_html)
            original_routes.plot_routes_html(original_routes_html)

        if not skip_compare:
            if not os.path.exists(manual_routes_img) or not os.path.exists(manual_routes_html):
                manual_routes = DisplayRoutes(manual_routes_file)
                manual_routes.make_dir(manual_routes_dir)
                # manual_routes.plot_routes_png(manual_routes_img)
                manual_routes.plot_routes_html_in_osrm(manual_routes_osrm_html)
                manual_routes.plot_routes_html(manual_routes_html)

            if os.path.exists(program_routes_file):
                if not os.path.exists(program_routes_img) or not os.path.exists(program_route_html):
                    prog_routes = DisplayRoutes(program_routes_file)
                    prog_routes.make_dir(program_routes_dir)
                    # prog_routes.plot_routes_png(program_routes_img)
                    prog_routes.plot_routes_html_in_osrm(program_route_osrm_html)
                    prog_routes.plot_routes_html(program_route_html)

        opt_routes = DisplayRoutes(optimized_routes_file)
        opt_routes.make_dir(optimized_routes_dir)
        # opt_routes.plot_routes_png(optimized_routes_img)
        opt_routes.plot_routes_html_in_osrm(optimized_routes_osrm_html)
        opt_routes.plot_routes_html(optimized_routes_html)

        end_time = time.time()
        print(f"路線視覺化執行時間: {end_time - start_time:.2f} 秒")

# -----------------------------------------------------------------------------------

    total_vehicles   = len(optimized_routes)
    vehicle_cost     = 2000
    support_vehicles = sum(1 for r in optimized_routes.values() if r['dc']['route_id'].isdigit())
    total_distance   = sum(r['dc']['distance'] for r in optimized_routes.values())
    total_duration   = sum(r['dc']['duration'] for r in optimized_routes.values()) / 3600
    total_stores     = sum(len(r['stores']) for r in optimized_routes.values())
    avg_distance     = total_distance / total_vehicles if total_vehicles else 0
    avg_duration     = total_duration / total_vehicles if total_vehicles else 0
    avg_load_rate    = (
        sum(r['dc']['load_rate'] for r in optimized_routes.values()) / total_vehicles
        if total_vehicles else 0
    )

    on_time = sum(
        1
        for r in optimized_routes.values()
        for s in r['stores']
        if s['earliest_time'] <= s['pred_time'] <= s['latest_time']
    )
    on_time_rate = on_time / total_stores if total_stores else 0

    optimized_cost = total_distance + (support_vehicles * vehicle_cost)

    return {
        "cost":            optimized_cost,
        "vehicle_num":     total_vehicles,
        "total_store_num": total_stores,
        "total_dist(km)":  round(total_distance,  4),
        "total_time(hr)":  round(total_duration,  4),
        "avg_dist(km)":    round(avg_distance,    4),
        "avg_time(hr)":    round(avg_duration,    4),
        "avg_load_rate":   round(avg_load_rate,   4),
        "on_time_rate":    round(on_time_rate,    4),
    }

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        args = parse_args()
        main(args.file_date, args.seed, args.test, args.google, args.comment, args.skip_compare, alb=args.alb)
    except KeyboardInterrupt:
        print("Execution interrupted by user.")
        exit(1)