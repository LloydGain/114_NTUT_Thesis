import os
import csv
import math
import random
import argparse
import sys
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from numba import njit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from setup import *
from solvers.support_line_ga import SupportLinePlanningGA

BEST_KNOWN_SOLUTIONS = {
    'c101': (10, 828.94), 'c102': (10, 828.94), 'c103': (10, 828.06), 'c104': (10, 824.78), 'c105': (10, 828.94),
    'c106': (10, 828.94), 'c107': (10, 828.94), 'c108': (10, 828.94), 'c109': (10, 828.94),
    'c201': (3, 591.56), 'c202': (3, 591.56), 'c203': (3, 591.17), 'c204': (3, 590.6), 'c205': (3, 588.85),
    'c206': (3, 588.49), 'c207': (3, 588.29), 'c208': (3, 588.32),
    'r101': (20, 1637.7), 'r102': (18, 1466.6), 'r103': (14, 1208.7), 'r104': (11, 976.61), 'r105': (15, 1355.3),
    'r106': (13, 1234.6), 'r107': (11, 1064.6), 'r108': (10, 938.2), 'r109': (13, 1146.9), 'r110': (12, 1068),
    'r111': (12, 1048.7), 'r112': (10, 953.63),
    'r201': (8, 1143.2), 'r202': (8, 1034.4), 'r203': (6, 874.87), 'r204': (5, 735.8), 'r205': (5, 954.16),
    'r206': (4, 879.86), 'r207': (4, 797.99), 'r208': (4, 705.33), 'r209': (5, 859.39), 'r210': (6, 905.21),
    'r211': (4, 753.15),
    'rc101': (14, 1619.8), 'rc102': (14, 1457.4), 'rc103': (11, 1258.0), 'rc104': (10, 1135.5), 'rc105': (15, 1513.7),
    'rc106': (13, 1378.0), 'rc107': (12, 1212.8), 'rc108': (11, 1117.5),
    'rc201': (9, 1261.8), 'rc202': (8, 1095.6), 'rc203': (5, 926.82), 'rc204': (4, 786.38), 'rc205': (7, 1157.6),
    'rc206': (7, 1054.6), 'rc207': (6, 966.08), 'rc208': (4, 778.93)
}

def export_routes(filename, best_solution, output_dir):
    route_file = os.path.join(output_dir, f"{filename}.json")
    route_manager = RouteManager(best_solution)
    route_manager.export_routes_info(route_file)

def plot_solution(filename, best_solution, nodes, distance, num_vehicles, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Dataset name in title with green-ish background
    ax.set_title(f"{filename.split('.')[0].upper()}", bbox={'facecolor':'#C1D5C0', 'pad':5, 'edgecolor':'gray'}, fontweight='bold', pad=20)
    
    # Subtitle
    ax.text(0.5, 1.02, f"travel distance: {distance:.2f}, number of vehicles: {num_vehicles}", 
            horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes, fontsize=10)
            
    # Node coordinates
    coords = {n['cust_no']: (n['x'], n['y']) for n in nodes}
    depot_x, depot_y = coords[0]
    
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    # Plot routes
    color_idx = 0
    for v_id, route in best_solution.items():
        stores = route['stores']
        if not stores: continue
        
        path_x = [depot_x]
        path_y = [depot_y]
        
        for s in stores:
            cust_no = int(s['store_id'])
            px, py = coords[cust_no]
            path_x.append(px)
            path_y.append(py)
            
        path_x.append(depot_x)
        path_y.append(depot_y)
        
        c = colors[color_idx % len(colors)]
        ax.plot(path_x, path_y, marker='.', markersize=6, color=c, linewidth=1, markerfacecolor='steelblue', markeredgecolor='steelblue')
        color_idx += 1
        
    # Plot depot distinctively
    ax.plot(depot_x, depot_y, marker='s', color='#2b2b2b', markersize=6)
    
    # Save fig
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{filename.split('.')[0]}_{num_vehicles}_{distance:.2f}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

@njit(cache=True)
def set_numba_seed(value):
    np.random.seed(value)


# Override config for DC before anything else
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import config

config.DC_CONFIG = {
    'store_id': 'dc', 
    'store_name': 'Solomon Depot', 
    'longitude': 0.0, 
    'latitude': 0.0
}

from models.route_manager import RouteManager
from solvers.support_line_aco import SupportLinePlanningACO
from solvers.support_line_ga import SupportLinePlanningGA

def parse_solomon_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    capacity = 0
    nodes = []
    
    data_section = False
    for i, line in enumerate(lines):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            capacity = float(parts[1])
            data_section = True
            continue
        
        if "CAPACITY" in line:
            idx = i + 1
            while idx < len(lines):
                val_parts = lines[idx].strip().split()
                if val_parts and len(val_parts) >= 2 and val_parts[0].isdigit():
                    capacity = float(val_parts[1])
                    break
                idx += 1

        if "READY" in line and "TIME" in line:
            data_section = True
            continue
            
        if data_section and len(parts) >= 7 and parts[0].isdigit():
            nodes.append({
                'cust_no': int(parts[0]),
                'x': float(parts[1]),
                'y': float(parts[2]),
                'demand': float(parts[3]),
                'ready_time': float(parts[4]),
                'due_date': float(parts[5]),
                'service_time': float(parts[6])
            })

    return capacity, nodes

def build_matrices(nodes):
    distance_matrix = {}
    time_matrix = {}
    for idx_a, node_a in enumerate(nodes):
        id_a = 'dc' if node_a['cust_no'] == 0 else str(node_a['cust_no'])
        distance_matrix[id_a] = {}
        time_matrix[id_a] = {}
        for idx_b, node_b in enumerate(nodes):
            id_b = 'dc' if node_b['cust_no'] == 0 else str(node_b['cust_no'])
            dist = math.hypot(node_a['x'] - node_b['x'], node_a['y'] - node_b['y'])
            distance_matrix[id_a][id_b] = dist
            time_matrix[id_a][id_b] = dist  # Speed is 1 unit
            
    return distance_matrix, time_matrix

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", type=str, choices=["aco", "ga"], default=None, help="Choose solver to run (aco or ga). If not specified, runs both.")
    parser.add_argument("--dataset", type=str, default=None, help="Specific dataset to run, e.g., 'c101.txt'.")
    parser.add_argument("--run-mode", type=str, choices=["only", "onward"], default="only", help="If --dataset is provided, 'only' runs just that dataset, 'onward' runs from that dataset to the end.")
    parser.add_argument("--seed", type=int, default=None, help="Specific seed to run.")
    parser.add_argument("--print-only", action="store_true", help="Only print results, do not output any files.")
    return parser.parse_args()

def run_single_aco_seed(args_tuple):
    run_idx, remaining_stores, distance_matrix, time_matrix, capacity, time_limit, target_cost = args_tuple
    import random
    import numpy as np
    import time
    
    seed_val = run_idx
    random.seed(seed_val)
    np.random.seed(seed_val)
    import run_solomon
    run_solomon.set_numba_seed(seed_val)
    
    from solvers.support_line_aco import SupportLinePlanningACO
    support = SupportLinePlanningACO(
        remaining_stores=remaining_stores,
        distance_matrix=distance_matrix,
        time_matrix=time_matrix,
        num_ants=len(remaining_stores) * 2,
        iterations=500, 
        early_stop_patience=100,
        support_capacity=capacity,
        vehicle_cost=2000,
        time_limit_per_route=time_limit,
        is_solomon=True,
        # target_cost=target_cost
    )
    
    try:
        start_time = time.time()
        best_cost, best_solution = support.run()
        duration = time.time() - start_time
        num_routes = len(best_solution) if best_solution else 0
        
        if isinstance(best_cost, tuple):
            distance = best_cost[1]
            best_cost_scalar = num_routes * 2000 + distance
        else:
            distance = best_cost - num_routes * 2000
            best_cost_scalar = best_cost
            
        return run_idx, seed_val, best_cost_scalar, distance, num_routes, duration, best_cost, best_solution
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e

def main():
    args = parse_args()
    solvers_to_run = [args.solver] if args.solver else ["aco", "ga"]
    solomon_dir = r"..\solomon-100"
    
    all_files = [f for f in os.listdir(solomon_dir) if f.endswith('.txt')]
    all_files.sort()
    
    if args.dataset is not None:
        dataset_name = args.dataset if args.dataset.endswith('.txt') else args.dataset + '.txt'
        if dataset_name in all_files:
            idx = all_files.index(dataset_name)
            if args.run_mode == "onward":
                files = all_files[idx:]
            else:
                files = [dataset_name]
        else:
            print(f"Dataset {dataset_name} not found in {solomon_dir}. Fallback to exact match.")
            files = [args.dataset]
    else:
        files = all_files
        
    print_only = args.print_only or (args.dataset is not None and args.seed is not None)
    
    base_dt = datetime(2024, 1, 1, 0, 0, 0)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for solver_name in solvers_to_run:
        print(f"\n{'='*30} RUNNING SOLVER: {solver_name.upper()} {'='*30}")
        NUM_RUNS = 2
        instance_results = {}
        
        if not print_only:
            output_dir = os.path.join(r"..\output\solomon", run_timestamp, solver_name)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"solomon_results_avg.csv")
            out_target = output_file
        else:
            output_dir = None
            out_target = os.devnull
            
        with open(out_target, 'w', newline='', encoding='utf-8') as csvf:
            writer = csv.writer(csvf)
            writer.writerow(["Dataset", "Nodes", "Capacity", "Best NV", "Best L", "Mean NV", "Min L", "Max L", "Med L", "Mean L", "Std. dev.", "Mean Cost", "Mean Time (s)"])
            
            for filename in files:
                base_name = filename.split('.')[0].lower()
                if not print_only:
                    dataset_csv_dir = os.path.join(output_dir, "datasets")
                    route_csv_dir = os.path.join(output_dir, "routes")
                    os.makedirs(dataset_csv_dir, exist_ok=True)
                    os.makedirs(route_csv_dir, exist_ok=True)
                    dataset_csv_path = os.path.join(dataset_csv_dir, f"{base_name}_results.csv")
                    dataset_csv_file = open(dataset_csv_path, 'w', newline='', encoding='utf-8')
                    dataset_csv_writer = csv.writer(dataset_csv_file)
                    dataset_csv_writer.writerow(["Dataset", "Nodes", "Capacity", "Run", "Seed", "Cost", "Num Routes", "Distance", "Time (s)"])
                else:
                    dataset_csv_file = None
                    dataset_csv_writer = None

                filepath = os.path.join(solomon_dir, filename)
                capacity, nodes = parse_solomon_file(filepath)
                instance_results[filename] = []

                if not nodes:
                    print(f"Skipping {filename}: no nodes found.")
                    continue
                    
                distance_matrix, time_matrix = build_matrices(nodes)
                
                depot = next((n for n in nodes if n['cust_no'] == 0), None)
                time_limit = depot['due_date'] if depot else 1000000.0
                
                remaining_stores = []
                for node in nodes:
                    if node['cust_no'] == 0:
                        continue 
                        
                    earliest = base_dt + timedelta(seconds=node['ready_time'])
                    latest = base_dt + timedelta(seconds=node['due_date'])
                    
                    remaining_stores.append({
                        'store_id': str(node['cust_no']),
                        'store_name': f"Store_{node['cust_no']}",
                        'route_code': str(node['cust_no']),
                        'volume': node['demand'],
                        'dwell_time': int(node['service_time']),
                        'earliest_time': earliest.isoformat(timespec='seconds'),
                        'latest_time': latest.isoformat(timespec='seconds'),
                        'sched_time': earliest.isoformat(timespec='seconds'), 
                        'pred_time': earliest.isoformat(timespec='seconds'),
                        'dist_group': 0, 
                        'region': 0, 
                    })
                    
                base_name = filename.split('.')[0].lower()
                target_cost = None
                if base_name in BEST_KNOWN_SOLUTIONS:
                    best_nv, best_dist = BEST_KNOWN_SOLUTIONS[base_name]
                    target_cost = (best_nv, best_dist)
                    
                print(f"--- Running {filename} ({solver_name.upper()}, Nodes: {len(remaining_stores)}, Capacity: {capacity}) ---")
                if target_cost is not None:
                    print(f"--- Best Known: NV={best_nv}, Dist={best_dist:.2f} ---")
                
                run_costs = []
                run_distances = []
                run_routes = []
                run_times = []
                successful_runs = 0
                
                overall_best_cost = (float('inf'), float('inf'))
                overall_best_sol = None
                overall_best_dist = 0
                overall_best_routes = 0
                
                runs_to_execute = [args.seed] if args.seed is not None else range(NUM_RUNS)
                
                if solver_name == "aco":
                    import concurrent.futures
                    max_workers = max(1, os.cpu_count() - 2) if os.cpu_count() else 4
                    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                        future_to_run_idx = {}
                        for run_idx in runs_to_execute:
                            args_tuple = (run_idx, remaining_stores, distance_matrix, time_matrix, capacity, time_limit, target_cost)
                            fut = executor.submit(run_single_aco_seed, args_tuple)
                            future_to_run_idx[fut] = run_idx
                            
                        for future in concurrent.futures.as_completed(future_to_run_idx):
                            run_idx = future_to_run_idx[future]
                            seed_val = run_idx
                            try:
                                run_idx_res, seed_val_res, best_cost_scalar, distance, num_routes, duration, best_cost, best_solution = future.result()
                                
                                print(f"  Run {run_idx+1}/{NUM_RUNS} (Seed {seed_val}): Cost: {best_cost_scalar:.2f}, Distance: {distance:.2f}, Routes: {num_routes}, Time: {duration:.2f}s")
                                
                                row_data = [filename, len(remaining_stores), capacity, run_idx+1, seed_val, round(best_cost_scalar, 2), num_routes, round(distance, 2), round(duration, 2)]
                                instance_results[filename].append(row_data)
                                
                                if dataset_csv_writer is not None:
                                    dataset_csv_writer.writerow(row_data)
                                    dataset_csv_file.flush()
                                
                                if best_cost < overall_best_cost:
                                    overall_best_cost = best_cost
                                    overall_best_sol = best_solution
                                    overall_best_dist = distance
                                    overall_best_routes = num_routes
                                    
                                run_costs.append(best_cost_scalar)
                                run_distances.append(distance)
                                run_routes.append(num_routes)
                                run_times.append(duration)
                                successful_runs += 1
                                
                            except Exception as e:
                                print(f"  Run {run_idx+1}/{NUM_RUNS} ERROR: {e}")
                                row_data = [filename, len(remaining_stores), capacity, run_idx+1, seed_val, "ERROR", "ERROR", "ERROR", "ERROR"]
                                instance_results[filename].append(row_data)
                                if dataset_csv_writer is not None:
                                    dataset_csv_writer.writerow(row_data)
                                    dataset_csv_file.flush()
                else:
                    for run_idx in runs_to_execute:
                        seed_val = run_idx
                        random.seed(seed_val)
                        np.random.seed(seed_val)
                        set_numba_seed(seed_val)
                        
                        support = SupportLinePlanningGA(
                            remaining_stores=remaining_stores,
                            distance_matrix=distance_matrix,
                            time_matrix=time_matrix,
                            population_size=1000,
                            generations=200, 
                            early_stop_patience=200,
                            support_capacity=capacity,
                            vehicle_cost=2000,
                            time_limit_per_route=time_limit,
                            is_solomon=True,
                            # target_cost=target_cost
                        )
                        
                        try:
                            start_time = time.time()
                            best_cost, best_solution = support.run()
                            duration = time.time() - start_time
                            num_routes = len(best_solution) if best_solution else 0
                            
                            if isinstance(best_cost, tuple):
                                distance = best_cost[1]
                                best_cost_scalar = num_routes * 2000 + distance
                            else:
                                distance = best_cost - num_routes * 2000
                                best_cost_scalar = best_cost
                                
                            print(f"  Run {run_idx+1}/{NUM_RUNS} (Seed {seed_val}): Cost: {best_cost_scalar:.2f}, Distance: {distance:.2f}, Routes: {num_routes}, Time: {duration:.2f}s")
                            
                            row_data = [filename, len(remaining_stores), capacity, run_idx+1, seed_val, round(best_cost_scalar, 2), num_routes, round(distance, 2), round(duration, 2)]
                            instance_results[filename].append(row_data)
                            
                            if dataset_csv_writer is not None:
                                dataset_csv_writer.writerow(row_data)
                                dataset_csv_file.flush()
                            
                            if best_cost < overall_best_cost:
                                overall_best_cost = best_cost
                                overall_best_sol = best_solution
                                overall_best_dist = distance
                                overall_best_routes = num_routes
                                
                            run_costs.append(best_cost_scalar)
                            run_distances.append(distance)
                            run_routes.append(num_routes)
                            run_times.append(duration)
                            successful_runs += 1
                            
                        except Exception as e:
                            import traceback
                            traceback.print_exc()
                            print(f"  Run {run_idx+1}/{NUM_RUNS} ERROR: {e}")
                            row_data = [filename, len(remaining_stores), capacity, run_idx+1, seed_val, "ERROR", "ERROR", "ERROR", "ERROR"]
                            instance_results[filename].append(row_data)
                            if dataset_csv_writer is not None:
                                dataset_csv_writer.writerow(row_data)
                                dataset_csv_file.flush()

                if successful_runs > 0:
                    min_l = np.min(run_distances)
                    max_l = np.max(run_distances)
                    med_l = np.median(run_distances)
                    mean_l = np.mean(run_distances)
                    std_l = np.std(run_distances, ddof=1) if len(run_distances) > 1 else 0.0
                    mean_nv = np.mean(run_routes)
                    mean_cost = np.mean(run_costs)
                    mean_time = np.mean(run_times)
                    best_idx = np.argmin(run_costs)
                    best_nv = run_routes[best_idx]
                    best_dist = run_distances[best_idx]
                    
                    print(f"=> [{solver_name.upper()}] {filename} Mean NV: {mean_nv:.1f}, Min L: {min_l:.2f}, Max L: {max_l:.2f}, Med L: {med_l:.2f}, Mean L: {mean_l:.2f}, Std L: {std_l:.2f}, Mean Time: {mean_time:.2f}s\n")
                    writer.writerow([filename, len(remaining_stores), capacity, f"{best_nv:.1f}", f"{best_dist:.2f}", f"{mean_nv:.1f}", f"{min_l:.2f}", f"{max_l:.2f}", f"{med_l:.2f}", f"{mean_l:.2f}", f"{std_l:.2f}", f"{mean_cost:.2f}", f"{mean_time:.2f}"])
                else:
                    writer.writerow([filename, len(remaining_stores), capacity, f"{best_nv:.1f}", f"{best_dist:.2f}", "ERROR", "ERROR", "ERROR", "ERROR", "ERROR", "ERROR", "ERROR", "ERROR"])
                csvf.flush()
                
                if dataset_csv_file is not None:
                    dataset_csv_file.close()

                if overall_best_sol and not print_only:
                    plot_out_dir = os.path.join(output_dir, "plots")
                    export_routes(filename, overall_best_sol, route_csv_dir)
                    plot_solution(filename, overall_best_sol, nodes, overall_best_dist, overall_best_routes, plot_out_dir)

        print(f"All datasets processed for solver {solver_name.upper()}.")

if __name__ == "__main__":
    main()
