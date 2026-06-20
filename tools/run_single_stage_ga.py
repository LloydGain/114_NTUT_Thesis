import argparse
import time
import random
import traceback
import copy
from pathlib import Path
import numpy as np
import pandas as pd

# Setup sys.path to find src/
from setup import *
from config import config
from data.store_data import StoreData
from data.origin_data import ODataManager
from models.route_manager import RouteManager
from solvers.single_stage_mainline_ga import SingleStageMainLineGA
from services.osrm import OSRM

# Metric columns matching run_experiments.py
METRIC_COLS = [
    "vehicle_num",
    "support_num",
    "total_dist(km)",
    "main_dist(km)",
    "sup_dist(km)",
    "total_time(hr)",
    "avg_load_rate",
    "on_time_rate",
    "running_time(s)",
    "fitness",
    "fitness_dist",
    "fitness_veh",
    "fitness_tw",
    "fitness_cap",
    "fitness_cross",
]

# Column order for Raw Results sheet
RAW_COLS = [
    "seed",
    *METRIC_COLS,
    "status",
    "error",
]

# Fixed output directory for single stage GA
OUTPUT_DIR = ROOT / "output" / "exp" / "alb" / "single_stage_ga"


def run_single_seed(file_date: str, seed: int, test_mode: bool, capacity: float, 
                    pop_size: int, generations: int, early_stop_patience: int, 
                    vehicle_cost: float, tw_penalty: float, cap_penalty: float,
                    cross_penalty: float, google: bool) -> dict:
    """Run a single seed of Single-Stage GA optimization."""
    random.seed(seed)
    np.random.seed(seed)

    result = {
        "seed": seed,
        "status": "ok",
        "error": "",
        **{col: None for col in METRIC_COLS},
    }

    try:
        t0 = time.time()
        
        # Load OSRM matrices
        dist_file = str(ROOT / "data/osrm/store_distance_matrix.json")
        time_file = str(ROOT / "data/osrm/store_time_matrix.json")
        store_info_file = str(ROOT / "data/store_info.xlsx")
        
        s_data = StoreData(store_info_file)
        distance_matrix, time_matrix = s_data.load_matrices_from_file(dist_file, time_file)

        # Load daily routes using ODataManager
        route_file = str(ROOT / f"data/{file_date}/{file_date}route.xlsx")
        route_network_file = str(ROOT / "data/route_network_and_dwell_times.xlsx")
        
        o_data = ODataManager([route_file, route_network_file, store_info_file], distance_matrix, time_matrix)
        
        # Extract all unique stores from all routes
        unique_stores = []
        seen_store_ids = set()
        for route_id, route_info in o_data.routes_info.items():
            for store in route_info['stores']:
                if store['store_id'] not in seen_store_ids:
                    seen_store_ids.add(store['store_id'])
                    unique_stores.append(store)
                    
        print(f"[Seed {seed}] Loaded {len(unique_stores)} unique stores to route.")
        
        # Configure test parameters if in test mode
        actual_pop = 10 if test_mode else pop_size
        actual_gens = 2 if test_mode else generations
        actual_patience = 1 if test_mode else early_stop_patience
        
        # Run SingleStageMainLineGA as the Single-Stage VRPTW Solver with main line concept
        solver = SingleStageMainLineGA(
            main_routes=o_data.routes_info,
            remaining_stores=unique_stores,
            distance_matrix=distance_matrix,
            time_matrix=time_matrix,
            population_size=actual_pop,
            generations=actual_gens,
            early_stop_patience=actual_patience,
            support_capacity=capacity,
            vehicle_cost=vehicle_cost,
            tw_penalty_weight=tw_penalty,
            cap_penalty_weight=cap_penalty,
            cross_penalty_weight=cross_penalty
        )
        
        best_cost, best_breakdown, best_solution, main_log = solver.run()
        total_fitness = best_cost
        total_breakdown = best_breakdown.copy()
        elapsed = time.time() - t0
        
        # Update solution info using RouteManager
        temp_rm = RouteManager(copy.deepcopy(best_solution), distance_matrix, time_matrix)
        
        # Always check and extract violating stores
        print("Checking for internal time window and capacity violations...")
        newly_extracted = temp_rm.extract_violating_stores(target='all', extract_original=True, check_capacity=True)
        
        if google:
            print("Validating all routes with Google Maps API...")
            new_ext_api = temp_rm.validate_and_extract_violating_stores(target='all', extract_original=True)
            newly_extracted.extend(new_ext_api)
            
        final_support_routes = {}
        for r_id, r_info in list(temp_rm.routes_info.items()):
            if str(r_id).isdigit():
                if r_info['stores']:
                    new_id = str(101 + len(final_support_routes))
                    r_info['dc']['route_id'] = new_id
                    r_info['dc']['route_code'] = new_id
                    for s in r_info['stores']:
                        s['route_id'] = new_id
                    final_support_routes[new_id] = r_info
                del temp_rm.routes_info[r_id]
                
        current_support_stores = newly_extracted
        
        while current_support_stores:
            print(f"Extracted {len(current_support_stores)} stores due to violations. Replanning with SingleStageMainLineGA...")
            support = SingleStageMainLineGA(
                main_routes={},
                remaining_stores=current_support_stores, 
                distance_matrix=distance_matrix, 
                time_matrix=time_matrix, 
                population_size=actual_pop,
                generations=actual_gens,
                early_stop_patience=actual_patience,
                support_capacity=capacity,
                vehicle_cost=vehicle_cost,
                tw_penalty_weight=tw_penalty,
                cap_penalty_weight=cap_penalty,
                cross_penalty_weight=cross_penalty
            )
            sup_cost, sup_breakdown, current_routes, sup_log = support.run()
            total_fitness += sup_cost
            for k in total_breakdown:
                total_breakdown[k] += sup_breakdown[k]
            
            sup_rm = RouteManager(copy.deepcopy(current_routes), distance_matrix, time_matrix)
            extracted_again = sup_rm.extract_violating_stores(target='support', extract_original=True, check_capacity=True)
            
            if google:
                print("Validating replanned support routes with Google Maps API...")
                extracted_api = sup_rm.validate_and_extract_violating_stores(target='support')
                extracted_again.extend(extracted_api)
            
            for _, r_info in sup_rm.routes_info.items():
                if r_info['stores']: 
                    new_id = str(101 + len(final_support_routes))
                    r_info['dc']['route_id'] = new_id
                    r_info['dc']['route_code'] = new_id
                    for s in r_info['stores']:
                        s['route_id'] = new_id
                    final_support_routes[new_id] = r_info
                    
            if extracted_again:
                current_support_stores = extracted_again
            else:
                break
                
        routes = {**temp_rm.routes_info, **final_support_routes}
        if not google:
            temp_rm.routes_info = routes
            temp_rm.update_all_routes_info()
            routes = temp_rm.routes_info
        
        temp_rm.routes_info = copy.deepcopy(routes)
        # Export detailed routes and maps
        from datetime import datetime
        from eval.display_routes import DisplayRoutes
        
        dt_folder = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
        out_base = ROOT / "output" / file_date / "alb" / "single_stage_ga" / dt_folder
        out_base.mkdir(parents=True, exist_ok=True)
        
        optimized_routes_file = str(out_base / "optimized_routes_info.json")
        optimized_routes_excel_file = str(out_base / "optimized_routes_info.xlsx")
        optimized_routes_dir = out_base / "optimized_routes"
        optimized_routes_dir.mkdir(parents=True, exist_ok=True)
        optimized_routes_osrm_html = str(optimized_routes_dir / "osrm_routes.html")
        optimized_routes_html = str(optimized_routes_dir / "routes.html")
        
        temp_rm.export_routes_info(optimized_routes_file)
        temp_rm.export_excel_file(optimized_routes_excel_file)
        
        opt_routes = DisplayRoutes(optimized_routes_file)
        opt_routes.plot_routes_html_in_osrm(optimized_routes_osrm_html)
        opt_routes.plot_routes_html(optimized_routes_html)
        
        ga_log_file = str(out_base / "ga_convergence_log.csv")
        pd.DataFrame(main_log).to_csv(ga_log_file, index=False)
        
        # Calculate metrics
        total_vehicles = len(routes)
        total_distance = sum(r['dc']['distance'] for r in routes.values())
        total_duration = sum(r['dc']['duration'] for r in routes.values()) / 3600
        total_stores = sum(len(r['stores']) for r in routes.values())
        avg_load_rate = (
            sum(r['dc']['load_rate'] for r in routes.values()) / total_vehicles
            if total_vehicles else 0
        )
        on_time = sum(
            1
            for r in routes.values()
            for s in r['stores']
            if s['earliest_time'] <= s['pred_time'] <= s['latest_time']
        )
        on_time_rate = on_time / total_stores if total_stores else 0
        
        main_routes_keys = [k for k in routes.keys() if not str(k).isdigit()]
        support_routes_keys = [k for k in routes.keys() if str(k).isdigit()]
        
        main_vehicles = len(main_routes_keys)
        support_vehicles = len(support_routes_keys)
        
        main_distance = sum(routes[k]['dc']['distance'] for k in main_routes_keys)
        support_distance = sum(routes[k]['dc']['distance'] for k in support_routes_keys)
        
        result["vehicle_num"] = total_vehicles
        result["support_num"] = support_vehicles
        result["total_dist(km)"] = round(total_distance, 4)
        result["main_dist(km)"] = round(main_distance, 4)
        result["sup_dist(km)"] = round(support_distance, 4)
        result["total_time(hr)"] = round(total_duration, 4)
        result["avg_load_rate"] = round(avg_load_rate, 4)
        result["on_time_rate"] = round(on_time_rate, 4)
        result["running_time(s)"] = round(elapsed, 2)
        result["fitness"] = round(total_fitness, 4)
        result["fitness_dist"] = round(total_breakdown['dist'], 4)
        result["fitness_veh"] = round(total_breakdown['veh'], 4)
        result["fitness_tw"] = round(total_breakdown['tw'], 4)
        result["fitness_cap"] = round(total_breakdown['cap'], 4)
        result["fitness_cross"] = round(total_breakdown['cross'], 4)
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        traceback.print_exc()
        
    return result


def summarize_and_save(new_results: list, data_name: str, output_dir: Path, params_used: dict):
    """Merge new results with any existing Excel data, then write the report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    excel_path = output_dir / f"{data_name}.xlsx"

    # Load previous raw results (if file already exists)
    if excel_path.exists():
        try:
            prev_df = pd.read_excel(excel_path, sheet_name="Raw Results")
            prev_records = prev_df.to_dict("records")
            print(f"[INFO] Loaded {len(prev_records)} existing row(s) from {excel_path}")
        except Exception:
            prev_records = []
    else:
        prev_records = []

    # Merge: index by seed; new results overwrite old ones with the same seed
    merged: dict[int, dict] = {r["seed"]: r for r in prev_records if "seed" in r}
    for r in new_results:
        merged[r["seed"]] = r

    # Reconstruct sorted list with fixed column order
    all_results = [merged[k] for k in sorted(merged)]
    df = pd.DataFrame(all_results).reindex(columns=RAW_COLS)

    # Split successful and failed runs
    ok_df   = df[df["status"] == "ok"].copy()
    fail_df = df[df["status"] != "ok"].copy()

    print(f"\n{'='*60}")
    print(f"  Dataset        : {data_name}")
    print(f"  Seeds in file  : {sorted(merged)}")
    print(f"  Success        : {len(ok_df)} / {len(df)}")
    print(f"{'='*60}")

    summary_rows = []
    if not ok_df.empty:
        # Print a quick console summary
        dist_data = ok_df["total_dist(km)"].astype(float)
        print(f"  total_dist(km) ->  mean={dist_data.mean():.4f}, std={dist_data.std(ddof=1) if len(ok_df)>1 else 0:.4f}, "
              f"min={dist_data.min():.4f}, max={dist_data.max():.4f}")
        print(f"  Time  ->  mean={ok_df['running_time(s)'].astype(float).mean():.2f}s")

        # Build summary statistics
        for col in METRIC_COLS:
            col_data = ok_df[col].astype(float)
            n = len(ok_df)
            summary_rows.append({
                "metric": col,
                "min":    round(col_data.min(),                          2),
                "max":    round(col_data.max(),                          2),
                "mean":   round(col_data.mean(),                         2),
                "median": round(col_data.median(),                       2),
                "std":    round(col_data.std(ddof=1) if n > 1 else 0.0,   2),
            })

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        # Sheet 1: raw result per seed
        df.to_excel(writer, sheet_name="Raw Results", index=False)

        # Sheet 2: hyperparameters used
        pd.DataFrame([params_used]).to_excel(writer, sheet_name="Hyper Params", index=False)

        # Sheet 3: summary statistics per metric
        if summary_rows:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)

        # Sheet 4: failed runs (if any)
        if not fail_df.empty:
            fail_df.to_excel(writer, sheet_name="Errors", index=False)

    print(f"\n[INFO] Results saved to: {excel_path}\n")
    return excel_path


def run(data_name: str, seed_list: list, test_mode: bool, force: bool, capacity: float,
        pop_size: int, generations: int, early_stop_patience: int, vehicle_cost: float,
        tw_penalty: float, cap_penalty: float, cross_penalty: float, google: bool):
    """Run multiple seeds and save results incrementally after each seed."""
    
    # Verify OSRM is running
    try:
        OSRM().check_osrm()
    except Exception as e:
        print(f"[ERROR] OSRM service check failed: {e}")
        return

    params_used = {
        "capacity": capacity,
        "pop_size": 10 if test_mode else pop_size,
        "generations": 2 if test_mode else generations,
        "early_stop_patience": 1 if test_mode else early_stop_patience,
        "vehicle_cost": vehicle_cost,
        "tw_penalty": tw_penalty,
        "cap_penalty": cap_penalty,
        "cross_penalty": cross_penalty,
        "test_mode": test_mode,
    }

    print(f"\n{'='*60}")
    print(f"  Dataset   : {data_name}")
    print(f"  Seeds     : {seed_list}")
    print(f"  Test mode : {test_mode}")
    print(f"  Params    : {{\"capacity\": {capacity}, \"pop_size\": {pop_size}, \"generations\": {generations}, \"early_stop_patience\": {early_stop_patience}, \"vehicle_cost\": {vehicle_cost}, \"tw_penalty\": {tw_penalty}, \"cap_penalty\": {cap_penalty}, \"cross_penalty\": {cross_penalty}, \"test_mode\": {test_mode}}}")
    print(f"  Output    : {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    excel_path = OUTPUT_DIR / f"{data_name}.xlsx"
    completed_seeds = set()
    if not force:
        if excel_path.exists():
            try:
                prev_df = pd.read_excel(excel_path, sheet_name="Raw Results")
                completed_seeds = set(prev_df[prev_df["status"] == "ok"]["seed"].tolist())
                print(f"  [INFO] Found {len(completed_seeds)} completed seed(s) in {excel_path.name}")
            except Exception:
                pass

        seed_list = [s for s in seed_list if s not in completed_seeds]
        if not seed_list:
            print(f"[INFO] All requested seeds are already completed for {data_name}. Skipping.")
            return excel_path

    total = len(seed_list)

    for i, seed in enumerate(seed_list, 1):
        print(f"[{i}/{total}] Running seed={seed} ...")
        result = run_single_seed(
            file_date=data_name,
            seed=seed,
            test_mode=test_mode,
            capacity=capacity,
            pop_size=pop_size,
            generations=generations,
            early_stop_patience=early_stop_patience,
            vehicle_cost=vehicle_cost,
            tw_penalty=tw_penalty,
            cap_penalty=cap_penalty,
            cross_penalty=cross_penalty,
            google=google
        )

        if result["status"] == "ok":
            print(f"  [OK]  seed={seed}: vehicle_num={result['vehicle_num']}, total_dist={result['total_dist(km)']}km, time={result['running_time(s)']}s")
        else:
            print(f"  [ERR] seed={seed}: {result['error']}")

        # Save immediately after each seed (merge with existing file)
        summarize_and_save([result], data_name, OUTPUT_DIR, params_used)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multiple seeds with Single-Stage GA and report statistics."
    )
    parser.add_argument("--file_date", type=str, required=True,
                        help="Dataset date string, e.g. 20221203")

    # Seed options
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seeds", type=int, default=30,
                             help="Number of seeds to run (default: 30), starting from 0")
    seed_group.add_argument("--seed_list", type=int, nargs="+",
                             help="Explicit list of seeds, e.g. --seed_list 0 1 2 3 4")

    # Test mode
    parser.add_argument("--test", action="store_true",
                        help="Run in test mode (reduced GA generations / populations)")
    parser.add_argument("--force", action="store_true",
                        help="Force run all seeds even if they are already completed")

    # GA hyperparams
    parser.add_argument("--capacity", type=float, default=7.2,
                        help="Vehicle capacity constraint to use for the single stage GA (default: 7.2)")
    parser.add_argument("--pop_size", type=int, default=500,
                        help="GA population size (default: 500)")
    parser.add_argument("--generations", type=int, default=5000,
                        help="GA generations (default: 1000)")
    parser.add_argument("--early_stop", type=int, default=100,
                        help="GA early stop patience (default: 50)")
    parser.add_argument("--vehicle_cost", type=float, default=2000.0,
                        help="Vehicle fixed cost penalty (default: 2000.0)")
    parser.add_argument("--tw_penalty", type=float, default=10000000.0,
                        help="Time window violation penalty weight (default: 1000.0)")
    parser.add_argument("--cap_penalty", type=float, default=10000000.0,
                        help="Capacity violation penalty weight for main lines (default: 1000000.0)")
    parser.add_argument("--cross_penalty", type=float, default=100.0,
                        help="Cross-route penalty weight to protect original routes (default: 100.0)")
    parser.add_argument("--use_vnd", action="store_true",
                        help="Enable VND local search (very slow for large datasets)")
    parser.add_argument("--google", action="store_true",
                        help="Update and validate routes via Google Maps API after optimization")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    import os
    if not args.use_vnd:
        os.environ["DISABLE_VND"] = "1"
        print("[INFO] VND local search is disabled via DISABLE_VND=1 environment variable.")

    # Determine seed list
    if args.seed_list is not None:
        seed_list = args.seed_list
    else:
        seed_list = list(range(args.seeds))

    # In test mode, only run 2 seeds
    if args.test:
        seed_list = seed_list[:2]
        print(f"[TEST MODE] Running only seeds: {seed_list}")

    run(
        data_name=args.file_date,
        seed_list=seed_list,
        test_mode=args.test,
        force=args.force,
        capacity=args.capacity,
        pop_size=args.pop_size,
        generations=args.generations,
        early_stop_patience=args.early_stop,
        vehicle_cost=args.vehicle_cost,
        tw_penalty=args.tw_penalty,
        cap_penalty=args.cap_penalty,
        cross_penalty=args.cross_penalty,
        google=args.google
    )
