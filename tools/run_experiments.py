import argparse
import time
import random
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from setup import *
from main import main


# ---------------------------------------------------------------------------
# Best hyperparameters determined from HPO across all datasets
# ---------------------------------------------------------------------------
BEST_PARAMS = {
    "ex_iter": 200,
    "ex_pop": 20,
    "ex_cx": 0.8,
    "ex_mut": 0.1,
    "al_iters": 200,
    "al_ants": 50,
    "al_alpha": 1,
    "al_beta": 1,
    "al_rho": 0.7,
    "time_limit": 100,
    "ants": 50,
    "beta": 7,
    "rho": 0.5,
}

# Fixed output directory
OUTPUT_DIR = ROOT / "output" / "exp"

# Metric columns to report (returned by main())
METRIC_COLS = [
    "vehicle_num",
    "total_dist(km)",
    "total_time(hr)",
    "avg_load_rate",
    "on_time_rate",
    "running_time(s)",
]

# Column order for Raw Results sheet
RAW_COLS = [
    "seed",
    *METRIC_COLS,
    "status",
    "error",
]


# ---------------------------------------------------------------------------
# Run a single seed
# ---------------------------------------------------------------------------
def run_single_seed(data_name: str, seed: int, test_mode: bool = False, google: bool = False, alb: list = None, skip_compare: bool = True) -> dict:
    """Run main() with the given seed and return a result dict."""
    if alb is None: alb = []
    random.seed(seed)
    np.random.seed(seed)

    result = {
        "seed":    seed,
        "status":  "ok",
        "error":   "",
        **{col: None for col in METRIC_COLS},
    }

    try:
        t0      = time.time()
        metrics = main(
            file_date=data_name,
            random_seed=seed,
            hyper_params=BEST_PARAMS,
            test_mode=test_mode,
            google=google,
            skip_compare=skip_compare,
            alb=alb,
        )
        elapsed = time.time() - t0

        for col in METRIC_COLS:
            if col == "running_time(s)":
                result[col] = round(elapsed, 2)
            else:
                result[col] = metrics.get(col)

    except Exception as e:
        result["status"] = "error"
        result["error"]  = str(e)
        traceback.print_exc()

    return result


# ---------------------------------------------------------------------------
# Aggregate statistics and save to Excel
# ---------------------------------------------------------------------------
def summarize_and_save(new_results: list, data_name: str, output_dir: Path):
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
        # Print a quick console summary for the key metric
        dist_data = ok_df["total_dist(km)"].astype(float)
        print(f"  total_dist(km) ->  mean={dist_data.mean():.4f}, std={dist_data.std(ddof=1) if len(ok_df)>1 else 0:.4f}, "
              f"min={dist_data.min():.4f}, max={dist_data.max():.4f}")
        print(f"  Time  ->  mean={ok_df['running_time(s)'].astype(float).mean():.2f}s")

        # Build per-metric summary rows: metric | min | max | mean | median | std
        for col in METRIC_COLS:
            col_data = ok_df[col].astype(float)
            n = len(ok_df)
            summary_rows.append({
                "metric": col,
                "min":    round(col_data.min(),                            2),
                "max":    round(col_data.max(),                            2),
                "mean":   round(col_data.mean(),                           2),
                "median": round(col_data.median(),                         2),
                "std":    round(col_data.std(ddof=1) if n > 1 else 0.0,   2),
            })

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        # Sheet 1: raw result per seed (all seeds, sorted, fixed column order)
        df.to_excel(writer, sheet_name="Raw Results", index=False)

        # Sheet 2: hyperparameters used
        pd.DataFrame([BEST_PARAMS]).to_excel(writer, sheet_name="Hyper Params", index=False)

        # Sheet 3: summary statistics per metric (min / max / mean / median / std)
        if summary_rows:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)

        # Sheet 4: failed runs (if any)
        if not fail_df.empty:
            fail_df.to_excel(writer, sheet_name="Errors", index=False)

    print(f"\n[INFO] Results saved to: {excel_path}\n")
    return excel_path


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------
def run(data_name: str, seed_list: list, test_mode: bool = False, google: bool = False, alb: list = None, skip_compare: bool = True):
    """Run multiple seeds and save results incrementally after each seed."""
    if alb is None: alb = []
    
    print(f"\n{'='*60}")
    print(f"  Dataset   : {data_name}")
    print(f"  Seeds     : {seed_list}")
    print(f"  Test mode : {test_mode}")
    print(f"  Google    : {google}")
    print(f"  Ablation  : {alb}")
    print(f"  Params    : {BEST_PARAMS}")
    print(f"  Output    : {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    current_out_dir = OUTPUT_DIR
    if alb:
        current_out_dir = OUTPUT_DIR / "alb" / '_'.join(alb)
    
    current_out_dir.mkdir(parents=True, exist_ok=True)
    
    excel_path = current_out_dir / f"{data_name}.xlsx"
    completed_seeds = set()
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
        result = run_single_seed(data_name, seed, test_mode=test_mode, google=google, alb=alb, skip_compare=skip_compare)

        if result["status"] == "ok":
            print(f"  [OK]  seed={seed}: total_dist={result['total_dist(km)']:.4f}km, time={result['running_time(s)']:.2f}s")
        else:
            print(f"  [ERR] seed={seed}: {result['error']}")

        # Save immediately after each seed (merge with existing file)
        summarize_and_save([result], data_name, current_out_dir)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multiple seeds with best hyperparameters and report statistics."
    )
    parser.add_argument("--file_date", type=str, required=True,
                        help="Dataset date string, e.g. 20241205")

    # Seed options (mutually exclusive)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seeds", type=int, default=30,
                             help="Number of seeds to run (default: 30), starting from 0")
    seed_group.add_argument("--seed_list", type=int, nargs="+",
                             help="Explicit list of seeds, e.g. --seed_list 0 1 2 3 4")

    # Test mode: runs with reduced parameters to verify the pipeline quickly
    parser.add_argument("--test", action="store_true",
                        help="Run in test mode (reduced GA generations / ACO iterations)")

    # Google Maps: update route distances/durations via Google Maps API
    parser.add_argument("--google", action="store_true",
                        help="Update routes via Google Maps API after optimization")

    # Ablation options
    parser.add_argument("--alb", type=str, nargs='+', choices=['extract', 'allocate', 'support'], default=[], 
                        help="Ablation options: extract, allocate, support")

    parser.add_argument("--skip_compare", action="store_true",
                        help="Skip comparison with manual routes")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Ensure output directory exists (create if missing)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine seed list
    if args.seed_list is not None:
        seed_list = args.seed_list
    else:
        seed_list = list(range(args.seeds))

    # In test mode, only run 2 seeds to keep it quick
    if args.test:
        seed_list = seed_list[:2]
        print(f"[TEST MODE] Running only seeds: {seed_list}")

    run(
        data_name=args.file_date,
        seed_list=seed_list,
        test_mode=args.test,
        google=args.google,
        alb=args.alb,
        skip_compare=args.skip_compare,
    )