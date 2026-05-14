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
    "ex_pop": 20,
    "ex_cx":  0.7,
    "ex_mut": 0.1,
    "al_pop": 50,
    "al_cx":  0.8,
    "al_mut": 0.05,
    "rho":    0.1,
    "alpha":  1,
    "beta":   7,
}

# Fixed output directory
OUTPUT_DIR = ROOT / "output" / "exp"

# Metric columns to report (returned by main())
METRIC_COLS = [
    "vehicle_num",
    "total_store_num",
    "total_dist(km)",
    "total_time(hr)",
    "avg_dist(km)",
    "avg_time(hr)",
    "avg_load_rate",
    "on_time_rate",
]


# Column order for Raw Results sheet
RAW_COLS = [
    "seed",
    *METRIC_COLS,
    "elapsed",
    "status",
    "error",
]


# ---------------------------------------------------------------------------
# Run a single seed
# ---------------------------------------------------------------------------
def run_single_seed(data_name: str, seed: int, test_mode: bool = False, google: bool = False) -> dict:
    """Run main() with the given seed and return a result dict."""
    random.seed(seed)
    np.random.seed(seed)

    result = {
        "seed":    seed,
        "elapsed": None,
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
        )
        elapsed = time.time() - t0

        result["elapsed"] = round(elapsed, 2)
        for col in METRIC_COLS:
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
        print(f"  Time  ->  mean={ok_df['elapsed'].astype(float).mean():.2f}s")

        # Build per-metric summary rows: metric | min | max | mean | median | std
        for col in METRIC_COLS:
            col_data = ok_df[col].astype(float)
            n = len(ok_df)
            summary_rows.append({
                "metric": col,
                "min":    round(col_data.min(),                            4),
                "max":    round(col_data.max(),                            4),
                "mean":   round(col_data.mean(),                           4),
                "median": round(col_data.median(),                         4),
                "std":    round(col_data.std(ddof=1) if n > 1 else 0.0,   4),
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
def run(data_name: str, seed_list: list, test_mode: bool = False, google: bool = False):
    """Run multiple seeds and save results incrementally after each seed."""
    print(f"\n{'='*60}")
    print(f"  Dataset   : {data_name}")
    print(f"  Seeds     : {seed_list}")
    print(f"  Test mode : {test_mode}")
    print(f"  Google    : {google}")
    print(f"  Params    : {BEST_PARAMS}")
    print(f"  Output    : {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    total = len(seed_list)

    for i, seed in enumerate(seed_list, 1):
        print(f"[{i}/{total}] Running seed={seed} ...")
        result = run_single_seed(data_name, seed, test_mode=test_mode, google=google)

        if result["status"] == "ok":
            print(f"  [OK]  seed={seed}: total_dist={result['total_dist(km)']:.4f}km, time={result['elapsed']:.2f}s")
        else:
            print(f"  [ERR] seed={seed}: {result['error']}")

        # Save immediately after each seed (merge with existing file)
        summarize_and_save([result], data_name, OUTPUT_DIR)


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
    )