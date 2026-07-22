# 114_NTUT_Thesis: Logistics Route Optimization

This project implements a hybrid metaheuristic approach combining **Cooperative Coevolutionary Genetic Algorithms (CCGA)**, **Hybrid Ant Colony Optimization (HACO)**, **Multiple Ant Colony System (MACS)**, **Solomon Insertion Heuristics**, and **Variable Neighborhood Descent (VND)** to optimize logistics delivery routes. The system is designed to handle complex constraints such as vehicle capacity, time windows, and support line allocation.

## Features

- **Store Extraction & Allocation**: Utilize CCGA (Cooperative Coevolutionary Genetic Algorithm) and HACO (Hybrid Ant Colony Optimization) for mainline sequence optimization and capacity constraint handling.
- **Multiple Ant Colony System (MACS)**: Advanced pheromone-based route allocation and support line planning.
- **Local Search (VND)**: Iterative improvement of routes using 2-opt, relocation, and swap operators.
- **Ablation Studies**: Built-in support to disable specific optimization stages (`--alb`) to evaluate their impact.
- **Statistical Analysis & Reporting**: Automated tools to aggregate convergence metrics and generate rigorous Excel statistical test reports.
- **Caching Mechanism**: `.pkl` state caching to speed up reruns, with bypass flags (`--ignore_pkl`).
- **Visualization**: Automatic generation of interactive HTML maps (Folium), PNG route visualizations (OSRM), and convergence plots (Matplotlib).

## Prerequisites

- **Python**: 3.12
- **OSRM Backend**: A local OSRM server running on `http://localhost:5000` is required for fast distance matrix calculations.
- **Google Maps API** (Optional): Update routes with real-world traffic data and precise durations.

## Installation & Setup

**Prerequisite:** Please ensure you have **Docker** installed and running on your system, as it is required for the OSRM routing engine.

1. **Clone the repository**

   ```bash
   git clone <repository_url>
   cd 114_NTUT_Thesis
   ```
2. **Environment Configuration**
   Create a `.env` file in the **project root** directory. This is required for configuring the OSRM server, Distribution Center (DC) coordinates, and optional Google Maps API.

   ```env
   # Optional: Google Maps API Key
   GOOGLE_API_KEY=your_google_maps_api_key

   # Optional: Fixed Random Seed
   RANDOM_SEED=42

   # OSRM Server URL (Default: http://localhost:5000)
   OSRM_HOST=http://localhost:5000

   # Distribution Center (DC) Coordinates
   DC_LONGITUDE=121.40712
   DC_LATITUDE=25.083282
   ```
3. **Run Setup Scripts**
   We provide two automated scripts to set up the Python environment and the OSRM backend server. Please run them sequentially:

   ```powershell
   # 1. Install Python dependencies
   scripts\setup\setup_python.bat

   # 2. Download map data and start OSRM server (requires Docker)
   scripts\setup\setup_osrm.bat
   ```

## Usage

The main entry point is `src/main.py`.

### Basic Command

```bash
cd src
python main.py --file_date <DATE_IDENTIFIER>
```

### Arguments

| Argument           | Type     | Required | Description                                                                       |
| ------------------ | -------- | -------- | --------------------------------------------------------------------------------- |
| `--file_date`    | `str`  | Yes      | The date for the input data files (e.g.,`20221203`, `20221205`).              |
| `--seed`         | `int`  | No       | Set a specific random seed for reproducibility. Overrides`.env`.                |
| `--test`         | Flag     | No       | Run in**test mode** with reduced iterations for quick debugging.            |
| `--google`       | Flag     | No       | Use**Google Maps API** for final distance/duration updates.                 |
| `--comment`      | `str`  | No       | Custom comment or identifier for the current run output folder.                   |
| `--skip_compare` | Flag     | No       | Skip comparison against Manual and Program routes to save time.                   |
| `--ignore_pkl`   | Flag     | No       | Ignore cached`.pkl` results and force full recalculation.                       |
| `--alb`          | `list` | No       | Run ablation study by disabling stages (e.g.,`--alb extract allocate support`). |

### Examples

**Run a standard optimization for data '20221203':**

```bash
python main.py --file_date 20221203
```

**Run a quick test with a fixed seed and custom comment:**

```bash
python main.py --file_date 20221203 --seed 123 --test --comment "baseline_test"
```

**Run an ablation study skipping cache and comparisons:**

```bash
python main.py --file_date 20221203 --ignore_pkl --skip_compare --alb extract
```

## Tools & Scripts

The repository includes extensive utilities to automate running experiments and analyzing statistical results.

### `tools/` Directory

- **Experiment Execution**:
  - `run_experiments.py`: Main script to execute batch experiments across multiple dataset dates and random seeds.
  - `run_solomon.py`: Executes the optimization algorithms on standard Solomon benchmark datasets for validation.
  - `run_single_stage_ga.py`: Executes batch experiments specifically focused on evaluating the new single-stage GA pipeline.
  - `run_tune_hpo.py`: Hyperparameter optimization script (e.g., using Optuna) to tune algorithm parameters.
  - `run_manual_only.py`: Executes and evaluates solely the manual (baseline) routing performance without running optimization.
- **Statistical Analysis & Reporting**:
  - `calculate_statistical_tests.py`: Automates the generation of comprehensive statistical Excel reports (e.g., Wilcoxon Signed-Rank tests) to compare algorithm performance.
  - `baseline_wilcoxon.py`: Executes Wilcoxon Signed-Rank tests specifically for baseline performance comparisons.
  - `dataset_analysis.py`: Analyzes dataset characteristics (e.g., volume, time windows) and outputs statistical summaries.
  - `compare_osrm_google.py`: Analyzes and compares routing results (distance and time) calculated by OSRM vs Google Maps API.
  - `format_route_results.py`: Formats the raw route results into a structured, readable layout for final reporting and analysis.
- **Visualization & Plotting**:
  - `convergence_analysis.py`: Reads .pkl history and aggregates iteration-level metrics to generate convergence charts for the optimization algorithms.
  - `plot_ablation.py`: Generates ablation study charts to visualize the impact of disabling specific optimization stages.
  - `generate_osrm_plots.py`: Automates the generation of map snapshots (e.g., HTML/PNG) for route visualizations using the OSRM backend.
- **Data Conversion & Utilities**:
  - `count_all_stores.py`: Helper utility to count and aggregate store quantities across datasets.
  - `export_error_routes.py`: Extracts and exports routes that encountered errors or violations during optimization.
  - `export_optimized_routes.py`: Helper script to format and export the final optimized routes into an Excel or JSON format.
  - `transfer_manual_to_origin.py`: Converts legacy manual route formats into the new origin format used by the system.
  - `transfer_store_info.py`: Utility to convert and migrate store information formats or structures.
  - `update_osrm_matrices.py`: Recalculates and updates the distance and time matrices using the local OSRM server.
  - `setup.py`: Standard Python setup script for package management and installation.

### `scripts/` Directory

Contains convenient `.bat` files for Windows batch execution, organized into subdirectories:

- **`experiments/`**:
  - `run_experiments.bat`: Standard execution script used for running the full hybrid optimization pipeline across multiple historical dataset dates.
  - `run_single_stage_ga.bat`: Executable script for running experiments focused purely on evaluating the new single-stage HGA-SIH
  - `run_hpo.bat`: Automated workflow for conducting Hyperparameter Optimization (e.g., using Optuna) to systematically tune algorithm parameters for better convergence and solution quality.
- **`setup/`**:
  - `setup_python.bat`: Installs all required Python dependencies from `requirements.txt`.
  - `setup_osrm.bat`: Automates OSRM backend setup and server startup using Docker.

## Project Structure

```text
114_NTUT_Thesis/
├── docs/           # Documentation, thesis materials, and statistical result reports
├── output/         # Experiment results, convergence plots, and cached states
├── scripts/        # Batch files for execution, organized by purpose (experiments, setup)
├── solomon-100/    # Standard Solomon benchmark datasets for algorithm validation
├── tools/          # Statistical analysis, plotting, and bulk execution tools
├── src/
│   ├── data/       # Data loading and processing (Base, Manual, Origin, Program)
│   ├── eval/       # Evaluation logic and visualization (Folium, Matplotlib)
│   ├── models/     # Core business logic (RouteManager)
│   ├── services/   # External API integrations (OSRM, Google Maps)
│   ├── solvers/    # Optimization algorithms (GA, MACS, VND, BaseACO)
│   ├── utils/      # Utilities (Logger, EarlyStopper)
│   └── main.py     # Application entry point
└── README.md
```
