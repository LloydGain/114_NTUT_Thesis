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

## Installation

1. **Clone the repository**

   ```bash
   git clone <repository_url>
   cd 115_NTUT_Thesis
   ```
2. **Install Dependencies**
   It is recommended to use a virtual environment.

   ```bash
   pip install -r requirements.txt
   ```
3. **Environment Configuration**
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

## OSRM Setup

This project relies on a local OSRM instance for calculating routing time and distance matrices and generating HTML maps. It is recommended to use **Docker**.

1. **Download Map Data**
   Download the OpenStreetMap data for Taiwan:

   ```powershell
   # Windows PowerShell
   Invoke-WebRequest -Uri http://download.geofabrik.de/asia/taiwan-latest.osm.pbf -OutFile taiwan-latest.osm.pbf
   ```
2. **Process Map Data**
   Run the following commands to extract, partition, and customize the map data:

   ```powershell
   # Windows PowerShell
   docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-extract -p /opt/car.lua /data/taiwan-latest.osm.pbf
   docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-partition /data/taiwan-latest.osrm
   docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-customize /data/taiwan-latest.osrm
   ```
3. **Run OSRM Server**
   Start the OSRM server on port 5000:

   ```powershell
   docker run -t -i -p 5000:5000 -v "${PWD}:/data" osrm/osrm-backend osrm-routed --algorithm mld /data/taiwan-latest.osrm
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
| `--file_date`    | `str`  | Yes      | The date for the input data files (e.g.,`1203`, `1205`).                      |
| `--seed`         | `int`  | No       | Set a specific random seed for reproducibility. Overrides`.env`.                |
| `--test`         | Flag     | No       | Run in**test mode** with reduced iterations for quick debugging.            |
| `--google`       | Flag     | No       | Use**Google Maps API** for final distance/duration updates.                 |
| `--comment`      | `str`  | No       | Custom comment or identifier for the current run output folder.                   |
| `--skip_compare` | Flag     | No       | Skip comparison against Manual and Program routes to save time.                   |
| `--ignore_pkl`   | Flag     | No       | Ignore cached`.pkl` results and force full recalculation.                       |
| `--alb`          | `list` | No       | Run ablation study by disabling stages (e.g.,`--alb extract allocate support`). |

### Examples

**Run a standard optimization for data '1203':**

```bash
python main.py --file_date 1203
```

**Run a quick test with a fixed seed and custom comment:**

```bash
python main.py --file_date 1203 --seed 123 --test --comment "baseline_test"
```

**Run an ablation study skipping cache and comparisons:**

```bash
python main.py --file_date 1203 --ignore_pkl --skip_compare --alb extract
```

## Tools & Scripts

The repository includes extensive utilities to automate running experiments and analyzing statistical results.

### `tools/` Directory

- **`run_experiments.py` / `run_solomon.py` / `run_single_stage_ga.py`**: Scripts to execute batch runs across multiple dates and seeds.
- **`calculate_statistical_tests.py` / `baseline_wilcoxon.py`**: Automates generating comprehensive statistical Excel reports (e.g., Wilcoxon Signed-Rank tests) to compare algorithm performance.
- **`convergence_analysis.py` / `plot_ablation.py`**: Reads `.pkl` history and aggregates iteration-level metrics to generate convergence charts.
- **`generate_osrm_plots.py`**: Automates map snapshot generation for route visualization.

### `scripts/` Directory

Contains convenient `.bat` files for Windows batch execution:

- `run_experiments.bat`: Standard bulk dataset execution.
- `run_single_stage_ga.bat`: Run experiments focused on the new GA pipeline.
- `run_test.bat` / `run_hpo.bat`: Quick testing and Hyperparameter Optimization flows.

## Project Structure

```text
115_NTUT_Thesis/
├── scripts/        # Batch files for automated execution (.bat)
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
