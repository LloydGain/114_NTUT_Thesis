# 115_NTUT_Thesis: Logistics Route Optimization

This project implements a hybrid metaheuristic approach combining **Genetic Algorithms (GA)**, **Ant Colony Optimization (ACO)**, and **Local Search** mechanisms to optimize logistics delivery routes. The system is designed to handle complex constraints such as vehicle capacity, time windows, and support line allocation.

## Features

- **Store Extraction (GA)**: Extract the stores from routes.
- **Store Allocation (ACO)**: Allocate stores to main routes.
- **Support Line Planning (ACO)**: Optimized routing for support vehicles.
- **Local Search**: Iterative improvement of routes using 2-opt ,relocation and swap operators.
- **Route Evaluation**: Comparison between Original, Manual, and Optimized strategies.
- **Visualization**: Automatic generation of interactive HTML maps and PNG route visualizations (OSRM).

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
   Create a `.env` file in the `src` directory if you plan to use Google Maps API or set a default seed.

   ```env
   GOOGLE_API_KEY=your_google_maps_api_key
   RANDOM_SEED=your_seed
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

| Argument        | Type    | Required | Description                                                             |
| --------------- | ------- | -------- | ----------------------------------------------------------------------- |
| `--file_date` | `str` | Yes      | The date for the input data files (e.g.,`1203`, `1205`). |
| `--seed`      | `int` | No       | Set a specific random seed for reproducibility. Overrides `.env`.     |
| `--test`      | Flag    | No       | Run in**test mode** with reduced iterations for quick debugging.  |
| `--google`    | Flag    | No       | Use**Google Maps API** for final distance/duration updates.       |

### Examples

**Run a standard optimization for data '1203':**

```bash
python main.py --file_date 1203
```

**Run a quick test with a fixed seed:**

```bash
python main.py --file_date 1203 --seed 123 --test
```

## Project Structure

```text
src/
├── data/           # Data loading and processing (Base, Manual, Origin, Program)
├── eval/           # Evaluation logic and visualization (Folium, Matplotlib)
├── models/         # Core business logic (RouteManager)
├── services/       # External API integrations (OSRM, Google Maps)
├── solvers/        # Optimization algorithms (BaseACO, AllocateACO, SupportACO, GA, LocalSearch)
├── utils/          # Utilities (Logger, EarlyStopper)
└── main.py         # Application entry point
```

## Code Quality

This project adheres to strict code quality standards:

- **Pylint**: Automated linting checks via GitHub Actions.

## License

NTUT Thesis Project.
