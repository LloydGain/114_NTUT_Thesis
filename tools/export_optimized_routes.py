import os
from setup import *
from datetime import datetime
from data.optimized_data import OPDataManager


def get_file_dates(base_dir):
    """
    Notes:
        Get file dates.
    
    Args:
        base_dir (str): Base directory.
    
    Returns:
        list: List of file dates.
    """
    file_dates = []
    for file_date in os.listdir(base_dir):
        if not file_date.isdigit():
            continue
        if os.path.isdir(f'{base_dir}/{file_date}'):
            file_dates.append(file_date)

    return file_dates


def get_latest_result_folder(base_dir, file_date):
    """
    Notes:
        Get latest result folder under file_date directory.

    Args:
        file_date (str): File date.

    Returns:
        str: Latest result folder name.
    """
    result_base_dir = os.path.join(base_dir, file_date)

    candidates = []

    for name in os.listdir(result_base_dir):
        path = os.path.join(result_base_dir, name)
        if not os.path.isdir(path):
            continue

        try:
            dt = datetime.strptime(name, "%Y-%m-%d_%H-%M-%S")
            candidates.append((dt, name))
        except ValueError:
            continue

    if not candidates:
        raise ValueError(f"No valid result folder found in {result_base_dir}")

    return max(candidates)[1]


def main():
    """
    Notes:
        Export Optimized Routes.
    """
    output_dir = '../output'
    optimized_route_file = f'{output_dir}/optimized_routes.xlsx'
    file_dates = get_file_dates(output_dir)

    if os.path.exists(optimized_route_file):
        os.remove(optimized_route_file)

    for file_date in file_dates:
        print(f'Processing {file_date}...')
        latest_folder = get_latest_result_folder(output_dir, file_date)
        optimized_data_file = f'{output_dir}/{file_date}/{latest_folder}/optimized_routes_info.json'
        o_data = OPDataManager(optimized_data_file)
        o_data.export_origin_excel_file(optimized_route_file, file_date)

    print('Done!')


if __name__ == "__main__":
    main()