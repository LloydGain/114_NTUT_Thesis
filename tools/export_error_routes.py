import os
from setup import *
from data.manual_data import MDataManager

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


def main():
    """
    Notes:
        Export Invalid Routes.
    """
    base_dir = '../data'
    output_dir = '../output'
    error_file = f'{output_dir}/error_routes.xlsx'
    route_network_file = f'{base_dir}/route_network_and_dwell_times.xlsx'
    store_info_file = f'{base_dir}/store_info.xlsx'
    file_dates = get_file_dates(base_dir)

    if os.path.exists(error_file):
        os.remove(error_file)

    for file_date in file_dates:
        print(f'Processing {file_date}...')
        manual_data_file = f'{base_dir}/{file_date}/{file_date}manual.xlsx'
        m_data = MDataManager([manual_data_file, route_network_file, store_info_file])
        m_data.export_invalid_routes(error_file, file_date)

    print('Done!')


if __name__ == "__main__":
    main()