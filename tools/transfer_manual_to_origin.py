import argparse
from setup import *
from data.store_data import StoreData
from data.manual_data import MDataManager

def parse_args():
    """
    Notes:
        Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_date", type=str, required=True)
    return parser.parse_args()


def main(file_date):
    """
    Notes:
        Transfer manual data to origin data.
    """
    data_dir = f'../data/{file_date}'
    source_manual_file = f'../data/manual_data/{file_date}manual.xlsx'
    origin_file = f'../data/{file_date}/{file_date}route.xlsx'
    dest_manual_file = f'../data/{file_date}/{file_date}manual.xlsx'
    dist_file = '../data/osrm/store_distance_matrix.json'
    time_file = '../data/osrm/store_time_matrix.json'
    route_network_file = '../data/route_network_and_dwell_times.xlsx'
    store_info_file = '../data/store_info.xlsx'

    s_data = StoreData(store_info_file)
    distance_matrix, time_matrix = s_data.load_matrices_from_file(dist_file, time_file)

    m_data = MDataManager([source_manual_file, route_network_file, store_info_file], distance_matrix, time_matrix)
    m_data.create_data_folder(data_dir, source_manual_file, dest_manual_file)
    m_data.export_origin_excel_file(origin_file)


if __name__ == "__main__":
    args = parse_args()
    main(args.file_date)
