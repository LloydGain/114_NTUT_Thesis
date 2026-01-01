import os
from data.manual_data import MDataManager

def main():
    """
    Notes:
        Export Invalid Routes.
    """
    error_file = '../data/error_routes.xlsx'
    route_network_file = '../data/route_network_and_dwell_times.xlsx'
    store_info_file = '../data/store_info.xlsx'
    file_dates = [
        '1203', '1205', '1207', '1208', '1209', '1210',
        '1212', '1213', '1214', '1215', '1216', '1217',
        '1219', '1220', '1221', '1222', '1223'
    ]

    if os.path.exists(error_file):
        os.remove(error_file)

    for file_date in file_dates:
        print(f'Processing {file_date}...')
        manual_data_file = f'../data/{file_date}/{file_date}manual.xlsx'
        m_data = MDataManager([manual_data_file, route_network_file, store_info_file])
        m_data.export_invalid_routes(error_file, file_date)


if __name__ == "__main__":
    main()
