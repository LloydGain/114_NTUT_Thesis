import pandas as pd
from data.base_data import BaseDataManager

class PDataManager(BaseDataManager):
    """
    Notes:
        Load route-related data from source files (manual).
    """
    def __init__(self, excel_files, distance_matrix, time_matrix):
        super().__init__(excel_files, distance_matrix, time_matrix)
        self.routes_df = pd.read_excel(self.excel_files[0], sheet_name=self._ROUTE_DATA_SHEET, dtype={'路線編號': str})
        self.routes_info = self._load_program_routes()
        self._update_routes_info()


    def _load_program_routes(self):
        """
        Notes:
            Load program routes from an Excel file into dict.

        Args:
            None.

        Returns:
            dict: routes info with route ID as key.
        """
        routes_info = {}

        current_main_route_id = None
        max_route_id_counter = 100
        is_main_line_mode = False

        for _, row in self.routes_df.iterrows():
            route_col_value = str(row['路線編號']).strip() if not pd.isna(row['路線編號']) else None
            store_name = row['店鋪名']

            if "主線" in str(store_name) or "主線" in str(route_col_value):
                is_main_line_mode = True
                continue

            if not route_col_value and not is_main_line_mode:
                continue

            if route_col_value == '爆量線':
                continue

            if not is_main_line_mode and pd.isna(store_name) and route_col_value:
                current_main_route_id = route_col_value
                if route_col_value.isdigit():
                    max_route_id_counter = max(max_route_id_counter, int(route_col_value))

                if current_main_route_id not in routes_info:
                    routes_info[current_main_route_id] = {
                        "dc": {
                            "route_id": current_main_route_id,
                            "store_id": "DC",
                            "store_name": "Distribution Center",
                            "load_rate": 0, "distance": 0, "duration": 0
                        },
                        "stores": []
                    }

                if not pd.isna(row['裝載率']):
                    routes_info[current_main_route_id]["dc"]["load_rate"] = row['裝載率']
                if not pd.isna(row['距離(公里)']):
                    routes_info[current_main_route_id]["dc"]["distance"] = row['距離(公里)']
                if not pd.isna(row['時間']):
                    routes_info[current_main_route_id]["dc"]["duration"] = row['時間']

                continue

            if is_main_line_mode:
                has_load_stats = not pd.isna(row['裝載率'])

                if has_load_stats:
                    max_route_id_counter += 1
                    current_main_route_id = str(max_route_id_counter)

                    routes_info[current_main_route_id] = {
                        "dc": {
                            "route_id": current_main_route_id,
                            "store_id": "DC",
                            "store_name": "Distribution Center",
                            "load_rate": 0, "distance": 0, "duration": 0
                        },
                        "stores": []
                    }

                    routes_info[current_main_route_id]["dc"]["load_rate"] = row['裝載率']
                    if not pd.isna(row['距離(公里)']):
                        routes_info[current_main_route_id]["dc"]["distance"] = row['距離(公里)']
                    if not pd.isna(row['時間']):
                        routes_info[current_main_route_id]["dc"]["duration"] = row['時間']

            if current_main_route_id:
                if pd.isna(store_name):
                    continue

                sub_route_code = route_col_value
                store_id = self._get_store_id_by_name(store_name)

                lng, lat = self._get_coordinates(store_id)
                dwell_time = self._get_dwell_time(store_id)

                sched_time = row['表定到店時間']
                sched_time = pd.to_datetime(sched_time).isoformat() if not pd.isna(sched_time) else None

                pred_time = row['預定到店']
                pred_time = pd.to_datetime(pred_time).isoformat() if not pd.isna(pred_time) else None

                volume = row['貨量'] if not pd.isna(row['貨量']) else 0

                routes_info[current_main_route_id]["stores"].append({
                    "route_id": current_main_route_id,
                    "route_code": sub_route_code,
                    "store_id": store_id,
                    "store_name": store_name,
                    "longitude": lng,
                    "latitude": lat,
                    "sched_time": sched_time,
                    "pred_time": pred_time,
                    "dwell_time": dwell_time,
                    "volume": volume
                })
            else:
                if not pd.isna(store_name):
                    print(f"Warning: Found store {store_name} without a main route header.")

        return routes_info
