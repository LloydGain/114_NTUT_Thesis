import os
import json
import pandas as pd
from route.route import RouteManager

class PDataManager:
    """
    Notes:
        Load route-related data from source files (manual).
    """
    _DC_CENTER = "DC"
    _ROUTE_DATA_SHEET = 0
    _DWELL_TIME_SHEET = 1
    _STORE_ID_SHEET = 1
    _STORE_COORD_SHEET = 2

    def __init__(self, excel_files, distance_matrix, time_matrix):
        self.dc = {'store_id': 'dc', 'latitude': 25.083282, 'longitude': 121.40712}
        self.excel_files = excel_files
        self.stores_info = self._load_store_coordinates()
        self.dwell_info = self._load_store_dwell_time()
        self.stores_id = self._load_store_id()
        self.avg_dwell_time = self._calculate_average_dwell_time()
        self.routes_info = self._load_program_routes()
        self.distance_matrix, self.time_matrix = distance_matrix, time_matrix
        self._update_routes_info()


    def _get_max_capacity_by_route_code(self, route_code):
        """
        Notes:
            Get the maximum vehicle capacity based on the route code.

        Args:
            route_code (str): Route code. (e.g., '2N', '2S', '2U', etc.)

        Returns:
            float: Maximum capacity for the vehicle.
        """
        if '2N' in route_code or '2S' in route_code:
            return 14.4
        elif '2U' in route_code:
            return 10
        else:
            return 7.2


    def _get_store_id_by_route_code(self, route_code):
        """
        Notes:
            Get the store ID based on the route code.

        Args:
            route_code (str): Route code.

        Returns:
            str: Store ID.
        """
        return self.stores_id.get(route_code, None)


    def _get_coordinates(self, store_id):
        """
        Notes:
            Extract longitude and latitude from a store information dictionary.

        Args:
            store_id (str): Store ID.

        Returns:
            tuple: (longitude, latitude)
        """
        store_info = self.stores_info.get(store_id)
        if not store_info:
            return self.dc['longitude'], self.dc['latitude']
        return store_info.get('longitude'), store_info.get('latitude')


    def _calculate_average_dwell_time(self):
        """
        Notes:
            Calculate the average dwell time (all stores).

        Args:
            None.

        Returns:
            float: The average dwell time in sec. Returns 0 if no dwell times are available.
        """
        if not self.dwell_info:
            return 0
        total_dwell_time = sum(self.dwell_info.values())
        average_dwell_time = round(total_dwell_time / len(self.dwell_info), 0)
        return average_dwell_time


    def _get_dwell_time(self, store_id):
        """
        Notes:
            Retrieve the average dwell time for the store.

        Args:
            store_id (str): Store ID.

        Returns:
            float: The dwell time in minutes. Returns 0 if the store ID is not found.
        """
        return self.dwell_info.get(store_id, self.avg_dwell_time)


    def _load_store_dwell_time(self):
        """
        Notes:
            Load dwell time information for each store from an sheet.

        Args:
            None.

        Returns:
            dict: A dictionary mapping store IDs to their average dwell times.
        """
        dwell_info = {}
        dwells_df = pd.read_excel(self.excel_files[1], sheet_name=self._DWELL_TIME_SHEET, skiprows=0)
        for _, row in dwells_df.iterrows():
            store_id = str(row['店舖ID'])
            dwell_time = row['平均滯店時間']
            dwell_info[store_id] = dwell_time
        
        return dwell_info


    def _load_store_coordinates(self):
        """
        Notes:
            Load store coordinates from an Excel file into dict.

        Args:
            file (str): Path to the Excel file.

        Returns:
            dict: Dictionary with store IDs as keys and their coordinates as values.
                Example: {'store_01': {'longitude': 121.5, 'latitude': 25.03}, ...}
        """
        stores_info = {}
        stores_df = pd.read_excel(self.excel_files[0], sheet_name=self._STORE_COORD_SHEET, skiprows=0)
        
        for _, row in stores_df.iterrows():
            store_id = str(row['店鋪編號'])
            longitude = row['經度']
            latitude = row['緯度']
            stores_info[store_id] = {'longitude': longitude, 'latitude': latitude}
        
        return stores_info


    def _load_store_id(self):
        """
        Notes:
            Load store IDs from an Excel file into dict.
        
        Args:
            None.
        
        Returns:
            dict: stores ID with route code as key.
        """
        stores_id = {}
        stores_df = pd.read_excel(self.excel_files[0], sheet_name=self._STORE_ID_SHEET, skiprows=0)

        for _, row in stores_df.iterrows():
            route_code = row['車次']
            if len(str(route_code)) > 2:
                store_id = str(int(row['ID']) if not pd.isna(row['ID']) else None)
                stores_id[route_code] = store_id

        return stores_id


    def _load_program_routes(self):
        routes_info = {}
        routes_df = pd.read_excel(self.excel_files[0], sheet_name=self._ROUTE_DATA_SHEET, dtype={'路線編號': str})
        
        current_main_route_id = None
        max_route_id_counter = 100
        is_main_line_mode = False

        for _, row in routes_df.iterrows():
            route_col_value = str(row['路線編號']).strip() if not pd.isna(row['路線編號']) else None
            store_name = row['店鋪名']
            
            # 0. 偵測主線模式
            if "主線" in str(store_name) or "主線" in str(route_col_value):
                is_main_line_mode = True
                continue 

            # 1. 略過無效行 (非主線模式下的空行)
            if not route_col_value and not is_main_line_mode:
                continue
            if route_col_value == '爆量線':
                continue

            # 2. 一般路線邏輯 (有標題行)
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
                
                # 處理標題行上的彙總數據
                if not pd.isna(row['裝載率']):
                    routes_info[current_main_route_id]["dc"]["load_rate"] = row['裝載率']
                if not pd.isna(row['距離(公里)']):
                    routes_info[current_main_route_id]["dc"]["distance"] = row['距離(公里)']
                if not pd.isna(row['時間']):
                    routes_info[current_main_route_id]["dc"]["duration"] = row['時間']
                
                continue

            # 3. 主線邏輯 (自動分車)
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

            # 4. 處理店鋪資料 (加入店鋪前的關鍵檢查)
            if current_main_route_id:
                # [重要修正] 如果這行沒有店鋪名，代表它是空行或無效行，直接跳過，避免產生 null 資料
                if pd.isna(store_name):
                    continue

                sub_route_code = route_col_value 
                store_id = self._get_store_id_by_route_code(sub_route_code)
                
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


    def _update_routes_info(self):
        """
        Notes:
            Update route information.

        Args:
            route (dict): Route information.

        Returns:
            None.
        """
        route_manager = RouteManager(self.routes_info, self.distance_matrix, self.time_matrix)
        route_manager._update_all_routes_info()


    def save_routes_to_json(self, json_file):
        """
        Notes:
            Save route data to a JSON file.

        Args:
            json_file: Path to the Json file.

        Returns:
            None.
        """
        class PandasJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if pd.isna(obj):
                    return None
                if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
                    return obj.isoformat()
                return super().default(obj)

        output_dir = os.path.dirname(json_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.routes_info, f, ensure_ascii=False, indent=4, cls=PandasJSONEncoder)