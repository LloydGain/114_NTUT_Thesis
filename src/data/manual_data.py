import os
import json
import copy
import shutil
import pandas as pd
from route.route import RouteManager

class MDataManager:
    """
    Notes:
        Load route-related data from source files (manual).
    """
    _DC_CENTER = "DC"
    _ROUTE_DATA_SHEET = 0
    _DWELL_TIME_SHEET = 1
    _STORE_ID_SHEET = 1
    _STORE_COORD_SHEET = 0             

    def __init__(self, excel_files, distance_matrix, time_matrix):
        self.dc = {'store_id': 'dc', 'latitude': 25.083282, 'longitude': 121.40712}
        self.excel_files = excel_files
        self.routes_df = pd.read_excel(self.excel_files[0], sheet_name=self._ROUTE_DATA_SHEET, skiprows=3)
        self.dwells_df = pd.read_excel(self.excel_files[1], sheet_name=self._DWELL_TIME_SHEET, skiprows=0)
        self.stores_df = pd.read_excel(self.excel_files[2], sheet_name=self._STORE_COORD_SHEET, skiprows=0)
        self.stores_info = self._load_store_coordinates()
        self.dwell_info = self._load_store_dwell_time()
        self.store_ids = self._load_store_id()
        self.avg_dwell_time = self._calculate_average_dwell_time()
        self.routes_info = self._load_manual_routes()
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


    def _get_store_id_by_name(self, store_name):
        """
        Notes:
            Get the store ID based on the store name.

        Args:
            store_name (str): Store Name.

        Returns:
            str: Store ID.
        """
        return self.store_ids.get(store_name, None)


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
        
        for _, row in self.dwells_df.iterrows():
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
        
        for _, row in self.stores_df.iterrows():
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
        store_ids = {}
        

        for _, row in self.stores_df.iterrows():
            store_id = row['店鋪編號']
            store_name = row['店鋪名稱']
            if not pd.isna(store_id):
                store_id = str(int(store_id))
                store_ids[store_name] = store_id

        return store_ids


    def _load_manual_routes(self):
        """
        Notes:
            Load routes data from excel & transfer to dict.

        Args:
            file (str): Path to the Excel File.

        Returns:
            dict: routes information, with route code as key.
        """
        routes_info = {}
        
        
        update_main_route = True
        main_route_code = None
        for _, row in self.routes_df.iterrows():
            route_code = str(row['車次'])

            if not route_code: continue

            store_name = row['店名']
            store_id = self._get_store_id_by_name(store_name)
            lng, lat = self._get_coordinates(store_id)
            dwell_time = self._get_dwell_time(store_id)
            sched_time = pd.to_datetime(row['表定時間']).isoformat()
            earliest_time = (pd.to_datetime(row['表定時間']) - pd.Timedelta(minutes=1)).isoformat()
            latest_time = (pd.to_datetime(row['表定時間']) + pd.Timedelta(hours=1)).isoformat()
            pred_time = pd.to_datetime(row['預定時間']).isoformat()
            volume = row['貨量']
            load_rate = row['裝載率']

            if main_route_code is None:
                main_route_code = route_code[:3] if route_code.isdigit() else route_code[:2]

            if update_main_route:
                main_route_code = route_code[:3] if route_code.isdigit() else route_code[:2]
                update_main_route = False

            if self._DC_CENTER in route_code:
                update_main_route = True

            max_capacity = self._get_max_capacity_by_route_code(main_route_code)

            if main_route_code not in routes_info:
                routes_info[main_route_code] = {"dc" : None, "stores": []}

            if self._DC_CENTER not in route_code and len(route_code) < 4:
                routes_info[main_route_code]["dc"] = {
                    "route_id": main_route_code,
                    "route_code": route_code,
                    "store_id": "DC",
                    "store_name": store_name,
                    "total_volume": volume,
                    "load_rate": load_rate,
                    "max_capacity": max_capacity,
                    "distance": 0,
                    "duration": 0
                }
            elif self._DC_CENTER not in route_code:
                routes_info[main_route_code]["stores"].append({
                    "route_id": main_route_code,
                    "route_code": route_code,
                    "store_id": store_id,
                    "store_name": store_name,
                    "longitude": lng, 
                    "latitude": lat,
                    "sched_time": sched_time,
                    "earliest_time": earliest_time,
                    "latest_time": latest_time,
                    "pred_time": pred_time,
                    "dwell_time": dwell_time,
                    "volume": volume
                })

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
        route_manager.update_all_routes_info()


    def create_data_folder(self, dst_dir, source_manual_file, dest_manual_file):
        """
        Notes:
            Create the data foler.
        
        Args:
            dst_dir (str): folder. 
            source_manual_file (str): Source Manual file.
            dest_manual_file (str): Destination Manual file.
        
        Returns:
            None.
        """
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(source_manual_file, dest_manual_file)


    def _get_origin_routes(self):
        """
        Note: 
            Transfer the manual route to the original route.
        
        Args:
            None.
        
        Returns:
            data (list): data row with routes info.
        """
        data = []
        routes = copy.deepcopy(self.routes_info)
        route_manager = RouteManager(routes, self.distance_matrix, self.time_matrix)
        route_manager.move_store_to_original_route()
        routes = route_manager.routes_info

        for route_id in routes:
            dc = routes[route_id]['dc']
            stores = routes[route_id]['stores']
            data.append({
                "不可大車": "",
                "車次": dc["route_code"],
                "ID": "",
                "店名": '林口DC',
                "表定時間": "",
                "預定時間": "",
                "貨量": dc["total_volume"],
                "裝載率": dc["load_rate"]
            })
            for s in stores:
                data.append({
                    "不可大車": False,
                    "車次": s["route_code"],
                    "ID": s["store_id"],
                    "店名": s["store_name"],
                    "表定時間": s["sched_time"],
                    "預定時間": s["sched_time"],
                    "貨量": s["volume"],
                    "裝載率": ""
                })
            data.append({
                "不可大車": False,
                "車次": f'{dc["route_code"]}DC',
                "ID": "",
                "店名": '林口DC',
                "表定時間": "",
                "預定時間": "",
                "貨量": 0,
                "裝載率": ""
            })

        return data


    def export_origin_excel_file(self, dest_file):
        """
        Notes:
            Export origin excel.

        Args:
            dest_file (str): original file.
        
        Returns:
            None.
        """
        data = self._get_origin_routes()
        columns = [
            "不可大車",
            "車次",
            "ID",
            "店名",
            "表定時間",
            "預定時間",
            "貨量",
            "裝載率"
        ]
        df = pd.DataFrame(data, columns=columns)
        with pd.ExcelWriter(dest_file, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Sheet1", index=False, startrow=3)


    def save_routes_to_json(self, json_file):
        """
        Notes:
            Save route data to a JSON file.

        Args:
            json_file: Path to the Json file.

        Returns:
            None.
        """
        output_dir = os.path.dirname(json_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.routes_info, f, ensure_ascii=False, indent=4)