import os
import json
import math
import pandas as pd
from api.osrm import OSRM

class ODataManager:
    """
    Notes:
        Load route-related data from source files.
    """
    _DC_CENTER = "DC"
    _ROUTE_DATA_SHEET = 0
    _DWELL_TIME_SHEET = 1
    _STORE_COORD_SHEET = 2

    def __init__(self, excel_files):
        osrm = OSRM()
        self.dc = {'store_id': 'dc', 'latitude': 25.083282, 'longitude': 121.40712}
        self.excel_files = excel_files
        self.stores_info = self._load_store_coordinates()
        self.dwell_info = self._load_store_dwell_time()
        self.store_ids = self._load_store_id()
        self.avg_dwell_time = self._calculate_average_dwell_time()
        self.routes_info = self._load_original_routes()
        self.distance_matrix, self.time_matrix = osrm._compute_cost_matrices(self.routes_info)
        self._classify_route_dist_group_and_region()
        self._classify_store_dist_group_and_region()    


    def _region_classification(self, lat, lng):
        """
        Note:
            Classify the region of a store to DC based on coordinates.
        
        Args:
            lat (float): Latitude of the store.
            lng (float): Longitude of the store.

        Returns:
            str: Store's region.
        """
        dc_lat, dc_lng = self.dc['latitude'], self.dc['longitude']
        dx = lng - dc_lng
        dy = lat - dc_lat

        theta = math.degrees(math.atan2(dy, dx)) % 360

        if 315 <= theta or theta < 45:
            region = 'east'
        elif 45 <= theta < 135:
            region = 'north'
        elif 135 <= theta < 225:
            region = 'west'
        else:
            region = 'south'

        return region


    def _distance_classification(self, store_id):
        """
        Note:
            Classify the distance group of a store to DC.
        
        Args:
            store_id (str): Store ID.

        Returns:
            str: Store's distance group.
        """
        dist = self.distance_matrix[self.dc['store_id']][store_id]
        if dist <= 3:
            group = 'near'
        elif 3 < dist <= 5:
            group ='mid'
        else:
            group ='far'

        return group


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
        stores_df = pd.read_excel(self.excel_files[0], sheet_name=self._STORE_COORD_SHEET, skiprows=0)

        for _, row in stores_df.iterrows():
            store_id = row['店鋪編號']
            store_name = row['店鋪名稱']
            if not pd.isna(store_id):
                store_id = str(int(store_id))
                store_ids[store_name] = store_id

        return store_ids


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


    def _load_original_routes(self):
        """
        Notes:
            Load routes data from excel & transfer to dict.

        Args:
            file (str): Path to the Excel File.

        Returns:
            dict: routes information, with route code as key.
        """
        routes_info = {}
        routes_df = pd.read_excel(self.excel_files[0], sheet_name=self._ROUTE_DATA_SHEET, skiprows=3)
        i = 0
        for _, row in routes_df.iterrows():
            route_code = str(row['車次'])
            store_name = row['店名']
            store_id = self._get_store_id_by_name(store_name)
            lng, lat = self._get_coordinates(store_id)
            dwell_time = self._get_dwell_time(store_id)
            sched_time = pd.to_datetime(row['表定時間']).isoformat()
            earliest_time = (pd.to_datetime(row['表定時間']) - pd.Timedelta(minutes=60)).isoformat()
            latest_time = (pd.to_datetime(row['表定時間']) + pd.Timedelta(minutes=30)).isoformat()
            pred_time = pd.to_datetime(row['預定時間']).isoformat()
            volume = row['貨量']
            load_rate = row['裝載率']

            main_route_code = route_code[:2]
            max_capacity = self._get_max_capacity_by_route_code(main_route_code)

            if main_route_code not in routes_info:
                routes_info[main_route_code] = {"dc" : None, "stores": []}

            if self._DC_CENTER not in route_code and len(route_code) == 2:
                routes_info[main_route_code]["dc"] = {
                    "route_id": main_route_code,
                    "route_code": route_code,
                    "store_id": "DC",
                    "store_name": store_name,
                    "total_volume": volume,
                    "load_rate": load_rate,
                    "max_capacity": max_capacity,
                    "region": "",
                    "distance": 0,
                    "duration": 0
                }
            elif self._DC_CENTER not in route_code:
                if dwell_time == 0:
                    i += 1
                    print(f"Route Code: {route_code}, store ID: {store_id}, store Name: {store_name}")
                routes_info[main_route_code]["stores"].append({
                    "route_id": main_route_code,
                    "route_code": route_code,
                    "store_id": store_id,
                    "store_name": store_name,
                    "longitude": lng, 
                    "latitude": lat,
                    "region": "",
                    "dist_group": "",
                    "sched_time": sched_time,
                    "earliest_time": earliest_time,
                    "latest_time": latest_time,
                    "pred_time": pred_time,
                    "dwell_time": dwell_time,
                    "volume": volume
                })

        return routes_info


    def _classify_route_dist_group_and_region(self):
        """
        Notes:
            Classify distance group and region for each route in routes_info.
        
        Args:
            None.

        Returns:
            None.
        """
        for route in self.routes_info.values():
            dc = route['dc']
            stores = route['stores']
            store_num = len(stores)
            route_lat = sum(store['latitude'] for store in stores) / store_num
            route_lng = sum(store['longitude'] for store in stores) / store_num
            route_region = self._region_classification(route_lat, route_lng)
            dc['region'] = route_region


    def _classify_store_dist_group_and_region(self):
        """
        Notes:
            Classify distance group and region for each store in routes_info.
        
        Args:
            None.

        Returns:
            None.
        """
        routes = self.routes_info.values()
        for route in routes:
            for store in route['stores']:
                store_group = self._distance_classification(store['store_id'])
                store_region = self._region_classification(store['latitude'], store['longitude'])
                store['dist_group'] = store_group
                store['region'] = store_region


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