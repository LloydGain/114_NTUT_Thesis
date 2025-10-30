import os
import json
import numpy as np
import pandas as pd
import haversine as hs

class DataManager:
    """
    Notes:
        Load route-related data from source files.
    
    Args:
        excel_files (list): Source files (Route information, Dwell information).
    
    Returns:
        None.
    """
    _DC_CENTER = "DC"
    _ROUTE_DATA_SHEET = 0
    _DWELL_TIME_SHEET = 1
    _STORE_COORD_SHEET = 2

    def __init__(self, excel_files):
        self.dc = {'store_id': 'dc', 'latitude': 25.083282, 'longitude': 121.40712}
        self.avg_speed_kmh = 40
        self.excel_files = excel_files
        self.stores_info = self._load_store_coordinates()
        self.dwell_info = self._load_store_dwell_time()
        self.avg_dwell_time = self._calculate_average_dwell_time()
        self.routes_info = self._load_original_routes()
        self.distance_matrix, self.time_matrix = self._compute_cost_matrices()        


    def _compute_cost_matrices(self):
        """
        Notes:
            Compute distance and time matrices.

        Args:
            None.
        
        Returns:
            tuple: (distance_matrix, time_matrix)
        """
        routes = self.routes_info.values()
        stores_id = [self.dc['store_id']] + [store['store_id'] for route in routes for store in route['stores']]
        coordinates = np.array([(self.dc['latitude'], self.dc['longitude'])] + [(store['latitude'], store['longitude']) for route in routes for store in route['stores']] )
        dist = hs.haversine_vector(coordinates, coordinates, comb=True, unit=hs.Unit.KILOMETERS)
        dist = dist * 1.5
        
        dist_matrix = {
            store_id: {
                store_id_j: dist[i][j] for j, store_id_j in enumerate(stores_id)
            } for i, store_id in enumerate(stores_id)
        }

        time_matrix = {
            store_id: {
                store_id_j: (dist[i][j] / self.avg_speed_kmh) * 60 for j, store_id_j in enumerate(stores_id)
            } for i, store_id in enumerate(stores_id)
        }

        return dist_matrix, time_matrix


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
            store_id = str(int(row['ID'])) if not pd.isna(row['ID']) else None
            store_name = row['店名']
            longitude, latitude = self._get_coordinates(store_id)
            dwell_time = self._get_dwell_time(store_id)
            sched_time = pd.to_datetime(row['表定時間']).isoformat()
            earliest_time = (pd.to_datetime(row['表定時間']) - pd.Timedelta(minutes=30)).isoformat()
            latest_time = (pd.to_datetime(row['表定時間']) + pd.Timedelta(hours=1)).isoformat()
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
                    "longitude": longitude, 
                    "latitude": latitude,
                    "sched_time": sched_time,
                    "earliest_time": earliest_time,
                    "latest_time": latest_time,
                    "pred_time": pred_time,
                    "dwell_time": dwell_time,
                    "volume": volume
                })

        return routes_info


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