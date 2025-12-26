import os
import json
import pandas as pd
from models.route_manager import RouteManager

class BaseDataManager:
    """
    Notes:
        Base class for loading route-related data from source files.
    """
    _DC_CENTER = "DC"
    _ROUTE_DATA_SHEET = 0
    _DWELL_TIME_SHEET = 1
    _STORE_ID_SHEET = 1
    _STORE_COORD_SHEET = 0

    def __init__(self, excel_files, distance_matrix, time_matrix):
        self.dc = {'store_id': 'dc', 'latitude': 25.083282, 'longitude': 121.40712}
        self.excel_files = excel_files
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix

        self.routes_df = pd.read_excel(self.excel_files[0], sheet_name=self._ROUTE_DATA_SHEET, skiprows=3)
        self.dwells_df = pd.read_excel(self.excel_files[1], sheet_name=self._DWELL_TIME_SHEET, skiprows=0)
        self.stores_df = pd.read_excel(self.excel_files[2], sheet_name=self._STORE_COORD_SHEET, skiprows=0)

        self.stores_info = self._load_store_coordinates()
        self.dwell_info = self._load_store_dwell_time()
        self.store_ids = self._load_store_id()
        self.avg_dwell_time = self._calculate_average_dwell_time()

        self.routes_info = {}

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
        if '2U' in route_code:
            return 10
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
            None

        Returns:
            int: Average dwell time.
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
            int: Average dwell time.
        """
        return self.dwell_info.get(store_id, self.avg_dwell_time)


    def _load_store_dwell_time(self):
        """
        Notes:
            Load dwell time information for each store from an sheet.

        Args:
            None

        Returns:
            dict: Dwell time information.
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
            None

        Returns:
            dict: Store coordinates.
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
            None

        Returns:
            dict: Store IDs.
        """
        store_ids = {}
        for _, row in self.stores_df.iterrows():
            store_id = row['店鋪編號']
            store_name = row['店鋪名稱']
            if not pd.isna(store_id):
                store_id = str(int(store_id))
                store_ids[store_name] = store_id

        return store_ids


    def _update_routes_info(self):
        """
        Notes:
            Update route information.

        Args:
            None

        Returns:
            None
        """
        route_manager = RouteManager(self.routes_info, self.distance_matrix, self.time_matrix)
        route_manager.update_all_routes_info()


    def save_routes_to_json(self, json_file):
        """
        Notes:
            Save route data to a JSON file.

        Args:
            json_file (str): Path to the JSON file.

        Returns:
            None
        """
        class PandasJSONEncoder(json.JSONEncoder):
            """
            Notes:
                Custom JSON encoder for handling pandas data types.
            """
            def default(self, o):
                if pd.isna(o):
                    return None
                if isinstance(o, (pd.Timestamp, pd.Timedelta)):
                    return o.isoformat()
                return super().default(o)

        output_dir = os.path.dirname(json_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.routes_info, f, ensure_ascii=False, indent=4, cls=PandasJSONEncoder)
