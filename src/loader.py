import json
import pandas as pd

class RouteManager:
    """
    
    """
    def __init__(self, excel_file):
        self.dc_center_long = 121.40712
        self.dc_center_lat = 25.083282
        self.excel_file = excel_file
        self.stores_info = self._load_store_coordinates()
        self.routes_info = self._load_original_routes()


    @staticmethod
    def get_max_capacity_by_route_code(route_code):
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


    @staticmethod
    def get_coordinates(store_info):
        """
        Extract longitude and latitude from a store information dictionary.

        Args:
            store_info (dict): A dictionary containing 'longitude' and 'latitude' keys.

        Returns:
            tuple: (longitude, latitude)
        """
        longitude = store_info.get('longitude')
        latitude = store_info.get('latitude')
        return longitude, latitude


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
        store_df = pd.read_excel(self.excel_file, sheet_name=2, skiprows=0)
        
        for _, row in store_df.iterrows():
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
        routes_df = pd.read_excel(self.excel_file, sheet_name=0, skiprows=3)
        
        for _, row in routes_df.iterrows():
            route_code = str(row['車次'])
            store_id = str(int(row['ID'])) if not pd.isna(row['ID']) else None
            store_name = row['店名']
            longitude, latitude = self.get_coordinates(self.stores_info[store_id]) if store_id is not None else (self.dc_center_long, self.dc_center_lat)
            sched_time = pd.to_datetime(row['表定時間']).isoformat()
            pred_time = pd.to_datetime(row['預定時間']).isoformat()
            volume = row['貨量']
            load_rate = row['裝載率']

            main_route_code = route_code[:2]
            max_capacity = self.get_max_capacity_by_route_code(main_route_code)

            if main_route_code not in routes_info:
                routes_info[main_route_code] = {"dc" : None, "shops": []}

            if "DC" not in route_code and len(route_code) == 2:
                routes_info[main_route_code]["dc"] = {
                    "route": main_route_code,
                    "route_code": route_code,
                    "store_id": "DC",
                    "store_name": store_name,
                    "sched_time": None,
                    "pred_time": None,
                    "volume": volume,
                    "load_rate": load_rate,
                    "max_capacity": max_capacity
                }
            routes_info[main_route_code]["shops"].append({
                "route": main_route_code,
                "route_code": route_code,
                "store_id": store_id,
                "store_name": store_name,
                "longitude": longitude, 
                "latitude": latitude,
                "sched_time": sched_time,
                "pred_time": pred_time,
                "volume": volume
            })
        # for route in routes_info:
        #     print(routes_info[route])
        return routes_info


    def save_routes_to_json(self, json_file):
        """
        Notes:
            Load routes data from excel & transfer to dict.

        Args:
            routes_info (dict): Routes information dict.
            json_file: Path to the Json file.

        Returns:
            None.
        """
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.routes_info, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    manager = RouteManager('../data/1203route.xlsx')
    manager.save_routes_to_json('../output/original_routes_info.json')