import math
import pandas as pd
from data.base_data import BaseDataManager

class ODataManager(BaseDataManager):
    """
    Notes:
        Load route-related data from source files.
    """
    def __init__(self, excel_files, distance_matrix, time_matrix):
        super().__init__(excel_files, distance_matrix, time_matrix)
        self.routes_info = self._load_original_routes()
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
        i = 0
        for _, row in self.routes_df.iterrows():
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
