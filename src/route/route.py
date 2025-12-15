import os
import json
from datetime import datetime, timedelta
from api.google_maps import GoogleRoutesAPI

class RouteManager:
    """
    Notes:
        Route Management.
    """
    def __init__(self, routes_info, distance_matrix=None, time_matrix=None):
        self.routes_info = routes_info
        self.dc = {'store_id': 'dc', 'longitude': 121.40712, 'latitude': 25.083282}
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix


    def add_store(self, route_id, store):
        """
        Notes:
            Add a new store to the specified route and update route info.

        Args:
            route_id (str): Route ID.
            store (dict): Store information.

        Returns:
            None
        """
        route = self.routes_info.get(route_id)

        if not route:
            return

        route['stores'].append(store)
        self._update_route_info(route)


    def insert_store(self, store, route_id, position):
        """
        Notes:
            Insert a store at a specific position in the route and update route info.

        Args:
            store (dict): Store information.
            route_id (str): Route ID.
            position (int): Position to insert the store.

        Returns:
            None
        """
        route = self.routes_info.get(route_id)

        if not route:
            return

        route['stores'].insert(position, store)
        self._update_route_info(route)


    def remove_store(self, route_id, removed_store):
        """
        Notes:
            Remove the store from the route and update route info.

        Args:
            route_id (str): Route ID.
            removed_store (dict): Removed store information.

        Returns:
            None
        """
        if route_id in self.routes_info:
            route = self.routes_info[route_id]
            stores = self.routes_info[route_id]['stores']
            self.routes_info[route_id]['stores'] = [store for store in stores if store['route_code'] != removed_store['route_code']]
            # self._remove_unused_route(route_id)
            self._update_route_info(route)


    def move_store_to_route(self, route1_id, store, route2_id, position):
        """
        """
        self.remove_store(route1_id, store)
        self.insert_store(store, route2_id, position)


    def replace_stores(self, route_id, stores):
        """
        Notes:
            Replace route's stores.
        
        Args:
            route_id (str): Route ID.
            stores (list): [store1, store2, ...]
        
        Returns:
            None.
        """
        route = self.routes_info.get(route_id)
        
        if not route: return

        route['stores'] = stores
        self._update_route_info(route)


    def remove_stores(self, route_id, removed_stores):
        """
        Notes:
            Remove multiple stores from the route and update route info.

        Args:
            route_id (str): Route ID.
            removed_stores (list): List of removed store information.

        Returns:
            None
        """
        if route_id in self.routes_info:
            route = self.routes_info[route_id]
            stores = self.routes_info[route_id]['stores']
            removed_route_codes = {store['route_code'] for store in removed_stores}
            self.routes_info[route_id]['stores'] = [store for store in stores if store['route_code'] not in removed_route_codes]
            self._update_route_info(route)


    def _remove_unused_routes(self):
        """
        Notes:
            Remove Unused route (0 store).
        
        Args:
            route_id (str): Route ID.

        Returns:
            None.
        """
        removed_route_ids = []
        for route_id in self.routes_info:
            stores = self.routes_info[route_id]['stores']
            if len(stores) == 0: 
                removed_route_ids.append(route_id)
                
        for route_id in removed_route_ids:
            self.routes_info.pop(route_id)


    def swap_stores(self, r1_id, s1, r2_id, s2):
        """
        Notes:
            Swap store.

        Args:
            r1_id (str): Route1 ID.
            s1 (dict): Store1 in Route1. 
            r2_id (str): Route2 ID.
            s2 (dict): Store2 in Route2.
        
        Returns:
            None.
        """
        r1 = self.routes_info[r1_id]
        r2 = self.routes_info[r2_id]
        r1_stores = r1['stores']
        r2_stores = r2['stores']

        i = r1_stores.index(s1)
        j = r2_stores.index(s2)

        r1_stores[i], r2_stores[j] = r2_stores[j], r1_stores[i]

        self._update_route_info(r1)
        self._update_route_info(r2)


    def get_route_info(self, route_id, field=None):
        """
        Notes:
            Get route information.

        Args:
            route_id (str): Route ID.

        Returns:
            dict: Route information with field (option).
        """
        if field:
            route_dc = self.routes_info.get(route_id).get('dc')
            if route_dc:
                return route_dc.get(field)
            return None
            
        return self.routes_info.get(route_id).get('dc')
            

    def _update_route_distance(self, route):
        """
        Notes:
            Calculate and update total distance of the route.
        
        Args:
            route (dict): Route information.

        Returns:
            None.
        """
        if self.distance_matrix is None:
            raise ValueError("Distance matrix must be provided to update route distance.")
        
        total_distance = 0
        stores = [self.dc] + route['stores'] + [self.dc]

        for prev, curr in zip(stores[:-1], stores[1:]):
            prev_id, cur_id = prev['store_id'], curr['store_id']
            distance = self.distance_matrix[prev_id][cur_id]
            total_distance += distance
        
        route['dc']['distance'] = total_distance


    def _update_route_time(self, route):
        """
        Notes:
            Calculate and update total time of the route.

        Args:
            route (dict): Route information.

        Returns:
            None.
        """
        if self.time_matrix is None:
            raise ValueError("Time matrix must be provided to update route time.")

        duration = 0
        stores = [self.dc] + route['stores'] + [self.dc]

        for prev, curr in zip(stores[:-1], stores[1:]):
            prev_id, cur_id = prev['store_id'], curr['store_id']
            travel_time = self.time_matrix[prev_id][cur_id]
            duration += travel_time

        total_dwell_time = sum(store['dwell_time'] for store in route['stores'])
        duration += total_dwell_time

        route['dc']['duration'] = duration


    def _update_all_stores_pred_time(self, route):
        """
        Notes:
            Update predicted time for all stores in all routes.
        
        Args:
            route (dict): Route information.
        
        Returns:    
            None.
        """
        if self.time_matrix is None:
            raise ValueError("Time matrix must be provided to update predicted times.")

        stores = route.get('stores', [])

        if not stores:
            return
        
        stores[0]['pred_time'] = stores[0]['sched_time']
        for prev, curr in zip(stores[:-1], stores[1:]):
            pre_id, cur_id = prev['store_id'], curr['store_id']
            travel_time = self.time_matrix[pre_id][cur_id]
            pre_dwell = prev['dwell_time']
            arrival_time = datetime.fromisoformat(prev['pred_time']) + timedelta(seconds=travel_time + pre_dwell)
            curr['pred_time'] = arrival_time.isoformat(timespec='seconds')


    def _update_route_volume(self, route):
        """
        Notes:
            Calculate and update total volume of the route.

        Args:
            route (dict): Route information.

        Returns:
            None
        """
        total_volume = sum(store.get('volume', 0) for store in route['stores'])
        route['dc']['total_volume'] = total_volume


    def _update_route_load_rate(self, route):
        """
        Notes:
            Calculate and update load rate of the route based on total volume and max capacity.

        Args:
            route (dict): Route information.

        Returns:
            None
        """
        load_rate = route['dc']['total_volume'] / route['dc'].get('max_capacity', 1)
        route['dc']['load_rate'] = load_rate


    def _update_route_id(self, route):
        """
        Notes:
            Update the store's route_id
        
        Args:
            route (dict): Route information.

        Returns:
            None
        """
        route_id = route['dc']['route_id']
        for store in route['stores']:
            store["route_id"] = route_id


    def _update_route_info(self, route):
        """
        Notes:
            Update both total volume and load rate for a route.

        Args:
            route (dict): Route information.

        Returns:
            None
        """
        self._update_route_distance(route)
        self._update_route_time(route)
        self._update_all_stores_pred_time(route)
        self._update_route_volume(route)
        self._update_route_load_rate(route)
        self._update_route_id(route)
    

    def _update_all_routes_info(self):
        """
        Notes:
            Update route information for all routes.

        Args:
            None.

        Returns:
            None.
        """
        for _, route in self.routes_info.items():
            self._update_route_info(route)


    # def compute_route_via_routes_api(self):
    #     """
    #     Notes:
    #         Compute route distance and duration via Google Maps Routes API.

    #     Args:
    #         None.

    #     Returns:
    #         None.
    #     """
    #     routes_api = GoogleRoutesAPI()

    #     for _, route in self.routes_info.items():
    #         waypoints = route['stores']
    #         distance, duration = routes_api.compute_route(waypoints)
    #         route['dc']['distance'] = distance
    #         route['dc']['duration'] = duration
    #     # pred_time update
        

    def export_routes_info(self, json_file='optimized_routes_info.json'):
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