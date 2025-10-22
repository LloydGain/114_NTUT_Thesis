import os
import json
from datetime import datetime, timedelta
from google_maps import GoogleRoutesAPI

class RouteManager:
    """
    Notes:
        Route Management.
    """
    def __init__(self, routes_info, distance_matrix=None, time_matrix=None):
        self.routes_info = routes_info
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
            

    def _update_route_time(self, route):
        """
        Notes:
            Calculate and update total time of the route.

        Args:
            route (dict): Route information.

        Returns:
            None
        """
        if self.time_matrix is None:
            raise ValueError("Time matrix must be provided to update route time.")

        total_time = 0

        prev_id = 'dc'
        for store in route['stores']:
            curr_id = store['store_id']
            total_time += self.time_matrix[prev_id][curr_id]
            total_time += store['dwell_time']
            prev_id = curr_id
        total_time += self.time_matrix[prev_id]['dc']

        route['dc']['duration'] = total_time


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
            travel_time = self.time_matrix[prev['store_id']][curr['store_id']]
            pre_dwell = prev['dwell_time']
            arrival_time = datetime.fromisoformat(prev['pred_time']) + timedelta(seconds=travel_time + pre_dwell)
            curr['pred_time'] = arrival_time.isoformat()


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


    def _update_route_info(self, route):
        """
        Notes:
            Update both total volume and load rate for a route.

        Args:
            route (dict): Route information.

        Returns:
            None
        """
        self._update_route_time(route)
        self._update_all_stores_pred_time(route)
        self._update_route_volume(route)
        self._update_route_load_rate(route)
    

    def _export_routes_info(self, json_file='optimized_routes_info.json'):
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