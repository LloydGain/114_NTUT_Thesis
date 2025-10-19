import os
import json

class RouteManager:
    """
    Notes:
        Route Management.
    """
    def __init__(self, routes_info):
        self.routes_info = routes_info


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