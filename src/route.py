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