import copy
from datetime import datetime, timedelta
from models.route_manager import RouteManager
import config

class LocalSearch:
    """
    Notes:
        Local Search for route optimization.
    """
    def __init__(self, distance_matrix, time_matrix):
        self.dc = config.DC_CONFIG
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix


    def _calculate_route_distance(self, stores):
        """
        Notes:
            Calculate the total distance of a single route.

        Args:
            stores (list): List of store dicts in the route.

        Returns:
            float: Total distance of the route in km.
        """
        total_dist = 0
        depot_id = self.dc['store_id']
        stores_id = [depot_id] + [store['store_id'] for store in stores] + [depot_id]

        for store_idx, store_idy in zip(stores_id[:-1], stores_id[1:]):
            dist = self.distance_matrix[store_idx][store_idy]
            total_dist += dist

        return total_dist


    def _calculate_routes_cost(self, routes):
        """
        Notes:
            Calculate the total cost of all routes.

        Args:
            routes (dict): Routes information.

        Returns:
            total_cost (float): Total cost of all routes.
        """
        total_cost = 0
        for route_id in routes:
            stores = routes[route_id]['stores']
            cost = self._calculate_route_distance(stores)
            total_cost += cost

        return total_cost


    def _is_within_time_window(self, arrival_time, earliest_time, latest_time):
        """
        Notes:
            Check if a given arrival time within store time window.

        Args:
            arrival_time (datetime): Arrival time.
            earliest_time (datetime): Earliest time (start of time window).
            latest_time (datetime): Latest time (end of time window).

        Returns:
            bool: True if within time window, False otherwise.
        """
        return earliest_time <= arrival_time <= latest_time


    def _check_time_constraint(self, stores):
        """
        Notes:
            Check if the route satisfies time window constraints.

        Args:
            stores (list): List of store dicts in the route.

        Returns:
            bool: True if time constraints are satisfied, False otherwise.
        """
        if len(stores) == 0:
            return True

        prev_store = stores[0]
        prev_dwell = prev_store['dwell_time']
        cur_time = datetime.fromisoformat(prev_store['sched_time'])
        cur_time = cur_time + timedelta(seconds=prev_dwell)

        for cur_store in stores[1:]:
            cur_earliest_time = datetime.fromisoformat(cur_store['earliest_time'])
            cur_lastest_time = datetime.fromisoformat(cur_store['latest_time'])
            travel_time = self.time_matrix[prev_store['store_id']][cur_store['store_id']]
            cur_time = cur_time + timedelta(seconds=travel_time)

            if not self._is_within_time_window(cur_time, cur_earliest_time, cur_lastest_time):
                return False

            cur_dwell = cur_store['dwell_time']
            cur_time = cur_time + timedelta(seconds=cur_dwell)
            prev_store = cur_store

        return True


    def _check_capacity_constraint(self, route, store):
        """
        Notes:
            Check if adding a store to the route satisfies capacity constraint.

        Args:
            route (dict): Route information.
            store (dict): Store to be added.

        Returns:
            bool: True if capacity constraint is satisfied, False otherwise.
        """
        vehicle_capacity = route['dc']['max_capacity']
        route_stores = route['stores']
        total_vol = sum(s['volume'] for s in route_stores)
        total_vol += store['volume']

        return total_vol <= vehicle_capacity


    def _check_order_principle(self, route_id, stores):
        """
        Notes:
            Check if the stores in the route follow the order principle based on route code.

        Args:
            stores (list): List of store dicts.

        Returns:
            bool: True if the stores follow the order principle, False otherwise.
        """
        route_codes = [store['route_code'][2:] for store in stores if store['route_code'].startswith(route_id)]
        return route_codes == sorted(route_codes)


    def _two_opt(self, stores, cost):
        """
        Notes:
            Intra-route 2-opt optimization.

        Args:
            stores (list): List of store dicts in the route.
            cost (float): Current cost of the route.

        Returns:
            tuple: (best_stores (list), best_cost (float))
        """
        best_stores = stores
        best_cost = cost
        n = len(stores)
        for i in range(n-1):
            for j in range(i+1, n):
                new_route_stores = best_stores[:]
                new_route_stores[i:j+1] = new_route_stores[i:j+1][::-1]

                new_cost = self._calculate_route_distance(new_route_stores)

                if new_cost < best_cost:
                    if self._check_time_constraint(new_route_stores):
                        best_cost = new_cost
                        best_stores = new_route_stores

        return best_stores, best_cost


    def _relocate(self, routes, route1_id, route2_id):
        """
        Notes:
            Inter-route relocate: move one store from route1 to route2.

        Args:
            routes (dict): Routes information.
            route1_id (str): Source route ID.
            route2_id (str): Destination route ID.

        Returns:
            tuple: (moved_store (dict or None), moved_position (int or -1))
        """
        r2 = routes[route2_id]
        r1_stores = routes[route1_id]['stores']
        r2_stores = routes[route2_id]['stores']

        moved_store = None
        moved_position = -1
        best_reduction = 0

        base_cost_r1 = self._calculate_route_distance(r1_stores)
        base_cost_r2 = self._calculate_route_distance(r2_stores)
        base_total_cost = base_cost_r1 + base_cost_r2

        # -----------------------------------------------
        if not route2_id.startswith('1'):
            return None, -1
        # -----------------------------------------------

        for idx, r1_store in enumerate(r1_stores):

            if not self._check_capacity_constraint(r2, r1_store):
                continue

            for idy, _ in enumerate(r2_stores):
                if len(r1_stores) == 1:
                    new_r1 = []
                    new_r2 = r2_stores[:idy] + [r1_store] + r2_stores[idy:]
                else:
                    new_r1 = r1_stores[:idx] + r1_stores[idx+1:]
                    new_r2 = r2_stores[:idy] + [r1_store] + r2_stores[idy:]

                new_total_cost = self._calculate_route_distance(new_r1) + self._calculate_route_distance(new_r2)

                diff = new_total_cost - base_total_cost

                if diff < best_reduction:
                    if self._check_time_constraint(new_r1) and self._check_time_constraint(new_r2):
                        if self._check_order_principle(route2_id, new_r2):
                            best_reduction = diff
                            moved_store = r1_store
                            moved_position = idy

        return moved_store, moved_position


    def _swap(self, routes, route1_id, route2_id):
        """
        Notes:
            Inter-route swap: swap one store from route1 with one store from route2.

        Args:
            routes (dict): Routes information.
            route1_id (str): First route ID.
            route2_id (str): Second route ID.

        Returns:
            tuple: (store_from_route1 (dict or None), store_from_route2 (dict or None))
        """
        r1 = routes[route1_id]
        r2 = routes[route2_id]
        r1_stores = r1['stores']
        r2_stores = r2['stores']
        cap1 = r1['dc']['max_capacity']
        cap2 = r2['dc']['max_capacity']

        best_reduction = 0
        best_pair = (None, None)

        base_cost = (
            self._calculate_route_distance(r1_stores)
            + self._calculate_route_distance(r2_stores)
        )

        # -----------------------------------------------
        if not route1_id.startswith('1') or not route2_id.startswith('1'):
            return (None, None)
        # -----------------------------------------------

        for i, s1 in enumerate(r1_stores):
            for j, s2 in enumerate(r2_stores):

                vol1 = sum(s['volume'] for s in r1_stores) - s1['volume'] + s2['volume']
                vol2 = sum(s['volume'] for s in r2_stores) - s2['volume'] + s1['volume']

                if vol1 > cap1 or vol2 > cap2:
                    continue

                new_r1 = r1_stores[:]
                new_r2 = r2_stores[:]
                new_r1[i] = s2
                new_r2[j] = s1

                if not self._check_time_constraint(new_r1):
                    continue
                if not self._check_time_constraint(new_r2):
                    continue

                if not self._check_order_principle(route1_id, new_r1):
                    continue
                if not self._check_order_principle(route2_id, new_r2):
                    continue

                new_cost = (
                    self._calculate_route_distance(new_r1)
                    + self._calculate_route_distance(new_r2)
                )

                diff = new_cost - base_cost

                if diff < best_reduction:
                    best_reduction = diff
                    best_pair = (s1, s2)

        return best_pair


    def optimize_intra_route(self, routes_info, routes_cost):
        """
        Notes:
            Intra-route optimization using 2-opt.

        Args:
            routes_info (dict): Routes information.
            routes_cost (float): Current total cost of all routes.

        Returns:
            tuple: (optimized_routes (dict), optimized_cost (float))
        """
        routes = copy.deepcopy(routes_info)
        route_manager = RouteManager(routes, self.distance_matrix, self.time_matrix)
        new_routes_cost = routes_cost

        improved = True
        while improved:
            improved = False
            for route_id, route_data in routes.items():

                if not route_id.startswith('1'):
                    continue

                original_route = route_data['dc']
                original_stores = route_data['stores']
                original_cost = original_route['distance']

                if len(original_stores) < 4:
                    continue

                new_stores, new_cost = self._two_opt(original_stores, original_cost)

                if new_cost < original_cost:
                    improved = True
                    # print(f'LS improved: {original_cost - new_cost}.')
                    # print([s['store_id'] for s in original_stores])
                    # print([s['store_id'] for s in new_stores])
                    route_manager.replace_stores(route_id, new_stores)
                    new_routes_cost = new_routes_cost - (original_cost - new_cost)

        return routes, new_routes_cost


    def optimize_inter_route(self, routes_info, routes_cost):
        """
        Notes:
            Inter-route optimization using relocate and swap.

        Args:
            routes_info (dict): Routes information.
            routes_cost (float): Current total cost of all routes.

        Returns:
            tuple: (optimized_routes (dict), optimized_cost (float))
        """
        routes = copy.deepcopy(routes_info)
        route_ids = list(routes.keys())
        route_manager = RouteManager(routes, self.distance_matrix, self.time_matrix)
        new_routes_cost = routes_cost

        improved = True
        while improved:
            improved = False
            for r1_id in route_ids:
                for r2_id in route_ids:
                    if r1_id == r2_id:
                        continue

                    store, position = self._relocate(routes, r1_id, r2_id)
                    if store is not None:
                        route_manager.move_store_to_route(r1_id, store, r2_id, position)
                        # print(f'Relocate {store["store_id"]} from {r1_id} to {r2_id} at {position}')
                        improved = True
                        break

                    s1, s2 = self._swap(routes, r1_id, r2_id)
                    if s1 is not None:
                        route_manager.swap_stores(r1_id, s1, r2_id, s2)
                        # print(f'Swap {s1["store_id"]} ({r1_id}) with {s2["store_id"]} ({r2_id})')
                        improved = True
                        break

                if improved:
                    break

        route_manager.update_all_routes_info()
        routes = route_manager.routes_info
        new_routes_cost = self._calculate_routes_cost(routes)

        return routes, new_routes_cost


    def optimize(self, routes_info, routes_cost):
        """
        Notes:
            Perform local search optimization (intra-route and inter-route).

        Args:
            routes_info (dict): Routes information.
            routes_cost (float): Current total cost of all routes.

        Returns:
            tuple: (optimized_routes (dict), optimized_cost (float))
        """
        routes_info, routes_cost = self.optimize_intra_route(routes_info, routes_cost)
        routes_info, routes_cost = self.optimize_inter_route(routes_info, routes_cost)
        routes_info, routes_cost = self.optimize_intra_route(routes_info, routes_cost)

        return routes_info, routes_cost
