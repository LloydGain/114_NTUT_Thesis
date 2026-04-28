import copy
from config import config
from datetime import datetime, timedelta
from models.route_manager import RouteManager

class VND:
    """
    Notes:
        VND for route optimization.
    """
    def __init__(self, distance_matrix, time_matrix, vehicle_cost=2000, is_solomon=False, improvement_strategy='best'):
        self.dc = config.DC_CONFIG
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix
        self.vehicle_cost = vehicle_cost
        self.is_solomon = is_solomon
        self.improvement_strategy = improvement_strategy


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
        num_vehicles = 0
        for route_id in routes:
            stores = routes[route_id]['stores']
            cost = self._calculate_route_distance(stores)
            total_cost += cost

            if route_id.startswith('1'):
                num_vehicles += 1

        if self.is_solomon:
            return (num_vehicles, total_cost)
        return total_cost + num_vehicles * self.vehicle_cost


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

            if self.is_solomon:
                if cur_time > cur_lastest_time:
                    return False
                if cur_time < cur_earliest_time:
                    cur_time = cur_earliest_time
            else:
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


    def _two_opt_first(self, stores, cost):
        """
        Notes:
            Intra-route 2-opt optimization (First Improvement).
        """
        n = len(stores)
        for i in range(n-1):
            for j in range(i+1, n):
                new_route = stores[:]
                new_route[i:j+1] = list(reversed(new_route[i:j+1]))
                new_dist = self._calculate_route_distance(new_route)

                if cost - new_dist > 1e-6:
                    if self._check_time_constraint(new_route):
                        return new_route, new_dist
        return stores, cost


    def _two_opt_best(self, stores, cost):
        """
        Notes:
            Intra-route 2-opt optimization (Best Improvement).
        """
        n = len(stores)
        best_route = stores
        best_cost = cost

        for i in range(n-1):
            for j in range(i+1, n):
                new_route = stores[:]
                new_route[i:j+1] = list(reversed(new_route[i:j+1]))
                new_dist = self._calculate_route_distance(new_route)

                if best_cost - new_dist > 1e-6:
                    if self._check_time_constraint(new_route):
                        best_route = new_route
                        best_cost = new_dist

        return best_route, best_cost


    def _relocate_first(self, routes, route1_id, route2_id):
        """
        Notes:
            Inter-route relocate (First Improvement).
        """
        r2 = routes[route2_id]
        r1_stores = routes[route1_id]['stores']
        r2_stores = routes[route2_id]['stores']
        depot = self.dc['store_id']

        for idx, r1_store in enumerate(r1_stores):
            if not self._check_capacity_constraint(r2, r1_store):
                continue

            s_id = r1_store['store_id']
            prev_i = depot if idx == 0 else r1_stores[idx-1]['store_id']
            next_i = depot if idx == len(r1_stores)-1 else r1_stores[idx+1]['store_id']

            delta_remove = (
                self.distance_matrix[prev_i][next_i]
                - self.distance_matrix[prev_i][s_id]
                - self.distance_matrix[s_id][next_i]
            )

            for idy in range(len(r2_stores) + 1):
                prev_j = depot if idy == 0 else r2_stores[idy-1]['store_id']
                next_j = depot if idy == len(r2_stores) else r2_stores[idy]['store_id']

                delta_insert = (
                    self.distance_matrix[prev_j][s_id]
                    + self.distance_matrix[s_id][next_j]
                    - self.distance_matrix[prev_j][next_j]
                )

                improvement = -(delta_remove + delta_insert)

                if improvement > 1e-6:
                    new_r1 = r1_stores[:idx] + r1_stores[idx+1:]
                    new_r2 = r2_stores[:idy] + [r1_store] + r2_stores[idy:]

                    if self._check_time_constraint(new_r1) and self._check_time_constraint(new_r2):
                        if self._check_order_principle(route2_id, new_r2):
                            return r1_store, idy, improvement

        return None, -1, 0.0


    def _relocate_best(self, routes, route1_id, route2_id):
        """
        Notes:
            Inter-route relocate (Best Improvement).
        """
        r2 = routes[route2_id]
        r1_stores = routes[route1_id]['stores']
        r2_stores = routes[route2_id]['stores']
        depot = self.dc['store_id']

        best_store = None
        best_pos = -1
        best_improvement = 0.0

        for idx, r1_store in enumerate(r1_stores):
            if not self._check_capacity_constraint(r2, r1_store):
                continue

            s_id = r1_store['store_id']
            prev_i = depot if idx == 0 else r1_stores[idx-1]['store_id']
            next_i = depot if idx == len(r1_stores)-1 else r1_stores[idx+1]['store_id']

            delta_remove = (
                self.distance_matrix[prev_i][next_i]
                - self.distance_matrix[prev_i][s_id]
                - self.distance_matrix[s_id][next_i]
            )

            for idy in range(len(r2_stores) + 1):
                prev_j = depot if idy == 0 else r2_stores[idy-1]['store_id']
                next_j = depot if idy == len(r2_stores) else r2_stores[idy]['store_id']

                delta_insert = (
                    self.distance_matrix[prev_j][s_id]
                    + self.distance_matrix[s_id][next_j]
                    - self.distance_matrix[prev_j][next_j]
                )

                improvement = -(delta_remove + delta_insert)

                if improvement > best_improvement:
                    new_r1 = r1_stores[:idx] + r1_stores[idx+1:]
                    new_r2 = r2_stores[:idy] + [r1_store] + r2_stores[idy:]

                    if self._check_time_constraint(new_r1) and self._check_time_constraint(new_r2):
                        if self._check_order_principle(route2_id, new_r2):
                            best_store = r1_store
                            best_pos = idy
                            best_improvement = improvement

        return best_store, best_pos, best_improvement


    def _swap_first(self, routes, route1_id, route2_id):
        """
        Notes:
            Inter-route swap (First Improvement).
        """
        r1 = routes[route1_id]
        r2 = routes[route2_id]
        r1_stores = r1['stores']
        r2_stores = r2['stores']
        cap1 = r1['dc']['max_capacity']
        cap2 = r2['dc']['max_capacity']
        
        current_vol1 = sum(s['volume'] for s in r1_stores)
        current_vol2 = sum(s['volume'] for s in r2_stores)

        for i, s1 in enumerate(r1_stores):
            for j, s2 in enumerate(r2_stores):

                vol1 = current_vol1 - s1['volume'] + s2['volume']
                vol2 = current_vol2 - s2['volume'] + s1['volume']
                if vol1 > cap1 or vol2 > cap2:
                    continue

                prev1 = self.dc['store_id'] if i == 0 else r1_stores[i-1]['store_id']
                next1 = self.dc['store_id'] if i == len(r1_stores) - 1 else r1_stores[i+1]['store_id']
                prev2 = self.dc['store_id'] if j == 0 else r2_stores[j-1]['store_id']
                next2 = self.dc['store_id'] if j == len(r2_stores) - 1 else r2_stores[j+1]['store_id']
                s1_id = s1['store_id']
                s2_id = s2['store_id']

                delta = (
                    - self.distance_matrix[prev1][s1_id]
                    - self.distance_matrix[s1_id][next1]
                    - self.distance_matrix[prev2][s2_id]
                    - self.distance_matrix[s2_id][next2]

                    + self.distance_matrix[prev1][s2_id]
                    + self.distance_matrix[s2_id][next1]

                    + self.distance_matrix[prev2][s1_id]
                    + self.distance_matrix[s1_id][next2]
                )

                improvement = -delta

                if improvement > 1e-6:
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
                    
                    return s1, s2, improvement

        return None, None, 0.0


    def _swap_best(self, routes, route1_id, route2_id):
        """
        Notes:
            Inter-route swap (Best Improvement).
        """
        r1 = routes[route1_id]
        r2 = routes[route2_id]
        r1_stores = r1['stores']
        r2_stores = r2['stores']
        cap1 = r1['dc']['max_capacity']
        cap2 = r2['dc']['max_capacity']
        
        current_vol1 = sum(s['volume'] for s in r1_stores)
        current_vol2 = sum(s['volume'] for s in r2_stores)

        best_s1, best_s2 = None, None
        best_improvement = 0.0

        for i, s1 in enumerate(r1_stores):
            for j, s2 in enumerate(r2_stores):

                vol1 = current_vol1 - s1['volume'] + s2['volume']
                vol2 = current_vol2 - s2['volume'] + s1['volume']
                if vol1 > cap1 or vol2 > cap2:
                    continue

                prev1 = self.dc['store_id'] if i == 0 else r1_stores[i-1]['store_id']
                next1 = self.dc['store_id'] if i == len(r1_stores) - 1 else r1_stores[i+1]['store_id']
                prev2 = self.dc['store_id'] if j == 0 else r2_stores[j-1]['store_id']
                next2 = self.dc['store_id'] if j == len(r2_stores) - 1 else r2_stores[j+1]['store_id']
                s1_id = s1['store_id']
                s2_id = s2['store_id']

                delta = (
                    - self.distance_matrix[prev1][s1_id]
                    - self.distance_matrix[s1_id][next1]
                    - self.distance_matrix[prev2][s2_id]
                    - self.distance_matrix[s2_id][next2]

                    + self.distance_matrix[prev1][s2_id]
                    + self.distance_matrix[s2_id][next1]

                    + self.distance_matrix[prev2][s1_id]
                    + self.distance_matrix[s1_id][next2]
                )

                improvement = -delta

                if improvement > best_improvement:
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
                    
                    best_s1, best_s2 = s1, s2
                    best_improvement = improvement

        return best_s1, best_s2, best_improvement


    def _cross_exchange_first(self, routes, route1_id, route2_id):
        """
        Notes:
            Inter-route Cross Exchange (First Improvement).
            Exchanges a segment of max length 3 from route1 with a segment from route2.
        """
        r1_stores = routes[route1_id]['stores']
        r2_stores = routes[route2_id]['stores']
        cap1 = routes[route1_id]['dc']['max_capacity']
        cap2 = routes[route2_id]['dc']['max_capacity']
        
        max_len = 3
        current_vol1 = sum(s['volume'] for s in r1_stores)
        current_vol2 = sum(s['volume'] for s in r2_stores)
        
        for i in range(len(r1_stores)):
            for l1 in range(1, max_len + 1):
                if i + l1 > len(r1_stores):
                    continue
                seg1 = r1_stores[i:i+l1]
                vol_seg1 = sum(s['volume'] for s in seg1)
                
                prev1 = self.dc['store_id'] if i == 0 else r1_stores[i-1]['store_id']
                next1 = self.dc['store_id'] if i + l1 == len(r1_stores) else r1_stores[i+l1]['store_id']
                start1, end1 = seg1[0]['store_id'], seg1[-1]['store_id']
                
                for j in range(len(r2_stores)):
                    for l2 in range(1, max_len + 1):
                        if j + l2 > len(r2_stores):
                            continue
                        seg2 = r2_stores[j:j+l2]
                        vol_seg2 = sum(s['volume'] for s in seg2)
                        
                        vol1 = current_vol1 - vol_seg1 + vol_seg2
                        vol2 = current_vol2 - vol_seg2 + vol_seg1
                        if vol1 > cap1 or vol2 > cap2:
                            continue
                            
                        prev2 = self.dc['store_id'] if j == 0 else r2_stores[j-1]['store_id']
                        next2 = self.dc['store_id'] if j + l2 == len(r2_stores) else r2_stores[j+l2]['store_id']
                        start2, end2 = seg2[0]['store_id'], seg2[-1]['store_id']
                        
                        delta = (
                            - self.distance_matrix[prev1][start1]
                            - self.distance_matrix[end1][next1]
                            - self.distance_matrix[prev2][start2]
                            - self.distance_matrix[end2][next2]
                            
                            + self.distance_matrix[prev1][start2]
                            + self.distance_matrix[end2][next1]
                            + self.distance_matrix[prev2][start1]
                            + self.distance_matrix[end1][next2]
                        )
                        improvement = -delta
                        
                        if improvement > 1e-6:
                            new_r1 = r1_stores[:i] + seg2 + r1_stores[i+l1:]
                            new_r2 = r2_stores[:j] + seg1 + r2_stores[j+l2:]
                            
                            if self._check_time_constraint(new_r1) and self._check_time_constraint(new_r2):
                                if self._check_order_principle(route1_id, new_r1) and self._check_order_principle(route2_id, new_r2):
                                    return new_r1, new_r2, improvement

        return None, None, 0.0


    def _cross_exchange_best(self, routes, route1_id, route2_id):
        """
        Notes:
            Inter-route cross exchange (Best Improvement).
        """
        r1 = routes[route1_id]
        r2 = routes[route2_id]
        r1_stores = r1['stores']
        r2_stores = r2['stores']
        cap1 = r1['dc']['max_capacity']
        cap2 = r2['dc']['max_capacity']
        
        current_vol1 = sum(s['volume'] for s in r1_stores)
        current_vol2 = sum(s['volume'] for s in r2_stores)

        best_r1, best_r2, best_improvement = None, None, 0.0

        for i in range(len(r1_stores)):
            for l1 in range(1, len(r1_stores) - i + 1):
                seg1 = r1_stores[i:i+l1]
                vol_seg1 = sum(s['volume'] for s in seg1)
                
                for j in range(len(r2_stores)):
                    for l2 in range(1, len(r2_stores) - j + 1):
                        seg2 = r2_stores[j:j+l2]
                        vol_seg2 = sum(s['volume'] for s in seg2)
                        
                        vol1 = current_vol1 - vol_seg1 + vol_seg2
                        vol2 = current_vol2 - vol_seg2 + vol_seg1
                        if vol1 > cap1 or vol2 > cap2:
                            continue
                            
                        prev1 = self.dc['store_id'] if i == 0 else r1_stores[i-1]['store_id']
                        next1 = self.dc['store_id'] if i + l1 == len(r1_stores) else r1_stores[i+l1]['store_id']
                        start1, end1 = seg1[0]['store_id'], seg1[-1]['store_id']
                        
                        prev2 = self.dc['store_id'] if j == 0 else r2_stores[j-1]['store_id']
                        next2 = self.dc['store_id'] if j + l2 == len(r2_stores) else r2_stores[j+l2]['store_id']
                        start2, end2 = seg2[0]['store_id'], seg2[-1]['store_id']
                        
                        delta = (
                            - self.distance_matrix[prev1][start1]
                            - self.distance_matrix[end1][next1]
                            - self.distance_matrix[prev2][start2]
                            - self.distance_matrix[end2][next2]
                            
                            + self.distance_matrix[prev1][start2]
                            + self.distance_matrix[end2][next1]
                            + self.distance_matrix[prev2][start1]
                            + self.distance_matrix[end1][next2]
                        )
                        improvement = -delta
                        
                        if improvement > best_improvement:
                            new_r1 = r1_stores[:i] + seg2 + r1_stores[i+l1:]
                            new_r2 = r2_stores[:j] + seg1 + r2_stores[j+l2:]
                            
                            if self._check_time_constraint(new_r1) and self._check_time_constraint(new_r2):
                                if self._check_order_principle(route1_id, new_r1) and self._check_order_principle(route2_id, new_r2):
                                    best_r1, best_r2, best_improvement = new_r1, new_r2, improvement

        return best_r1, best_r2, best_improvement


    def _neighborhood_intra_first(self, routes, route_manager):
        """
        Notes:
            Intra-route optimization using 2-opt.

        Args:
            routes (dict): Routes information.
            route_manager (RouteManager): Route manager.

        Returns:
            bool: True if any optimization is made, False otherwise.
        """
        for route_id, route_data in routes.items():
            original_dist = self._calculate_route_distance(route_data['stores'])
            new_stores, new_dist = self._two_opt_first(route_data['stores'], original_dist)

            if original_dist - new_dist > 1e-6:
                route_manager.replace_stores(route_id, new_stores)
                return True

        return False


    def _neighborhood_intra_best(self, routes, route_manager):
        """
        Notes:
            Intra-route optimization using 2-opt (Best Improvement).
        """
        best_route_id, best_new_stores, max_improvement = None, None, 1e-6

        for route_id, route_data in routes.items():
            original_dist = self._calculate_route_distance(route_data['stores'])
            new_stores, new_dist = self._two_opt_best(route_data['stores'], original_dist)
            improvement = original_dist - new_dist

            if improvement > max_improvement:
                max_improvement, best_route_id, best_new_stores = improvement, route_id, new_stores

        if best_route_id:
            route_manager.replace_stores(best_route_id, best_new_stores)
            return True

        return False


    def _neighborhood_inter_first(self, routes, route_manager):
        """
        Notes:
            Inter-route optimization using relocate and swap.

        Args:
            routes (dict): Routes information.
            route_manager (RouteManager): Route manager.

        Returns:
            bool: True if any optimization is made, False otherwise.
        """
        route_ids = list(routes.keys())

        for i in range(len(route_ids)):
            for j in range(len(route_ids)):
                if i == j: continue
                moved_store, pos, impr = self._relocate_first(routes, route_ids[i], route_ids[j])
                if impr > 1e-6:
                    route_manager.move_store_to_route(route_ids[i], moved_store, route_ids[j], pos)
                    return True

        for i in range(len(route_ids)):
            for j in range(i + 1, len(route_ids)):
                s1, s2, impr = self._swap_first(routes, route_ids[i], route_ids[j])
                if impr > 1e-6:
                    route_manager.swap_stores(route_ids[i], s1, route_ids[j], s2)
                    return True

        for i in range(len(route_ids)):
            for j in range(i + 1, len(route_ids)):
                nr1, nr2, impr = self._cross_exchange_first(routes, route_ids[i], route_ids[j])
                if impr > 1e-6:
                    route_manager.replace_stores(route_ids[i], nr1)
                    route_manager.replace_stores(route_ids[j], nr2)
                    return True

        return False


    def _neighborhood_inter_best(self, routes, route_manager):
        """
        Notes:
            Inter-route optimization using Relocate, Swap, and Cross Exchange sequentially (First Improvement behavior).
            Returns True immediately after finding any improving step to massively accelerate searches.
        """
        route_ids = list(routes.keys())

        best_impr, best_move = 1e-6, None
        for i in range(len(route_ids)):
            for j in range(len(route_ids)):
                if i == j: continue
                r1_id, r2_id = route_ids[i], route_ids[j]
                moved_store, pos, impr = self._relocate_best(routes, r1_id, r2_id)
                if impr > best_impr:
                    best_impr = impr
                    best_move = ('relocate', r1_id, moved_store, r2_id, pos)

        if best_move:
            _, r1_id, moved_store, r2_id, pos = best_move
            route_manager.move_store_to_route(r1_id, moved_store, r2_id, pos)
            return True

        for i in range(len(route_ids)):
            for j in range(i + 1, len(route_ids)):
                r1_id, r2_id = route_ids[i], route_ids[j]
                s1, s2, impr = self._swap_best(routes, r1_id, r2_id)
                if impr > best_impr:
                    best_impr = impr
                    best_move = ('swap', r1_id, s1, r2_id, s2)

        if best_move:
            _, r1_id, s1, r2_id, s2 = best_move
            route_manager.swap_stores(r1_id, s1, r2_id, s2)
            return True

        for i in range(len(route_ids)):
            for j in range(i + 1, len(route_ids)):
                r1_id, r2_id = route_ids[i], route_ids[j]
                nr1, nr2, impr = self._cross_exchange_best(routes, r1_id, r2_id)
                if impr > best_impr:
                    best_impr = impr
                    best_move = ('cross', r1_id, nr1, r2_id, nr2)

        if best_move:
            _, r1_id, nr1, r2_id, nr2 = best_move
            route_manager.replace_stores(r1_id, nr1)
            route_manager.replace_stores(r2_id, nr2)
            return True

        return False


    def optimize(self, routes_info):
        """
        Notes:
            Perform VND (Variable Neighborhood Descent) optimization.

            Neighborhoods:
                1. Intra-route 2-opt
                2. Inter-route relocate
                3. Inter-route swap

            Strategy:
                'best'  -> Best Improvement (2-opt best + relocate/swap/cross-exchange first)
                'first' -> First Improvement (2-opt first + relocate/swap first)

        Args:
            routes_info (dict): Routes information.

        Returns:
            tuple: (optimized_routes (dict), optimized_cost (float))
        """
        current_routes = copy.deepcopy(routes_info)
        route_manager = RouteManager(current_routes, self.distance_matrix, self.time_matrix)

        improved = True
        while improved:
            improved = False

            if self.improvement_strategy == 'first':
                if self._neighborhood_intra_first(current_routes, route_manager):
                    improved = True
                    current_routes = route_manager.routes_info
                    continue

                if self._neighborhood_inter_first(current_routes, route_manager):
                    improved = True
                    current_routes = route_manager.routes_info
                    continue
            else:
                if self._neighborhood_intra_best(current_routes, route_manager):
                    improved = True
                    current_routes = route_manager.routes_info
                    continue

                if self._neighborhood_inter_best(current_routes, route_manager):
                    improved = True
                    current_routes = route_manager.routes_info
                    continue

        route_manager.update_all_routes_info()
        final_routes = route_manager.routes_info
        final_cost = self._calculate_routes_cost(final_routes)

        return final_routes, final_cost