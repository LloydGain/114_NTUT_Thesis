import copy
import numpy as np
from config import config
from datetime import datetime, timedelta
from models.route_manager import RouteManager
from numba import njit
from numba.typed import List

@njit(cache=True)
def _njit_calculate_route_distance(route, dist_matrix, depot_idx):
    if len(route) == 0:
        return 0.0
    total_dist = dist_matrix[depot_idx, route[0]]
    for i in range(len(route) - 1):
        total_dist += dist_matrix[route[i], route[i+1]]
    total_dist += dist_matrix[route[-1], depot_idx]
    return total_dist

@njit(cache=True)
def _njit_check_time_constraint(route, time_matrix, dwell_array, earliest_array, latest_array, sched_array, is_solomon, depot_idx):
    if len(route) == 0:
        return True
    
    prev_idx = route[0]
    prev_dwell = dwell_array[prev_idx]
    cur_time = sched_array[prev_idx] + prev_dwell

    for i in range(1, len(route)):
        cur_idx = route[i]
        cur_earliest = earliest_array[cur_idx]
        cur_latest = latest_array[cur_idx]
        travel_time = time_matrix[prev_idx, cur_idx]
        cur_time += travel_time

        if is_solomon:
            if cur_time > cur_latest:
                return False
            if cur_time < cur_earliest:
                cur_time = cur_earliest
        else:
            if cur_time < cur_earliest or cur_time > cur_latest:
                return False

        cur_dwell = dwell_array[cur_idx]
        cur_time += cur_dwell
        prev_idx = cur_idx

    return True

@njit(cache=True)
def _njit_check_capacity_constraint(route, new_store_idx, volume_array, max_capacity):
    total_vol = volume_array[new_store_idx]
    for i in range(len(route)):
        total_vol += volume_array[route[i]]
    return total_vol <= max_capacity

@njit(cache=True)
def _njit_check_order_principle(route_id_int, route, order_match_matrix, order_rank_array):
    last_rank = -1
    for i in range(len(route)):
        store_idx = route[i]
        if order_match_matrix[store_idx, route_id_int]:
            rank = order_rank_array[store_idx]
            if rank < last_rank:
                return False
            last_rank = rank
    return True

@njit(cache=True)
def _njit_two_opt(route_id_int, route, cost, dist_matrix, time_matrix, dwell_array, earliest_array, latest_array, sched_array, is_solomon, depot_idx, order_match_matrix, order_rank_array):
    n = len(route)
    for i in range(n - 1):
        for j in range(i + 1, n):
            new_route = route.copy()
            for k in range((j - i + 1) // 2):
                tmp = new_route[i + k]
                new_route[i + k] = new_route[j - k]
                new_route[j - k] = tmp
            
            new_dist = _njit_calculate_route_distance(new_route, dist_matrix, depot_idx)
            if new_dist < cost - 1e-6:
                if _njit_check_time_constraint(new_route, time_matrix, dwell_array, earliest_array, latest_array, sched_array, is_solomon, depot_idx):
                    if _njit_check_order_principle(route_id_int, new_route, order_match_matrix, order_rank_array):
                        return True, new_route, new_dist
    return False, route, cost

@njit(cache=True)
def _njit_two_opt_best(route_id_int, route, cost, dist_matrix, time_matrix, dwell_array, earliest_array, latest_array, sched_array, is_solomon, depot_idx, order_match_matrix, order_rank_array):
    n = len(route)
    best_route = route.copy()
    best_cost = cost
    improved = False

    for i in range(n - 1):
        for j in range(i + 1, n):
            new_route = route.copy()
            for k in range((j - i + 1) // 2):
                tmp = new_route[i + k]
                new_route[i + k] = new_route[j - k]
                new_route[j - k] = tmp
            
            new_dist = _njit_calculate_route_distance(new_route, dist_matrix, depot_idx)
            if best_cost - new_dist > 1e-6:
                if _njit_check_time_constraint(new_route, time_matrix, dwell_array, earliest_array, latest_array, sched_array, is_solomon, depot_idx):
                    if _njit_check_order_principle(route_id_int, new_route, order_match_matrix, order_rank_array):
                        best_route = new_route
                        best_cost = new_dist
                        improved = True

    return improved, best_route, best_cost

@njit(cache=True)
def _njit_relocate(r1_id_int, r2_id_int, r1_stores, r2_stores, cap2,
                   dist_matrix, time_matrix, volume_array, dwell_array,
                   earliest_array, latest_array, sched_array, order_match_matrix, order_rank_array,
                   is_solomon, depot_idx):
    n1 = len(r1_stores)
    n2 = len(r2_stores)
    
    for idx in range(n1):
        s_id = r1_stores[idx]
        if not _njit_check_capacity_constraint(r2_stores, s_id, volume_array, cap2):
            continue
            
        prev_i = depot_idx if idx == 0 else r1_stores[idx - 1]
        next_i = depot_idx if idx == n1 - 1 else r1_stores[idx + 1]
        
        delta_remove = (dist_matrix[prev_i, next_i] 
                        - dist_matrix[prev_i, s_id] 
                        - dist_matrix[s_id, next_i])
        
        for idy in range(n2 + 1):
            prev_j = depot_idx if idy == 0 else r2_stores[idy - 1]
            next_j = depot_idx if idy == n2 else r2_stores[idy]
            
            delta_insert = (dist_matrix[prev_j, s_id]
                            + dist_matrix[s_id, next_j]
                            - dist_matrix[prev_j, next_j])
            
            diff = delta_remove + delta_insert
            
            if diff < -1e-6:
                new_r1 = np.empty(n1 - 1, dtype=r1_stores.dtype)
                for k in range(idx): new_r1[k] = r1_stores[k]
                for k in range(idx + 1, n1): new_r1[k - 1] = r1_stores[k]
                    
                new_r2 = np.empty(n2 + 1, dtype=r2_stores.dtype)
                for k in range(idy): new_r2[k] = r2_stores[k]
                new_r2[idy] = s_id
                for k in range(idy, n2): new_r2[k + 1] = r2_stores[k]
                    
                if _njit_check_time_constraint(new_r1, time_matrix, dwell_array, earliest_array, latest_array, sched_array, is_solomon, depot_idx) and \
                   _njit_check_time_constraint(new_r2, time_matrix, dwell_array, earliest_array, latest_array, sched_array, is_solomon, depot_idx):
                    if _njit_check_order_principle(r1_id_int, new_r1, order_match_matrix, order_rank_array) and \
                       _njit_check_order_principle(r2_id_int, new_r2, order_match_matrix, order_rank_array):
                        return True, new_r1, new_r2
                        
    return False, r1_stores, r2_stores

@njit(cache=True)
def _njit_swap(r1_id_int, r2_id_int, r1_stores, r2_stores, cap1, cap2,
               dist_matrix, time_matrix, volume_array, dwell_array,
               earliest_array, latest_array, sched_array, order_match_matrix, order_rank_array,
               is_solomon, depot_idx):
    n1 = len(r1_stores)
    n2 = len(r2_stores)
    
    vol1_base = 0.0
    for s in r1_stores: vol1_base += volume_array[s]
    vol2_base = 0.0
    for s in r2_stores: vol2_base += volume_array[s]
    
    for i in range(n1):
        for j in range(n2):
            s1 = r1_stores[i]
            s2 = r2_stores[j]
            
            v1 = vol1_base - volume_array[s1] + volume_array[s2]
            v2 = vol2_base - volume_array[s2] + volume_array[s1]
            if v1 > cap1 or v2 > cap2:
                continue
                
            prev1 = depot_idx if i == 0 else r1_stores[i - 1]
            next1 = depot_idx if i == n1 - 1 else r1_stores[i + 1]
            prev2 = depot_idx if j == 0 else r2_stores[j - 1]
            next2 = depot_idx if j == n2 - 1 else r2_stores[j + 1]
            
            delta = (
                - dist_matrix[prev1, s1] - dist_matrix[s1, next1]
                - dist_matrix[prev2, s2] - dist_matrix[s2, next2]
                + dist_matrix[prev1, s2] + dist_matrix[s2, next1]
                + dist_matrix[prev2, s1] + dist_matrix[s1, next2]
            )
            
            if delta < -1e-6:
                new_r1 = r1_stores.copy()
                new_r2 = r2_stores.copy()
                new_r1[i] = s2
                new_r2[j] = s1
                
                if _njit_check_time_constraint(new_r1, time_matrix, dwell_array, earliest_array, latest_array, sched_array, is_solomon, depot_idx) and \
                   _njit_check_time_constraint(new_r2, time_matrix, dwell_array, earliest_array, latest_array, sched_array, is_solomon, depot_idx):
                    if _njit_check_order_principle(r1_id_int, new_r1, order_match_matrix, order_rank_array) and \
                       _njit_check_order_principle(r2_id_int, new_r2, order_match_matrix, order_rank_array):
                        return True, new_r1, new_r2
                        
    return False, r1_stores, r2_stores

@njit(cache=True)
def _njit_cross_exchange(r1_id_int, r2_id_int, r1_stores, r2_stores, cap1, cap2,
                         dist_matrix, time_matrix, volume_array, dwell_array,
                         earliest_array, latest_array, sched_array, order_match_matrix, order_rank_array,
                         is_solomon, depot_idx):
    n1 = len(r1_stores)
    n2 = len(r2_stores)
    
    vol1_base = 0.0
    for s in r1_stores: vol1_base += volume_array[s]
    vol2_base = 0.0
    for s in r2_stores: vol2_base += volume_array[s]
    
    max_len = 3
    
    for i in range(n1):
        for l1 in range(1, max_len + 1):
            if i + l1 > n1:
                continue
            
            vol_seg1 = 0.0
            for k in range(i, i + l1):
                vol_seg1 += volume_array[r1_stores[k]]
                
            prev1 = depot_idx if i == 0 else r1_stores[i - 1]
            next1 = depot_idx if i + l1 == n1 else r1_stores[i + l1]
            start1 = r1_stores[i]
            end1 = r1_stores[i + l1 - 1]
            
            for j in range(n2):
                for l2 in range(1, max_len + 1):
                    if j + l2 > n2:
                        continue
                        
                    vol_seg2 = 0.0
                    for k in range(j, j + l2):
                        vol_seg2 += volume_array[r2_stores[k]]
                        
                    v1 = vol1_base - vol_seg1 + vol_seg2
                    v2 = vol2_base - vol_seg2 + vol_seg1
                    if v1 > cap1 or v2 > cap2:
                        continue
                        
                    prev2 = depot_idx if j == 0 else r2_stores[j - 1]
                    next2 = depot_idx if j + l2 == n2 else r2_stores[j + l2]
                    start2 = r2_stores[j]
                    end2 = r2_stores[j + l2 - 1]
                    
                    delta = (
                        - dist_matrix[prev1, start1] - dist_matrix[end1, next1]
                        - dist_matrix[prev2, start2] - dist_matrix[end2, next2]
                        + dist_matrix[prev1, start2] + dist_matrix[end2, next1]
                        + dist_matrix[prev2, start1] + dist_matrix[end1, next2]
                    )
                    
                    if delta < -1e-6:
                        new_r1 = np.empty(n1 - l1 + l2, dtype=r1_stores.dtype)
                        idx1 = 0
                        for k in range(i): new_r1[idx1] = r1_stores[k]; idx1 += 1
                        for k in range(j, j + l2): new_r1[idx1] = r2_stores[k]; idx1 += 1
                        for k in range(i + l1, n1): new_r1[idx1] = r1_stores[k]; idx1 += 1
                        
                        new_r2 = np.empty(n2 - l2 + l1, dtype=r2_stores.dtype)
                        idx2 = 0
                        for k in range(j): new_r2[idx2] = r2_stores[k]; idx2 += 1
                        for k in range(i, i + l1): new_r2[idx2] = r1_stores[k]; idx2 += 1
                        for k in range(j + l2, n2): new_r2[idx2] = r2_stores[k]; idx2 += 1
                        
                        if _njit_check_time_constraint(new_r1, time_matrix, dwell_array, earliest_array, latest_array, sched_array, is_solomon, depot_idx) and \
                           _njit_check_time_constraint(new_r2, time_matrix, dwell_array, earliest_array, latest_array, sched_array, is_solomon, depot_idx):
                            if _njit_check_order_principle(r1_id_int, new_r1, order_match_matrix, order_rank_array) and \
                               _njit_check_order_principle(r2_id_int, new_r2, order_match_matrix, order_rank_array):
                                return True, new_r1, new_r2
                                
    return False, r1_stores, r2_stores

@njit(cache=True)
def _njit_optimize(routes_list, route_ids_int, route_caps, strategy,
                   dist_matrix, time_matrix, volume_array, dwell_array,
                   earliest_array, latest_array, sched_array, order_match_matrix, order_rank_array,
                   is_solomon, depot_idx):
    improved = True
    num_routes = len(routes_list)
    
    while improved:
        improved = False
        
        # 1. Intra-route optimization
        if strategy == 0: # first
            for r_idx in range(num_routes):
                r_id_int = route_ids_int[r_idx]
                r_stores = routes_list[r_idx]
                if len(r_stores) == 0: continue
                cost = _njit_calculate_route_distance(r_stores, dist_matrix, depot_idx)
                r_improved, new_stores, new_cost = _njit_two_opt(r_id_int, r_stores, cost, dist_matrix, time_matrix, dwell_array, earliest_array, latest_array, sched_array, is_solomon, depot_idx, order_match_matrix, order_rank_array)
                if r_improved:
                    routes_list[r_idx] = new_stores
                    improved = True
                    break # Restart while loop
        else: # best
            best_r_idx = -1
            best_stores = routes_list[0] # dummy
            max_impr = 1e-6
            for r_idx in range(num_routes):
                r_id_int = route_ids_int[r_idx]
                r_stores = routes_list[r_idx]
                if len(r_stores) == 0: continue
                cost = _njit_calculate_route_distance(r_stores, dist_matrix, depot_idx)
                r_improved, new_stores, new_cost = _njit_two_opt_best(r_id_int, r_stores, cost, dist_matrix, time_matrix, dwell_array, earliest_array, latest_array, sched_array, is_solomon, depot_idx, order_match_matrix, order_rank_array)
                impr = cost - new_cost
                if r_improved and impr > max_impr:
                    max_impr = impr
                    best_r_idx = r_idx
                    best_stores = new_stores
            if best_r_idx != -1:
                routes_list[best_r_idx] = best_stores
                improved = True
                continue
                
        if improved: continue

        # 2. Inter-route optimization
        if strategy == 0: # first
            # relocate
            relocate_improved = False
            for i in range(num_routes):
                for j in range(num_routes):
                    if i == j: continue
                    impr_bool, new_r1, new_r2 = _njit_relocate(route_ids_int[i], route_ids_int[j], routes_list[i], routes_list[j], route_caps[j], dist_matrix, time_matrix, volume_array, dwell_array, earliest_array, latest_array, sched_array, order_match_matrix, order_rank_array, is_solomon, depot_idx)
                    if impr_bool:
                        routes_list[i] = new_r1
                        routes_list[j] = new_r2
                        improved = True
                        relocate_improved = True
                        break
                if relocate_improved: break
            if relocate_improved: continue

            # swap
            swap_improved = False
            for i in range(num_routes):
                for j in range(i + 1, num_routes):
                    impr_bool, new_r1, new_r2 = _njit_swap(route_ids_int[i], route_ids_int[j], routes_list[i], routes_list[j], route_caps[i], route_caps[j], dist_matrix, time_matrix, volume_array, dwell_array, earliest_array, latest_array, sched_array, order_match_matrix, order_rank_array, is_solomon, depot_idx)
                    if impr_bool:
                        routes_list[i] = new_r1
                        routes_list[j] = new_r2
                        improved = True
                        swap_improved = True
                        break
                if swap_improved: break
            if swap_improved: continue
            
        else: # best (first improvement sequentially)
            # relocate
            relocate_improved = False
            for i in range(num_routes):
                for j in range(i + 1, num_routes):
                    # i to j
                    impr_bool, new_r1, new_r2 = _njit_relocate(route_ids_int[i], route_ids_int[j], routes_list[i], routes_list[j], route_caps[j], dist_matrix, time_matrix, volume_array, dwell_array, earliest_array, latest_array, sched_array, order_match_matrix, order_rank_array, is_solomon, depot_idx)
                    if impr_bool:
                        routes_list[i] = new_r1
                        routes_list[j] = new_r2
                        improved = True
                        relocate_improved = True
                        break
                    # j to i
                    impr_bool, new_r2, new_r1 = _njit_relocate(route_ids_int[j], route_ids_int[i], routes_list[j], routes_list[i], route_caps[i], dist_matrix, time_matrix, volume_array, dwell_array, earliest_array, latest_array, sched_array, order_match_matrix, order_rank_array, is_solomon, depot_idx)
                    if impr_bool:
                        routes_list[j] = new_r2
                        routes_list[i] = new_r1
                        improved = True
                        relocate_improved = True
                        break
                if relocate_improved: break
            if relocate_improved: continue
            
            # swap
            swap_improved = False
            for i in range(num_routes):
                for j in range(i + 1, num_routes):
                    impr_bool, new_r1, new_r2 = _njit_swap(route_ids_int[i], route_ids_int[j], routes_list[i], routes_list[j], route_caps[i], route_caps[j], dist_matrix, time_matrix, volume_array, dwell_array, earliest_array, latest_array, sched_array, order_match_matrix, order_rank_array, is_solomon, depot_idx)
                    if impr_bool:
                        routes_list[i] = new_r1
                        routes_list[j] = new_r2
                        improved = True
                        swap_improved = True
                        break
                if swap_improved: break
            if swap_improved: continue
            
            # cross exchange
            cross_improved = False
            for i in range(num_routes):
                for j in range(i + 1, num_routes):
                    impr_bool, new_r1, new_r2 = _njit_cross_exchange(route_ids_int[i], route_ids_int[j], routes_list[i], routes_list[j], route_caps[i], route_caps[j], dist_matrix, time_matrix, volume_array, dwell_array, earliest_array, latest_array, sched_array, order_match_matrix, order_rank_array, is_solomon, depot_idx)
                    if impr_bool:
                        routes_list[i] = new_r1
                        routes_list[j] = new_r2
                        improved = True
                        cross_improved = True
                        break
                if cross_improved: break
            if cross_improved: continue

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

    def _calculate_routes_cost(self, routes):
        """
        Notes:
            Calculate the total cost of all routes.
        """
        total_cost = 0
        num_vehicles = 0
        
        # Build local maps if we just want a quick cost based on route manager dicts
        s2i = {self.dc['store_id']: 0}
        i2s = {0: self.dc}
        idx = 1
        for route_id in routes:
            for s in routes[route_id]['stores']:
                if s['store_id'] not in s2i:
                    s2i[s['store_id']] = idx
                    i2s[idx] = s
                    idx += 1
                    
        n = len(s2i)
        dist_matrix = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            s_i = i2s[i]['store_id']
            for j in range(n):
                s_j = i2s[j]['store_id']
                if s_i in self.distance_matrix and s_j in self.distance_matrix[s_i]:
                    dist_matrix[i, j] = self.distance_matrix[s_i][s_j]
                    
        for route_id in routes:
            stores = routes[route_id]['stores']
            if not stores: continue
            
            s_arr = np.array([s2i[s['store_id']] for s in stores], dtype=np.int64)
            cost = _njit_calculate_route_distance(s_arr, dist_matrix, 0)
            total_cost += cost

            if route_id.startswith('1'):
                num_vehicles += 1

        if self.is_solomon:
            return (num_vehicles, total_cost)
        return total_cost + num_vehicles * self.vehicle_cost

    def optimize(self, routes_info):
        current_routes = copy.deepcopy(routes_info)
        route_manager = RouteManager(current_routes, self.distance_matrix, self.time_matrix)

        # 1. Map Stores
        s2i = {self.dc['store_id']: 0}
        i2s = {0: self.dc}
        store_idx = 1
        for route in current_routes.values():
            for s in route['stores']:
                s_id = s['store_id']
                if s_id not in s2i:
                    s2i[s_id] = store_idx
                    i2s[store_idx] = s
                    store_idx += 1
        num_stores = store_idx

        # 2. Map Routes
        route_keys = list(current_routes.keys())
        r2i = {r_id: idx for idx, r_id in enumerate(route_keys)}
        i2r = {idx: r_id for idx, r_id in enumerate(route_keys)}
        num_routes = len(route_keys)

        if num_routes == 0 or num_stores <= 1:
            return current_routes, self._calculate_routes_cost(current_routes)

        # 3. Create Numba Arrays
        np_dist = np.zeros((num_stores, num_stores), dtype=np.float64)
        np_time = np.zeros((num_stores, num_stores), dtype=np.float64)
        np_vol = np.zeros(num_stores, dtype=np.float64)
        np_dwell = np.zeros(num_stores, dtype=np.float64)
        np_earliest = np.zeros(num_stores, dtype=np.float64)
        np_latest = np.zeros(num_stores, dtype=np.float64)
        np_sched = np.zeros(num_stores, dtype=np.float64)
        np_route_match = np.zeros((num_stores, num_routes), dtype=np.bool_)
        np_route_rank = np.zeros(num_stores, dtype=np.int64)
        route_caps = np.zeros(num_routes, dtype=np.float64)
        route_ids_int = np.arange(num_routes, dtype=np.int64)

        for i in range(num_stores):
            s_id = i2s[i]['store_id']
            for j in range(num_stores):
                s_j = i2s[j]['store_id']
                if s_id in self.distance_matrix and s_j in self.distance_matrix[s_id]:
                    np_dist[i, j] = self.distance_matrix[s_id][s_j]
                if s_id in self.time_matrix and s_j in self.time_matrix[s_id]:
                    np_time[i, j] = self.time_matrix[s_id][s_j]

            if i > 0:
                s = i2s[i]
                np_vol[i] = s.get('volume', 0.0)
                np_dwell[i] = s.get('dwell_time', 0.0)
                
                if 'earliest_time' in s and s['earliest_time']:
                    np_earliest[i] = datetime.fromisoformat(s['earliest_time']).timestamp()
                if 'latest_time' in s and s['latest_time']:
                    np_latest[i] = datetime.fromisoformat(s['latest_time']).timestamp()
                if 'sched_time' in s and s['sched_time']:
                    np_sched[i] = datetime.fromisoformat(s['sched_time']).timestamp()
                    
                if 'route_code' in s:
                    rc = s['route_code']
                    for r_idx in range(num_routes):
                        if rc.startswith(i2r[r_idx]):
                            np_route_match[i, r_idx] = True

        unique_suffixes = sorted(list(set(s['route_code'][2:] for s in i2s.values() if 'route_code' in s)))
        suffix_to_rank = {suf: idx for idx, suf in enumerate(unique_suffixes)}
        for i in range(1, num_stores):
            s = i2s[i]
            if 'route_code' in s:
                np_route_rank[i] = suffix_to_rank.get(s['route_code'][2:], -1)

        routes_list = List()
        for r_id in route_keys:
            r_idx = r2i[r_id]
            route_caps[r_idx] = current_routes[r_id]['dc']['max_capacity']
            s_arr = np.array([s2i[s['store_id']] for s in current_routes[r_id]['stores']], dtype=np.int64)
            routes_list.append(s_arr)

        strategy_int = 0 if self.improvement_strategy == 'first' else 1
        
        # 4. Execute Numba JIT
        _njit_optimize(routes_list, route_ids_int, route_caps, strategy_int,
                       np_dist, np_time, np_vol, np_dwell, np_earliest, np_latest, np_sched,
                       np_route_match, np_route_rank, self.is_solomon, 0)
        
        # 5. Rebuild routes
        for r_idx, r_id in enumerate(route_keys):
            opt_stores = routes_list[r_idx]
            new_store_list = [i2s[idx] for idx in opt_stores]
            route_manager.replace_stores(r_id, new_store_list)

        route_manager.update_all_routes_info()
        final_routes = route_manager.routes_info
        final_cost = self._calculate_routes_cost(final_routes)

        return final_routes, final_cost

# Numba pre-compile warmup
try:
    _njit_calculate_route_distance(np.zeros(0, dtype=np.int64), np.zeros((1,1)), 0)
    _njit_check_capacity_constraint(np.zeros(0, dtype=np.int64), 0, np.zeros(1), 10.0)
except Exception:
    pass