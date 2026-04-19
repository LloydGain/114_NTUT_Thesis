import copy
import numpy as np
from numba import njit
from config import config
from datetime import datetime, timedelta
from models.route_manager import RouteManager

@njit(cache=True)
def _njit_route_distance(path, dist_matrix):
    total = 0.0
    for k in range(len(path) - 1):
        total += dist_matrix[path[k], path[k + 1]]
    return total

@njit(cache=True)
def _njit_relocate_delta(r1_path, r2_path, idx, idy, dist_matrix):
    s_id = r1_path[idx]
    prev_i = r1_path[idx - 1]
    next_i = r1_path[idx + 1]
    prev_j = r2_path[idy - 1]
    next_j = r2_path[idy]

    delta_remove = (dist_matrix[prev_i, next_i]
                    - dist_matrix[prev_i, s_id]
                    - dist_matrix[s_id, next_i])
    delta_insert = (dist_matrix[prev_j, s_id]
                    + dist_matrix[s_id, next_j]
                    - dist_matrix[prev_j, next_j])
    return delta_remove + delta_insert

@njit(cache=True)
def _njit_swap_delta(r1_path, r2_path, i, j, dist_matrix):
    s1_id = r1_path[i]
    prev1 = r1_path[i - 1]
    next1 = r1_path[i + 1]
    s2_id = r2_path[j]
    prev2 = r2_path[j - 1]
    next2 = r2_path[j + 1]

    return (- dist_matrix[prev1, s1_id] - dist_matrix[s1_id, next1]
            - dist_matrix[prev2, s2_id] - dist_matrix[s2_id, next2]
            + dist_matrix[prev1, s2_id] + dist_matrix[s2_id, next1]
            + dist_matrix[prev2, s1_id] + dist_matrix[s1_id, next2])

@njit(cache=True)
def _njit_cross_exchange_delta(r1_path, r2_path, i, l1, j, l2, dist_matrix):
    prev1 = r1_path[i - 1]
    start1 = r1_path[i]
    end1 = r1_path[i + l1 - 1]
    next1 = r1_path[i + l1]

    prev2 = r2_path[j - 1]
    start2 = r2_path[j]
    end2 = r2_path[j + l2 - 1]
    next2 = r2_path[j + l2]

    return (- dist_matrix[prev1, start1] - dist_matrix[end1, next1]
            - dist_matrix[prev2, start2] - dist_matrix[end2, next2]
            + dist_matrix[prev1, start2] + dist_matrix[end2, next1]
            + dist_matrix[prev2, start1] + dist_matrix[end1, next2])


def _warmup_njit():
    try:
        dm = np.zeros((3, 3), dtype=np.float64)
        p2 = np.array([0, 1, 2, 0], dtype=np.int64)
        _njit_route_distance(p2, dm)
        _njit_relocate_delta(p2, p2, 1, 1, dm)
        _njit_swap_delta(p2, p2, 1, 1, dm)
        _njit_cross_exchange_delta(p2, p2, 1, 1, 1, 1, dm)
    except Exception:
        pass

_warmup_njit()


class VND:
    def __init__(self, distance_matrix, time_matrix, vehicle_cost=2000,
                 is_solomon=False, vnd_strategy='best'):
        self.dc = config.DC_CONFIG
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix
        self.vehicle_cost = vehicle_cost
        self.is_solomon = is_solomon
        self.vnd_strategy = vnd_strategy

        self._np_dist = None
        self._np_time = None
        self._s2i = {}
        self._i2s = {}
        self._np_dwell = None
        self._np_earliest = None
        self._np_latest = None

    def _ensure_np_matrices(self, all_store_ids):
        dc_id = self.dc['store_id']
        needed = [sid for sid in all_store_ids if sid not in self._s2i]
        if not needed and self._np_dist is not None:
            return

        ids = [dc_id] + [sid for sid in all_store_ids if sid != dc_id]
        self._s2i = {sid: i for i, sid in enumerate(ids)}
        self._i2s = {i: sid for i, sid in enumerate(ids)}
        n = len(ids)

        self._np_dist = np.zeros((n, n), dtype=np.float64)
        self._np_time = np.zeros((n, n), dtype=np.float64)

        for i, si in enumerate(ids):
            row_d = self.distance_matrix.get(si, {})
            row_t = self.time_matrix.get(si, {})
            for j, sj in enumerate(ids):
                self._np_dist[i, j] = row_d.get(sj, 0.0)
                self._np_time[i, j] = row_t.get(sj, 0.0)

    def _stores_to_path(self, stores, include_trailing_dc=True):
        dc_idx = self._s2i[self.dc['store_id']]
        indices = [dc_idx] + [self._s2i[s['store_id']] for s in stores]
        if include_trailing_dc:
            indices.append(dc_idx)
        return np.array(indices, dtype=np.int64)

    def _calculate_route_distance(self, stores):
        if not stores:
            return 0.0
        path = self._stores_to_path(stores, include_trailing_dc=True)
        return float(_njit_route_distance(path, self._np_dist))

    def _calculate_routes_cost(self, routes):
        total_cost = 0.0
        num_vehicles = 0
        for route_id, route_data in routes.items():
            total_cost += self._calculate_route_distance(route_data['stores'])
            if route_id.startswith('1'):
                num_vehicles += 1
        if self.is_solomon:
            return (num_vehicles, total_cost)
        return total_cost + num_vehicles * self.vehicle_cost


    def _check_time_constraint(self, stores):
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
                if not (cur_earliest_time <= cur_time <= cur_lastest_time):
                    return False

            cur_dwell = cur_store['dwell_time']
            cur_time = cur_time + timedelta(seconds=cur_dwell)
            prev_store = cur_store

        return True

    def _check_capacity_constraint(self, route, store):
        cap = route['dc']['max_capacity']
        total = sum(s['volume'] for s in route['stores']) + store['volume']
        return total <= cap

    def _check_order_principle(self, route_id, stores):
        codes = [s['route_code'][2:] for s in stores if s['route_code'].startswith(route_id)]
        return codes == sorted(codes)

    def _two_opt(self, stores, cost):
        if len(stores) < 2: return stores, cost
        n = len(stores)
        best_stores, best_dist = stores, cost
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_stores = stores[:]
                new_stores[i:j + 1] = list(reversed(new_stores[i:j + 1]))
                new_dist = self._calculate_route_distance(new_stores)
                if new_dist < best_dist:
                    if self._check_time_constraint(new_stores):
                        best_stores, best_dist = new_stores, new_dist
        return best_stores, best_dist

    def _two_opt_best(self, stores, cost):
        n = len(stores)
        best_stores, best_dist = stores, cost
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_stores = stores[:]
                new_stores[i:j + 1] = list(reversed(new_stores[i:j + 1]))
                new_dist = self._calculate_route_distance(new_stores)
                if best_dist - new_dist > 1e-6:
                    if self._check_time_constraint(new_stores):
                        best_stores, best_dist = new_stores, new_dist
        return best_stores, best_dist


    def _relocate(self, routes, route1_id, route2_id):
        r1_stores, r2_stores, r2 = routes[route1_id]['stores'], routes[route2_id]['stores'], routes[route2_id]
        dc_idx = self._s2i[self.dc['store_id']]
        for idx, r1_store in enumerate(r1_stores):
            if not self._check_capacity_constraint(r2, r1_store): continue
            r1_path = np.empty(len(r1_stores) + 2, dtype=np.int64)
            r1_path[0] = r1_path[-1] = dc_idx
            for k, s in enumerate(r1_stores): r1_path[k + 1] = self._s2i[s['store_id']]
            r2_path = np.empty(len(r2_stores) + 2, dtype=np.int64)
            r2_path[0] = r2_path[-1] = dc_idx
            for k, s in enumerate(r2_stores): r2_path[k + 1] = self._s2i[s['store_id']]
            r1_inner = idx + 1
            for idy in range(len(r2_stores) + 1):
                r2_inner = idy + 1
                delta = _njit_relocate_delta(r1_path, r2_path, r1_inner, r2_inner, self._np_dist)
                if delta >= -1e-6: continue
                new_r1, new_r2 = r1_stores[:idx] + r1_stores[idx+1:], r2_stores[:idy] + [r1_store] + r2_stores[idy:]
                if (self._check_time_constraint(new_r1) and self._check_time_constraint(new_r2) and self._check_order_principle(route2_id, new_r2)):
                    return r1_store, idy
        return None, -1

    def _relocate_first(self, routes, route1_id, route2_id):
        r1_stores, r2_stores, r2 = routes[route1_id]['stores'], routes[route2_id]['stores'], routes[route2_id]
        dc_idx = self._s2i[self.dc['store_id']]
        r1_path = np.empty(len(r1_stores) + 2, dtype=np.int64)
        r1_path[0] = r1_path[-1] = dc_idx
        for k, s in enumerate(r1_stores): r1_path[k + 1] = self._s2i[s['store_id']]
        r2_path = np.empty(len(r2_stores) + 2, dtype=np.int64)
        r2_path[0] = r2_path[-1] = dc_idx
        for k, s in enumerate(r2_stores): r2_path[k + 1] = self._s2i[s['store_id']]
        for idx, r1_store in enumerate(r1_stores):
            if not self._check_capacity_constraint(r2, r1_store): continue
            r1_inner = idx + 1
            for idy in range(len(r2_stores) + 1):
                r2_inner = idy + 1
                delta = _njit_relocate_delta(r1_path, r2_path, r1_inner, r2_inner, self._np_dist)
                improvement = -delta
                if improvement <= 1e-6: continue
                new_r1, new_r2 = r1_stores[:idx] + r1_stores[idx+1:], r2_stores[:idy] + [r1_store] + r2_stores[idy:]
                if (self._check_time_constraint(new_r1) and self._check_time_constraint(new_r2) and self._check_order_principle(route2_id, new_r2)):
                    return r1_store, idy, improvement
        return None, -1, 0.0


    def _swap(self, routes, route1_id, route2_id):
        r1, r2 = routes[route1_id], routes[route2_id]
        r1_stores, r2_stores = r1['stores'], r2['stores']
        cap1, cap2 = r1['dc']['max_capacity'], r2['dc']['max_capacity']
        vol1_total, vol2_total = sum(s['volume'] for s in r1_stores), sum(s['volume'] for s in r2_stores)
        dc_idx = self._s2i[self.dc['store_id']]
        r1_path = np.empty(len(r1_stores) + 2, dtype=np.int64)
        r1_path[0] = r1_path[-1] = dc_idx
        for k, s in enumerate(r1_stores): r1_path[k + 1] = self._s2i[s['store_id']]
        r2_path = np.empty(len(r2_stores) + 2, dtype=np.int64)
        r2_path[0] = r2_path[-1] = dc_idx
        for k, s in enumerate(r2_stores): r2_path[k + 1] = self._s2i[s['store_id']]
        for i, s1 in enumerate(r1_stores):
            for j, s2 in enumerate(r2_stores):
                if (vol1_total - s1['volume'] + s2['volume'] > cap1 or vol2_total - s2['volume'] + s1['volume'] > cap2): continue
                delta = _njit_swap_delta(r1_path, r2_path, i + 1, j + 1, self._np_dist)
                if delta >= -1e-6: continue
                new_r1, new_r2 = r1_stores[:], r2_stores[:]
                new_r1[i], new_r2[j] = s2, s1
                if (self._check_time_constraint(new_r1) and self._check_time_constraint(new_r2) and self._check_order_principle(route1_id, new_r1) and self._check_order_principle(route2_id, new_r2)):
                    return s1, s2
        return None, None

    def _swap_first(self, routes, route1_id, route2_id):
        r1, r2 = routes[route1_id], routes[route2_id]
        r1_stores, r2_stores = r1['stores'], r2['stores']
        cap1, cap2 = r1['dc']['max_capacity'], r2['dc']['max_capacity']
        vol1_total, vol2_total = sum(s['volume'] for s in r1_stores), sum(s['volume'] for s in r2_stores)
        dc_idx = self._s2i[self.dc['store_id']]
        r1_path = np.empty(len(r1_stores) + 2, dtype=np.int64)
        r1_path[0] = r1_path[-1] = dc_idx
        for k, s in enumerate(r1_stores): r1_path[k + 1] = self._s2i[s['store_id']]
        r2_path = np.empty(len(r2_stores) + 2, dtype=np.int64)
        r2_path[0] = r2_path[-1] = dc_idx
        for k, s in enumerate(r2_stores): r2_path[k + 1] = self._s2i[s['store_id']]
        for i, s1 in enumerate(r1_stores):
            for j, s2 in enumerate(r2_stores):
                if (vol1_total - s1['volume'] + s2['volume'] > cap1 or vol2_total - s2['volume'] + s1['volume'] > cap2): continue
                delta = _njit_swap_delta(r1_path, r2_path, i + 1, j + 1, self._np_dist)
                improvement = -delta
                if improvement <= 1e-6: continue
                new_r1, new_r2 = r1_stores[:], r2_stores[:]
                new_r1[i], new_r2[j] = s2, s1
                if (self._check_time_constraint(new_r1) and self._check_time_constraint(new_r2) and self._check_order_principle(route1_id, new_r1) and self._check_order_principle(route2_id, new_r2)):
                    return s1, s2, improvement
        return None, None, 0.0

    def _cross_exchange_first(self, routes, route1_id, route2_id):
        r1_stores, r2_stores = routes[route1_id]['stores'], routes[route2_id]['stores']
        cap1, cap2 = routes[route1_id]['dc']['max_capacity'], routes[route2_id]['dc']['max_capacity']
        vol1_total, vol2_total = sum(s['volume'] for s in r1_stores), sum(s['volume'] for s in r2_stores)
        dc_idx, max_len = self._s2i[self.dc['store_id']], 3
        r1_path = np.empty(len(r1_stores) + 2, dtype=np.int64)
        r1_path[0] = r1_path[-1] = dc_idx
        for k, s in enumerate(r1_stores): r1_path[k + 1] = self._s2i[s['store_id']]
        r2_path = np.empty(len(r2_stores) + 2, dtype=np.int64)
        r2_path[0] = r2_path[-1] = dc_idx
        for k, s in enumerate(r2_stores): r2_path[k + 1] = self._s2i[s['store_id']]
        for i in range(len(r1_stores)):
            for l1 in range(1, max_len + 1):
                if i + l1 > len(r1_stores): continue
                seg1, vol_seg1 = r1_stores[i:i + l1], sum(s['volume'] for s in r1_stores[i:i + l1])
                for j in range(len(r2_stores)):
                    for l2 in range(1, max_len + 1):
                        if j + l2 > len(r2_stores): continue
                        seg2, vol_seg2 = r2_stores[j:j + l2], sum(s['volume'] for s in r2_stores[j:j + l2])
                        if (vol1_total - vol_seg1 + vol_seg2 > cap1 or vol2_total - vol_seg2 + vol_seg1 > cap2): continue
                        delta = _njit_cross_exchange_delta(r1_path, r2_path, i + 1, l1, j + 1, l2, self._np_dist)
                        improvement = -delta
                        if improvement <= 1e-6: continue
                        new_r1, new_r2 = r1_stores[:i] + seg2 + r1_stores[i + l1:], r2_stores[:j] + seg1 + r2_stores[j + l2:]
                        if (self._check_time_constraint(new_r1) and self._check_time_constraint(new_r2) and self._check_order_principle(route1_id, new_r1) and self._check_order_principle(route2_id, new_r2)):
                            return new_r1, new_r2, improvement
        return None, None, 0.0

    def _neighborhood_intra(self, routes, route_manager):
        for route_id, route_data in routes.items():
            original_dist = self._calculate_route_distance(route_data['stores'])
            new_stores, new_dist = self._two_opt(route_data['stores'], original_dist)
            if new_dist < original_dist - 1e-6:
                route_manager.replace_stores(route_id, new_stores)
                return True
        return False

    def _neighborhood_inter(self, routes, route_manager):
        route_ids = list(routes.keys())
        for r1_id in route_ids:
            for r2_id in route_ids:
                if r1_id == r2_id: continue
                moved_store, position = self._relocate(routes, r1_id, r2_id)
                if moved_store is not None:
                    route_manager.move_store_to_route(r1_id, moved_store, r2_id, position)
                    return True
        visited_pairs = set()
        for i in range(len(route_ids)):
            for j in range(i + 1, len(route_ids)):
                r1_id, r2_id = route_ids[i], route_ids[j]
                if (r1_id, r2_id) in visited_pairs: continue
                visited_pairs.add((r1_id, r2_id))
                s1, s2 = self._swap(routes, r1_id, r2_id)
                if s1 is not None:
                    route_manager.swap_stores(r1_id, s1, r2_id, s2)
                    return True
        return False

    def _neighborhood_intra_best(self, routes, route_manager):
        best_route_id, best_new_stores, max_improvement = None, None, 1e-6
        for route_id, route_data in routes.items():
            original_dist = self._calculate_route_distance(route_data['stores'])
            new_stores, new_dist = self._two_opt_best(route_data['stores'], original_dist)
            improvement = original_dist - new_dist
            if improvement > max_improvement:
                max_improvement, best_route_id, best_new_stores = improvement, route_id, new_stores
        if best_route_id is not None:
            route_manager.replace_stores(best_route_id, best_new_stores)
            return True
        return False

    def _neighborhood_inter_best(self, routes, route_manager):
        route_ids = list(routes.keys())
        for i in range(len(route_ids)):
            for j in range(i + 1, len(route_ids)):
                r1_id, r2_id = route_ids[i], route_ids[j]
                for ra, rb in [(r1_id, r2_id), (r2_id, r1_id)]:
                    moved_store, position, impr = self._relocate_first(routes, ra, rb)
                    if impr > 1e-6:
                        route_manager.move_store_to_route(ra, moved_store, rb, position)
                        return True
        for i in range(len(route_ids)):
            for j in range(i + 1, len(route_ids)):
                r1_id, r2_id = route_ids[i], route_ids[j]
                s1, s2, impr = self._swap_first(routes, r1_id, r2_id)
                if impr > 1e-6:
                    route_manager.swap_stores(r1_id, s1, r2_id, s2)
                    return True
        for i in range(len(route_ids)):
            for j in range(i + 1, len(route_ids)):
                r1_id, r2_id = route_ids[i], route_ids[j]
                new_r1, new_r2, impr = self._cross_exchange_first(routes, r1_id, r2_id)
                if impr > 1e-6:
                    route_manager.replace_stores(r1_id, new_r1)
                    route_manager.replace_stores(r2_id, new_r2)
                    return True
        return False

    def optimize(self, routes_info):
        current_routes = copy.deepcopy(routes_info)
        all_ids = [s['store_id'] for rd in current_routes.values() for s in rd['stores']]
        self._ensure_np_matrices(all_ids)
        route_manager = RouteManager(current_routes, self.distance_matrix, self.time_matrix)
        improved = True
        while improved:
            improved = False
            if self.vnd_strategy == 'first':
                if self._neighborhood_intra(current_routes, route_manager):
                    improved = True
                    current_routes = route_manager.routes_info
                    continue
                if self._neighborhood_inter(current_routes, route_manager):
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