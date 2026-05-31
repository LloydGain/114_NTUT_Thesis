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
                 is_solomon=False, improvement_strategy='best', vnd_strategy=None, time_limit=1e12, verbose=False):
        self.dc = config.DC_CONFIG
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix
        self.vehicle_cost = vehicle_cost
        self.is_solomon = is_solomon
        self.improvement_strategy = vnd_strategy if vnd_strategy is not None else improvement_strategy
        self.time_limit = time_limit
        self.verbose = verbose

        self._np_dist = None
        self._np_time = None
        self._s2i = {}
        self._i2s = {}

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
            if route_id.startswith('1') or route_id.startswith('V'): # Support both '1' and 'V' prefix
                num_vehicles += 1
        if self.is_solomon:
            return (num_vehicles, total_cost)
        return total_cost + num_vehicles * self.vehicle_cost

    def _check_time_constraint(self, stores):
        if len(stores) == 0:
            return True

        if self.is_solomon:
            # Solomon logic: start at depot time=0 (minutes)
            # Determine base_dt as midnight of the planning day
            if isinstance(stores[0].get('earliest_time'), str):
                base_dt = datetime.fromisoformat(stores[0]['earliest_time']).replace(
                    hour=0, minute=0, second=0, microsecond=0)
            else:
                base_dt = stores[0]['earliest_time'].replace(
                    hour=0, minute=0, second=0, microsecond=0)

            depot_id = self.dc['store_id']

            # Depot ready_time / due_time (if provided in DC_CONFIG)
            depot_ready = base_dt + timedelta(minutes=self.dc.get('earliest_time', 0))
            depot_due   = base_dt + timedelta(minutes=self.dc.get('latest_time', 1e9))

            # Vehicle departs depot no earlier than depot ready_time
            cur_time = depot_ready

            for i, curr in enumerate(stores):
                prev_id = depot_id if i == 0 else stores[i - 1]['store_id']
                travel_time = self.time_matrix[prev_id][curr['store_id']]

                # Add previous node's service (dwell) time before travelling
                if i > 0:
                    cur_time = cur_time + timedelta(minutes=stores[i - 1]['dwell_time'])

                # Arrive at current store
                cur_time = cur_time + timedelta(minutes=travel_time)

                # Parse time window
                if isinstance(curr['latest_time'], str):
                    cur_latest   = datetime.fromisoformat(curr['latest_time'])
                    cur_earliest = datetime.fromisoformat(curr['earliest_time'])
                else:
                    cur_latest   = curr['latest_time']
                    cur_earliest = curr['earliest_time']

                # Hard constraint: must arrive before latest_time
                if cur_time > cur_latest:
                    return False

                # Wait if arrived before time window opens
                if cur_time < cur_earliest:
                    cur_time = cur_earliest

            # Service last store, then return to depot
            last_s = stores[-1]
            cur_time = cur_time + timedelta(minutes=last_s['dwell_time'])
            return_travel = self.time_matrix[last_s['store_id']][depot_id]
            cur_time = cur_time + timedelta(minutes=return_travel)

            # Depot due_time check (vehicle must be back before depot closes)
            if cur_time > depot_due:
                return False

            # Optional global time-limit check
            total_duration_mins = (cur_time - depot_ready).total_seconds() / 60
            if total_duration_mins > getattr(self, 'time_limit', 1e12):
                return False

            return True
        else:
            # Original non-Solomon logic: start from first store's sched_time, seconds
            prev_store = stores[0]
            if isinstance(prev_store.get('sched_time', prev_store.get('pred_time', prev_store['earliest_time'])), str):
                start_time = datetime.fromisoformat(prev_store.get('sched_time', prev_store.get('pred_time', prev_store['earliest_time'])))
            else:
                start_time = prev_store.get('sched_time', prev_store.get('pred_time', prev_store['earliest_time']))
            
            cur_time = start_time + timedelta(seconds=prev_store['dwell_time'])

            for cur_store in stores[1:]:
                if isinstance(cur_store['earliest_time'], str):
                    cur_earliest_time = datetime.fromisoformat(cur_store['earliest_time'])
                    cur_latest_time = datetime.fromisoformat(cur_store['latest_time'])
                else:
                    cur_earliest_time = cur_store['earliest_time']
                    cur_latest_time = cur_store['latest_time']
                    
                travel_time = self.time_matrix[prev_store['store_id']][cur_store['store_id']]
                cur_time = cur_time + timedelta(seconds=travel_time)

                if not (cur_earliest_time <= cur_time <= cur_latest_time):
                    return False

                cur_dwell = cur_store['dwell_time']
                cur_time = cur_time + timedelta(seconds=cur_dwell)
                prev_store = cur_store
                
            return_travel = self.time_matrix[stores[-1]['store_id']][self.dc['store_id']]
            end_time = cur_time + timedelta(seconds=return_travel)
            
            depot_to_first = self.time_matrix[self.dc['store_id']][stores[0]['store_id']]
            true_start = start_time - timedelta(seconds=depot_to_first)
            
            total_duration = (end_time - true_start).total_seconds()
            if total_duration > getattr(self, 'time_limit', 1e12):
                return False
                
            return True

    def _check_capacity_constraint(self, route, store):
        cap = route['dc']['max_capacity']
        total = sum(s['volume'] for s in route['stores']) + store['volume']
        return total <= cap

    def _check_region_constraint(self, stores):
        """Check that no two consecutive stores are in directly opposing regions.
        e.g. north->east->south is OK, but north->south is NOT OK."""
        if self.is_solomon: return True
        OPPOSITES = {'north': 'south', 'south': 'north', 'east': 'west', 'west': 'east'}
        for prev_s, curr_s in zip(stores[:-1], stores[1:]):
            prev_r = prev_s.get('region', '').lower()
            curr_r = curr_s.get('region', '').lower()
            if prev_r and curr_r and OPPOSITES.get(prev_r) == curr_r:
                return False
        return True

    def _is_movable(self, store, route_id):
        """Returns True if the store does NOT belong to route_id (i.e., can be moved).
        A store 'belongs' to a route if its route_code prefix matches the route_id.
        In Solomon mode all stores are considered movable."""
        if self.is_solomon: return True
        rc = store.get('route_code', '')
        prefix = rc[:2] if rc and len(rc) >= 2 else ''
        return not (prefix and route_id.startswith(prefix))

    def _check_order_principle(self, route_id, stores):
        if self.is_solomon:
            return True
        codes = [s['route_code'][2:] for s in stores if s['route_code'].startswith(route_id)]
        return codes == sorted(codes)

    def _two_opt_first(self, stores, cost, movable_stores=None):
        n = len(stores)
        for i in range(n - 1):
            for j in range(i + 1, n):
                if movable_stores is not None and any(s["store_id"] not in movable_stores for s in stores[i:j+1]): continue
                new_stores = stores[:]
                new_stores[i:j + 1] = list(reversed(new_stores[i:j + 1]))
                new_dist = self._calculate_route_distance(new_stores)
                if cost - new_dist > 1e-6:
                    if self._check_time_constraint(new_stores):
                        return new_stores, new_dist
        return stores, cost

    def _two_opt_best(self, stores, cost, movable_stores=None):
        n = len(stores)
        best_stores, best_dist = stores, cost
        for i in range(n - 1):
            for j in range(i + 1, n):
                if movable_stores is not None and any(s["store_id"] not in movable_stores for s in stores[i:j+1]): continue
                new_stores = stores[:]
                new_stores[i:j + 1] = list(reversed(new_stores[i:j + 1]))
                new_dist = self._calculate_route_distance(new_stores)
                if best_dist - new_dist > 1e-6:
                    if self._check_time_constraint(new_stores):
                        best_stores, best_dist = new_stores, new_dist
        return best_stores, best_dist

    def _intra_route_relocate_first(self, r_stores, original_dist, movable_stores=None):
        n = len(r_stores)
        if n <= 1:
            return r_stores, original_dist

        for i in range(n):
            node = r_stores[i]
            if movable_stores is not None and node["store_id"] not in movable_stores: continue
            tmp = r_stores[:i] + r_stores[i+1:]

            for j in range(n):
                if j == i or j == i + 1:
                    continue

                new_r = tmp[:j] + [node] + tmp[j:]
                new_dist = self._calculate_route_distance(new_r)
                improvement = original_dist - new_dist

                if improvement > 1e-6 and self._check_time_constraint(new_r):
                    return new_r, new_dist

        return r_stores, original_dist

    def _intra_route_relocate_best(self, r_stores, original_dist, movable_stores=None):
        n = len(r_stores)
        if n <= 1:
            return r_stores, original_dist
        best_r = r_stores
        best_dist = original_dist
        best_improvement = 0.0

        for i in range(n):
            node = r_stores[i]
            if movable_stores is not None and node["store_id"] not in movable_stores: continue
            tmp = r_stores[:i] + r_stores[i+1:]

            for j in range(n):
                if j == i or j == i + 1:
                    continue

                new_r = tmp[:j] + [node] + tmp[j:]
                new_dist = self._calculate_route_distance(new_r)
                improvement = original_dist - new_dist

                if improvement > best_improvement and improvement > 1e-6:
                    if self._check_time_constraint(new_r):
                        best_r, best_dist = new_r, new_dist
                        best_improvement = improvement

        return best_r, best_dist

    def _relocate_first(self, routes, route1_id, route2_id, movable_stores=None):
        r1_stores, r2_stores, r2 = routes[route1_id]['stores'], routes[route2_id]['stores'], routes[route2_id]
        dc_idx = self._s2i[self.dc['store_id']]
        r1_path = np.empty(len(r1_stores) + 2, dtype=np.int64)
        r1_path[0] = r1_path[-1] = dc_idx
        for k, s in enumerate(r1_stores): r1_path[k + 1] = self._s2i[s['store_id']]
        r2_path = np.empty(len(r2_stores) + 2, dtype=np.int64)
        r2_path[0] = r2_path[-1] = dc_idx
        for k, s in enumerate(r2_stores): r2_path[k + 1] = self._s2i[s['store_id']]
        
        for idx, r1_store in enumerate(r1_stores):
            if movable_stores is not None and r1_store["store_id"] not in movable_stores: continue
            if not self._is_movable(r1_store, route1_id): continue
            if not self._check_capacity_constraint(r2, r1_store): continue
            r1_inner = idx + 1
            for idy in range(len(r2_stores) + 1):
                r2_inner = idy + 1
                delta = _njit_relocate_delta(r1_path, r2_path, r1_inner, r2_inner, self._np_dist)
                improvement = -delta
                if improvement <= 1e-6: continue
                new_r1 = r1_stores[:idx] + r1_stores[idx+1:]
                new_r2 = r2_stores[:idy] + [r1_store] + r2_stores[idy:]
                if (self._check_time_constraint(new_r1) and self._check_time_constraint(new_r2) and self._check_order_principle(route2_id, new_r2)):
                    return r1_store, idy, improvement
        return None, -1, 0.0

    def _relocate_best(self, routes, route1_id, route2_id, movable_stores=None):
        r1_stores, r2_stores, r2 = routes[route1_id]['stores'], routes[route2_id]['stores'], routes[route2_id]
        dc_idx = self._s2i[self.dc['store_id']]
        r1_path = np.empty(len(r1_stores) + 2, dtype=np.int64)
        r1_path[0] = r1_path[-1] = dc_idx
        for k, s in enumerate(r1_stores): r1_path[k + 1] = self._s2i[s['store_id']]
        r2_path = np.empty(len(r2_stores) + 2, dtype=np.int64)
        r2_path[0] = r2_path[-1] = dc_idx
        for k, s in enumerate(r2_stores): r2_path[k + 1] = self._s2i[s['store_id']]
        
        best_store, best_pos, best_improvement = None, -1, 0.0
        for idx, r1_store in enumerate(r1_stores):
            if movable_stores is not None and r1_store["store_id"] not in movable_stores: continue
            if not self._is_movable(r1_store, route1_id): continue
            if not self._check_capacity_constraint(r2, r1_store): continue
            r1_inner = idx + 1
            for idy in range(len(r2_stores) + 1):
                r2_inner = idy + 1
                delta = _njit_relocate_delta(r1_path, r2_path, r1_inner, r2_inner, self._np_dist)
                improvement = -delta
                if improvement <= best_improvement or improvement <= 1e-6: continue
                new_r1 = r1_stores[:idx] + r1_stores[idx+1:]
                new_r2 = r2_stores[:idy] + [r1_store] + r2_stores[idy:]
                if (self._check_time_constraint(new_r1) and self._check_time_constraint(new_r2) and self._check_order_principle(route2_id, new_r2)):
                    best_store, best_pos, best_improvement = r1_store, idy, improvement
        return best_store, best_pos, best_improvement

    def _swap_first(self, routes, route1_id, route2_id, movable_stores=None):
        r1, r2 = routes[route1_id], routes[route2_id]
        r1_stores, r2_stores = r1['stores'], r2['stores']
        cap1, cap2 = r1['dc']['max_capacity'], r2['dc']['max_capacity']
        vol1_total = sum(s['volume'] for s in r1_stores)
        vol2_total = sum(s['volume'] for s in r2_stores)
        dc_idx = self._s2i[self.dc['store_id']]
        r1_path = np.empty(len(r1_stores) + 2, dtype=np.int64)
        r1_path[0] = r1_path[-1] = dc_idx
        for k, s in enumerate(r1_stores): r1_path[k + 1] = self._s2i[s['store_id']]
        r2_path = np.empty(len(r2_stores) + 2, dtype=np.int64)
        r2_path[0] = r2_path[-1] = dc_idx
        for k, s in enumerate(r2_stores): r2_path[k + 1] = self._s2i[s['store_id']]
        
        for i, s1 in enumerate(r1_stores):
            if movable_stores is not None and s1["store_id"] not in movable_stores: continue
            if not self._is_movable(s1, route1_id): continue
            for j, s2 in enumerate(r2_stores):
                if movable_stores is not None and s2["store_id"] not in movable_stores: continue
                if not self._is_movable(s2, route2_id): continue
                if (vol1_total - s1['volume'] + s2['volume'] > cap1 or vol2_total - s2['volume'] + s1['volume'] > cap2): continue
                delta = _njit_swap_delta(r1_path, r2_path, i + 1, j + 1, self._np_dist)
                improvement = -delta
                if improvement <= 1e-6: continue
                new_r1, new_r2 = r1_stores[:], r2_stores[:]
                new_r1[i], new_r2[j] = s2, s1
                if (self._check_time_constraint(new_r1) and self._check_time_constraint(new_r2) and self._check_order_principle(route1_id, new_r1) and self._check_order_principle(route2_id, new_r2) and self._check_region_constraint(new_r1) and self._check_region_constraint(new_r2)):
                    return s1, s2, improvement
        return None, None, 0.0

    def _swap_best(self, routes, route1_id, route2_id, movable_stores=None):
        r1, r2 = routes[route1_id], routes[route2_id]
        r1_stores, r2_stores = r1['stores'], r2['stores']
        cap1, cap2 = r1['dc']['max_capacity'], r2['dc']['max_capacity']
        vol1_total = sum(s['volume'] for s in r1_stores)
        vol2_total = sum(s['volume'] for s in r2_stores)
        dc_idx = self._s2i[self.dc['store_id']]
        r1_path = np.empty(len(r1_stores) + 2, dtype=np.int64)
        r1_path[0] = r1_path[-1] = dc_idx
        for k, s in enumerate(r1_stores): r1_path[k + 1] = self._s2i[s['store_id']]
        r2_path = np.empty(len(r2_stores) + 2, dtype=np.int64)
        r2_path[0] = r2_path[-1] = dc_idx
        for k, s in enumerate(r2_stores): r2_path[k + 1] = self._s2i[s['store_id']]
        
        best_s1, best_s2, best_improvement = None, None, 0.0
        for i, s1 in enumerate(r1_stores):
            if movable_stores is not None and s1["store_id"] not in movable_stores: continue
            if not self._is_movable(s1, route1_id): continue
            for j, s2 in enumerate(r2_stores):
                if movable_stores is not None and s2["store_id"] not in movable_stores: continue
                if not self._is_movable(s2, route2_id): continue
                if (vol1_total - s1['volume'] + s2['volume'] > cap1 or vol2_total - s2['volume'] + s1['volume'] > cap2): continue
                delta = _njit_swap_delta(r1_path, r2_path, i + 1, j + 1, self._np_dist)
                improvement = -delta
                if improvement <= best_improvement or improvement <= 1e-6: continue
                new_r1, new_r2 = r1_stores[:], r2_stores[:]
                new_r1[i], new_r2[j] = s2, s1
                if (self._check_time_constraint(new_r1) and self._check_time_constraint(new_r2) and self._check_order_principle(route1_id, new_r1) and self._check_order_principle(route2_id, new_r2) and self._check_region_constraint(new_r1) and self._check_region_constraint(new_r2)):
                    best_s1, best_s2, best_improvement = s1, s2, improvement
        return best_s1, best_s2, best_improvement

    def _cross_exchange_first(self, routes, route1_id, route2_id, movable_stores=None):
        r1_stores, r2_stores = routes[route1_id]['stores'], routes[route2_id]['stores']
        cap1, cap2 = routes[route1_id]['dc']['max_capacity'], routes[route2_id]['dc']['max_capacity']
        vol1_total, vol2_total = sum(s['volume'] for s in r1_stores), sum(s['volume'] for s in r2_stores)
        dc_idx, max_len = self._s2i[self.dc['store_id']], 5
        r1_path = np.empty(len(r1_stores) + 2, dtype=np.int64)
        r1_path[0] = r1_path[-1] = dc_idx
        for k, s in enumerate(r1_stores): r1_path[k + 1] = self._s2i[s['store_id']]
        r2_path = np.empty(len(r2_stores) + 2, dtype=np.int64)
        r2_path[0] = r2_path[-1] = dc_idx
        for k, s in enumerate(r2_stores): r2_path[k + 1] = self._s2i[s['store_id']]
        
        for i in range(len(r1_stores)):
            for l1 in range(1, max_len + 1):
                if i + l1 > len(r1_stores): continue
                seg1 = r1_stores[i:i + l1]
                if movable_stores is not None and any(s["store_id"] not in movable_stores for s in seg1): continue
                if any(not self._is_movable(s, route1_id) for s in seg1): continue
                vol_seg1 = sum(s['volume'] for s in seg1)
                for j in range(len(r2_stores)):
                    for l2 in range(1, max_len + 1):
                        if j + l2 > len(r2_stores): continue
                        seg2 = r2_stores[j:j + l2]
                        if movable_stores is not None and any(s["store_id"] not in movable_stores for s in seg2): continue
                        if any(not self._is_movable(s, route2_id) for s in seg2): continue
                        vol_seg2 = sum(s['volume'] for s in seg2)
                        if (vol1_total - vol_seg1 + vol_seg2 > cap1 or vol2_total - vol_seg2 + vol_seg1 > cap2): continue
                        delta = _njit_cross_exchange_delta(r1_path, r2_path, i + 1, l1, j + 1, l2, self._np_dist)
                        improvement = -delta
                        if improvement <= 1e-6: continue
                        new_r1, new_r2 = r1_stores[:i] + seg2 + r1_stores[i + l1:], r2_stores[:j] + seg1 + r2_stores[j + l2:]
                        if (self._check_time_constraint(new_r1) and self._check_time_constraint(new_r2) and self._check_order_principle(route1_id, new_r1) and self._check_order_principle(route2_id, new_r2) and self._check_region_constraint(new_r1) and self._check_region_constraint(new_r2)):
                            return new_r1, new_r2, improvement
        return None, None, 0.0

    def _cross_exchange_best(self, routes, route1_id, route2_id, movable_stores=None):
        r1_stores, r2_stores = routes[route1_id]['stores'], routes[route2_id]['stores']
        cap1, cap2 = routes[route1_id]['dc']['max_capacity'], routes[route2_id]['dc']['max_capacity']
        vol1_total, vol2_total = sum(s['volume'] for s in r1_stores), sum(s['volume'] for s in r2_stores)
        dc_idx, max_len = self._s2i[self.dc['store_id']], 5
        r1_path = np.empty(len(r1_stores) + 2, dtype=np.int64)
        r1_path[0] = r1_path[-1] = dc_idx
        for k, s in enumerate(r1_stores): r1_path[k + 1] = self._s2i[s['store_id']]
        r2_path = np.empty(len(r2_stores) + 2, dtype=np.int64)
        r2_path[0] = r2_path[-1] = dc_idx
        for k, s in enumerate(r2_stores): r2_path[k + 1] = self._s2i[s['store_id']]
        
        best_r1, best_r2, best_improvement = None, None, 0.0
        for i in range(len(r1_stores)):
            for l1 in range(1, max_len + 1):
                if i + l1 > len(r1_stores): continue
                seg1 = r1_stores[i:i + l1]
                if movable_stores is not None and any(s["store_id"] not in movable_stores for s in seg1): continue
                if any(not self._is_movable(s, route1_id) for s in seg1): continue
                vol_seg1 = sum(s['volume'] for s in seg1)
                for j in range(len(r2_stores)):
                    for l2 in range(1, max_len + 1):
                        if j + l2 > len(r2_stores): continue
                        seg2 = r2_stores[j:j + l2]
                        if movable_stores is not None and any(s["store_id"] not in movable_stores for s in seg2): continue
                        if any(not self._is_movable(s, route2_id) for s in seg2): continue
                        vol_seg2 = sum(s['volume'] for s in seg2)
                        if (vol1_total - vol_seg1 + vol_seg2 > cap1 or vol2_total - vol_seg2 + vol_seg1 > cap2): continue
                        delta = _njit_cross_exchange_delta(r1_path, r2_path, i + 1, l1, j + 1, l2, self._np_dist)
                        improvement = -delta
                        if improvement <= best_improvement or improvement <= 1e-6: continue
                        new_r1, new_r2 = r1_stores[:i] + seg2 + r1_stores[i+l1:], r2_stores[:j] + seg1 + r2_stores[j+l2:]
                        if (self._check_time_constraint(new_r1) and self._check_time_constraint(new_r2) and self._check_order_principle(route1_id, new_r1) and self._check_order_principle(route2_id, new_r2) and self._check_region_constraint(new_r1) and self._check_region_constraint(new_r2)):
                            best_r1, best_r2, best_improvement = new_r1, new_r2, improvement
        return best_r1, best_r2, best_improvement


    def optimize(self, routes_info, movable_stores=None, active_neighborhoods=None):
        """
        Standard VND (Variable Neighborhood Descent):

            k = 0
            while k < kmax:
                x' = best (or first) solution in N_k(x)
                if f(x') < f(x):
                    x = x'
                    k = 0          # improvement → restart from N_0
                else:
                    k += 1         # no improvement → try next neighborhood

        Neighborhoods (in order):
            0: Intra-route 2-opt
            1: Intra-route Relocate
            2: Inter-route Relocate
            3: Inter-route Swap
            4: Inter-route Cross Exchange
        """
        current_routes = copy.deepcopy(routes_info)
        all_ids = [s['store_id'] for rd in current_routes.values() for s in rd['stores']]
        self._ensure_np_matrices(all_ids)
        route_manager = RouteManager(current_routes, self.distance_matrix, self.time_matrix, is_solomon=self.is_solomon)

        use_first = (self.improvement_strategy == 'first')

        # Each entry: callable(routes, route_manager) → bool
        if use_first:
            nb_dict = {
                'two_opt': self._nb_intra_two_opt_first,
                'intra_relocate': self._nb_intra_relocate_first,
                'inter_relocate': self._nb_inter_relocate_first,
                'inter_swap': self._nb_inter_swap_first,
                'cross_exchange': self._nb_inter_cross_first,
            }
        else:
            nb_dict = {
                'two_opt': self._nb_intra_two_opt_best,
                'intra_relocate': self._nb_intra_relocate_best,
                'inter_relocate': self._nb_inter_relocate_best,
                'inter_swap': self._nb_inter_swap_best,
                'cross_exchange': self._nb_inter_cross_best,
            }

        if active_neighborhoods:
            neighborhoods = [nb_dict[name] for name in active_neighborhoods if name in nb_dict]
        else:
            neighborhoods = list(nb_dict.values())

        k = 0
        while k < len(neighborhoods):
            improved = neighborhoods[k](current_routes, route_manager, movable_stores)
            if improved:
                current_routes = route_manager.routes_info
                k = 0          # restart from first neighborhood
            else:
                k += 1         # move to next neighborhood

        route_manager.update_all_routes_info()
        final_routes = route_manager.routes_info
        final_cost = self._calculate_routes_cost(final_routes)
        return final_routes, final_cost

    # ------------------------------------------------------------------
    # Neighborhood wrappers — each calls the corresponding primitive and
    # returns True/False so the VND loop can stay clean.
    # ------------------------------------------------------------------

    def _nb_intra_two_opt_first(self, routes, route_manager, movable_stores=None):
        for route_id, route_data in routes.items():
            original_dist = self._calculate_route_distance(route_data['stores'])
            new_stores, new_dist = self._two_opt_first(route_data['stores'], original_dist, movable_stores)
            if original_dist - new_dist > 1e-6:
                route_manager.replace_stores(route_id, new_stores)
                if self.verbose: print(f"      [VND] 2-opt reversed segment in {route_id}")
                return True
        return False

    def _nb_intra_two_opt_best(self, routes, route_manager, movable_stores=None):
        best_route_id, best_new_stores, max_improvement = None, None, 1e-6
        for route_id, route_data in routes.items():
            original_dist = self._calculate_route_distance(route_data['stores'])
            new_stores, new_dist = self._two_opt_best(route_data['stores'], original_dist, movable_stores)
            improvement = original_dist - new_dist
            if improvement > max_improvement:
                max_improvement, best_route_id, best_new_stores = improvement, route_id, new_stores
        if best_route_id is not None:
            route_manager.replace_stores(best_route_id, best_new_stores)
            if self.verbose: print(f"      [VND] 2-opt reversed segment in {best_route_id}")
            return True
        return False

    def _nb_intra_relocate_first(self, routes, route_manager, movable_stores=None):
        for route_id, route_data in routes.items():
            original_dist = self._calculate_route_distance(route_data['stores'])
            new_stores, new_dist = self._intra_route_relocate_first(route_data['stores'], original_dist, movable_stores)
            if original_dist - new_dist > 1e-6:
                route_manager.replace_stores(route_id, new_stores)
                if self.verbose: print(f"      [VND] 2-opt reversed segment in {route_id}")
                return True
        return False

    def _nb_intra_relocate_best(self, routes, route_manager, movable_stores=None):
        best_route_id, best_new_stores, max_improvement = None, None, 1e-6
        for route_id, route_data in routes.items():
            original_dist = self._calculate_route_distance(route_data['stores'])
            new_stores, new_dist = self._intra_route_relocate_best(route_data['stores'], original_dist, movable_stores)
            improvement = original_dist - new_dist
            if improvement > max_improvement:
                max_improvement, best_route_id, best_new_stores = improvement, route_id, new_stores
        if best_route_id is not None:
            route_manager.replace_stores(best_route_id, best_new_stores)
            if self.verbose: print(f"      [VND] 2-opt reversed segment in {best_route_id}")
            return True
        return False

    def _nb_inter_relocate_first(self, routes, route_manager, movable_stores=None):
        route_ids = list(routes.keys())
        for i in range(len(route_ids)):
            for j in range(len(route_ids)):
                if i == j:
                    continue
                moved_store, position, impr = self._relocate_first(routes, route_ids[i], route_ids[j], movable_stores)
                if impr > 1e-6:
                    route_manager.move_store_to_route(route_ids[i], moved_store, route_ids[j], position)
                    if self.verbose: print(f"      [VND] Relocated store {moved_store['store_id']} from {route_ids[i]} to {route_ids[j]}")
                    return True
        return False

    def _nb_inter_relocate_best(self, routes, route_manager, movable_stores=None):
        route_ids = list(routes.keys())
        best_impr, best_move = 1e-6, None
        for i in range(len(route_ids)):
            for j in range(len(route_ids)):
                if i == j:
                    continue
                r1_id, r2_id = route_ids[i], route_ids[j]
                moved_store, pos, impr = self._relocate_best(routes, r1_id, r2_id, movable_stores)
                if impr > best_impr:
                    best_impr, best_move = impr, (r1_id, moved_store, r2_id, pos)
        if best_move:
            r1_id, moved_store, r2_id, pos = best_move
            route_manager.move_store_to_route(r1_id, moved_store, r2_id, pos)
            if self.verbose: print(f"      [VND] Relocated store {moved_store['store_id']} from {r1_id} to {r2_id}")
            return True
        return False

    def _nb_inter_swap_first(self, routes, route_manager, movable_stores=None):
        route_ids = list(routes.keys())
        for i in range(len(route_ids)):
            for j in range(i + 1, len(route_ids)):
                s1, s2, impr = self._swap_first(routes, route_ids[i], route_ids[j], movable_stores)
                if impr > 1e-6:
                    route_manager.swap_stores(route_ids[i], s1, route_ids[j], s2)
                    if self.verbose: print(f"      [VND] Swapped store {s1['store_id']} ({route_ids[i]}) with store {s2['store_id']} ({route_ids[j]})")
                    return True
        return False

    def _nb_inter_swap_best(self, routes, route_manager, movable_stores=None):
        route_ids = list(routes.keys())
        best_impr, best_move = 1e-6, None
        for i in range(len(route_ids)):
            for j in range(i + 1, len(route_ids)):
                r1_id, r2_id = route_ids[i], route_ids[j]
                s1, s2, impr = self._swap_best(routes, r1_id, r2_id, movable_stores)
                if impr > best_impr:
                    best_impr, best_move = impr, (r1_id, s1, r2_id, s2)
        if best_move:
            r1_id, s1, r2_id, s2 = best_move
            route_manager.swap_stores(r1_id, s1, r2_id, s2)
            if self.verbose: print(f"      [VND] Swapped store {s1['store_id']} ({r1_id}) with store {s2['store_id']} ({r2_id})")
            return True
        return False

    def _nb_inter_cross_first(self, routes, route_manager, movable_stores=None):
        route_ids = list(routes.keys())
        for i in range(len(route_ids)):
            for j in range(i + 1, len(route_ids)):
                nr1, nr2, impr = self._cross_exchange_first(routes, route_ids[i], route_ids[j], movable_stores)
                if impr > 1e-6:
                    route_manager.replace_stores(route_ids[i], nr1)
                    route_manager.replace_stores(route_ids[j], nr2)
                    if self.verbose: print(f"      [VND] Cross-exchanged segments between {route_ids[i]} and {route_ids[j]}")
                    return True
        return False

    def _nb_inter_cross_best(self, routes, route_manager, movable_stores=None):
        route_ids = list(routes.keys())
        best_impr, best_move = 1e-6, None
        for i in range(len(route_ids)):
            for j in range(i + 1, len(route_ids)):
                r1_id, r2_id = route_ids[i], route_ids[j]
                nr1, nr2, impr = self._cross_exchange_best(routes, r1_id, r2_id, movable_stores)
                if impr > best_impr:
                    best_impr, best_move = impr, (r1_id, nr1, r2_id, nr2)
        if best_move:
            r1_id, nr1, r2_id, nr2 = best_move
            route_manager.replace_stores(r1_id, nr1)
            route_manager.replace_stores(r2_id, nr2)
            if self.verbose: print(f"      [VND] Cross-exchanged segments between {r1_id} and {r2_id}")
            return True
        return False