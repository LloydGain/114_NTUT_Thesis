import copy
import random
import hashlib
import numpy as np
from datetime import datetime, timedelta
from multiprocessing import cpu_count, Manager
import concurrent.futures
from numba import njit
from numba.typed import List as NumbaList

from config import config
from models.route_manager import RouteManager
from utils.early_stopper import EarlyStopper
from solvers.support_line_macs import SupportLinePlanningMACS

@njit(cache=True)
def _njit_check_time_constraint(full_route, dc_idx, distance_matrix, time_matrix,
                                dwell_arr, earliest_arr, latest_arr, sched_arr, time_limit):
    first_store_idx = full_route[1]
    
    curr_time = sched_arr[first_store_idx]
    
    dc_depart_time = curr_time - time_matrix[dc_idx, first_store_idx]

    for i in range(1, len(full_route) - 1):
        prev_idx = full_route[i - 1]
        curr_idx = full_route[i]

        if i > 1:
            travel_time = time_matrix[prev_idx, curr_idx]
            pre_dwell = dwell_arr[prev_idx]
            curr_time += travel_time + pre_dwell
        if curr_time < earliest_arr[curr_idx]:
            return False

        if curr_time > latest_arr[curr_idx]:
            return False

    last_store_idx = full_route[-2]
    curr_time += dwell_arr[last_store_idx] + time_matrix[last_store_idx, dc_idx]

    total_duration = curr_time - dc_depart_time
    if total_duration > time_limit:
        return False

    return True

@njit(cache=True)
def _njit_get_store_insertion_cost_pos(store_idx, route_stores, dc_idx, dc_region, route_vol, route_cap,
                                       store_region, store_vol, dist_matrix, time_matrix,
                                       dwell_arr, earliest_arr, latest_arr, sched_arr, time_limit):
    if dc_region == 0 and store_region == 1: return -1.0, -1
    if dc_region == 1 and store_region == 0: return -1.0, -1
    if dc_region == 2 and store_region == 3: return -1.0, -1
    if dc_region == 3 and store_region == 2: return -1.0, -1

    if route_vol + store_vol > route_cap:
        return -1.0, -1

    L = len(route_stores)
    best_pos = -1
    min_cost = 1e12

    full_route = np.zeros(L + 2, dtype=np.int64)
    full_route[0]     = dc_idx
    full_route[L + 1] = dc_idx
    for i in range(L):
        full_route[i + 1] = route_stores[i]

    test_route = np.zeros(L + 3, dtype=np.int64)

    for pos in range(1, len(full_route)):
        prev_idx = full_route[pos - 1]
        next_idx = full_route[pos]
        insert_cost = (dist_matrix[prev_idx, store_idx]
                       + dist_matrix[store_idx, next_idx]
                       - dist_matrix[prev_idx, next_idx])

        if 0 < insert_cost < min_cost:
            for i in range(pos):
                test_route[i] = full_route[i]
            test_route[pos] = store_idx
            for i in range(pos, len(full_route)):
                test_route[i + 1] = full_route[i]

            if _njit_check_time_constraint(test_route, dc_idx, dist_matrix, time_matrix,
                                           dwell_arr, earliest_arr, latest_arr,
                                           sched_arr, time_limit):
                best_pos = pos - 1
                min_cost = insert_cost

    return min_cost, best_pos

@njit(cache=True)
def _njit_total_distance(route_paths, dist_matrix):
    total = 0.0
    for k in range(len(route_paths)):
        path = route_paths[k]
        for m in range(len(path) - 1):
            total += dist_matrix[path[m], path[m + 1]]
    return total

try:
    _njit_check_time_constraint(
        np.array([0, 0, 0], dtype=np.int64), 0,
        np.zeros((1, 1)), np.zeros((1, 1)),
        np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]), 10.0
    )
    _njit_get_store_insertion_cost_pos(
        0, np.array([0], dtype=np.int64), 0, 0, 0.0, 10.0, 0, 0.0,
        np.zeros((1, 1)), np.zeros((1, 1)),
        np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]), 10.0
    )
    _dummy = NumbaList()
    _dummy.append(np.array([0, 0], dtype=np.int64))
    _njit_total_distance(_dummy, np.zeros((1, 1)))
except Exception:
    pass

ALLOC_ROUTES      = None
ALLOC_DIST        = None
ALLOC_TIME        = None
ALLOC_GA_INSTANCE = None

def init_alloc_worker(main_routes, distance_matrix, time_matrix, ga_instance):
    global ALLOC_ROUTES, ALLOC_DIST, ALLOC_TIME, ALLOC_GA_INSTANCE
    ALLOC_ROUTES      = main_routes
    ALLOC_DIST        = distance_matrix
    ALLOC_TIME        = time_matrix
    ALLOC_GA_INSTANCE = ga_instance

def alloc_fitness_worker(args):
    chromo, key, shared_cache = args
    if key in shared_cache:
        return shared_cache[key]
    ga   = ALLOC_GA_INSTANCE
    cost, routes, support, repaired, vn = ga._evaluate_individual(chromo)
    result = {'cost': cost, 'repaired': repaired, 'vn': vn}
    shared_cache[key] = result
    return result

class StoreAllocationGA:
    _cached_mappings = None

    def __init__(self, main_routes, remaining_stores, distance_matrix, time_matrix,
                 population_size=50, elite_rate=0.1, generations=50,
                 cross_rate=0.8, mutation_rate=0.2, early_stop_patience=100):
        self.main_routes      = main_routes
        self.remaining_stores = remaining_stores
        self.distance_matrix  = distance_matrix
        self.time_matrix      = time_matrix
        self.dc               = config.DC_CONFIG
        self.population_size  = population_size
        self.elite_size       = int(population_size * elite_rate)
        self.generations      = generations
        self.cross_rate       = cross_rate
        self.mutation_rate    = mutation_rate
        self.early_stop_patience = early_stop_patience
        self.route_choices    = list(self.main_routes.keys()) + ['SUPPORT']
        self.time_limit_per_route = 5 * 60 * 60
        self.best_cost        = float('inf')
        self.best_solution    = None
        self.best_remaining_solution = None
        self.best_individual  = None
        self.log              = []

        self._init_numpy_mappings()
        self.remaining_stores = self._sort_stores_by_insertion_cost(remaining_stores)
                                                                
    def _init_numpy_mappings(self):
        if StoreAllocationGA._cached_mappings is not None:
            (self.s2i, self.i2s, self.np_dist, self.np_time,
             self.np_volume, self.np_dwell, self.np_earliest,
             self.np_latest, self.np_region, self.np_sched) = StoreAllocationGA._cached_mappings
            return

        self.s2i = {self.dc['store_id']: 0}
        self.i2s = {0: self.dc}
        idx = 1
        all_stores = list(self.remaining_stores)
        for r in self.main_routes.values():
            all_stores.extend(r['stores'])

        for s in all_stores:
            if s['store_id'] not in self.s2i:
                self.s2i[s['store_id']] = idx
                self.i2s[idx] = s
                idx += 1

        n = len(self.s2i)
        self.np_dist     = np.zeros((n, n), dtype=np.float64)
        self.np_time     = np.zeros((n, n), dtype=np.float64)
        self.np_volume   = np.zeros(n, dtype=np.float64)
        self.np_dwell    = np.zeros(n, dtype=np.float64)
        self.np_earliest = np.zeros(n, dtype=np.float64)
        self.np_latest   = np.zeros(n, dtype=np.float64)
        self.np_region   = np.full(n, -1, dtype=np.int64)
        self.np_sched    = np.zeros(n, dtype=np.float64)

        region_map = {'north': 0, 'south': 1, 'east': 2, 'west': 3}
        for i in range(1, n):
            s_i    = self.i2s[i]
            s_i_id = s_i['store_id']
            self.np_volume[i]   = s_i.get('volume', 0.0)
            self.np_dwell[i]    = float(s_i.get('dwell_time', 0))
            self.np_earliest[i] = float(int(datetime.fromisoformat(s_i['earliest_time']).timestamp()))
            self.np_latest[i]   = float(int(datetime.fromisoformat(s_i['latest_time']).timestamp()))
            self.np_region[i]   = region_map.get(s_i.get('region', ''), -1)
            
            sched_str = s_i.get('sched_time', s_i.get('pred_time', s_i['earliest_time']))
            self.np_sched[i]    = float(int(datetime.fromisoformat(sched_str).timestamp()))

            for j in range(n):
                s_j_id = self.i2s[j]['store_id']
                if s_i_id in self.distance_matrix and s_j_id in self.distance_matrix[s_i_id]:
                    self.np_dist[i, j] = self.distance_matrix[s_i_id][s_j_id]
                if s_i_id in self.time_matrix and s_j_id in self.time_matrix[s_i_id]:
                    self.np_time[i, j] = self.time_matrix[s_i_id][s_j_id]

        for j in range(n):
            s_j_id = self.i2s[j]['store_id']
            dc_id  = self.dc['store_id']
            if dc_id in self.distance_matrix and s_j_id in self.distance_matrix[dc_id]:
                self.np_dist[0, j] = self.distance_matrix[dc_id][s_j_id]
            if dc_id in self.time_matrix and s_j_id in self.time_matrix[dc_id]:
                self.np_time[0, j] = self.time_matrix[dc_id][s_j_id]

        StoreAllocationGA._cached_mappings = (
            self.s2i, self.i2s, self.np_dist, self.np_time,
            self.np_volume, self.np_dwell, self.np_earliest,
            self.np_latest, self.np_region, self.np_sched
        )

    def _get_store_insertion_cost_and_pos(self, route_info, store):
        dc      = route_info['dc']
        stores  = route_info['stores']
        store_idx    = self.s2i[store['store_id']]
        route_stores = np.array([self.s2i[s['store_id']] for s in stores], dtype=np.int64)

        dc_region_map  = {'north': 0, 'south': 1, 'east': 2, 'west': 3}
        dc_region_int  = dc_region_map.get(dc.get('region', ''), -1)
        store_region_int = self.np_region[store_idx]

        cost, pos = _njit_get_store_insertion_cost_pos(
            store_idx, route_stores, 0, dc_region_int,
            float(dc.get('total_volume', 0)), float(dc.get('max_capacity', 1e9)),
            store_region_int, float(store.get('volume', 0)),
            self.np_dist, self.np_time, self.np_dwell,
            self.np_earliest, self.np_latest,
            self.np_sched, float(self.time_limit_per_route)
        )

        if cost < 0:
            return float('inf'), -1
        return cost, pos

    def _fast_get_store_insertion_cost_and_pos(self, r_id, store_idx, route_stores_idx, route_vols, route_caps, route_regions):
        cost, pos = _njit_get_store_insertion_cost_pos(
            store_idx, np.array(route_stores_idx[r_id], dtype=np.int64), 0, route_regions[r_id],
            route_vols[r_id], route_caps[r_id],
            self.np_region[store_idx], self.np_volume[store_idx],
            self.np_dist, self.np_time, self.np_dwell,
            self.np_earliest, self.np_latest,
            self.np_sched, float(self.time_limit_per_route)
        )
        if cost < 0:
            return float('inf'), -1
        return cost, pos

    def _sort_stores_by_insertion_cost(self, stores):
        store_with_costs = []
        for store in stores:
            min_cost = float('inf')
            for r_id, r_info in self.main_routes.items():
                cost, pos = self._get_store_insertion_cost_and_pos(r_info, store)
                if pos != -1 and cost < min_cost:
                    min_cost = cost
            store_with_costs.append((store, min_cost))
        return [s[0] for s in sorted(store_with_costs, key=lambda x: x[1])]

    def _generate_greedy_individual(self, return_routes=False):
        greedy_chromo = []
        support_pool_indices = []

        route_stores_idx = {r_id: [self.s2i[s['store_id']] for s in r_info['stores']] for r_id, r_info in self.main_routes.items()}
        route_vols = {r_id: float(r_info['dc'].get('total_volume', 0)) for r_id, r_info in self.main_routes.items()}
        route_caps = {r_id: float(r_info['dc'].get('max_capacity', 1e9)) for r_id, r_info in self.main_routes.items()}
        dc_region_map  = {'north': 0, 'south': 1, 'east': 2, 'west': 3}
        route_regions = {r_id: dc_region_map.get(r_info['dc'].get('region', ''), -1) for r_id, r_info in self.main_routes.items()}

        for i, store in enumerate(self.remaining_stores):
            store_idx = self.s2i[store['store_id']]
            best_r    = 'SUPPORT'
            min_cost  = float('inf')
            best_pos  = -1
            for r_id in self.main_routes.keys():
                cost, pos = self._fast_get_store_insertion_cost_and_pos(
                    r_id, store_idx, route_stores_idx, route_vols, route_caps, route_regions)
                if pos != -1 and cost < min_cost:
                    min_cost = cost
                    best_r   = r_id
                    best_pos = pos

            greedy_chromo.append(best_r)
            if best_r != 'SUPPORT':
                route_stores_idx[best_r].insert(best_pos, store_idx)
                route_vols[best_r] += self.np_volume[store_idx]
            else:
                support_pool_indices.append(i)

        paths = NumbaList()
        for r_id, s_indices in route_stores_idx.items():
            if not s_indices: continue
            path = np.empty(len(s_indices) + 2, dtype=np.int64)
            path[0] = 0
            for k, idx in enumerate(s_indices):
                path[k + 1] = idx
            path[-1] = 0
            paths.append(path)
            
        main_cost = float(_njit_total_distance(paths, self.np_dist))
        
        support_pool = [self.remaining_stores[i] for i in support_pool_indices]
        macs = SupportLinePlanningMACS(support_pool, self.distance_matrix, self.time_matrix, time_limit=0, verbose=False)

        if return_routes:
            macs_cost, support_routes = macs.run()
            vn = len(support_routes)
            total_cost = main_cost + macs_cost

            temp_routes = self._copy_routes_info(self.main_routes)
            rm = RouteManager(temp_routes, self.distance_matrix, self.time_matrix)
            for i, target in enumerate(greedy_chromo):
                if target != 'SUPPORT':
                    store = self.remaining_stores[i]
                    _, pos = self._get_store_insertion_cost_and_pos(rm.routes_info[target], store)
                    rm.insert_store(store, target, pos)
            return greedy_chromo, total_cost, rm.routes_info, support_pool, vn

        macs_cost = macs.gb_cost
        vn = len(macs.gb_routes)
        total_cost = main_cost + macs_cost
        return greedy_chromo, total_cost, None, None, vn

    def _generate_random_individual(self):
        """Probabilistic construction weighted by inverse insertion cost.
        Returns (chromo, total_cost, routes_info, support_pool, vn)."""
        temp_routes   = self._copy_routes_info(self.main_routes)
        route_manager = RouteManager(temp_routes, self.distance_matrix, self.time_matrix)
        chromo       = []
        support_pool = []

        for store in self.remaining_stores:
            feasible_routes = []
            feasible_costs  = []
            for r_id in self.main_routes.keys():
                cost, pos = self._get_store_insertion_cost_and_pos(
                    route_manager.routes_info[r_id], store)
                if pos != -1:
                    feasible_routes.append((r_id, pos))
                    feasible_costs.append(cost)

            if not feasible_routes:
                chromo.append('SUPPORT')
                support_pool.append(store)
                continue

            inv_costs = np.array([1.0 / (c + 1e-6) for c in feasible_costs])
            probs     = inv_costs / inv_costs.sum()
            chosen_idx           = np.random.choice(len(feasible_routes), p=probs)
            chosen_r, chosen_pos = feasible_routes[chosen_idx]

            chromo.append(chosen_r)
            route_manager.insert_store(store, chosen_r, chosen_pos)

        main_cost = self._calculate_total_distance(route_manager.routes_info)
        macs = SupportLinePlanningMACS(support_pool, self.distance_matrix, self.time_matrix, time_limit=0, verbose=False)
        macs_cost, support_routes = macs.run()
        vn = len(support_routes)
        total_cost = main_cost + macs_cost
        return chromo, total_cost, route_manager.routes_info, support_pool, vn

    def _copy_routes_info(self, routes):
        return {
            route_id: {
                "dc": route_data["dc"].copy(),
                "stores": [s.copy() for s in route_data["stores"]]
            }
            for route_id, route_data in routes.items()
        }

    def _encode_individual(self, individual):
        s = ','.join(map(str, individual))
        return hashlib.md5(s.encode()).hexdigest()

    def _routes_to_paths(self, routes_info):
        dc_idx = self.s2i[self.dc['store_id']]
        paths  = NumbaList()
        for route_data in routes_info.values():
            stores = route_data['stores']
            if not stores:
                continue
            path = np.empty(len(stores) + 2, dtype=np.int64)
            path[0] = dc_idx
            for k, s in enumerate(stores):
                path[k + 1] = self.s2i[s['store_id']]
            path[-1] = dc_idx
            paths.append(path)
        return paths

    def _calculate_total_distance(self, routes):
        paths = self._routes_to_paths(routes)
        if len(paths) == 0:
            return 0.0
        return float(_njit_total_distance(paths, self.np_dist))

    def _repair_individual(self, individual):
        """
        Repair Operator using DROP Phase and ADD Phase based on Pseudo-utility Ratio.
        u_j = v_j / (c_j^2 - c_j^1)
        """
        new_individual = list(individual)
        
        # 0. Pre-calculate utility for each store
        temp_routes = self._copy_routes_info(self.main_routes)
        rm = RouteManager(temp_routes, self.distance_matrix, self.time_matrix)
        
        utilities = {}
        for idx in range(len(self.remaining_stores)):
            store = self.remaining_stores[idx]
            store_idx = self.s2i[store['store_id']]
            
            c_j2 = 2 * self.np_dist[0, store_idx]
            
            c_j1 = float('inf')
            for r_id in self.main_routes.keys():
                cost, pos = self._get_store_insertion_cost_and_pos(rm.routes_info[r_id], store)
                if pos != -1 and cost < c_j1:
                    c_j1 = cost
                    
            if c_j1 == float('inf'):
                c_j1 = c_j2
                
            v_j = max(0, c_j2 - c_j1) 
            capacity = self.np_volume[store_idx]
            
            utilities[idx] = v_j / (capacity + 1e-6)

        def get_utility(idx):
            return utilities[idx]

        # 1. DROP Phase: Restore feasibility for each route
        route_to_indices = {r_id: [] for r_id in self.main_routes.keys()}
        for i, target in enumerate(new_individual):
            if target != 'SUPPORT':
                route_to_indices[target].append(i)

        for r_id, indices in route_to_indices.items():
            if not indices: continue
            
            # Sort by utility ascending (lowest utility first for DROP)
            indices.sort(key=get_utility)
            
            while indices:
                temp_route_info = self._copy_routes_info({r_id: self.main_routes[r_id]})
                rm = RouteManager(temp_route_info, self.distance_matrix, self.time_matrix)
                
                success = True
                for idx in indices:
                    s = self.remaining_stores[idx]
                    cost, pos = self._get_store_insertion_cost_and_pos(rm.routes_info[r_id], s)
                    if pos != -1:
                        rm.insert_store(s, r_id, pos)
                    else:
                        success = False
                        break
                
                if success:
                    break
                else:
                    dropped_idx = indices.pop(0)
                    new_individual[dropped_idx] = 'SUPPORT'

        # 2. ADD Phase: Improve quality by re-inserting unassigned nodes
        support_indices = [i for i, target in enumerate(new_individual) if target == 'SUPPORT']
        # Sort by utility descending (highest utility first for ADD)
        support_indices.sort(key=get_utility, reverse=True)
        
        current_routes = self._copy_routes_info(self.main_routes)
        rm = RouteManager(current_routes, self.distance_matrix, self.time_matrix)
        for i, target in enumerate(new_individual):
            if target != 'SUPPORT':
                s = self.remaining_stores[i]
                cost, pos = self._get_store_insertion_cost_and_pos(rm.routes_info[target], s)
                if pos != -1: rm.insert_store(s, target, pos)

        for idx in support_indices:
            store = self.remaining_stores[idx]
            best_r, best_pos, min_inc = None, -1, float('inf')
            
            for r_id in self.main_routes.keys():
                cost, pos = self._get_store_insertion_cost_and_pos(rm.routes_info[r_id], store)
                if pos != -1 and cost < min_inc:
                    min_inc, best_r, best_pos = cost, r_id, pos
            
            if best_r:
                rm.insert_store(store, best_r, best_pos)
                new_individual[idx] = best_r

        return new_individual

    def _evaluate_individual(self, individual, return_routes=False):
        """Evaluate a chromosome directly. Chromosomes produced by the initialisation,
        BCRC crossover and swap mutation are feasible by construction; any residual
        infeasibility (pos == -1) is handled inline by falling back to SUPPORT."""
        chromo = list(individual)

        route_stores_idx = {r_id: [self.s2i[s['store_id']] for s in r_info['stores']] for r_id, r_info in self.main_routes.items()}
        route_vols = {r_id: float(r_info['dc'].get('total_volume', 0)) for r_id, r_info in self.main_routes.items()}
        route_caps = {r_id: float(r_info['dc'].get('max_capacity', 1e9)) for r_id, r_info in self.main_routes.items()}
        dc_region_map  = {'north': 0, 'south': 1, 'east': 2, 'west': 3}
        route_regions = {r_id: dc_region_map.get(r_info['dc'].get('region', ''), -1) for r_id, r_info in self.main_routes.items()}

        support_pool_indices = []

        for i, target in enumerate(chromo):
            store = self.remaining_stores[i]
            if target == 'SUPPORT':
                support_pool_indices.append(i)
                continue
            
            store_idx = self.s2i[store['store_id']]
            cost, pos = self._fast_get_store_insertion_cost_and_pos(
                target, store_idx, route_stores_idx, route_vols, route_caps, route_regions)
            
            if pos != -1:
                route_stores_idx[target].insert(pos, store_idx)
                route_vols[target] += self.np_volume[store_idx]
            else:
                support_pool_indices.append(i)
                chromo[i] = 'SUPPORT'

        paths = NumbaList()
        for r_id, s_indices in route_stores_idx.items():
            if not s_indices: continue
            path = np.empty(len(s_indices) + 2, dtype=np.int64)
            path[0] = 0
            for k, idx in enumerate(s_indices):
                path[k + 1] = idx
            path[-1] = 0
            paths.append(path)
            
        main_cost = float(_njit_total_distance(paths, self.np_dist))
        
        support_pool = [self.remaining_stores[i] for i in support_pool_indices]
        macs = SupportLinePlanningMACS(support_pool, self.distance_matrix, self.time_matrix, time_limit=0, verbose=False)

        if return_routes:
            macs_cost, support_routes = macs.run()
            vn = len(support_routes)
            total_cost = main_cost + macs_cost

            temp_routes = self._copy_routes_info(self.main_routes)
            rm = RouteManager(temp_routes, self.distance_matrix, self.time_matrix)
            for i, target in enumerate(chromo):
                if target != 'SUPPORT':
                    store = self.remaining_stores[i]
                    _, pos = self._get_store_insertion_cost_and_pos(rm.routes_info[target], store)
                    rm.insert_store(store, target, pos)
            return total_cost, rm.routes_info, support_pool, chromo, vn

        macs_cost = macs.gb_cost
        vn = len(macs.gb_routes)
        total_cost = main_cost + macs_cost
        return total_cost, None, None, chromo, vn

    def _bcrc_crossover(self, parent1, parent2):
        """Best Cost Route Crossover (BCRC).

        Step a) Randomly select one route from each parent.
        Step b) Remove those customers from the *other* parent → pending re-insertion.
        Step c) Re-insert the removed customers in random order at the cheapest
                feasible position in the child, or assign to SUPPORT if none exists.

        Returns (child1, child2).
        """
        # Collect route → [store indices] mapping for each parent
        def routes_with_stores(parent):
            mapping = {}
            for i, r_id in enumerate(parent):
                if r_id != 'SUPPORT':
                    mapping.setdefault(r_id, []).append(i)
            return mapping

        p1_map = routes_with_stores(parent1)
        p2_map = routes_with_stores(parent2)

        # Fall back to uniform crossover if a parent has no real routes
        if not p1_map or not p2_map:
            child = list(parent1)
            for i in range(len(parent1)):
                if random.random() < 0.5:
                    child[i] = parent2[i]
            return child, list(parent2)

        # Step a: select one route randomly from each parent
        r_from_p1 = random.choice(list(p1_map.keys()))  # will be removed from P2 → C2
        r_from_p2 = random.choice(list(p2_map.keys()))  # will be removed from P1 → C1

        p1_selected_indices = set(p1_map[r_from_p1])  # stores to pull out of P2
        p2_selected_indices = set(p2_map[r_from_p2])  # stores to pull out of P1

        def make_child(base_parent, remove_indices):
            """Build a child from base_parent, removing the given store indices,
            then re-inserting them at the best feasible location."""
            child = list(base_parent)

            # Mark removed stores as unassigned
            for idx in remove_indices:
                child[idx] = 'SUPPORT'

            # Reconstruct current route state from the (non-removed) assignments
            temp_routes   = self._copy_routes_info(self.main_routes)
            route_manager = RouteManager(temp_routes, self.distance_matrix, self.time_matrix)

            for i, r_id in enumerate(child):
                if r_id == 'SUPPORT':
                    continue
                store = self.remaining_stores[i]
                cost, pos = self._get_store_insertion_cost_and_pos(
                    route_manager.routes_info[r_id], store)
                if pos != -1:
                    route_manager.insert_store(store, r_id, pos)
                else:
                    child[i] = 'SUPPORT'  # can't fit: mark as unassigned

            # Step b & c: re-insert removed stores in random order at best cost
            to_insert = list(remove_indices)
            random.shuffle(to_insert)

            for idx in to_insert:
                store = self.remaining_stores[idx]
                best_r, best_pos, min_cost = 'SUPPORT', -1, float('inf')
                for r_id in self.main_routes.keys():
                    cost, pos = self._get_store_insertion_cost_and_pos(
                        route_manager.routes_info[r_id], store)
                    if pos != -1 and cost < min_cost:
                        min_cost = cost
                        best_r   = r_id
                        best_pos = pos
                child[idx] = best_r
                if best_r != 'SUPPORT':
                    route_manager.insert_store(store, best_r, best_pos)

            return child

        child1 = make_child(parent1, p2_selected_indices)
        child2 = make_child(parent2, p1_selected_indices)
        return child1, child2

    def _crossover(self, parent1, parent2):
        if random.random() < self.cross_rate:
            c1, c2 = self._bcrc_crossover(parent1, parent2)
            return c1, c2
        return copy.deepcopy(parent1), copy.deepcopy(parent2)


    def _mutate(self, individual):
        """Swap Mutation: randomly pick two different groups (real route or SUPPORT),
        swap one store from each. Repair handles any resulting infeasibility."""
        if random.random() >= self.mutation_rate:
            return individual

        # Build ALL groups → [store indices] mapping
        group_to_indices = {}
        for i, r_id in enumerate(individual):
            group_to_indices.setdefault(r_id, []).append(i)

        groups = [g for g in group_to_indices if group_to_indices[g]]
        if len(groups) < 2:
            return individual

        # Pick two distinct groups
        group_a, group_b = random.sample(groups, 2)

        idx_a = random.choice(group_to_indices[group_a])
        idx_b = random.choice(group_to_indices[group_b])

        # Perform the swap
        individual[idx_a], individual[idx_b] = individual[idx_b], individual[idx_a]

        # Repair infeasibilities introduced by the swap
        individual = self._repair_individual(individual)

        return individual


    def run(self, return_routes=True):
        greedy_chromo, g_cost, g_routes, g_support, g_vn = self._generate_greedy_individual(return_routes=(return_routes and self.generations == 0))
        self.best_cost               = g_cost
        self.best_solution           = g_routes
        self.best_remaining_solution = g_support
        self.best_individual         = greedy_chromo
        self.best_vn                 = g_vn

        if self.generations == 0:
            return self.best_cost, self.best_solution, self.best_remaining_solution, self.best_vn

        self.log.append({
            'generation': 0,
            'iter_worst_cost': float(g_cost),
            'iter_best_cost': float(g_cost),
            'iter_avg_cost': float(g_cost),
            'std_cost': float(0.0),
            'best_cost': g_cost,
        })
        print(f'Store Allocation: iteration{0} -> vn = {g_vn}, cost = {g_cost - g_vn * 2000:.2f}, fitness = {g_cost:.2f}')

        # Build initial population; pre-evaluate inline so gen-0 needs no workers
        population = [greedy_chromo]
        pre_cache  = {self._encode_individual(greedy_chromo): {'cost': g_cost, 'repaired': greedy_chromo, 'vn': g_vn}}

        while len(population) < self.population_size:
            rand_chromo, rand_cost, _, _, rand_vn = self._generate_random_individual()
            rand_key = self._encode_individual(rand_chromo)
            pre_cache[rand_key] = {'cost': rand_cost, 'repaired': rand_chromo, 'vn': rand_vn}
            population.append(rand_chromo)

        early_stopper = EarlyStopper(patience=self.early_stop_patience)
        with Manager() as manager:
            shared_cache = manager.dict()
            shared_cache.update(pre_cache)  # seed with pre-evaluated initial population
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max(1, cpu_count() - 2),
                initializer=init_alloc_worker,
                initargs=(self.main_routes, self.distance_matrix, self.time_matrix, self)
            ) as pool:

                for gen_idx in range(self.generations):
                    unique_tasks    = []
                    individual_keys = []

                    for chromo in population:
                        key = self._encode_individual(chromo)
                        individual_keys.append(key)
                        if key not in shared_cache:
                            unique_tasks.append((chromo, key, shared_cache))

                    if unique_tasks:
                        try:
                            list(pool.map(alloc_fitness_worker, unique_tasks, timeout=120,
                                          chunksize=max(1, len(unique_tasks) // (cpu_count() - 2) * 2)))
                        except concurrent.futures.TimeoutError:
                            print('Timeout error')
                            for _, key, _ in unique_tasks:
                                if key not in shared_cache:
                                    shared_cache[key] = {'cost': float('inf')}

                    fitnesses    = []
                    evaluated_pop = []
                    for i, chromo in enumerate(population):
                        key = individual_keys[i]
                        res = shared_cache[key]
                        evaluated_pop.append({'individual': chromo, 'cost': res['cost'], 'vn': res['vn']})
                        fitnesses.append(res['cost'])

                    evaluated_pop.sort(key=lambda x: x['cost'])
                    current_best = evaluated_pop[0]

                    if current_best['cost'] < self.best_cost:
                        self.best_cost       = current_best['cost']
                        self.best_individual = copy.deepcopy(shared_cache[individual_keys[0]]['repaired'])

                    # Update population with repaired individuals
                    for i in range(len(population)):
                        population[i] = shared_cache[individual_keys[i]]['repaired']

                    self.log.append({
                        'generation': gen_idx + 1,
                        'iter_worst_cost': float(np.max(fitnesses)),
                        'iter_best_cost': float(current_best['cost']),
                        'iter_avg_cost': float(np.mean(fitnesses)),
                        'std_cost': float(np.std(fitnesses)),
                        'best_cost': self.best_cost,
                    })
                    print(f'Store Allocation: iteration{gen_idx + 1} -> vn = {current_best["vn"]}, cost = {current_best["cost"] - current_best["vn"] * 2000:.2f}, fitness = {self.best_cost:.2f}')

                    if early_stopper.check(self.best_cost):
                        break

                    elites  = [copy.deepcopy(ind['individual']) for ind in evaluated_pop[:self.elite_size]]
                    weights = [max(1.0 / ind['cost'], 1e-12) for ind in evaluated_pop]

                    children_generated = []
                    while len(children_generated) < (self.population_size - self.elite_size):
                        p1, p2 = random.choices(evaluated_pop, weights=weights, k=2)
                        c1, c2 = self._crossover(copy.deepcopy(p1['individual']),
                                            copy.deepcopy(p2['individual']))
                        c1 = self._mutate(list(c1))
                        c2 = self._mutate(list(c2))
                        children_generated.append(c1)
                        children_generated.append(c2)

                    task_candidates = []
                    for c in children_generated:
                            k = self._encode_individual(c)
                            if k not in shared_cache:
                                task_candidates.append((c, k, shared_cache))

                    if task_candidates:
                        try:
                            num_procs = max(1, cpu_count() - 2)
                            list(pool.map(alloc_fitness_worker, task_candidates, timeout=120,
                                          chunksize=max(1, len(task_candidates) // num_procs * 2)))
                        except concurrent.futures.TimeoutError:
                            print("Timeout Error during candidate evaluation")
                            for _, k, _ in task_candidates:
                                if k not in shared_cache:
                                    shared_cache[k] = {'cost': float('inf')}

                    population = elites + children_generated

        if self.best_individual is not None:
            _, self.best_solution, self.best_remaining_solution, _, self.best_vn = self._evaluate_individual(self.best_individual, return_routes=return_routes)

        return self.best_cost, self.best_solution, self.best_remaining_solution, self.best_vn