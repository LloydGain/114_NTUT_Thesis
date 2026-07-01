import copy
import os
import random
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from multiprocessing import cpu_count, Manager
import concurrent.futures
from numba import njit
from numba.typed import List as NumbaList

from config import config
from models.route_manager import RouteManager
from utils.early_stopper import EarlyStopper
from solvers.support_line_macs import SupportLinePlanningMACS
from solvers.allocate_ga import _njit_get_store_insertion_cost_pos_ranged
from solvers.vnd import VND

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
        if curr_time < earliest_arr[curr_idx] - 1e-4:
            return False
        if curr_time > latest_arr[curr_idx] + 1e-4:
            return False

    last_store_idx = full_route[-2]
    curr_time += dwell_arr[last_store_idx] + time_matrix[last_store_idx, dc_idx]

    total_duration = curr_time - dc_depart_time
    if total_duration > time_limit + 1e-4:
        return False
    return True

@njit(cache=True)
def _njit_greedy_function_attractions(store_idx, route_stores, dc_idx, route_vol, route_cap,
                                      store_vol, dist_matrix, time_matrix,
                                      dwell_arr, earliest_arr, latest_arr, sched_arr, region_arr, time_limit):
    """
    Returns an array of heuristic desirability (attraction) for each possible insertion position.
    attractions[pos] = value (if valid), or -1.0 (if invalid).
    Returns an array of shape (len(route_stores) + 1,).
    """
    L = len(route_stores)
    attractions = np.full(L + 1, -1.0, dtype=np.float64)

    if route_vol + store_vol > route_cap:
        return attractions

    full_route = np.zeros(L + 2, dtype=np.int64)
    full_route[0] = dc_idx
    full_route[L + 1] = dc_idx
    for i in range(L):
        full_route[i + 1] = route_stores[i]

    test_route = np.zeros(L + 3, dtype=np.int64)

    for pos in range(1, len(full_route)):
        prev_idx = full_route[pos - 1]
        next_idx = full_route[pos]
        
        # Check Region constraint (no direct N<->S or E<->W between consecutive stores)
        valid_region = True
        store_r = region_arr[store_idx]
        
        if prev_idx != dc_idx:
            prev_r = region_arr[prev_idx]
            if (prev_r == 0 and store_r == 1) or (prev_r == 1 and store_r == 0) or \
               (prev_r == 2 and store_r == 3) or (prev_r == 3 and store_r == 2):
                valid_region = False
                
        if valid_region and next_idx != dc_idx:
            next_r = region_arr[next_idx]
            if (store_r == 0 and next_r == 1) or (store_r == 1 and next_r == 0) or \
               (store_r == 2 and next_r == 3) or (store_r == 3 and next_r == 2):
                valid_region = False
                
        if not valid_region:
            continue
        
        # C0: Distance increase
        c0 = dist_matrix[prev_idx, store_idx] + dist_matrix[store_idx, next_idx] - dist_matrix[prev_idx, next_idx]
        
        if c0 > 0:
            for i in range(pos):
                test_route[i] = full_route[i]
            test_route[pos] = store_idx
            for i in range(pos, len(full_route)):
                test_route[i + 1] = full_route[i]

            if _njit_check_time_constraint(test_route, dc_idx, dist_matrix, time_matrix,
                                           dwell_arr, earliest_arr, latest_arr,
                                           sched_arr, time_limit):
                # C1: Vehicle capacity utilization
                c1 = route_cap - route_vol - store_vol
                
                attraction = 1 / (0.5 * c0 + 0.5 * c1 + 1e-6)
                
                # Desirability must be positive for probability calculation, offset if needed
                attraction = max(1e-6, attraction)
                
                attractions[pos - 1] = attraction

    return attractions

@njit(cache=True)
def _njit_total_distance(route_paths, dist_matrix):
    total = 0.0
    for k in range(len(route_paths)):
        path = route_paths[k]
        if len(path) > 1:
            for m in range(len(path) - 1):
                total += dist_matrix[path[m], path[m + 1]]
    return total

@njit(cache=True)
def _njit_global_nearest_neighbor(support_indices, np_dist, np_time, np_volume, np_dwell, np_earliest, np_latest, np_sched, capacity, time_limit, np_group, np_region):
    """Fast Nearest Neighbor using global indices to avoid rebuilding numpy arrays."""
    n_support = len(support_indices)
    unvisited = np.ones(n_support, dtype=np.bool_)
    unvisited_count = n_support
    
    total_dist = 0.0
    v_idx = 0
    
    while unvisited_count > 0:
        cur = 0 # DC index
        cur_vol = 0.0
        cur_duration = 0.0
        prev_pred_time_epoch = 0.0
        
        route_found = False
        while unvisited_count > 0:
            best_dist = 1e12
            best_i = -1
            best_global_idx = -1
            
            for i in range(n_support):
                if not unvisited[i]: continue
                cid = support_indices[i]
                
                if cur_vol + np_volume[cid] > capacity: continue
                
                # Region constraint (simplified for NN, identical to MACS)
                last_g, last_r = np_group[cur], np_region[cur]
                store_g, store_r = np_group[cid], np_region[cid]
                if last_g == 2:
                    if (last_r == 0 and store_r == 1) or (last_r == 1 and store_r == 0) or (last_r == 2 and store_r == 3) or (last_r == 3 and store_r == 2): continue
                if last_g == 2 and store_g not in (0, 1, 2): continue
                elif last_g == 1 and store_g not in (0, 1): continue
                elif last_g == 0 and store_g != 0: continue
                
                pre_to_cur = np_time[cur, cid]
                prev_dwell = np_dwell[cur] if cur != 0 else 0
                
                if cur == 0: 
                    arrival_time = np_sched[cid]
                else: 
                    arrival_time = prev_pred_time_epoch + pre_to_cur + prev_dwell
                    
                if arrival_time < np_earliest[cid] - 1e-4 or arrival_time > np_latest[cid] + 1e-4: continue
                
                if cur == 0:
                    new_dur = np_time[0, cid] + np_dwell[cid] + np_time[cid, 0]
                else:
                    new_dur = cur_duration + np_time[cur, cid] + np_time[cid, 0] - np_time[cur, 0] + np_dwell[cid]
                    
                if new_dur > time_limit + 1e-4: continue
                
                if np_dist[cur, cid] < best_dist:
                    best_dist = np_dist[cur, cid]
                    best_i = i
                    best_global_idx = cid
                    
            if best_i == -1: break
            
            total_dist += best_dist
            cur_vol += np_volume[best_global_idx]
            
            pre_to_cur = np_time[cur, best_global_idx]
            prev_dwell = np_dwell[cur] if cur != 0 else 0
            
            if cur == 0:
                cur_duration = np_time[0, best_global_idx] + np_dwell[best_global_idx] + np_time[best_global_idx, 0]
                prev_pred_time_epoch = np_sched[best_global_idx]
            else:
                cur_duration += np_time[cur, best_global_idx] + np_time[best_global_idx, 0] - np_time[cur, 0] + np_dwell[best_global_idx]
                prev_pred_time_epoch = prev_pred_time_epoch + pre_to_cur + prev_dwell
                
            unvisited[best_i] = False
            unvisited_count -= 1
            cur = best_global_idx
            route_found = True
            
        if route_found:
            v_idx += 1
            
        total_dist += np_dist[cur, 0]
        
        if not route_found and unvisited_count > 0:
            for i in range(n_support):
                if unvisited[i]:
                    cid = support_indices[i]
                    total_dist += np_dist[0, cid] + np_dist[cid, 0]
                    unvisited[i] = False
                    unvisited_count -= 1
                    v_idx += 1
                    break
                    
    return total_dist, v_idx

ALLOC_ROUTES = None
ALLOC_DIST = None
ALLOC_TIME = None
ALLOC_ACO_INSTANCE = None

def init_alloc_worker(main_routes, distance_matrix, time_matrix, aco_instance):
    global ALLOC_ROUTES, ALLOC_DIST, ALLOC_TIME, ALLOC_ACO_INSTANCE
    ALLOC_ROUTES = main_routes
    ALLOC_DIST = distance_matrix
    ALLOC_TIME = time_matrix
    ALLOC_ACO_INSTANCE = aco_instance

def ant_worker(args):
    ant_idx, shared_cache = args
    aco = ALLOC_ACO_INSTANCE
    ant_solution, ant_routes_info, remaining_pool = aco._ant_route_construction()
    cost, vn = aco._evaluate_solution(ant_solution, ant_routes_info, remaining_pool)
    
    compact_routes = {r_id: [s['store_id'] for s in r_info['stores']] for r_id, r_info in ant_routes_info.items()}
    compact_support = [s['store_id'] for s in remaining_pool]
    
    return {
        'solution': ant_solution, 
        'cost': cost, 
        'vn': vn, 
        'compact_routes': compact_routes, 
        'compact_support': compact_support
    }

class AllocateVND(VND):
    def __init__(self, distance_matrix, time_matrix, vehicle_cost=2000, time_limit=5*60*60, verbose=False):
        super().__init__(distance_matrix, time_matrix, vehicle_cost=vehicle_cost, is_solomon=False, time_limit=time_limit, verbose=verbose)


class FastFitnessEvaluator:
    _cached_mappings = None

    def __init__(self, main_routes, distance_matrix, time_matrix):
        self.main_routes = main_routes
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix
        self.dc = config.DC_CONFIG
        self.time_limit_per_route = 5 * 60 * 60
        self._init_numpy_mappings()

    def _init_numpy_mappings(self):
        if FastFitnessEvaluator._cached_mappings is not None:
            (self.s2i, self.i2s, self.np_dist, self.np_time,
             self.np_volume, self.np_dwell, self.np_earliest,
             self.np_latest, self.np_region, self.np_group, self.np_sched) = FastFitnessEvaluator._cached_mappings
            return

        self.s2i = {self.dc['store_id']: 0}
        self.i2s = {0: self.dc}
        idx = 1
        all_stores = []
        for r in self.main_routes.values():
            all_stores.extend(r['stores'])

        for s in all_stores:
            if s['store_id'] not in self.s2i:
                self.s2i[s['store_id']] = idx
                self.i2s[idx] = s
                idx += 1

        n = len(self.s2i)
        self.np_dist = np.zeros((n, n), dtype=np.float64)
        self.np_time = np.zeros((n, n), dtype=np.float64)
        self.np_volume = np.zeros(n, dtype=np.float64)
        self.np_dwell = np.zeros(n, dtype=np.float64)
        self.np_earliest = np.zeros(n, dtype=np.float64)
        self.np_latest = np.zeros(n, dtype=np.float64)
        self.np_region = np.full(n, -1, dtype=np.int64)
        self.np_group = np.full(n, -1, dtype=np.int64)
        self.np_sched = np.zeros(n, dtype=np.float64)

        region_map = {'north': 0, 'south': 1, 'east': 2, 'west': 3}
        group_map = {'T': 0, 'C': 1, 'H': 2}
        for i in range(1, n):
            s_i = self.i2s[i]
            s_i_id = s_i['store_id']
            self.np_volume[i] = s_i.get('volume', 0.0)
            self.np_dwell[i] = float(s_i.get('dwell_time', 0))
            self.np_earliest[i] = float(int(datetime.fromisoformat(s_i['earliest_time']).timestamp()))
            self.np_latest[i] = float(int(datetime.fromisoformat(s_i['latest_time']).timestamp()))
            self.np_region[i] = region_map.get(s_i.get('region', ''), -1)
            self.np_group[i] = group_map.get(s_i.get('dist_group', ''), -1)
            
            sched_str = s_i.get('sched_time', s_i.get('pred_time', s_i['earliest_time']))
            self.np_sched[i] = float(int(datetime.fromisoformat(sched_str).timestamp()))

            for j in range(n):
                s_j_id = self.i2s[j]['store_id']
                if s_i_id in self.distance_matrix and s_j_id in self.distance_matrix[s_i_id]:
                    self.np_dist[i, j] = self.distance_matrix[s_i_id][s_j_id]
                if s_i_id in self.time_matrix and s_j_id in self.time_matrix[s_i_id]:
                    self.np_time[i, j] = self.time_matrix[s_i_id][s_j_id]

        for j in range(n):
            s_j_id = self.i2s[j]['store_id']
            dc_id = self.dc['store_id']
            if dc_id in self.distance_matrix and s_j_id in self.distance_matrix[dc_id]:
                self.np_dist[0, j] = self.distance_matrix[dc_id][s_j_id]
            if dc_id in self.time_matrix and s_j_id in self.time_matrix[dc_id]:
                self.np_time[0, j] = self.time_matrix[dc_id][s_j_id]

        FastFitnessEvaluator._cached_mappings = (
            self.s2i, self.i2s, self.np_dist, self.np_time,
            self.np_volume, self.np_dwell, self.np_earliest,
            self.np_latest, self.np_region, self.np_group, self.np_sched
        )

    def _compute_route_code_position_range(self, route_id, store_idx, route_stores_idx):
        store = self.i2s[store_idx]
        store_rc = store.get('route_code', '')
        if not store_rc or not store_rc.startswith(route_id) or len(store_rc) < 2:
            return 1, len(route_stores_idx) + 2

        try:
            store_seq = int(store_rc[2:])
        except ValueError:
            return 1, len(route_stores_idx) + 2

        pos_low = 1
        pos_high = len(route_stores_idx) + 2

        for i, curr_idx in enumerate(route_stores_idx):
            curr_store = self.i2s[curr_idx]
            curr_rc = curr_store.get('route_code', '')
            
            if curr_rc and curr_rc.startswith(route_id):
                try:
                    curr_seq = int(curr_rc[2:])
                    if curr_seq < store_seq:
                        pos_low = max(pos_low, i + 2)
                    elif curr_seq > store_seq:
                        pos_high = min(pos_high, i + 2)
                except ValueError:
                    pass

        return pos_low, pos_high

    def _fast_get_store_insertion_cost_and_pos(self, r_id, store_idx, route_stores_idx, route_vols, route_caps, route_regions):
        pos_low, pos_high = self._compute_route_code_position_range(r_id, store_idx, route_stores_idx[r_id])
        
        cost, pos = _njit_get_store_insertion_cost_pos_ranged(
            store_idx, np.array(route_stores_idx[r_id], dtype=np.int64), 0, route_regions[r_id],
            route_vols[r_id], route_caps[r_id],
            self.np_region[store_idx], self.np_volume[store_idx],
            self.np_dist, self.np_time, self.np_dwell,
            self.np_earliest, self.np_latest,
            self.np_sched, float(self.time_limit_per_route),
            pos_low, pos_high
        )
        if cost < 0:
            return float('inf'), -1
        return cost, pos

    def fast_evaluate_fitness(self, route_stores_dicts, extracted_stores):
        route_stores_idx = {r_id: [self.s2i[s['store_id']] for s in r_info['stores']] for r_id, r_info in route_stores_dicts.items()}
        route_vols = {r_id: float(r_info['dc'].get('total_volume', 0)) for r_id, r_info in route_stores_dicts.items()}
        route_caps = {r_id: float(r_info['dc'].get('max_capacity', 1e9)) for r_id, r_info in route_stores_dicts.items()}
        dc_region_map = {'north': 0, 'south': 1, 'east': 2, 'west': 3}
        route_regions = {r_id: dc_region_map.get(r_info['dc'].get('region', ''), -1) for r_id, r_info in route_stores_dicts.items()}

        support_pool_indices = []

        store_with_costs = []
        for store in extracted_stores:
            store_idx = self.s2i[store['store_id']]
            min_cost = float('inf')
            
            for r_id in route_stores_dicts.keys():
                cost, pos = self._fast_get_store_insertion_cost_and_pos(
                    r_id, store_idx, route_stores_idx, route_vols, route_caps, route_regions)
                if pos != -1 and cost < min_cost:
                    min_cost = cost
                    
            store_with_costs.append({
                'store': store,
                'store_idx': store_idx,
                'min_cost': min_cost
            })
            
        store_with_costs.sort(key=lambda x: x['min_cost'])
        
        for item in store_with_costs:
            store_idx = item['store_idx']
            best_r = 'SUPPORT'
            min_cost = float('inf')
            best_pos = -1
            
            for r_id in route_stores_dicts.keys():
                cost, pos = self._fast_get_store_insertion_cost_and_pos(
                    r_id, store_idx, route_stores_idx, route_vols, route_caps, route_regions)
                if pos != -1 and cost < min_cost:
                    min_cost = cost
                    best_r = r_id
                    best_pos = pos
                    
            if best_r != 'SUPPORT':
                route_stores_idx[best_r].insert(best_pos, store_idx)
                route_vols[best_r] += self.np_volume[store_idx]
            else:
                support_pool_indices.append(store_idx)

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

        if not support_pool_indices:
            return main_cost, 0

        support_capacity = float(config.DC_CONFIG.get('support_capacity', 7.2) if hasattr(config, 'DC_CONFIG') else 7.2)
        
        support_arr = np.array(support_pool_indices, dtype=np.int64)
        nn_dist, nn_vn = _njit_global_nearest_neighbor(
            support_arr, self.np_dist, self.np_time, self.np_volume, self.np_dwell, 
            self.np_earliest, self.np_latest, self.np_sched, support_capacity, float(self.time_limit_per_route),
            self.np_group, self.np_region
        )

        nn_cost = nn_dist + (nn_vn * 2000.0)
        return main_cost + nn_cost, nn_vn

class StoreAllocationACO:
    _cached_mappings = None

    def __init__(self, main_routes, remaining_stores, distance_matrix, time_matrix,
                 num_ants=5, iterations=100, early_stop_patience=20, 
                 mode='aco', output_dir=None, verbose=True, **kwargs):
        self.mode = mode
        self.output_dir = output_dir
        self.verbose = verbose
        self.main_routes = main_routes
        self.remaining_stores = remaining_stores
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix
        self.dc = config.DC_CONFIG
        
        # Support both names for parameters
        self.num_ants = num_ants
        self.iterations = iterations
        self.early_stop_patience = early_stop_patience
        
        self.q0 = kwargs.get('q0', 0.8)
        self.rho = kwargs.get('rho', 0.2)
        self.alpha1 = kwargs.get('alpha1', 0.25)
        self.alpha2 = kwargs.get('alpha2', 0.25)
        self.alpha3 = kwargs.get('alpha3', 0.25)
        self.alpha4 = kwargs.get('alpha4', 0.25)
        
        self.route_choices = list(self.main_routes.keys())
        self.time_limit_per_route = 5 * 60 * 60
        self.best_cost = float('inf')
        self.best_solution = None
        self.best_remaining_solution = None
        self.best_ant_choices = None
        self.best_vn = float('inf')
        self.log = []

        self.stores_by_id = {s['store_id']: s for s in self.remaining_stores}
        for r in self.main_routes.values():
            for s in r['stores']:
                self.stores_by_id[s['store_id']] = s
        
        self._init_numpy_mappings()
        # self.remaining_stores = self._sort_stores_by_insertion_cost(remaining_stores)
        self.pheromone_matrix = {}

    def _init_numpy_mappings(self):
        if StoreAllocationACO._cached_mappings is not None:
            (self.s2i, self.i2s, self.np_dist, self.np_time,
             self.np_volume, self.np_dwell, self.np_earliest,
             self.np_latest, self.np_region, self.np_group, self.np_sched) = StoreAllocationACO._cached_mappings
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
        self.np_dist = np.zeros((n, n), dtype=np.float64)
        self.np_time = np.zeros((n, n), dtype=np.float64)
        self.np_volume = np.zeros(n, dtype=np.float64)
        self.np_dwell = np.zeros(n, dtype=np.float64)
        self.np_earliest = np.zeros(n, dtype=np.float64)
        self.np_latest = np.zeros(n, dtype=np.float64)
        self.np_region = np.full(n, -1, dtype=np.int64)
        self.np_group = np.full(n, -1, dtype=np.int64)
        self.np_sched = np.zeros(n, dtype=np.float64)

        region_map = {'north': 0, 'south': 1, 'east': 2, 'west': 3}
        group_map = {'T': 0, 'C': 1, 'H': 2}
        for i in range(1, n):
            s_i = self.i2s[i]
            s_i_id = s_i['store_id']
            self.np_volume[i] = s_i.get('volume', 0.0)
            self.np_dwell[i] = float(s_i.get('dwell_time', 0))
            self.np_earliest[i] = float(int(datetime.fromisoformat(s_i['earliest_time']).timestamp()))
            self.np_latest[i] = float(int(datetime.fromisoformat(s_i['latest_time']).timestamp()))
            self.np_region[i] = region_map.get(s_i.get('region', ''), -1)
            self.np_group[i] = group_map.get(s_i.get('dist_group', ''), -1)
            
            sched_str = s_i.get('sched_time', s_i.get('pred_time', s_i['earliest_time']))
            self.np_sched[i] = float(int(datetime.fromisoformat(sched_str).timestamp()))

            for j in range(n):
                s_j_id = self.i2s[j]['store_id']
                if s_i_id in self.distance_matrix and s_j_id in self.distance_matrix[s_i_id]:
                    self.np_dist[i, j] = self.distance_matrix[s_i_id][s_j_id]
                if s_i_id in self.time_matrix and s_j_id in self.time_matrix[s_i_id]:
                    self.np_time[i, j] = self.time_matrix[s_i_id][s_j_id]

        for j in range(n):
            s_j_id = self.i2s[j]['store_id']
            dc_id = self.dc['store_id']
            if dc_id in self.distance_matrix and s_j_id in self.distance_matrix[dc_id]:
                self.np_dist[0, j] = self.distance_matrix[dc_id][s_j_id]
            if dc_id in self.time_matrix and s_j_id in self.time_matrix[dc_id]:
                self.np_time[0, j] = self.time_matrix[dc_id][s_j_id]

        StoreAllocationACO._cached_mappings = (
            self.s2i, self.i2s, self.np_dist, self.np_time,
            self.np_volume, self.np_dwell, self.np_earliest,
            self.np_latest, self.np_region, self.np_group, self.np_sched
        )

    def _copy_routes_info(self, routes):
        return {
            route_id: {
                "dc": route_data["dc"].copy(),
                "stores": [s.copy() for s in route_data["stores"]]
            }
            for route_id, route_data in routes.items()
        }



    def _evaluate_solution(self, solution, route_manager_routes, support_pool, proxy=False):
        paths = NumbaList()
        for r_id, r_info in route_manager_routes.items():
            if not r_info['stores']: continue
            s_indices = [self.s2i[s['store_id']] for s in r_info['stores']]
            path = np.empty(len(s_indices) + 2, dtype=np.int64)
            path[0] = 0
            for k, idx in enumerate(s_indices):
                path[k + 1] = idx
            path[-1] = 0
            paths.append(path)
            
        main_cost = float(_njit_total_distance(paths, self.np_dist))
        
        if proxy:
            support_pool_indices = [self.s2i[s['store_id']] for s in support_pool]
            if not support_pool_indices:
                return main_cost, 0
                
            support_capacity = float(self.dc.get('support_capacity', 7.2))
            support_arr = np.array(support_pool_indices, dtype=np.int64)
            
            nn_dist, nn_vn = _njit_global_nearest_neighbor(
                support_arr, self.np_dist, self.np_time, self.np_volume, self.np_dwell, 
                self.np_earliest, self.np_latest, self.np_sched, support_capacity, float(self.time_limit_per_route),
                self.np_group, self.np_region
            )
            return main_cost + nn_dist + (nn_vn * 2000.0), nn_vn
        else:
            macs = SupportLinePlanningMACS(support_pool, self.distance_matrix, self.time_matrix, time_limit=0, verbose=False)
            macs_cost, support_routes = macs.run()
            vn = len(support_routes)
            total_cost = main_cost + macs_cost
            return total_cost, vn

    def _ant_route_construction(self):
        solution = {}
        temp_routes = self._copy_routes_info(self.main_routes)
        route_manager = RouteManager(temp_routes, self.distance_matrix, self.time_matrix)
        
        route_vols = {r_id: float(r_info['dc'].get('total_volume', 0)) for r_id, r_info in self.main_routes.items()}
        route_caps = {r_id: float(r_info['dc'].get('max_capacity', 1e9)) for r_id, r_info in self.main_routes.items()}
        
        support_pool = []
        
        Cs = list(self.remaining_stores)
        Vk = list(self.route_choices)
        random.shuffle(Vk)
        
        # Initialize route_stores_idx and route_stores_np ONCE before loop
        route_stores_idx = {
            r_id: [self.s2i[s['store_id']] for s in route_manager.routes_info[r_id]['stores']]
            for r_id in self.main_routes.keys()
        }
        route_stores_np = {
            r_id: np.array(route_stores_idx[r_id], dtype=np.int64)
            for r_id in self.main_routes.keys()
        }
        
        for k in Vk:
            if not Cs:
                break
                
            Select_cust = True
            
            while Cs and Select_cust:
                Select_cust = False
                q = random.random()
                
                feasible_moves = []
                
                for store in Cs:
                    store_idx = self.s2i[store['store_id']]
                    attractions = _njit_greedy_function_attractions(
                        store_idx, route_stores_np[k], 0,
                        route_vols[k], route_caps[k],
                        self.np_volume[store_idx],
                        self.np_dist, self.np_time, self.np_dwell,
                        self.np_earliest, self.np_latest, self.np_sched, self.np_region, float(self.time_limit_per_route)
                    )
                    
                    for pos in range(len(attractions)):
                        attraction = attractions[pos]
                        if attraction > 0:
                            r_stores = route_stores_idx[k]
                            prev_idx = r_stores[pos - 1] if pos > 0 else 0
                            tau = self.pheromone_matrix[prev_idx, store_idx]
                            feasible_moves.append({
                                'store': store,
                                'pos': pos,
                                'attraction': attraction,
                                'tau': tau,
                                'prev_idx': prev_idx
                            })
                
                if feasible_moves:
                    if q <= self.q0:
                        # Exploitation
                        best_move = max(feasible_moves, key=lambda x: x['tau'] * x['attraction'])
                    else:
                        # Exploration (Roulette Wheel)
                        probs = np.array([x['tau'] * x['attraction'] for x in feasible_moves])
                        if probs.sum() > 0:
                            probs = probs / probs.sum()
                            best_move = random.choices(feasible_moves, weights=probs, k=1)[0]
                        else:
                            best_move = random.choice(feasible_moves)
                            
                    # Insert store
                    store = best_move['store']
                    pos = best_move['pos']
                    store_idx = self.s2i[store['store_id']]
                    
                    solution[store['store_id']] = k
                    Cs.remove(store)
                    
                    route_manager.insert_store(store, k, pos, fast_update=True)
                    route_vols[k] += self.np_volume[store_idx]
                    
                    route_stores_idx[k].insert(pos, store_idx)
                    route_stores_np[k] = np.array(route_stores_idx[k], dtype=np.int64)
                    
                    Select_cust = True
                    
        # Any stores left in Cs go to support
        for store in Cs:
            solution[store['store_id']] = 'SUPPORT'
            support_pool.append(store)
            
        # Local Pheromone Update (decay traversed edges back toward initial_tau)
        for s_id, r_id in solution.items():
            if r_id == 'SUPPORT': continue
            s_idx = self.s2i[s_id]
            route_list = route_stores_idx[r_id]
            pos = route_list.index(s_idx)
            prev_i = route_list[pos - 1] if pos > 0 else 0
            self.pheromone_matrix[prev_i, s_idx] = (
                (1 - self.rho) * self.pheromone_matrix[prev_i, s_idx] + self.rho * self.initial_tau
            )
            
        return solution, route_manager.routes_info, support_pool

    def _evaluate_proxy_cost(self, rm_routes, support_list):
        paths = NumbaList()
        for r_id, r_info in rm_routes.items():
            if not r_info['stores']: continue
            s_indices = [self.s2i[s['store_id']] for s in r_info['stores']]
            path = np.empty(len(s_indices) + 2, dtype=np.int64)
            path[0] = 0
            for k, idx in enumerate(s_indices):
                path[k + 1] = idx
            path[-1] = 0
            paths.append(path)
            
        main_cost = float(_njit_total_distance(paths, self.np_dist))
        # Proxy penalty: 2000 for each store relegated to support
        return main_cost + len(support_list) * 2000

    def _calculate_route_distance(self, stores):
        if not stores: return 0.0
        dc_id = self.dc['store_id']
        total = self.distance_matrix[dc_id][stores[0]['store_id']]
        for prev, curr in zip(stores[:-1], stores[1:]):
            total += self.distance_matrix[prev['store_id']][curr['store_id']]
        total += self.distance_matrix[stores[-1]['store_id']][dc_id]
        return total

    def _compute_route_code_position_range(self, route_id, store_idx, route_stores_idx):
        store = self.i2s[store_idx]
        store_rc = store.get('route_code', '')
        
        if not store_rc or not store_rc.startswith(route_id):
            return 1, len(route_stores_idx) + 2

        try:
            store_seq = int(store_rc[2:])
        except ValueError:
            return 1, len(route_stores_idx) + 2

        pos_low = 1
        pos_high = len(route_stores_idx) + 2

        for i, curr_idx in enumerate(route_stores_idx):
            curr_store = self.i2s[curr_idx]
            curr_rc = curr_store.get('route_code', '')
            
            if curr_rc and curr_rc.startswith(route_id):
                try:
                    curr_seq = int(curr_rc[2:])
                    if curr_seq < store_seq:
                        pos_low = max(pos_low, i + 2)
                    elif curr_seq > store_seq:
                        pos_high = min(pos_high, i + 2)
                except ValueError:
                    pass

        return pos_low, pos_high

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

    def _get_store_insertion_cost_and_pos(self, route_info, store):
        dc = route_info['dc']
        stores = route_info['stores']
        store_idx = self.s2i[store['store_id']]
        route_stores_idx = [self.s2i[s['store_id']] for s in stores]
        route_stores = np.array(route_stores_idx, dtype=np.int64)

        dc_region_map = {'north': 0, 'south': 1, 'east': 2, 'west': 3}
        dc_region_int = dc_region_map.get(dc.get('region', ''), -1)
        store_region_int = self.np_region[store_idx]

        route_id = ""
        prefix_counts = {}
        for s in stores:
            rc = s.get('route_code', '')
            if rc and len(rc) >= 2:
                prefix = rc[:2]
                prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
        
        if prefix_counts:
            route_id = max(prefix_counts, key=prefix_counts.get)

        pos_low, pos_high = self._compute_route_code_position_range(route_id, store_idx, route_stores_idx)

        cost, pos = _njit_get_store_insertion_cost_pos_ranged(
            store_idx, route_stores, 0, dc_region_int,
            float(dc.get('total_volume', 0)), float(dc.get('max_capacity', 1e9)),
            store_region_int, float(store.get('volume', 0)),
            self.np_dist, self.np_time, self.np_dwell,
            self.np_earliest, self.np_latest,
            self.np_sched, float(self.time_limit_per_route),
            pos_low, pos_high
        )

        if cost < 0:
            return float('inf'), -1
        return cost, pos

    def _fast_get_store_insertion_cost_and_pos(self, r_id, store_idx, route_stores_idx, route_vols, route_caps, route_regions):
        pos_low, pos_high = self._compute_route_code_position_range(r_id, store_idx, route_stores_idx[r_id])
        
        cost, pos = _njit_get_store_insertion_cost_pos_ranged(
            store_idx, np.array(route_stores_idx[r_id], dtype=np.int64), 0, route_regions[r_id],
            route_vols[r_id], route_caps[r_id],
            self.np_region[store_idx], self.np_volume[store_idx],
            self.np_dist, self.np_time, self.np_dwell,
            self.np_earliest, self.np_latest,
            self.np_sched, float(self.time_limit_per_route),
            pos_low, pos_high
        )
        if cost < 0:
            return float('inf'), -1
        return cost, pos

    def _generate_greedy_solution(self, return_routes=False):
        greedy_choices = []
        support_pool_indices = []

        route_stores_idx = {r_id: [self.s2i[s['store_id']] for s in r_info['stores']] for r_id, r_info in self.main_routes.items()}
        route_vols = {r_id: float(r_info['dc'].get('total_volume', 0)) for r_id, r_info in self.main_routes.items()}
        route_caps = {r_id: float(r_info['dc'].get('max_capacity', 1e9)) for r_id, r_info in self.main_routes.items()}
        dc_region_map  = {'north': 0, 'south': 1, 'east': 2, 'west': 3}
        route_regions = {r_id: dc_region_map.get(r_info['dc'].get('region', ''), -1) for r_id, r_info in self.main_routes.items()}

        # 1. Sort remaining stores by insertion cost (Identical to MACS behavior and fast_evaluate_fitness)
        store_with_costs = []
        for i, store in enumerate(self.remaining_stores):
            store_idx = self.s2i[store['store_id']]
            min_cost = float('inf')
            
            for r_id in self.main_routes.keys():
                cost, pos = self._fast_get_store_insertion_cost_and_pos(
                    r_id, store_idx, route_stores_idx, route_vols, route_caps, route_regions)
                if pos != -1 and cost < min_cost:
                    min_cost = cost
                    
            store_with_costs.append({
                'store': store,
                'orig_idx': i,
                'store_idx': store_idx,
                'min_cost': min_cost
            })
            
        store_with_costs.sort(key=lambda x: x['min_cost'])
        
        greedy_choices = ['SUPPORT'] * len(self.remaining_stores)

        for item in store_with_costs:
            store_idx = item['store_idx']
            orig_idx = item['orig_idx']
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

            greedy_choices[orig_idx] = best_r
            if best_r != 'SUPPORT':
                route_stores_idx[best_r].insert(best_pos, store_idx)
                route_vols[best_r] += self.np_volume[store_idx]
            else:
                support_pool_indices.append(orig_idx)

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
        
        # Use exact same NN evaluation as fast_evaluate_fitness
        if not support_pool_indices:
            nn_dist, nn_vn = 0.0, 0
        else:
            support_capacity = float(self.dc.get('support_capacity', 7.2))
            support_arr = np.array([self.s2i[s['store_id']] for s in support_pool], dtype=np.int64)
            nn_dist, nn_vn = _njit_global_nearest_neighbor(
                support_arr, self.np_dist, self.np_time, self.np_volume, self.np_dwell, 
                self.np_earliest, self.np_latest, self.np_sched, support_capacity, float(self.time_limit_per_route),
                self.np_group, self.np_region
            )
            
        total_cost = main_cost + nn_dist + (nn_vn * 2000.0)
        vn = nn_vn

        if return_routes:
            temp_routes = self._copy_routes_info(self.main_routes)
            for r_id, indices in route_stores_idx.items():
                temp_routes[r_id]['stores'] = [self.i2s[idx] for idx in indices]
                vol = sum(self.i2s[idx].get('volume', 0) for idx in indices)
                temp_routes[r_id]['dc']['total_volume'] = vol
                cap = temp_routes[r_id]['dc'].get('max_capacity', 1e9)
                temp_routes[r_id]['dc']['load_rate'] = vol / cap if cap > 0 else 0
                
            return greedy_choices, total_cost, temp_routes, support_pool, vn

        return greedy_choices, total_cost, None, None, vn

    def fast_evaluate_fitness(self, copy_routes, extracted_stores):
        """
        Fast evaluation function replacing object instantiation in fitness_worker.
        Uses cached numpy arrays to perform greedy insertion and Nearest Neighbor.
        """
        route_stores_idx = {r_id: [self.s2i[s['store_id']] for s in r_info['stores']] for r_id, r_info in copy_routes.items()}
        route_vols = {r_id: float(r_info['dc'].get('total_volume', 0)) for r_id, r_info in copy_routes.items()}
        route_caps = {r_id: float(r_info['dc'].get('max_capacity', 1e9)) for r_id, r_info in copy_routes.items()}
        dc_region_map = {'north': 0, 'south': 1, 'east': 2, 'west': 3}
        route_regions = {r_id: dc_region_map.get(r_info['dc'].get('region', ''), -1) for r_id, r_info in copy_routes.items()}

        support_pool_indices = []

        # 1. Sort extracted stores by insertion cost (Identical to MACS behavior)
        store_with_costs = []
        for store in extracted_stores:
            store_idx = self.s2i[store['store_id']]
            min_cost = float('inf')
            
            for r_id in copy_routes.keys():
                cost, pos = self._fast_get_store_insertion_cost_and_pos(
                    r_id, store_idx, route_stores_idx, route_vols, route_caps, route_regions)
                if pos != -1 and cost < min_cost:
                    min_cost = cost
                    
            store_with_costs.append({
                'store': store,
                'store_idx': store_idx,
                'min_cost': min_cost
            })
            
        store_with_costs.sort(key=lambda x: x['min_cost'])
        
        # 1.5 Greedy Insertion in sorted order
        for item in store_with_costs:
            store_idx = item['store_idx']
            best_r = 'SUPPORT'
            min_cost = float('inf')
            best_pos = -1
            
            # Re-evaluate insertion cost because routes might have changed!
            for r_id in copy_routes.keys():
                cost, pos = self._fast_get_store_insertion_cost_and_pos(
                    r_id, store_idx, route_stores_idx, route_vols, route_caps, route_regions)
                if pos != -1 and cost < min_cost:
                    min_cost = cost
                    best_r = r_id
                    best_pos = pos
                    
            if best_r != 'SUPPORT':
                route_stores_idx[best_r].insert(best_pos, store_idx)
                route_vols[best_r] += self.np_volume[store_idx]
            else:
                support_pool_indices.append(store_idx)

        # 2. Main Route Distance
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

        # 3. Fast Nearest Neighbor for Support Lines
        if not support_pool_indices:
            return main_cost, 0

        # default support capacity
        support_capacity = float(config.DC_CONFIG.get('support_capacity', 7.2) if hasattr(config, 'DC_CONFIG') else 7.2)
        
        support_arr = np.array(support_pool_indices, dtype=np.int64)
        nn_dist, nn_vn = _njit_global_nearest_neighbor(
            support_arr, self.np_dist, self.np_time, self.np_volume, self.np_dwell, 
            self.np_earliest, self.np_latest, self.np_sched, support_capacity, float(self.time_limit_per_route),
            self.np_group, self.np_region
        )

        # MACS cost normally includes the vehicle_cost for each support line
        nn_cost = nn_dist + (nn_vn * 2000.0)

        return main_cost + nn_cost, nn_vn

    def _vnd(self, solution, route_manager_routes, support_pool, best_cost):
        """
        Integrates AllocateVND local search.
        Moves stores from SUPPORT to MAIN routes first (greedily),
        then creates temporary support routes from the remaining support pool.
        VND is then applied across both main routes and support routes.
        """
        current_solution = solution.copy()
        temp_routes = self._copy_routes_info(route_manager_routes)
        rm = RouteManager(temp_routes, self.distance_matrix, self.time_matrix)
        current_support = [s for s in support_pool]
        
        # 1. Identify movable stores (stores not in their assigned route based on prefix)
        movable_stores = set()
        for s in self.remaining_stores:
            s_id = s['store_id']
            curr_route = current_solution.get(s_id, 'SUPPORT')
            rc = s.get('route_code', '')
            prefix = rc[:2] if rc and len(rc) >= 2 else ''
            
            if curr_route != 'SUPPORT' and curr_route.startswith(prefix):
                continue
            movable_stores.add(s_id)

        if not movable_stores:
            return current_solution, rm.routes_info, current_support, best_cost, len(current_support)


        if self.mode == 'aco_vnd':
            # 3. Call AllocateVND to optimize the main routes
            vnd_solver = AllocateVND(self.distance_matrix, self.time_matrix, time_limit=self.time_limit_per_route, verbose=False)
            optimized_routes, vnd_cost = vnd_solver.optimize(rm.routes_info, movable_stores=movable_stores)
            
            # Update current_solution to reflect VND changes
            for r_id, r_data in optimized_routes.items():
                for s in r_data['stores']:
                    current_solution[s['store_id']] = r_id
                    
            rm.routes_info = optimized_routes
        
        # After VND, recalculate true MACS cost
        final_cost, vn = self._evaluate_solution(current_solution, rm.routes_info, current_support)
        return current_solution, rm.routes_info, current_support, final_cost, vn
        
    def run(self, return_routes=True):
        if self.iterations == 0 and self.num_ants == 0:
            if not return_routes:
                _, greedy_cost, _, _, greedy_vn = self._generate_greedy_solution(return_routes=False)
                return greedy_cost, None, None, greedy_vn
                
        # 1. Greedy initialization to find tau0 using the GA greedy heuristic
        greedy_choices, greedy_cost, greedy_routes_info, greedy_pool, greedy_vn = self._generate_greedy_solution(return_routes=True)
        greedy_solution = {self.remaining_stores[idx]['store_id']: r_id for idx, r_id in enumerate(greedy_choices)}
        
        self.best_cost = greedy_cost
        self.best_solution = greedy_routes_info
        self.best_ant_choices = greedy_solution
        self.best_remaining_solution = greedy_pool
        self.best_vn = greedy_vn
        
        greedy_cost_main = greedy_cost - greedy_vn * 2000
        if self.verbose:
            print(f'Store Allocation ACO: iteration0 -> vn = {greedy_vn}, cost = {greedy_cost_main:.2f}, fitness = {self.best_cost:.2f}')
        
        self.initial_tau = 1.0 / (len(self.remaining_stores) * greedy_cost) if greedy_cost > 0 else 1e-4
        
        # Reset edge pheromone matrix  (n × n, all nodes including depot)
        n = len(self.s2i)
        self.pheromone_matrix = np.full((n, n), self.initial_tau, dtype=np.float64)
            
        early_stopper = EarlyStopper(patience=self.early_stop_patience)
 
        for i in range(self.iterations):
            iter_results = []
            for ant_idx in range(self.num_ants):
                ant_solution, ant_routes_info, remaining_pool = self._ant_route_construction()
                cost, vn = self._evaluate_solution(ant_solution, ant_routes_info, remaining_pool, proxy=True)
                
                # Apply VND to every ant (true cost evaluation + local search)
                # true_cost, true_vn = self._evaluate_solution(ant_solution, ant_routes_info, remaining_pool, proxy=False)
                # vnts_solution, vnts_routes, vnts_support, vnts_cost, vnts_vn = self._vnts(
                #     ant_solution, ant_routes_info, remaining_pool, true_cost
                # )
                # iter_results.append({
                #     'solution': vnts_solution,
                #     'routes_info': vnts_routes,
                #     'support_pool': vnts_support,
                #     'cost': vnts_cost,
                #     'vn': vnts_vn
                # })

                # Only evaluate true cost for every ant, no VND here
                true_cost, true_vn = self._evaluate_solution(ant_solution, ant_routes_info, remaining_pool, proxy=False)
                iter_results.append({
                    'solution': ant_solution,
                    'routes_info': ant_routes_info,
                    'support_pool': remaining_pool,
                    'cost': true_cost,
                    'vn': true_vn
                })
                
            iter_best = min(iter_results, key=lambda x: x['cost'])
            
            # iter_best is already VND-optimized; update global best directly
            # vnts_cost = iter_best['cost']
            # vnts_vn = iter_best['vn']
            # vnts_solution = iter_best['solution']
            # vnts_routes = iter_best['routes_info']
            # vnts_support = iter_best['support_pool']

            # Apply VND only to iter_best if mode is 'aco_vnd'
            if self.mode == 'aco_vnd':
                vnd_solution, vnd_routes, vnd_support, vnd_cost, vnd_vn = self._vnd(
                    iter_best['solution'], iter_best['routes_info'], iter_best['support_pool'], iter_best['cost']
                )
            else:
                vnd_solution, vnd_routes, vnd_support, vnd_cost, vnd_vn = (
                    iter_best['solution'], iter_best['routes_info'], iter_best['support_pool'], iter_best['cost'], iter_best['vn']
                )
            
            if vnd_cost < self.best_cost:
                self.best_cost = vnd_cost
                self.best_solution = vnd_routes
                self.best_ant_choices = vnd_solution
                self.best_remaining_solution = vnd_support
                self.best_vn = vnd_vn
                
            # Global Pheromone Update: reinforce edges in best solution routes
            delta_tau = 1.0 / self.best_cost
            for r_id, r_data in self.best_solution.items():
                s_indices = [self.s2i[s['store_id']] for s in r_data['stores']]
                if not s_indices: continue
                path = [0] + s_indices + [0]  # depot → stores → depot
                for k in range(len(path) - 1):
                    from_node, to_node = path[k], path[k + 1]
                    self.pheromone_matrix[from_node, to_node] = (
                        (1 - self.rho) * self.pheromone_matrix[from_node, to_node] + self.rho * delta_tau
                    )
                
            best_cost_main = self.best_cost - self.best_vn * 2000
            if self.verbose:
                print(f'Store Allocation ACO: iteration{i + 1} -> vn = {self.best_vn}, cost = {best_cost_main:.2f}, fitness = {self.best_cost:.2f}')
            
            fitnesses = [res['cost'] for res in iter_results]
            self.log.append({
                'iteration': i + 1,
                'iter_worst_cost': float(np.max(fitnesses)),
                'iter_best_cost': float(vnd_cost),
                'iter_avg_cost': float(np.mean(fitnesses)),
                'std_cost': float(np.std(fitnesses)),
                'best_cost': self.best_cost,
            })
            
            if early_stopper.check(self.best_cost):
                if self.verbose:
                    print(f"Store Allocation ACO: Early stop triggered at iteration {i + 1}.")
                break
            
        # Update all route details for the final best solution before returning
        rm = RouteManager(self.best_solution, self.distance_matrix, self.time_matrix)
        rm.update_all_routes_info()
        
        macs = SupportLinePlanningMACS(self.best_remaining_solution, self.distance_matrix, self.time_matrix, time_limit=0, verbose=False)
        return self.best_cost, self.best_solution, self.best_remaining_solution, self.best_vn