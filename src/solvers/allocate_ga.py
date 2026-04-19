import copy
import random
import hashlib
import numpy as np
from config import config
from datetime import datetime, timedelta
from multiprocessing import cpu_count, Manager
import concurrent.futures
from numba import njit
from models.route_manager import RouteManager
from utils.early_stopper import EarlyStopper
from solvers.support_line_aco import SupportLinePlanningACO, _njit_get_feasible_stores, _njit_greedy_selection
from solvers.support_line_ga import SupportLinePlanningGA

@njit(cache=True)
def _njit_check_time_constraint(full_route, dc_idx, distance_matrix, time_matrix, 
                                dwell_arr, earliest_arr, latest_arr, first_node_pred_time_epoch, time_limit):
    # full_route: [dc_idx, store1, ..., storeN, dc_idx]
    
    # Check duration first:
    duration_sec = time_matrix[dc_idx, full_route[1]] + time_matrix[full_route[-2], dc_idx]
    
    # For arrival times: starts from first actual store (index 1)
    curr_pred_time = first_node_pred_time_epoch
    
    for i in range(2, len(full_route) - 1):
        prev_idx = full_route[i - 1]
        curr_idx = full_route[i]
        
        travel_time = time_matrix[prev_idx, curr_idx]
        pre_dwell = dwell_arr[prev_idx]
        
        arrival_time = curr_pred_time + travel_time + pre_dwell
        
        if arrival_time < earliest_arr[curr_idx] or arrival_time > latest_arr[curr_idx]:
            return False
            
        duration_sec += travel_time + pre_dwell
        curr_pred_time = arrival_time
        
    if duration_sec > time_limit:
        return False
        
    return True

@njit(cache=True)
def _njit_get_store_insertion_cost_pos(store_idx, route_stores, dc_idx, dc_region, route_vol, route_cap,
                                       store_region, store_vol, dist_matrix, time_matrix,
                                       dwell_arr, earliest_arr, latest_arr, first_pred_epoch, time_limit):
                                       
    # Region constraint
    if dc_region == 0 and store_region == 1: return -1.0, -1
    if dc_region == 1 and store_region == 0: return -1.0, -1
    if dc_region == 2 and store_region == 3: return -1.0, -1
    if dc_region == 3 and store_region == 2: return -1.0, -1
    
    # Capacity constraint
    if route_vol + store_vol > route_cap:
        return -1.0, -1
        
    L = len(route_stores)
    best_pos = -1
    min_cost = 1e12
    
    # Construct base full route [DC, ..., DC]
    full_route = np.zeros(L + 2, dtype=np.int64)
    full_route[0] = dc_idx
    full_route[L+1] = dc_idx
    for i in range(L):
        full_route[i+1] = route_stores[i]
        
    # Test insertions at pos = 1..L+1 (since POS corresponds to inserting before existing full_route[pos])
    test_route = np.zeros(L + 3, dtype=np.int64)
    
    for pos in range(1, len(full_route)):
        prev_idx = full_route[pos - 1]
        next_idx = full_route[pos]
        
        insert_cost = dist_matrix[prev_idx, store_idx] + dist_matrix[store_idx, next_idx] - dist_matrix[prev_idx, next_idx]
        
        if 0 < insert_cost < min_cost:
            # Build test route
            for i in range(pos):
                test_route[i] = full_route[i]
            test_route[pos] = store_idx
            for i in range(pos, len(full_route)):
                test_route[i+1] = full_route[i]
                
            # If the original route had no stores, the first_pred_epoch doesn't exist.
            # However, logic in python ensures we parse `route[1]` which IS the store we just inserted.
            if _njit_check_time_constraint(test_route, dc_idx, dist_matrix, time_matrix, 
                                           dwell_arr, earliest_arr, latest_arr, first_pred_epoch, time_limit):
                best_pos = pos - 1
                min_cost = insert_cost
                
    return min_cost, best_pos

# ---------------- NUMBA WARMUP ----------------
try:
    _njit_check_time_constraint(np.array([0, 0, 0], dtype=np.int64), 0, np.zeros((1, 1)), np.zeros((1, 1)), np.array([0.0]), np.array([0.0]), np.array([0.0]), 0.0, 10.0)
    _njit_get_store_insertion_cost_pos(0, np.array([0], dtype=np.int64), 0, 0, 0.0, 10.0, 0, 0.0, np.zeros((1, 1)), np.zeros((1, 1)), 
                                       np.array([0.0]), np.array([0.0]), np.array([0.0]), 0.0, 10.0)
except Exception:
    pass
# -----------------------------------------------

ALLOC_ROUTES = None
ALLOC_DIST = None
ALLOC_TIME = None
ALLOC_GA_INSTANCE = None

def init_alloc_worker(main_routes, distance_matrix, time_matrix, ga_instance):
    global ALLOC_ROUTES, ALLOC_DIST, ALLOC_TIME, ALLOC_GA_INSTANCE
    ALLOC_ROUTES = main_routes
    ALLOC_DIST = distance_matrix
    ALLOC_TIME = time_matrix
    ALLOC_GA_INSTANCE = ga_instance

def alloc_fitness_worker(args):
    chromo, key, shared_cache = args

    if key in shared_cache:
        return shared_cache[key]

    ga = ALLOC_GA_INSTANCE
    cost = ga._evaluate_individual_fast(chromo)

    result = {
        'cost': cost
    }

    shared_cache[key] = result
    return result

class StoreAllocationGA:
    """
    Store Allocation using Genetic Algorithm (GA) with Strict Time Window & Capacity Constraints (Numba Optimized).
    """
    _cached_mappings = None

    def __init__(self, main_routes, remaining_stores, distance_matrix, time_matrix, population_size=50, elite_rate=0.1, generations=50, cross_rate=0.8, mutation_rate=0.2, early_stop_patience=100):
        self.main_routes = main_routes
        self.remaining_stores = remaining_stores
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix
        self.dc = config.DC_CONFIG
        self.population_size = population_size
        self.elite_size = int(population_size * elite_rate)
        self.generations = generations
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.early_stop_patience = early_stop_patience
        self.route_choices = list(self.main_routes.keys()) + ['SUPPORT']
        self.time_limit_per_route = 5 * 60 * 60
        self.best_cost = float('inf')
        self.best_solution = None
        self.best_remaining_solution = None
        self.best_individual = None
        self.log = []
        
        self._init_numpy_mappings()
        self.remaining_stores = self._sort_stores_by_insertion_cost(remaining_stores)

    def _init_numpy_mappings(self):
        """Map generic dict data to isolated numpy arrays for njit evaluation calls."""
        if StoreAllocationGA._cached_mappings is not None:
            self.s2i, self.i2s, self.np_dist, self.np_time, self.np_volume, self.np_dwell, \
            self.np_earliest, self.np_latest, self.np_region, self.np_group = StoreAllocationGA._cached_mappings
            return

        self.s2i = {self.dc['store_id']: 0}
        self.i2s = {0: self.dc}
        idx = 1
        # Include ALL stores from main routes AND remaining stores
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
        
        region_map = {'north': 0, 'south': 1, 'east': 2, 'west': 3}
        group_map = {'near': 0, 'mid': 1, 'far': 2}
        for i in range(1, n):
            s_i = self.i2s[i]
            s_i_id = s_i['store_id']
            self.np_volume[i] = s_i.get('volume', 0.0)
            self.np_dwell[i] = float(s_i.get('dwell_time', 0))
            self.np_earliest[i] = float(int(datetime.fromisoformat(s_i['earliest_time']).timestamp()))
            self.np_latest[i] = float(int(datetime.fromisoformat(s_i['latest_time']).timestamp()))
            self.np_region[i] = region_map.get(s_i.get('region', ''), -1)
            self.np_group[i] = group_map.get(s_i.get('dist_group', ''), -1)
            
            for j in range(n):
                s_j_id = self.i2s[j]['store_id']
                if s_i_id in self.distance_matrix and s_j_id in self.distance_matrix[s_i_id]:
                    self.np_dist[i, j] = self.distance_matrix[s_i_id][s_j_id]
                if s_i_id in self.time_matrix and s_j_id in self.time_matrix[s_i_id]:
                    self.np_time[i, j] = self.time_matrix[s_i_id][s_j_id]
                    
        # Fix DC -> elements
        for j in range(n):
            s_j_id = self.i2s[j]['store_id']
            if self.dc['store_id'] in self.distance_matrix and s_j_id in self.distance_matrix[self.dc['store_id']]:
                self.np_dist[0, j] = self.distance_matrix[self.dc['store_id']][s_j_id]
            if self.dc['store_id'] in self.time_matrix and s_j_id in self.time_matrix[self.dc['store_id']]:
                self.np_time[0, j] = self.time_matrix[self.dc['store_id']][s_j_id]
                
        StoreAllocationGA._cached_mappings = (
            self.s2i, self.i2s, self.np_dist, self.np_time, self.np_volume, 
            self.np_dwell, self.np_earliest, self.np_latest, self.np_region, self.np_group
        )


    def _get_store_insertion_cost_and_pos(self, route_info, store):
        dc = route_info['dc']
        stores = route_info['stores']
        
        store_idx = self.s2i[store['store_id']]
        route_stores = np.array([self.s2i[s['store_id']] for s in stores], dtype=np.int64)
        dc_region_map = {'north': 0, 'south': 1, 'east': 2, 'west': 3}
        dc_region_int = dc_region_map.get(dc.get('region', ''), -1)
        store_region_int = self.np_region[store_idx]
        
        # Calculate base pred time
        if len(stores) > 0:
            first_pred_epoch = float(int(datetime.fromisoformat(stores[0]['pred_time']).timestamp()))
        else:
            first_pred_epoch = float(int(datetime.fromisoformat(store['sched_time']).timestamp()))
            
        cost, pos = _njit_get_store_insertion_cost_pos(
            store_idx,
            route_stores,
            0,
            dc_region_int,
            float(dc.get('total_volume', 0)),
            float(dc.get('max_capacity', 1e9)),
            store_region_int,
            float(store.get('volume', 0)),
            self.np_dist,
            self.np_time,
            self.np_dwell,
            self.np_earliest,
            self.np_latest,
            first_pred_epoch,
            float(self.time_limit_per_route)
        )
        
        if cost < 0:
            return float('inf'), -1
        return cost, pos

    def _sort_stores_by_insertion_cost(self, stores):
        """Sort stores by regret-2 cost (descending).

        Regret cost = 2nd_best_insertion_cost - best_insertion_cost.
        Stores with higher regret are placed first so that stores
        which suffer the most from NOT being assigned to their best
        route are prioritised early.
        """
        route_keys = list(self.main_routes.keys())

        main_route_stores = []
        main_route_vols = []
        main_route_first_epochs = []
        main_route_caps = []
        main_route_dc_regions = []

        dc_region_map = {'north': 0, 'south': 1, 'east': 2, 'west': 3}

        for r_id in route_keys:
            r_info = self.main_routes[r_id]
            r_stores = [self.s2i[s['store_id']] for s in r_info['stores']]
            main_route_stores.append(r_stores)
            main_route_vols.append(float(r_info['dc'].get('total_volume', 0)))
            if r_stores:
                epoch = float(int(datetime.fromisoformat(r_info['stores'][0]['pred_time']).timestamp()))
            else:
                epoch = 0.0  # dummy
            main_route_first_epochs.append(epoch)
            main_route_caps.append(float(r_info['dc'].get('max_capacity', 1e9)))
            main_route_dc_regions.append(dc_region_map.get(r_info['dc'].get('region', ''), -1))

        store_regrets = []

        from solvers.allocate_ga import _njit_get_store_insertion_cost_pos

        for store in stores:
            store_idx = self.s2i[store['store_id']]
            best_cost = float('inf')
            second_best_cost = float('inf')

            for r_idx in range(len(route_keys)):
                route_stores_arr = np.array(main_route_stores[r_idx], dtype=np.int64)

                if len(main_route_stores[r_idx]) == 0:
                    epoch = float(int(datetime.fromisoformat(store['sched_time']).timestamp()))
                else:
                    epoch = main_route_first_epochs[r_idx]

                cost, pos = _njit_get_store_insertion_cost_pos(
                    store_idx,
                    route_stores_arr,
                    0,
                    main_route_dc_regions[r_idx],
                    main_route_vols[r_idx],
                    main_route_caps[r_idx],
                    self.np_region[store_idx],
                    self.np_volume[store_idx],
                    self.np_dist,
                    self.np_time,
                    self.np_dwell,
                    self.np_earliest,
                    self.np_latest,
                    epoch,
                    float(self.time_limit_per_route)
                )

                if pos != -1 and cost < best_cost:
                    second_best_cost = best_cost
                    best_cost = cost
                elif pos != -1 and cost < second_best_cost:
                    second_best_cost = cost

            # Regret = difference between 2nd-best and best insertion cost.
            # If only one feasible route exists, regret = inf (must place now).
            # If no feasible route exists, regret = -inf (hopeless; place last).
            if best_cost == float('inf'):
                regret = float('-inf')
            elif second_best_cost == float('inf'):
                regret = float('inf')
            else:
                regret = second_best_cost - best_cost

            store_regrets.append((store, regret))

        # Sort descending: highest regret first
        store_regrets.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in store_regrets]



    def _generate_greedy_individual(self):
        route_keys = list(self.main_routes.keys())
        
        main_route_stores = []
        main_route_vols = []
        main_route_first_epochs = []
        main_route_caps = []
        main_route_dc_regions = []
        
        dc_region_map = {'north': 0, 'south': 1, 'east': 2, 'west': 3}
        
        for r_id in route_keys:
            r_info = self.main_routes[r_id]
            stores = [self.s2i[s['store_id']] for s in r_info['stores']]
            main_route_stores.append(stores)
            main_route_vols.append(float(r_info['dc'].get('total_volume', 0)))
            if stores:
                epoch = float(int(datetime.fromisoformat(r_info['stores'][0]['pred_time']).timestamp()))
            else:
                epoch = 0.0 # dummy
            main_route_first_epochs.append(epoch)
            main_route_caps.append(float(r_info['dc'].get('max_capacity', 1e9)))
            main_route_dc_regions.append(dc_region_map.get(r_info['dc'].get('region', ''), -1))
            
        greedy_chromo = []
        
        from solvers.allocate_ga import _njit_get_store_insertion_cost_pos
        
        for store in self.remaining_stores:
            store_idx = self.s2i[store['store_id']]
            best_r_idx = -1
            min_cost = float('inf')
            best_pos = -1
            
            for r_idx in range(len(route_keys)):
                route_stores_arr = np.array(main_route_stores[r_idx], dtype=np.int64)
                
                if len(main_route_stores[r_idx]) == 0:
                    epoch = float(int(datetime.fromisoformat(store['sched_time']).timestamp()))
                else:
                    epoch = main_route_first_epochs[r_idx]
                    
                cost, pos = _njit_get_store_insertion_cost_pos(
                    store_idx,
                    route_stores_arr,
                    0,
                    main_route_dc_regions[r_idx],
                    main_route_vols[r_idx],
                    main_route_caps[r_idx],
                    self.np_region[store_idx],
                    self.np_volume[store_idx],
                    self.np_dist,
                    self.np_time,
                    self.np_dwell,
                    self.np_earliest,
                    self.np_latest,
                    epoch,
                    float(self.time_limit_per_route)
                )
                
                if pos != -1 and cost < min_cost:
                    min_cost = cost
                    best_r_idx = r_idx
                    best_pos = pos
                    
            if best_r_idx != -1:
                greedy_chromo.append(route_keys[best_r_idx])
                main_route_stores[best_r_idx].insert(best_pos, store_idx)
                main_route_vols[best_r_idx] += self.np_volume[store_idx]
                if best_pos == 0:
                    main_route_first_epochs[best_r_idx] = float(int(datetime.fromisoformat(store['sched_time']).timestamp()))
            else:
                greedy_chromo.append('SUPPORT')

        return greedy_chromo


    def _copy_routes_info(self, routes):
        return {
            route_id: {
                "dc": route_data["dc"].copy(),
                "stores": [store.copy() for store in route_data["stores"]]
            } for route_id, route_data in routes.items()
        }


    def _encode_individual(self, individual):
        s = ','.join(map(str, individual))
        return hashlib.md5(s.encode()).hexdigest()

    def _calculate_total_distance(self, routes):
        total_cost = 0
        for _, route_info in routes.items():
            prev = self.dc
            for store in route_info['stores']:
                total_cost += self.distance_matrix[prev['store_id']][store['store_id']]
                prev = store
            total_cost += self.distance_matrix[prev['store_id']][self.dc['store_id']]

        return total_cost

    def _evaluate_individual(self, individual):
        temp_routes = self._copy_routes_info(self.main_routes)
        route_manager = RouteManager(temp_routes, self.distance_matrix, self.time_matrix)
        support_pool = []

        for i, target in enumerate(individual):
            store = self.remaining_stores[i]
            if target == 'SUPPORT':
                support_pool.append(store)
                continue

            _, pos = self._get_store_insertion_cost_and_pos(route_manager.routes_info[target], store)
            if pos != -1:
                route_manager.insert_store(store, target, pos)
            else:
                support_pool.append(store)

        main_cost = self._calculate_total_distance(route_manager.routes_info)
        support_pool_indices = [self.s2i[s['store_id']] for s in support_pool]
        support_cost = self._fast_support_cost(support_pool_indices)

        return main_cost + support_cost, route_manager.routes_info, support_pool

    def _fast_support_cost(self, support_pool_indices):
        if not support_pool_indices:
            return 0.0

        num_vehicles = 0
        total_distance = 0.0
        
        unvisited_idx = set(support_pool_indices)
        
        # Dummy pheromone matrix for greedy
        dummy_pheromone = np.ones((len(self.s2i), len(self.s2i)), dtype=np.float64)

        while unvisited_idx:
            num_vehicles += 1
            
            feasible_arr = np.array(list(unvisited_idx), dtype=np.int64)
            current_idx = _njit_greedy_selection(0, feasible_arr, self.np_dist, dummy_pheromone, 1.0, 1.0)
            
            # Init route state
            route_vol = self.np_volume[current_idx]
            
            # depart time
            depart_time = float(self.np_earliest[current_idx]) - self.np_time[0, current_idx]
            curr_time = float(self.np_earliest[current_idx])
            ret_time = curr_time + self.np_dwell[current_idx] + self.np_time[current_idx, 0]
            curr_duration = ret_time - depart_time
            
            prev_pred_time_epoch = float(self.np_earliest[current_idx])
            
            total_distance += self.np_dist[0, current_idx]
            unvisited_idx.remove(current_idx)
            
            while unvisited_idx:
                unv_arr = np.array(list(unvisited_idx), dtype=np.int64)
                feasible_arr = _njit_get_feasible_stores(
                    unv_arr, current_idx, route_vol, curr_duration,
                    prev_pred_time_epoch, 0, float(config.DC_CONFIG.get('max_capacity', 7.2)), float(self.time_limit_per_route),
                    self.np_group, self.np_region, self.np_volume, self.np_time, self.np_dwell,
                    self.np_earliest, self.np_latest, False
                )
                
                if len(feasible_arr) == 0:
                    break
                    
                next_idx = _njit_greedy_selection(current_idx, feasible_arr, self.np_dist, dummy_pheromone, 1.0, 1.0)
                
                # Update route state
                route_vol += self.np_volume[next_idx]
                arr_time = prev_pred_time_epoch + self.np_time[current_idx, next_idx] + self.np_dwell[current_idx]
                curr_time = arr_time # since is_solomon=False
                ret_time = curr_time + self.np_dwell[next_idx] + self.np_time[next_idx, 0]
                curr_duration = ret_time - depart_time
                prev_pred_time_epoch = curr_time
                
                total_distance += self.np_dist[current_idx, next_idx]
                
                current_idx = next_idx
                unvisited_idx.remove(next_idx)
                
            total_distance += self.np_dist[current_idx, 0]

        return total_distance + num_vehicles * 2000.0

    def _evaluate_individual_fast(self, individual):
        route_keys = list(self.main_routes.keys())
        
        main_route_stores = []
        main_route_vols = []
        main_route_first_epochs = []
        main_route_caps = []
        main_route_dc_regions = []
        
        dc_region_map = {'north': 0, 'south': 1, 'east': 2, 'west': 3}
        
        for r_id in route_keys:
            r_info = self.main_routes[r_id]
            stores = [self.s2i[s['store_id']] for s in r_info['stores']]
            main_route_stores.append(stores)
            main_route_vols.append(float(r_info['dc'].get('total_volume', 0)))
            if stores:
                epoch = float(int(datetime.fromisoformat(r_info['stores'][0]['pred_time']).timestamp()))
            else:
                epoch = 0.0 # dummy
            main_route_first_epochs.append(epoch)
            main_route_caps.append(float(r_info['dc'].get('max_capacity', 1e9)))
            main_route_dc_regions.append(dc_region_map.get(r_info['dc'].get('region', ''), -1))
            
        support_pool_indices = []
        
        for i, target in enumerate(individual):
            store = self.remaining_stores[i]
            store_idx = self.s2i[store['store_id']]
            
            if target == 'SUPPORT':
                support_pool_indices.append(store_idx)
                continue
                
            r_idx = route_keys.index(target)
            route_stores_arr = np.array(main_route_stores[r_idx], dtype=np.int64)
            
            if len(main_route_stores[r_idx]) == 0:
                epoch = float(int(datetime.fromisoformat(store['sched_time']).timestamp()))
            else:
                epoch = main_route_first_epochs[r_idx]
                
            cost, pos = _njit_get_store_insertion_cost_pos(
                store_idx,
                route_stores_arr,
                0,
                main_route_dc_regions[r_idx],
                main_route_vols[r_idx],
                main_route_caps[r_idx],
                self.np_region[store_idx],
                self.np_volume[store_idx],
                self.np_dist,
                self.np_time,
                self.np_dwell,
                self.np_earliest,
                self.np_latest,
                epoch,
                float(self.time_limit_per_route)
            )
            
            if pos != -1:
                main_route_stores[r_idx].insert(pos, store_idx)
                main_route_vols[r_idx] += self.np_volume[store_idx]
                if pos == 0:
                    main_route_first_epochs[r_idx] = float(int(datetime.fromisoformat(store['sched_time']).timestamp()))
            else:
                support_pool_indices.append(store_idx)
                
        main_cost = 0.0
        for r_idx in range(len(route_keys)):
            stores = main_route_stores[r_idx]
            if len(stores) == 0:
                continue
            main_cost += self.np_dist[0, stores[0]]
            for j in range(len(stores) - 1):
                main_cost += self.np_dist[stores[j], stores[j+1]]
            main_cost += self.np_dist[stores[-1], 0]
            
        support_cost = self._fast_support_cost(support_pool_indices)
        
        return main_cost + support_cost

    def _uniform_crossover(self, parent1, parent2):
        child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]
        return child1, child2

    def _crossover(self, parent1, parent2):
        if random.random() < self.cross_rate:
            return self._uniform_crossover(parent1, parent2)
        return copy.deepcopy(parent1), copy.deepcopy(parent2)

    def _mutate(self, individual):
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] = random.choice(self.route_choices)
        return individual

    def run(self):
        greedy_chromo = self._generate_greedy_individual()
        
        if self.generations == 0:
            g_cost = self._evaluate_individual_fast(greedy_chromo)
            return g_cost, None, None

        g_cost, g_routes, g_support = self._evaluate_individual(greedy_chromo)
        self.best_cost = g_cost
        self.best_solution = g_routes
        self.best_remaining_solution = g_support
        self.best_individual = greedy_chromo

        population = [greedy_chromo]
        num_stores = len(self.remaining_stores)
        while len(population) < self.population_size:
            population.append([random.choice(self.route_choices) for _ in range(num_stores)])

        early_stopper = EarlyStopper(patience=self.early_stop_patience)
        with Manager() as manager:
            shared_cache = manager.dict()
            with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, cpu_count() - 2), initializer=init_alloc_worker, initargs=(self.main_routes, self.distance_matrix, self.time_matrix, self)) as pool:
                for gen_idx in range(self.generations):
                    unique_tasks = []
                    individual_keys = []

                    for chromo in population:
                        key = self._encode_individual(chromo)
                        individual_keys.append(key)
                        if key not in shared_cache:
                            unique_tasks.append((chromo, key, shared_cache))

                    if len(unique_tasks) > 0:
                        try:
                            list(pool.map(alloc_fitness_worker, unique_tasks, timeout=120, chunksize=max(1, len(unique_tasks) // (cpu_count() - 2) * 2)))
                        except concurrent.futures.TimeoutError:
                            print('Timeout error')
                            for _, _, key in unique_tasks:
                                if key not in shared_cache:
                                    print(f"Skipping individual {key}")
                                    shared_cache[key] = {'cost': float('inf')}

                    fitnesses = []
                    evaluated_pop = []
                    for i, chromo in enumerate(population):
                        key = individual_keys[i]
                        res = shared_cache[key]
                        evaluated_pop.append({
                            'individual': chromo,
                            'cost': res['cost']
                        })
                        fitnesses.append(res['cost'])

                    evaluated_pop.sort(key=lambda x: x['cost'])
                    current_best = evaluated_pop[0]

                    if current_best['cost'] < self.best_cost:
                        self.best_cost = current_best['cost']
                        self.best_individual = copy.deepcopy(current_best['individual'])

                    self.log.append({
                        'generation': gen_idx + 1,
                        'iter_worst_cost': float(np.max(fitnesses)),
                        'iter_best_cost': float(current_best['cost']),
                        'iter_avg_cost': float(np.mean(fitnesses)),
                        'std_cost': float(np.std(fitnesses)),
                        'best_cost': self.best_cost,
                    })

                    print(f'Store Allocation: iteration{gen_idx+1} -> best cost = {self.best_cost}')

                    if early_stopper.check(self.best_cost):
                        break

                    elites = [copy.deepcopy(ind['individual']) for ind in evaluated_pop[:self.elite_size]]
                    weights = [max(1.0 / ind['cost'], 1e-12) for ind in evaluated_pop]

                    child_pairs = []
                    while len(child_pairs) < (self.population_size - self.elite_size):
                        p1, p2 = random.choices(evaluated_pop, weights=weights, k=2)
                        c1, c2 = self._crossover(copy.deepcopy(p1['individual']), copy.deepcopy(p2['individual']))
                        c1 = self._mutate(list(c1))
                        c2 = self._mutate(list(c2))
                        child_pairs.append((c1, c2))
                        
                    task_candidates = []
                    for c1, c2 in child_pairs:
                        for c in (c1, c2):
                            k = self._encode_individual(c)
                            if k not in shared_cache:
                                task_candidates.append((c, k, shared_cache))
                                
                    if len(task_candidates) > 0:
                        try:
                            num_procs = max(1, cpu_count() - 2)
                            list(pool.map(alloc_fitness_worker, task_candidates, timeout=120, chunksize=max(1, len(task_candidates) // num_procs * 2)))
                        except concurrent.futures.TimeoutError:
                            print("Timeout Error during candidate evaluation")
                            for _, k, _ in task_candidates:
                                if k not in shared_cache:
                                    shared_cache[k] = {'cost': float('inf')}
                                    
                    childrens = []
                    for c1, c2 in child_pairs:
                        k1 = self._encode_individual(c1)
                        k2 = self._encode_individual(c2)
                        
                        cost1 = shared_cache[k1]['cost']
                        cost2 = shared_cache[k2]['cost']
                        
                        if cost1 < cost2:
                            childrens.append(c1)
                        else:
                            childrens.append(c2)

                    population = elites + childrens

        if self.best_individual is not None:
             _, self.best_solution, self.best_remaining_solution = self._evaluate_individual(self.best_individual)

        return self.best_cost, self.best_solution, self.best_remaining_solution