import copy
import random
import hashlib
import numpy as np
from datetime import datetime
from multiprocessing import cpu_count, Manager

from config import config
from models.route_manager import RouteManager
from utils.early_stopper import EarlyStopper
from numba import njit
from solvers.vnd import VND

@njit(cache=True)
def _njit_evaluate_chromosome_mainline(
    permutation, 
    N, M, 
    np_volume, np_earliest, np_latest, np_dwell, np_time, np_dist, np_region, np_group, np_orig_route, np_sched, np_orig_seq,
    main_route_caps,
    support_capacity, 
    vehicle_cost,
    tw_penalty_weight,
    cap_penalty_weight,
    cross_penalty_weight,
    order_penalty_weight
):
    acc_dist = 0.0
    acc_cross = 0.0
    valid_vehicles = 0
    support_vehicles = 0
    tw_penalty = 0.0
    cap_penalty = 0.0
    order_penalty = 0.0
    
    # 1. Collect stores for each route
    main_stores = np.full((M, N), -1, dtype=np.int64)
    main_counts = np.zeros(M, dtype=np.int64)
    
    support_stores = np.full(N, -1, dtype=np.int64)
    support_count = 0
    
    curr_route = -1
    for i in range(len(permutation)):
        g = permutation[i]
        if g > N:
            curr_route = g - N - 1 # 0 to M-1
        else:
            if curr_route == -1:
                support_stores[support_count] = g
                support_count += 1
            else:
                main_stores[curr_route, main_counts[curr_route]] = g
                main_counts[curr_route] += 1
                
    # 2. Evaluate Main Routes
    for r in range(M):
        count = main_counts[r]
        if count == 0:
            continue
            
        valid_vehicles += 1
        
        route_vol = 0.0
        route_dist = 0.0
        cross_route_penalty = 0.0
        route_tw_penalty = 0.0
        has_inserted_stores = False
        
        curr_time = 0.0
        prev_idx = 0
        last_seq = -1
        for i in range(count):
            curr_idx = main_stores[r, i]
            
            if np_orig_route[curr_idx] != -1 and np_orig_route[curr_idx] != r:
                cross_route_penalty += cross_penalty_weight
                has_inserted_stores = True
            elif np_orig_route[curr_idx] == -1:
                has_inserted_stores = True
            elif np_orig_route[curr_idx] == r:
                curr_seq = np_orig_seq[curr_idx]
                if curr_seq < last_seq:
                    order_penalty += 1.0
                else:
                    last_seq = curr_seq
                
            route_vol += np_volume[curr_idx]
            
            if prev_idx == 0:
                arrival_time = np_sched[curr_idx]
            else:
                travel = np_time[prev_idx, curr_idx]
                dwell = np_dwell[prev_idx] if prev_idx != 0 else 0
                arrival_time = curr_time + round(travel + dwell)
                
            if arrival_time > np_latest[curr_idx]:
                route_tw_penalty += (arrival_time - np_latest[curr_idx])
            elif arrival_time < np_earliest[curr_idx]:
                route_tw_penalty += (np_earliest[curr_idx] - arrival_time)
                
            curr_time = arrival_time
            route_dist += np_dist[prev_idx, curr_idx]
            prev_idx = curr_idx
            
        route_dist += np_dist[prev_idx, 0] # return to DC
        
        if route_vol > main_route_caps[r]:
            cap_penalty += (route_vol - main_route_caps[r])
            
        if has_inserted_stores:
            tw_penalty += route_tw_penalty
            
        acc_dist += route_dist
        acc_cross += cross_route_penalty
        
    # 3. Evaluate Support Routes (Greedy Split)
    if support_count > 0:
        curr_vol = 0.0
        curr_time = 0.0
        prev_idx = 0
        curr_region = -1
        curr_group = -1
        
        valid_vehicles += 1
        support_vehicles += 1
        
        for i in range(support_count):
            curr_idx = support_stores[i]
            r_reg = np_region[curr_idx]
            g_grp = np_group[curr_idx]
            
            if np_orig_route[curr_idx] != -1:
                acc_cross += cross_penalty_weight
                
            if curr_vol == 0.0:
                curr_region = r_reg
                curr_group = g_grp
                
            if prev_idx == 0:
                test_arrival = np_sched[curr_idx]
            else:
                test_travel = np_time[prev_idx, curr_idx] + (np_dwell[prev_idx] if prev_idx != 0 else 0)
                test_arrival = curr_time + round(test_travel)
                
            is_feasible = True
            if curr_vol + np_volume[curr_idx] > support_capacity:
                is_feasible = False
            elif curr_region != -1 and r_reg != -1 and r_reg != curr_region:
                is_feasible = False
            elif curr_group != -1 and g_grp != -1 and g_grp != curr_group:
                is_feasible = False
            elif test_arrival > np_latest[curr_idx] or test_arrival < np_earliest[curr_idx]:
                is_feasible = False
                    
            if not is_feasible:
                # Close vehicle
                acc_dist += np_dist[prev_idx, 0]
                
                # New vehicle
                valid_vehicles += 1
                support_vehicles += 1
                curr_vol = 0.0
                curr_time = 0.0
                prev_idx = 0
                curr_region = r_reg
                curr_group = g_grp
                
            curr_vol += np_volume[curr_idx]
            
            if prev_idx == 0:
                arrival_time = np_sched[curr_idx]
            else:
                travel = np_time[prev_idx, curr_idx]
                dwell = np_dwell[prev_idx] if prev_idx != 0 else 0
                arrival_time = curr_time + round(travel + dwell)
                
            if arrival_time > np_latest[curr_idx]:
                tw_penalty += (arrival_time - np_latest[curr_idx])
            elif arrival_time < np_earliest[curr_idx]:
                tw_penalty += (np_earliest[curr_idx] - arrival_time)
                
            curr_time = arrival_time
            acc_dist += np_dist[prev_idx, curr_idx]
            prev_idx = curr_idx
            
        # Close last vehicle
        acc_dist += np_dist[prev_idx, 0]
        
    acc_tw = tw_penalty * tw_penalty_weight
    acc_cap = cap_penalty * cap_penalty_weight
    acc_order = order_penalty * order_penalty_weight
    acc_veh = support_vehicles * vehicle_cost
    
    final_cost = acc_dist + acc_cross + acc_tw + acc_cap + acc_order + acc_veh
    
    return final_cost, acc_dist, acc_veh, acc_tw, acc_cap, acc_cross, acc_order


def init_ga_worker(ga_instance):
    global ALLOC_GA_INSTANCE
    ALLOC_GA_INSTANCE = ga_instance

def ga_fitness_worker(args):
    chromo, key, shared_cache, apply_vnd = args

    if key in shared_cache:
        return shared_cache[key]

    ga = ALLOC_GA_INSTANCE
    
    fast_cost = ga._evaluate_individual_fast(chromo)
    final_chromo = chromo
    
    # We could apply VND here, but VND relies on `RouteManager` structure.
    # For now, since SingleStageMainLineGA allows soft constraints, VND might fail 
    # to understand the penalties unless we modify VND. We'll skip VND in this fast path.

    result = {
        'cost': fast_cost,
        'routes': None,
        'optimized_chromo': final_chromo
    }

    shared_cache[key] = result
    return result


class SingleStageMainLineGA:
    """
    Single-Stage GA that maintains Main Lines concept via dividers in the chromosome.
    Uses soft time windows (and soft capacity on main lines) with penalties.
    """
    def __init__(self, main_routes, remaining_stores, distance_matrix, time_matrix, 
                 population_size=1000, elite_rate=0.1, generations=1000, 
                 cross_rate=0.8, mutation_rate=0.1, early_stop_patience=500, 
                 support_capacity=7.2, vehicle_cost=2000, 
                 tw_penalty_weight=10000.0, cap_penalty_weight=10000.0,
                 cross_penalty_weight=100.0, order_penalty_weight=10000.0,
                 target_cost=None):
        self.dc = config.DC_CONFIG
        self.main_routes = main_routes
        self.remaining_stores = remaining_stores
        self.orig_distance_matrix = distance_matrix
        self.orig_time_matrix = time_matrix
        
        self.population_size = population_size
        self.elite_size = max(1, int(population_size * elite_rate))
        self.generations = generations
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.early_stop_patience = early_stop_patience
        self.support_capacity = float(support_capacity)
        self.vehicle_cost = float(vehicle_cost)
        
        self.tw_penalty_weight = tw_penalty_weight
        self.cap_penalty_weight = cap_penalty_weight
        self.cross_penalty_weight = cross_penalty_weight
        self.order_penalty_weight = order_penalty_weight
        
        self.target_cost = target_cost
        
        self.log = []
        self.best_cost = float('inf')
        self.best_solution = None

        self._init_numpy_mappings()

    def _init_numpy_mappings(self):
        self.s2i = {self.dc['store_id']: 0}
        self.i2s = {0: self.dc}
        idx = 1
        for s in self.remaining_stores:
            self.s2i[s['store_id']] = idx
            self.i2s[idx] = s
            idx += 1
            
        self.N = len(self.remaining_stores)
        self.route_keys = list(self.main_routes.keys())
        self.M = len(self.route_keys)
        
        n = len(self.s2i)
        self.np_dist = np.zeros((n, n), dtype=np.float64)
        self.np_time = np.zeros((n, n), dtype=np.float64)
        self.np_volume = np.zeros(n, dtype=np.float64)
        self.np_dwell = np.zeros(n, dtype=np.float64)
        self.np_earliest = np.zeros(n, dtype=np.float64)
        self.np_latest = np.zeros(n, dtype=np.float64)
        self.np_sched = np.zeros(n, dtype=np.float64)
        self.np_group = np.full(n, -1, dtype=np.int64)
        self.np_region = np.full(n, -1, dtype=np.int64)
        self.np_orig_route = np.full(n, -1, dtype=np.int64)
        self.np_orig_seq = np.full(n, -1, dtype=np.int64)
        
        for r_id, r_info in self.main_routes.items():
            for seq_idx, s in enumerate(r_info['stores']):
                s_id = s['store_id']
                if s_id in self.s2i:
                    self.np_orig_seq[self.s2i[s_id]] = seq_idx
        
        region_map = {'north': 0, 'south': 1, 'east': 2, 'west': 3}
        group_map = {'near': 0, 'mid': 1, 'far': 2}
        
        for i in range(n):
            s_i = self.i2s[i]
            s_i_id = s_i['store_id']
            if i > 0:
                self.np_volume[i] = s_i.get('volume', 0.0)
                self.np_dwell[i] = float(s_i.get('dwell_time', 0))
                self.np_earliest[i] = float(int(datetime.fromisoformat(s_i['earliest_time']).timestamp()))
                self.np_latest[i] = float(int(datetime.fromisoformat(s_i['latest_time']).timestamp()))
                sched_str = s_i.get('sched_time', s_i.get('pred_time', s_i['earliest_time']))
                self.np_sched[i] = float(int(datetime.fromisoformat(sched_str).timestamp()))
                self.np_group[i] = group_map.get(s_i.get('dist_group', ''), -1)
                self.np_region[i] = region_map.get(s_i.get('region', ''), -1)
                
                orig_route_code = s_i.get('route_code', '')
                orig_route_id = orig_route_code[:2] if orig_route_code else ''
                if orig_route_id in self.route_keys:
                    self.np_orig_route[i] = self.route_keys.index(orig_route_id)
                
            for j in range(n):
                s_j_id = self.i2s[j]['store_id']
                if s_i_id in self.orig_distance_matrix and s_j_id in self.orig_distance_matrix[s_i_id]:
                    self.np_dist[i, j] = self.orig_distance_matrix[s_i_id][s_j_id]
                if s_i_id in self.orig_time_matrix and s_j_id in self.orig_time_matrix[s_i_id]:
                    self.np_time[i, j] = self.orig_time_matrix[s_i_id][s_j_id]
                    
        self.main_route_caps = np.zeros(self.M, dtype=np.float64)
        for r in range(self.M):
            r_id = self.route_keys[r]
            self.main_route_caps[r] = float(self.main_routes[r_id]['dc'].get('max_capacity', 1e9))

    def _evaluate_individual_fast(self, permutation):
        perm_arr = np.array(permutation, dtype=np.int64)
        c, dist, veh, tw, cap, cross, order = _njit_evaluate_chromosome_mainline(
            perm_arr, self.N, self.M,
            self.np_volume, self.np_earliest, self.np_latest,
            self.np_dwell, self.np_time, self.np_dist, self.np_region,
            self.np_group, self.np_orig_route, self.np_sched, self.np_orig_seq,
            self.main_route_caps, self.support_capacity, self.vehicle_cost,
            self.tw_penalty_weight, self.cap_penalty_weight, self.cross_penalty_weight, self.order_penalty_weight
        )
        return c, {'dist': dist, 'veh': veh, 'tw': tw, 'cap': cap, 'cross': cross, 'order': order}

    def _decode_to_routes(self, permutation):
        solution = {}
        main_stores = [[] for _ in range(self.M)]
        support_stores = []
        
        curr_route = -1
        for g in permutation:
            if g > self.N:
                curr_route = g - self.N - 1
            else:
                if curr_route == -1:
                    support_stores.append(g)
                else:
                    main_stores[curr_route].append(g)
                    
        # Build Main Routes
        for r in range(self.M):
            r_id = self.route_keys[r]
            dc_info = self.main_routes[r_id]['dc'].copy()
            dc_info['total_volume'] = 0.0
            stores = []
            for g in main_stores[r]:
                s = self.i2s[g].copy()
                stores.append(s)
                dc_info['total_volume'] += s.get('volume', 0.0)
            dc_info['load_rate'] = dc_info['total_volume'] / float(dc_info.get('max_capacity', 1))
            
            # 即使店家是空的，也必須保留主線
            solution[r_id] = {'dc': dc_info, 'stores': stores}
                
        # Build Support Routes
        vehicle_num = 101
        if support_stores:
            curr_vol = 0.0
            curr_region = -1
            curr_group = -1
            curr_time = 0.0
            prev_idx = 0
            curr_veh_stores = []
            
            for g in support_stores:
                s = self.i2s[g].copy()
                r_reg = self.np_region[g]
                g_grp = self.np_group[g]
                vol = s.get('volume', 0.0)
                
                if curr_vol == 0.0:
                    curr_region = r_reg
                    curr_group = g_grp
                    
                is_feasible = True
                if curr_vol + vol > self.support_capacity:
                    is_feasible = False
                elif curr_region != -1 and r_reg != -1 and r_reg != curr_region:
                    is_feasible = False
                elif curr_group != -1 and g_grp != -1 and g_grp != curr_group:
                    is_feasible = False
                
                # Check Time Window Feasibility
                if is_feasible:
                    if prev_idx == 0:
                        test_arrival = self.np_sched[g]
                    else:
                        test_travel = self.np_time[prev_idx][g] + (self.np_dwell[prev_idx] if prev_idx != 0 else 0)
                        test_arrival = curr_time + round(test_travel)
                        
                    if test_arrival > self.np_latest[g] or test_arrival < self.np_earliest[g]:
                        is_feasible = False
                        
                if not is_feasible:
                    v_id = f'{vehicle_num}'
                    solution[v_id] = {
                        'dc': {'route_id': v_id, 'route_code': v_id, 'store_id': 'DC', 'total_volume': curr_vol, 'load_rate': curr_vol/self.support_capacity, 'max_capacity': self.support_capacity},
                        'stores': curr_veh_stores
                    }
                    vehicle_num += 1
                    curr_veh_stores = []
                    curr_vol = 0.0
                    curr_region = r_reg
                    curr_group = g_grp
                    curr_time = 0.0
                    prev_idx = 0
                    
                curr_veh_stores.append(s)
                curr_vol += vol
                
                # Update curr_time
                if prev_idx == 0:
                    curr_time = self.np_sched[g]
                else:
                    travel = self.np_time[prev_idx][g]
                    dwell = self.np_dwell[prev_idx] if prev_idx != 0 else 0
                    curr_time = curr_time + round(travel + dwell)
                    
                prev_idx = g
                    
            if curr_veh_stores:
                v_id = f'{vehicle_num}'
                solution[v_id] = {
                    'dc': {'route_id': v_id, 'route_code': v_id, 'store_id': 'DC', 'total_volume': curr_vol, 'load_rate': curr_vol/self.support_capacity, 'max_capacity': self.support_capacity},
                    'stores': curr_veh_stores
                }
                
        # Calculate full distances using RouteManager
        temp_rm = RouteManager(solution, self.orig_distance_matrix, self.orig_time_matrix)
        temp_rm.update_all_routes_info()
        return temp_rm.routes_info

    def _encode_individual(self, individual):
        s = ','.join(map(str, individual))
        return hashlib.md5(s.encode()).hexdigest()

    def _pmx_crossover(self, parent1, parent2):
        size = len(parent1)
        if size < 2:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
            
        p1, p2 = random.sample(range(size), 2)
        start, end = min(p1, p2), max(p1, p2)
        
        child1 = [-1] * size
        child2 = [-1] * size
        
        child1[start:end+1] = parent1[start:end+1]
        child2[start:end+1] = parent2[start:end+1]
        
        mapping1 = {parent1[i]: parent2[i] for i in range(start, end+1)}
        mapping2 = {parent2[i]: parent1[i] for i in range(start, end+1)}
        
        def fill(child, parent, mapping):
            for i in range(size):
                if child[i] == -1:
                    val = parent[i]
                    while val in mapping:
                        val = mapping[val]
                    child[i] = val
                    
        fill(child1, parent2, mapping1)
        fill(child2, parent1, mapping2)
        
        return child1, child2

    def _crossover(self, parent1, parent2):
        if random.random() < self.cross_rate:
            return self._pmx_crossover(parent1, parent2)
        return copy.deepcopy(parent1), copy.deepcopy(parent2)

    def _mutate(self, individual):
        if random.random() < self.mutation_rate:
            size = len(individual)
            if size >= 2:
                p1, p2 = sorted(random.sample(range(size), 2))
                individual[p1:p2+1] = reversed(individual[p1:p2+1])
        return individual

    def _generate_initial_individual(self, randomize=False):
        chromo = []
        support_pool = []
        
        main_store_ids = set()
        visited_s_idx = set()
        
        # 1. Simulate Main Routes and extract violating stores
        main_stores_kept = [[] for _ in range(self.M)]
        
        for r in range(self.M):
            r_id = self.route_keys[r]
            cap = self.main_route_caps[r]
            
            curr_vol = 0.0
            curr_time = 0.0
            prev_idx = 0
            has_extracted = False
            
            for s in self.main_routes[r_id]['stores']:
                s_idx = self.s2i[s['store_id']]
                
                if s_idx in visited_s_idx:
                    continue
                visited_s_idx.add(s_idx)
                
                main_store_ids.add(s['store_id'])
                
                vol = self.np_volume[s_idx]
                
                # Capacity is always checked
                if curr_vol + vol > cap:
                    support_pool.append(s_idx)
                    has_extracted = True
                    continue
                    
                # Calculate arrival time
                if prev_idx == 0:
                    arrival_time = self.np_sched[s_idx]
                else:
                    travel = self.np_time[prev_idx, s_idx]
                    dwell = self.np_dwell[prev_idx] if prev_idx != 0 else 0
                    arrival_time = curr_time + round(travel + dwell)
                    
                # If we have modified this route (extracted something), enforce Time Window
                if has_extracted:
                    if arrival_time > self.np_latest[s_idx] or arrival_time < self.np_earliest[s_idx]:
                        support_pool.append(s_idx)
                        # has_extracted is already True
                        continue
                
                # Store is valid, keep it
                curr_vol += vol
                curr_time = arrival_time
                prev_idx = s_idx
                main_stores_kept[r].append(s_idx)
                
        # 2. Add stores not in main routes to support pool
        for s in self.remaining_stores:
            if s['store_id'] not in main_store_ids:
                support_pool.append(self.s2i[s['store_id']])
                
        def check_tw_feasibility(route_stores):
            curr_time = 0.0
            prev_idx = 0
            for s_idx in route_stores:
                if prev_idx == 0:
                    arrival_time = self.np_sched[s_idx]
                else:
                    travel = self.np_time[prev_idx, s_idx]
                    dwell = self.np_dwell[prev_idx] if prev_idx != 0 else 0
                    arrival_time = curr_time + round(travel + dwell)
                    
                if arrival_time > self.np_latest[s_idx] or arrival_time < self.np_earliest[s_idx]:
                    return False
                curr_time = arrival_time
                prev_idx = s_idx
            return True

        def calc_route_distance(route_stores):
            if not route_stores:
                return 0.0
            dist = self.np_dist[0, route_stores[0]]
            for i in range(len(route_stores) - 1):
                dist += self.np_dist[route_stores[i], route_stores[i+1]]
            dist += self.np_dist[route_stores[-1], 0]
            return dist

        # 3. Mainline Insertion (Solomon Heuristic)
        main_order = list(range(self.M))
        random.shuffle(main_order)
        
        for r in main_order:
            cap = self.main_route_caps[r]
            while True:
                best_insertions = []
                
                curr_route = main_stores_kept[r]
                curr_vol = sum(self.np_volume[s] for s in curr_route)
                curr_dist = calc_route_distance(curr_route)
                
                for s_idx in support_pool:
                    vol = self.np_volume[s_idx]
                    if curr_vol + vol > cap:
                        continue
                        
                    for k in range(len(curr_route) + 1):
                        test_route = curr_route[:k] + [s_idx] + curr_route[k:]
                        if check_tw_feasibility(test_route):
                            test_dist = calc_route_distance(test_route)
                            cost_diff = test_dist - curr_dist
                            best_insertions.append((cost_diff, s_idx, k))
                                
                if best_insertions:
                    best_insertions.sort(key=lambda x: x[0])
                    if randomize:
                        top_k = min(3, len(best_insertions))
                        chosen = random.choice(best_insertions[:top_k])
                    else:
                        chosen = best_insertions[0]
                        
                    main_stores_kept[r].insert(chosen[2], chosen[1])
                    support_pool.remove(chosen[1])
                else:
                    break
                    
        # 4. Support Line Generation
        support_routes = []
        while support_pool:
            # Seed point: farthest from depot
            seed_store = max(support_pool, key=lambda x: self.np_dist[0, x])
            curr_route = [seed_store]
            support_pool.remove(seed_store)
            
            curr_region = self.np_region[seed_store]
            curr_group = self.np_group[seed_store]
            
            while True:
                best_insertions = []
                
                curr_vol = sum(self.np_volume[s] for s in curr_route)
                curr_dist = calc_route_distance(curr_route)
                
                for s_idx in support_pool:
                    vol = self.np_volume[s_idx]
                    if curr_vol + vol > self.support_capacity:
                        continue
                    
                    r_reg = self.np_region[s_idx]
                    g_grp = self.np_group[s_idx]
                    if curr_region != -1 and r_reg != -1 and r_reg != curr_region:
                        continue
                    if curr_group != -1 and g_grp != -1 and g_grp != curr_group:
                        continue
                        
                    for k in range(len(curr_route) + 1):
                        test_route = curr_route[:k] + [s_idx] + curr_route[k:]
                        if check_tw_feasibility(test_route):
                            test_dist = calc_route_distance(test_route)
                            cost_diff = test_dist - curr_dist
                            best_insertions.append((cost_diff, s_idx, k))
                                
                if best_insertions:
                    best_insertions.sort(key=lambda x: x[0])
                    if randomize:
                        top_k = min(3, len(best_insertions))
                        chosen = random.choice(best_insertions[:top_k])
                    else:
                        chosen = best_insertions[0]
                        
                    curr_route.insert(chosen[2], chosen[1])
                    support_pool.remove(chosen[1])
                else:
                    break
                    
            support_routes.append(curr_route)
            
        # 5. Build Chromosome
        for sr in support_routes:
            for s_idx in sr:
                chromo.append(s_idx)
            
        # Add dividers and kept main stores
        for r in range(self.M):
            div_val = self.N + r + 1
            chromo.append(div_val)
            for s_idx in main_stores_kept[r]:
                chromo.append(s_idx)
                
        return chromo

    def run(self):
        if not self.remaining_stores:
            return 0, {}
            
        base_chromo = self._generate_initial_individual(randomize=False)
        population = [base_chromo]
        
        # Inject diverse but high-quality individuals
        grasp_count = min(self.population_size // 10, 30)
        for _ in range(grasp_count):
            population.append(self._generate_initial_individual(randomize=True))
        
        while len(population) < self.population_size:
            mutated = list(base_chromo)
            # Completely shuffle the chromosome for true randomness
            random.shuffle(mutated)
            population.append(mutated)
            
        self.best_cost = float('inf')
        best_chromo = base_chromo
        early_stopper = EarlyStopper(patience=self.early_stop_patience)
        
        shared_cache = {}
        
        for gen_idx in range(self.generations):
            evaluated_pop = []
            fitnesses = []
            
            for chromo in population:
                key = self._encode_individual(chromo)
                
                if key not in shared_cache:
                    fast_cost, fast_breakdown = self._evaluate_individual_fast(chromo)
                    shared_cache[key] = {
                        'cost': fast_cost,
                        'breakdown': fast_breakdown,
                        'optimized_chromo': chromo
                    }
                    
                res = shared_cache[key]
                opt_chromo = res.get('optimized_chromo', chromo)
                
                evaluated_pop.append({
                    'individual': opt_chromo,
                    'cost': res['cost'],
                    'breakdown': res['breakdown']
                })
                fit_val = res['cost']
                fitnesses.append(fit_val)
                
            evaluated_pop.sort(key=lambda x: x['cost'])
            current_best = evaluated_pop[0]
                
            if current_best['cost'] < self.best_cost:
                self.best_cost = current_best['cost']
                self.best_breakdown = current_best['breakdown']
                best_chromo = copy.deepcopy(current_best['individual'])
                
            print(f'Iteration {gen_idx+1} | Best Cost: {self.best_cost:.4f}')
            
            self.log.append({
                'iteration': gen_idx + 1,
                'iter_worst_cost': float(np.max(fitnesses)),
                'iter_best_cost': float(current_best['cost']),
                'iter_avg_cost': float(np.mean(fitnesses)),
                'std_cost': float(np.std(fitnesses)),
                'best_cost': self.best_cost
            })
            
            if early_stopper.check(self.best_cost):
                break
                
            if self.target_cost is not None:
                if self.best_cost <= self.target_cost + 1e-4:
                    print(f"GA Early Stop: Reached target cost {self.target_cost:.4f} (Best Known)")
                    break
                
            # Elitism
            elites = [copy.deepcopy(ind['individual']) for ind in evaluated_pop[:self.elite_size]]
            weights = [max(1.0 / ind['cost'], 1e-12) for ind in evaluated_pop]
            
            child_pairs = []
            while len(child_pairs) < (self.population_size - self.elite_size):
                p1, p2 = random.choices(evaluated_pop, weights=weights, k=2)
                c1, c2 = self._crossover(copy.deepcopy(p1['individual']), copy.deepcopy(p2['individual']))
                c1 = self._mutate(list(c1))
                c2 = self._mutate(list(c2))
                child_pairs.append((c1, c2))
                
            childrens = []
            for c1, c2 in child_pairs:
                # Evaluate children if not in cache to select the best
                for c in (c1, c2):
                    k = self._encode_individual(c)
                    if k not in shared_cache:
                        fast_cost, fast_breakdown = self._evaluate_individual_fast(c)
                        shared_cache[k] = {
                            'cost': fast_cost,
                            'breakdown': fast_breakdown,
                            'optimized_chromo': c
                        }
                        
                k1 = self._encode_individual(c1)
                k2 = self._encode_individual(c2)
                res1 = shared_cache[k1]
                res2 = shared_cache[k2]
                
                c1_opt = res1.get('optimized_chromo', c1)
                c2_opt = res2.get('optimized_chromo', c2)
                
                if res1['cost'] < res2['cost']:
                    childrens.append(c1_opt)
                else:
                    childrens.append(c2_opt)

            population = elites + childrens[:(self.population_size - self.elite_size)]

        self.best_solution = self._decode_to_routes(best_chromo)
        return self.best_cost, self.best_breakdown, self.best_solution, self.log
