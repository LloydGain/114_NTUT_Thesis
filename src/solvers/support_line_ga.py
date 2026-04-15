import copy
import random
import hashlib
import numpy as np
from datetime import datetime
from multiprocessing import Pool, cpu_count, Manager
from multiprocessing import TimeoutError as MPTimeoutError

from config import config
from models.route_manager import RouteManager
from utils.early_stopper import EarlyStopper
from numba import njit
from solvers.support_line_aco import _njit_get_feasible_stores, _njit_greedy_selection

@njit(cache=True)
def _njit_evaluate_insertion(route_arr, insert_idx, u_idx, capacity, time_limit,
                             np_volume, np_earliest, np_latest, np_dwell, np_time, np_dist, np_region, np_group, is_solomon):
    total_vol = np_volume[u_idx]
    for idx in route_arr:
        total_vol += np_volume[idx]
    if total_vol > capacity:
        return False, 0.0
        
    route_region = -1
    route_group = -1
    first_store_idx = u_idx if insert_idx == 0 else route_arr[0]
    
    T = 0.0
    prev_idx = 0
    route_dist = 0.0
    
    length = len(route_arr)
    for i in range(length + 1):
        if i < insert_idx:
            curr_idx = route_arr[i]
        elif i == insert_idx:
            curr_idx = u_idx
        else:
            curr_idx = route_arr[i - 1]
            
        g = np_group[curr_idx]
        r = np_region[curr_idx]
        if not is_solomon:
            if route_region == -1 and r != -1:
                route_region = r
            elif r != -1 and r != route_region:
                return False, 0.0
                
            if route_group == -1 and g != -1:
                route_group = g
            elif g != -1 and g != route_group:
                return False, 0.0
            
        if i == 0:
            T = float(np_earliest[curr_idx])
        else:
            T = T + np_dwell[prev_idx] + np_time[prev_idx, curr_idx]
            if is_solomon:
                if T > np_latest[curr_idx]:
                    return False, 0.0
                if T < np_earliest[curr_idx]:
                    T = float(np_earliest[curr_idx])
            else:
                if T < np_earliest[curr_idx] or T > np_latest[curr_idx]:
                    return False, 0.0
                
        route_dist += np_dist[prev_idx, curr_idx]
        prev_idx = curr_idx
        
    T_end = T + np_dwell[prev_idx] + np_time[prev_idx, 0]
    depart_time = float(np_earliest[first_store_idx]) - np_time[0, first_store_idx]
    
    if T_end - depart_time > time_limit:
        return False, 0.0
        
    route_dist += np_dist[prev_idx, 0]
    
    base_dist = 0.0
    if length > 0:
        p_idx = 0
        for i in range(length):
            c_idx = route_arr[i]
            base_dist += np_dist[p_idx, c_idx]
            p_idx = c_idx
        base_dist += np_dist[p_idx, 0]
        
    return True, route_dist - base_dist

@njit
def _njit_find_best_insertion(route_arr, unvisited_arr, capacity, time_limit, np_volume, np_earliest, np_latest, np_dwell, np_time, np_dist, np_region, np_group, is_solomon):
    best_u = -1
    best_p = -1
    best_cost = 1e12
    
    for i in range(len(unvisited_arr)):
        u = unvisited_arr[i]
        for p in range(len(route_arr) + 1):
            is_feasible, cost = _njit_evaluate_insertion(
                route_arr, p, u, capacity, time_limit,
                np_volume, np_earliest, np_latest, np_dwell, 
                np_time, np_dist, np_region, np_group, is_solomon
            )
            if is_feasible and cost < best_cost:
                best_cost = cost
                best_u = u
                best_p = p
                
    return best_u, best_p

@njit
def _njit_evaluate_chromosome_cost(permutation, np_volume, np_earliest, np_latest, np_dwell, np_time, np_dist, np_region, np_group, support_capacity, time_limit, vehicle_cost, is_solomon):
    total_cost = 0.0
    valid_vehicles = 0
    
    current_vol = 0.0
    current_time = 0.0
    current_dist = 0.0
    current_region = -1
    current_group = -1
    prev_idx = 0
    depart_time = 0.0
    
    i = 0
    while i < len(permutation):
        curr_idx = permutation[i]
        is_feasible = True
        
        if prev_idx != 0:
            new_vol = current_vol + np_volume[curr_idx]
            if new_vol > support_capacity:
                is_feasible = False
            else:
                r = np_region[curr_idx]
                g = np_group[curr_idx]
                if not is_solomon and current_region != -1 and r != -1 and r != current_region:
                    is_feasible = False
                elif not is_solomon and current_group != -1 and g != -1 and g != current_group:
                    is_feasible = False
                else:
                    arr_time = current_time + np_dwell[prev_idx] + np_time[prev_idx, curr_idx]
                    
                    if is_solomon:
                        if arr_time > np_latest[curr_idx]:
                            is_feasible = False
                        else:
                            start_time = arr_time if arr_time > float(np_earliest[curr_idx]) else float(np_earliest[curr_idx])
                            ret_time = start_time + np_dwell[curr_idx] + np_time[curr_idx, 0]
                            if ret_time - depart_time > time_limit:
                                is_feasible = False
                    else:
                        if arr_time < float(np_earliest[curr_idx]) or arr_time > float(np_latest[curr_idx]):
                            is_feasible = False
                        else:
                            start_time = arr_time
                            ret_time = start_time + np_dwell[curr_idx] + np_time[curr_idx, 0]
                            if ret_time - depart_time > time_limit:
                                is_feasible = False
                            
        if is_feasible and prev_idx != 0:
            current_vol += np_volume[curr_idx]
            r = np_region[curr_idx]
            g = np_group[curr_idx]
            if not is_solomon:
                if current_region == -1 and r != -1:
                    current_region = r
                if current_group == -1 and g != -1:
                    current_group = g
            arr_time = current_time + np_dwell[prev_idx] + np_time[prev_idx, curr_idx]
            
            if is_solomon:
                current_time = arr_time if arr_time > float(np_earliest[curr_idx]) else float(np_earliest[curr_idx])
            else:
                current_time = arr_time
            current_dist += np_dist[prev_idx, curr_idx]
            prev_idx = curr_idx
            i += 1
        elif is_feasible and prev_idx == 0:
            valid_vehicles += 1
            current_vol = np_volume[curr_idx]
            current_region = np_region[curr_idx]
            current_group = np_group[curr_idx]
            depart_time = float(np_earliest[curr_idx]) - np_time[0, curr_idx]
            current_time = float(np_earliest[curr_idx])
            current_dist += np_dist[0, curr_idx]
            prev_idx = curr_idx
            i += 1
        else:
            current_dist += np_dist[prev_idx, 0]
            total_cost += current_dist
            
            # Reset for next vehicle
            current_vol = 0.0
            current_time = 0.0
            current_dist = 0.0
            current_region = -1
            current_group = -1
            prev_idx = 0
            depart_time = 0.0
            
    if prev_idx != 0:
        current_dist += np_dist[prev_idx, 0]
        total_cost += current_dist
        
    if is_solomon:
        return float(valid_vehicles), total_cost
    else:
        return 0.0, total_cost + valid_vehicles * vehicle_cost

def init_ga_worker(ga_instance):
    global ALLOC_GA_INSTANCE
    ALLOC_GA_INSTANCE = ga_instance


def ga_fitness_worker(args):
    chromo, key, shared_cache = args

    if key in shared_cache:
        return shared_cache[key]

    ga = ALLOC_GA_INSTANCE
    cost = ga._evaluate_individual_fast(chromo)

    result = {
        'cost': cost,
        'routes': None
    }

    shared_cache[key] = result
    return result


class SupportLinePlanningGA:
    """
    Genetic Algorithm for Support Line Planning (Numba Optimized Evaluator).
    Created as an extra robust file for comparison against ACO.
    """
    def __init__(self, remaining_stores, distance_matrix, time_matrix, population_size=1000, elite_rate=0.1, generations=200, cross_rate=0.8, mutation_rate=0.01, early_stop_patience=50, support_capacity=7.2, vehicle_cost=2000, time_limit_per_route=5 * 60 * 60, is_solomon=False, target_cost=None):
        self.dc = config.DC_CONFIG
        self.remaining_stores = remaining_stores
        self.orig_distance_matrix = distance_matrix
        self.orig_time_matrix = time_matrix
        self.store_count = len(remaining_stores)
        self.population_size = population_size
        self.elite_size = max(1, int(population_size * elite_rate))
        self.generations = generations
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.early_stop_patience = early_stop_patience
        self.support_capacity = float(support_capacity)
        self.vehicle_cost = float(vehicle_cost)
        self.time_limit_per_route = time_limit_per_route
        self.is_solomon = is_solomon
        self.target_cost = target_cost
        
        self.log = []
        self.best_cost = (float('inf'), float('inf')) if self.is_solomon else float('inf')
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
            
        n = len(self.s2i)
        self.np_dist = np.zeros((n, n), dtype=np.float64)
        self.np_time = np.zeros((n, n), dtype=np.float64)
        self.np_volume = np.zeros(n, dtype=np.float64)
        self.np_dwell = np.zeros(n, dtype=np.int64)
        self.np_earliest = np.zeros(n, dtype=np.int64)
        self.np_latest = np.zeros(n, dtype=np.int64)
        self.np_group = np.full(n, -1, dtype=np.int64)
        self.np_region = np.full(n, -1, dtype=np.int64)
        
        region_map = {'north': 0, 'south': 1, 'east': 2, 'west': 3}
        group_map = {'near': 0, 'mid': 1, 'far': 2}
        
        for i in range(n):
            s_i = self.i2s[i]
            s_i_id = s_i['store_id']
            if i > 0:
                self.np_volume[i] = s_i.get('volume', 0.0)
                self.np_dwell[i] = s_i.get('dwell_time', 0)
                self.np_earliest[i] = int(datetime.fromisoformat(s_i['earliest_time']).timestamp())
                self.np_latest[i] = int(datetime.fromisoformat(s_i['latest_time']).timestamp())
                self.np_group[i] = group_map.get(s_i.get('dist_group', ''), -1)
                self.np_region[i] = region_map.get(s_i.get('region', ''), -1)
                
            for j in range(n):
                s_j_id = self.i2s[j]['store_id']
                if s_i_id in self.orig_distance_matrix and s_j_id in self.orig_distance_matrix[s_i_id]:
                    self.np_dist[i, j] = self.orig_distance_matrix[s_i_id][s_j_id]
                if s_i_id in self.orig_time_matrix and s_j_id in self.orig_time_matrix[s_i_id]:
                    self.np_time[i, j] = self.orig_time_matrix[s_i_id][s_j_id]

    def _initial_route(self, vehicle_id):
        return {
            "dc": {
                "route_id": vehicle_id,
                "route_code": vehicle_id,
                "store_id": "DC",
                "store_name": "林口ＤＣ",
                "total_volume": 0.0,
                "load_rate": 0.0,
                "max_capacity": self.support_capacity,
                "distance": 0.0,
                "duration": 0.0
            },
            "stores": []
        }

    def _cost_function(self, solution):
        total_cost = 0
        valid_vehicles = 0
        for vehicle_id in solution:
            stores = solution[vehicle_id]['stores']
            if not stores:
                continue
            valid_vehicles += 1
            total_cost += self.orig_distance_matrix[self.dc['store_id']][stores[0]['store_id']]
            for i in range(len(stores) - 1):
                total_cost += self.orig_distance_matrix[stores[i]['store_id']][stores[i+1]['store_id']]
            total_cost += self.orig_distance_matrix[stores[-1]['store_id']][self.dc['store_id']]
            
        if self.is_solomon:
            return (valid_vehicles, total_cost)
            
        total_cost += valid_vehicles * self.vehicle_cost
        return total_cost

    def _generate_solomon_individual(self, random_seed=True):
        chromosome = []
        unvisited = set(range(1, self.store_count + 1))
        all_routes = []
        
        while unvisited:
            if random_seed:
                # Select a seed customer s from UnroutedCustomers (randomly to ensure population diversity)
                best_seed = random.choice(list(unvisited))
            else:
                max_dist = -1
                best_seed = -1
                for u in unvisited:
                    d = self.np_dist[0, u]
                    if d > max_dist:
                        max_dist = d
                        best_seed = u
            
            # Create a new route r <- [best_seed]
            route_arr = [best_seed]
            unvisited.remove(best_seed)
            
            while True:
                np_route = np.array(route_arr, dtype=np.int64)
                unvisited_arr = np.array(list(unvisited), dtype=np.int64)
                
                best_u, best_p = _njit_find_best_insertion(
                    np_route, unvisited_arr, self.support_capacity, self.time_limit_per_route,
                    self.np_volume, self.np_earliest, self.np_latest, self.np_dwell, 
                    self.np_time, self.np_dist, self.np_region, self.np_group, self.is_solomon
                )
                            
                # if BestCandidate is null then break
                if best_u == -1:
                    break
                    
                # Insert u into route r at best position
                route_arr.insert(best_p, best_u)
                unvisited.remove(best_u)
                
            all_routes.append(route_arr)
            
        for r in all_routes:
            chromosome.extend(r)
            
        return chromosome

    def _evaluate_individual(self, permutation):
        solution = {}
        vehicle_num = 101
        route_manager = RouteManager(solution, self.orig_distance_matrix, self.orig_time_matrix, is_solomon=self.is_solomon)
        
        vehicle_id = f'{vehicle_num}'
        route_manager.routes_info[vehicle_id] = self._initial_route(vehicle_id)
        
        for idx in permutation:
            store = self.i2s[idx]
            route_info = route_manager.routes_info[vehicle_id]
            
            if len(route_info['stores']) == 0:
                is_feasible = True
            else:
                last_store = route_info['stores'][-1]
                last_idx = self.s2i[last_store['store_id']]
                route_vol = route_info['dc']['total_volume']
                curr_duration = route_info['dc']['duration']
                prev_pred_time_epoch = int(datetime.fromisoformat(last_store['pred_time']).timestamp())
                
                unv_arr = np.array([idx], dtype=np.int64)
                feasible = _njit_get_feasible_stores(
                    unv_arr, last_idx, route_vol, curr_duration,
                    prev_pred_time_epoch, 0, self.support_capacity, self.time_limit_per_route,
                    self.np_group, self.np_region, self.np_volume, self.np_time, self.np_dwell,
                    self.np_earliest, self.np_latest, self.is_solomon
                )
                is_feasible = len(feasible) > 0
                
            if is_feasible:
                route_manager.add_store(vehicle_id, store)
            else:
                vehicle_num += 1
                vehicle_id = f'{vehicle_num}'
                route_manager.routes_info[vehicle_id] = self._initial_route(vehicle_id)
                route_manager.add_store(vehicle_id, store)
                
        cost = self._cost_function(route_manager.routes_info)
        return cost, route_manager.routes_info

    def _evaluate_individual_fast(self, permutation):
        perm_arr = np.array(permutation, dtype=np.int64)
        v, c = _njit_evaluate_chromosome_cost(
            perm_arr, self.np_volume, self.np_earliest, self.np_latest,
            self.np_dwell, self.np_time, self.np_dist, self.np_region,
            self.np_group, self.support_capacity, self.time_limit_per_route,
            self.vehicle_cost, self.is_solomon
        )
        if self.is_solomon:
            return (int(v), c)
        return c


    def _encode_individual(self, individual):
        s = ','.join(map(str, individual))
        return hashlib.md5(s.encode()).hexdigest()

    def _pmx_crossover(self, parent1, parent2):
        size = len(parent1)
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
            p1, p2 = random.sample(range(size), 2)
            start, end = min(p1, p2), max(p1, p2)
            # Invert the genetic sequence between the two cut points
            individual[start:end+1] = individual[start:end+1][::-1]
        return individual

    def _local_search_sih(self, chromosome):
        new_chromo = list(chromosome)
        
        # 1. 隨機移除少量節點 (例如 2-3 個)
        num_to_remove = random.randint(1, 3)
        removed_nodes = []
        for _ in range(num_to_remove):
            if not new_chromo:
                break
            idx = random.randrange(len(new_chromo))
            removed_nodes.append(new_chromo.pop(idx))
        
        # 2. 針對被移除的節點，利用你已有的 _njit_find_best_insertion 重新插回
        # 這裡可以迭代 removed_nodes，每次找出最適合插入的點與位置
        for node in removed_nodes:
            np_route = np.array(new_chromo, dtype=np.int64)
            unvisited_arr = np.array([node], dtype=np.int64)
            
            # 利用你寫好的 Numba 函式找最佳位置 [cite: 243, 245, 246]
            best_u, best_p = _njit_find_best_insertion(
                np_route, unvisited_arr, self.support_capacity, self.time_limit_per_route,
                self.np_volume, self.np_earliest, self.np_latest, self.np_dwell, 
                self.np_time, self.np_dist, self.np_region, self.np_group, self.is_solomon
            )
            
            if best_p != -1:
                new_chromo.insert(best_p, node)
            else:
                new_chromo.append(node) # 若找不到可行位置則塞回尾端
                
        return new_chromo

    def run(self):
        if not self.remaining_stores:
            return 0, {}
            
        # 1. 初始化族群：結合 Solomon SIH 與 隨機生成
        greedy_chromo = self._generate_solomon_individual(random_seed=False)
        
        if self.generations == 0:
            self.best_cost = self._evaluate_individual_fast(greedy_chromo)
            _, self.best_solution = self._evaluate_individual(greedy_chromo)
            return self.best_cost, self.best_solution
            
        population = [greedy_chromo]
        num_solomon = int(self.population_size * 0.1)

        while len(population) < num_solomon:
            population.append(self._generate_solomon_individual(random_seed=True))
            
        all_indices = np.arange(1, self.store_count + 1, dtype=np.int64)
        while len(population) < self.population_size:
            population.append(np.random.permutation(all_indices).tolist())
            
        self.best_cost = (float('inf'), float('inf')) if self.is_solomon else float('inf')
        best_chromo = greedy_chromo
        early_stopper = EarlyStopper(patience=self.early_stop_patience)
        
        with Manager() as manager:
            shared_cache = manager.dict()
            num_procs = max(1, cpu_count() - 2)

            with Pool(processes=num_procs, initializer=init_ga_worker, initargs=(self,)) as pool:
                for gen_idx in range(self.generations):
                    # --- A. 評估目前族群 ---
                    unique_tasks = []
                    keys = []
                    
                    for chromo in population:
                        key = self._encode_individual(chromo)
                        keys.append(key)
                        if key not in shared_cache:
                            unique_tasks.append((chromo, key, shared_cache))
                            
                    if unique_tasks:
                        try:
                            pool.map_async(ga_fitness_worker, unique_tasks, chunksize=max(1, len(unique_tasks) // num_procs * 2)).get(timeout=120)
                        except MPTimeoutError:
                            print("Timeout Error")
                            for _, key, _ in unique_tasks:
                                if key not in shared_cache:
                                    shared_cache[key] = {'cost': float('inf'), 'routes': {}}
                                    
                    evaluated_pop = []
                    fitnesses = []
                    for i, chromo in enumerate(population):
                        key = keys[i]
                        res = shared_cache[key]
                        evaluated_pop.append({
                            'individual': chromo,
                            'cost': res['cost'],
                        })
                        fit_val = res['cost'][1] if self.is_solomon else res['cost']
                        fitnesses.append(fit_val)
                        
                    evaluated_pop.sort(key=lambda x: x['cost'])
                    current_best = evaluated_pop[0]
                        
                    if current_best['cost'] < self.best_cost:
                        self.best_cost = current_best['cost']
                        best_chromo = copy.deepcopy(current_best['individual'])
                        
                    if self.is_solomon:
                        print(f'Iteration {gen_idx+1} | Best Cost: {self.best_cost[0]} vehicles, {self.best_cost[1]:.4f} distance')
                    else:
                        print(f'Iteration {gen_idx+1} | Best Cost: {self.best_cost:.4f}')
                    
                    self.log.append({
                        'iteration': gen_idx + 1,
                        'iter_worst_cost': float(np.max(fitnesses)),
                        'iter_best_cost': float(current_best['cost'][1] if self.is_solomon else current_best['cost']),
                        'iter_avg_cost': float(np.mean(fitnesses)),
                        'std_cost': float(np.std(fitnesses)),
                        'best_cost': self.best_cost
                    })
                    
                    if early_stopper.check(self.best_cost):
                        break
                        
                    if self.target_cost is not None:
                        if self.is_solomon:
                            if self.best_cost <= (self.target_cost[0], self.target_cost[1] + 1e-4):
                                print(f"GA Early Stop: Reached target cost NV={self.target_cost[0]}, Dist={self.target_cost[1]:.4f} (Best Known)")
                                break
                        else:
                            if self.best_cost <= self.target_cost + 1e-4:
                                print(f"GA Early Stop: Reached target cost {self.target_cost:.4f} (Best Known)")
                                break
                        
                    # --- B. 選擇與演化 (精英保留) ---
                    elites = [copy.deepcopy(ind['individual']) for ind in evaluated_pop[:self.elite_size]]
                    weights = [max(1.0 / (ind['cost'][1] if self.is_solomon else ind['cost']), 1e-12) for ind in evaluated_pop]
                    
                    child_pairs = []
                    while len(child_pairs) < (self.population_size - self.elite_size):
                        p1, p2 = random.choices(evaluated_pop, weights=weights, k=2)
                        c1, c2 = self._crossover(copy.deepcopy(p1['individual']), copy.deepcopy(p2['individual']))
                        c1 = self._mutate(list(c1))
                        c2 = self._mutate(list(c2))
                        child_pairs.append((c1, c2))
                        
                    # --- C. 評估子代候選者 ---
                    task_candidates = []
                    for c1, c2 in child_pairs:
                        for c in (c1, c2):
                            k = self._encode_individual(c)
                            if k not in shared_cache:
                                task_candidates.append((c, k, shared_cache))
                                
                    if task_candidates:
                        try:
                            pool.map_async(ga_fitness_worker, task_candidates, chunksize=max(1, len(task_candidates) // num_procs * 2)).get(timeout=120)
                        except MPTimeoutError:
                            print("Timeout Error during candidate evaluation")
                            for _, k, _ in task_candidates:
                                if k not in shared_cache:
                                    shared_cache[k] = {'cost': float('inf'), 'routes': None}
                                    
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
                            
                    # --- D. 突變與進展 (區域搜尋) ---
                    for i in range(len(childrens)):
                        if random.random() < 0.1:
                            childrens[i] = self._local_search_sih(childrens[i])

                    population = elites + childrens[:(self.population_size - self.elite_size)]

        # 最終解碼
        _, self.best_solution = self._evaluate_individual(best_chromo)
        return self.best_cost, self.best_solution