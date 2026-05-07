import random
import numpy as np
from datetime import datetime, timedelta
from numba import njit
from config import config
from models.route_manager import RouteManager
from solvers.vnd import VND
from utils.early_stopper import EarlyStopper

@njit
def _njit_dist_heuristic(current_idx, next_idx, distance_matrix):
    if current_idx == next_idx:
        return 0.0
    return 1.0 / (distance_matrix[current_idx, next_idx] + 1e-12)

@njit(cache=True)
def _njit_transition_value(current_idx, next_idx, distance_matrix, pheromone_matrix, alpha, beta):
    tau = pheromone_matrix[current_idx, next_idx]
    eta = _njit_dist_heuristic(current_idx, next_idx, distance_matrix)
    return (tau ** alpha) * (eta ** beta)

@njit
def _njit_greedy_selection(current_idx, feasible_indices, distance_matrix, pheromone_matrix, alpha, beta):
    best_idx = -1
    best_val = -1e12
    for idx in feasible_indices:
        val = _njit_transition_value(current_idx, idx, distance_matrix, pheromone_matrix, alpha, beta)
        if val > best_val:
            best_val = val
            best_idx = idx
    return best_idx

@njit
def _njit_roulette_selection(current_idx, feasible_indices, distance_matrix, pheromone_matrix, alpha, beta, q0, rand_q, rand_r):
    if len(feasible_indices) == 1:
        return feasible_indices[0]

    if rand_q < q0:
        return _njit_greedy_selection(current_idx, feasible_indices, distance_matrix, pheromone_matrix, alpha, beta)

    n_feas = len(feasible_indices)
    probs = np.zeros(n_feas, dtype=np.float64)
    sum_prob = 0.0
    for i in range(n_feas):
        val = _njit_transition_value(current_idx, feasible_indices[i], distance_matrix, pheromone_matrix, alpha, beta)
        probs[i] = val
        sum_prob += val

    if sum_prob == 0:
        return feasible_indices[0]

    r = rand_r * sum_prob
    cum = 0.0
    for i in range(n_feas):
        cum += probs[i]
        if r <= cum:
            return feasible_indices[i]

    return feasible_indices[-1]


@njit
def _njit_get_feasible_stores(unvisited_indices, last_idx, route_vol, curr_duration,
                               prev_pred_time_epoch, dc_idx, support_capacity, time_limit,
                               dist_group, region, volume, time_matrix, dwell_time, 
                               earliest_time, latest_time, is_solomon):
    feasible = []
    last_g = dist_group[last_idx]
    last_r = region[last_idx]
    
    for i in range(len(unvisited_indices)):
        store_idx = unvisited_indices[i]
        store_g = dist_group[store_idx]
        store_r = region[store_idx]
        
        # Region constraint
        if not is_solomon:
            if last_g == 2:
                if (last_r == 0 and store_r == 1) or \
                   (last_r == 1 and store_r == 0) or \
                   (last_r == 2 and store_r == 3) or \
                   (last_r == 3 and store_r == 2):
                    continue
                    
            if last_g == 2 and store_g not in (0, 1, 2):
                continue
            elif last_g == 1 and store_g not in (0, 1):
                continue
            elif last_g == 0 and store_g != 0:
                continue
            
        # Capacity
        if route_vol + volume[store_idx] > support_capacity:
            continue
            
        # Time constraints
        pre_to_cur = time_matrix[last_idx, store_idx]
        prev_dwell = dwell_time[last_idx] if last_idx != dc_idx else 0
        arrival_time = prev_pred_time_epoch + pre_to_cur + prev_dwell
        
        if is_solomon:
            if arrival_time > latest_time[store_idx]:
                continue
                
            start_time = max(arrival_time, earliest_time[store_idx])
            pre_to_dc = time_matrix[last_idx, dc_idx]
            cur_to_dc = time_matrix[store_idx, dc_idx]
            # new_duration in minutes
            new_duration = curr_duration + (pre_to_cur + cur_to_dc - pre_to_dc) + (start_time - arrival_time) + dwell_time[store_idx]
        else:
            # Non-Solomon: strict time window — arrival must be within [earliest, latest]
            if arrival_time < earliest_time[store_idx]:
                continue
            if arrival_time > latest_time[store_idx]:
                continue

            pre_to_dc = time_matrix[last_idx, dc_idx]
            cur_to_dc = time_matrix[store_idx, dc_idx]
            # new_duration in seconds (no waiting allowed)
            new_duration = curr_duration + (pre_to_cur + cur_to_dc - pre_to_dc) + dwell_time[store_idx]

        if new_duration > time_limit:
            continue
            
        feasible.append(store_idx)
        
    return np.array(feasible, dtype=np.int64)


class SupportLinePlanningACO:
    """
    Notes:
        Ant Colony Optimization for Support Line Planning (Numba Optimized).
    """
    _cached_mappings = None

    def __init__(self, remaining_stores, distance_matrix, time_matrix, num_ants=None, iterations=1, alpha=1.0, beta=1.0, rho=0.1, q=1.0, q0=0.8, early_stop_patience=10, support_capacity=7.2, vehicle_cost=2000, time_limit_per_route=5 * 60 * 60, is_solomon=False, target_cost=None, vnd_strategy='best'):
        self.dc = config.DC_CONFIG
        self.remaining_stores = remaining_stores
        self.orig_distance_matrix = distance_matrix
        self.orig_time_matrix = time_matrix
        self.num_ants = num_ants if num_ants is not None else len(remaining_stores) * 2
        self.store_count = len(remaining_stores)
        self.iterations = iterations
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rho = float(rho)
        self.q = float(q)
        self.tau0 = 0.0
        self.q0 = float(q0)
        self.early_stop_patience = early_stop_patience
        self.support_capacity = float(support_capacity)
        self.vehicle_cost = float(vehicle_cost)
        self.time_limit_per_route = time_limit_per_route
        self.is_solomon = is_solomon
        self.target_cost = target_cost
        self.vnd_strategy = vnd_strategy
        self.log = []
        self.best_cost = (float('inf'), float('inf')) if self.is_solomon else float('inf')
        self.best_solution = None

        self.vnd = VND(self.orig_distance_matrix, self.orig_time_matrix, vehicle_cost=self.vehicle_cost, is_solomon=self.is_solomon, vnd_strategy=self.vnd_strategy)

        # Apply mapping
        self._init_numpy_mappings()


    def _init_numpy_mappings(self):
        """Map dictionaries to numpy arrays for numba compilation"""
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
        self.pheromone_matrix = np.zeros((n, n), dtype=np.float64)
        
        region_map = {'north': 0, 'south': 1, 'east': 2, 'west': 3}
        group_map = {'near': 0, 'mid': 1, 'far': 2}
        
        # Populate arrays
        base_dt = datetime(2024, 1, 1, 0, 0, 0)
        for i in range(n):
            s_i = self.i2s[i]
            s_i_id = s_i['store_id']
            if i > 0:
                self.np_volume[i] = s_i.get('volume', 0.0)
                self.np_dwell[i] = s_i.get('dwell_time', 0)
                
                dt_e = datetime.fromisoformat(s_i['earliest_time'])
                dt_l = datetime.fromisoformat(s_i['latest_time'])
                
                if self.is_solomon:
                    self.np_earliest[i] = int((dt_e - base_dt).total_seconds() / 60)
                    self.np_latest[i] = int((dt_l - base_dt).total_seconds() / 60)
                else:
                    self.np_earliest[i] = int(dt_e.timestamp())
                    self.np_latest[i] = int(dt_l.timestamp())
                
                self.np_group[i] = group_map.get(s_i.get('dist_group', ''), -1)
                self.np_region[i] = region_map.get(s_i.get('region', ''), -1)
                
            for j in range(n):
                s_j_id = self.i2s[j]['store_id']
                if s_i_id in self.orig_distance_matrix and s_j_id in self.orig_distance_matrix[s_i_id]:
                    self.np_dist[i, j] = self.orig_distance_matrix[s_i_id][s_j_id]
                if s_i_id in self.orig_time_matrix and s_j_id in self.orig_time_matrix[s_i_id]:
                    self.np_time[i, j] = self.orig_time_matrix[s_i_id][s_j_id]

        # DC departure time: for non-Solomon, use the earliest store opening time
        if not self.is_solomon and self.store_count > 0:
            self.dc_departure_time = int(np.min(self.np_earliest[1:self.store_count + 1]))
        else:
            self.dc_departure_time = 0

    # Re-wrap original functions referencing Numpy mappings
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

    def _feasible_stores_idx(self, route, unvisited_idx):
        base_dt = datetime(2024, 1, 1, 0, 0, 0)
        if len(route['stores']) == 0:
            last_idx = 0 # DC
            route_vol = 0.0
            curr_duration = 0.0
            prev_pred_time_epoch = self.dc_departure_time if not self.is_solomon else 0
        else:
            last_store = route['stores'][-1]
            last_idx = self.s2i[last_store['store_id']]
            route_vol = route['dc']['total_volume']
            curr_duration = route['dc']['duration']
            
            dt = datetime.fromisoformat(last_store['pred_time'])
            if self.is_solomon:
                prev_pred_time_epoch = int((dt - base_dt).total_seconds() / 60)
            else:
                prev_pred_time_epoch = int(dt.timestamp())

        unv_arr = np.array(list(unvisited_idx), dtype=np.int64)
        
        return _njit_get_feasible_stores(
            unv_arr, last_idx, route_vol, curr_duration,
            prev_pred_time_epoch, 0, self.support_capacity, self.time_limit_per_route,
            self.np_group, self.np_region, self.np_volume, self.np_time, self.np_dwell,
            self.np_earliest, self.np_latest, self.is_solomon
        )

    def _greedy_solution(self):
        if not self.remaining_stores:
            return {}

        solution = {}
        vehicle_num = 101
        unvisited_idx = set(range(1, self.store_count + 1))
        
        route_manager = RouteManager(solution, self.orig_distance_matrix, self.orig_time_matrix, is_solomon=self.is_solomon)

        while unvisited_idx:
            vehicle_id = f'{vehicle_num}'
            route = self._initial_route(vehicle_id)
            solution[vehicle_id] = route

            feasible_arr = self._feasible_stores_idx(route, unvisited_idx)
            if len(feasible_arr) == 0:
                # No feasible stores from DC: force-assign one store regardless of time window
                # so no store is left behind
                forced_idx = next(iter(unvisited_idx))
                forced_store = self.i2s[forced_idx]
                route_manager.add_store(vehicle_id, forced_store)
                unvisited_idx.remove(forced_idx)
                vehicle_num += 1
                continue

            current_idx = _njit_greedy_selection(0, feasible_arr, self.np_dist, self.pheromone_matrix, self.alpha, self.beta)
            current_store = self.i2s[current_idx]

            route_manager.add_store(vehicle_id, current_store)
            unvisited_idx.remove(current_idx)

            while unvisited_idx:
                feasible_arr = self._feasible_stores_idx(route, unvisited_idx)
                if len(feasible_arr) == 0:
                    break

                next_idx = _njit_greedy_selection(current_idx, feasible_arr, self.np_dist, self.pheromone_matrix, self.alpha, self.beta)
                next_store = self.i2s[next_idx]
                route_manager.add_store(vehicle_id, next_store)
                current_idx = next_idx
                unvisited_idx.remove(next_idx)
            
            vehicle_num += 1

        return solution

    def _cost_function(self, solution):
        total_cost = 0
        for vehicle_id in solution:
            stores = solution[vehicle_id]['stores']
            if not stores:
                continue

            total_cost += self.orig_distance_matrix[self.dc['store_id']][stores[0]['store_id']]
            for i in range(len(stores) - 1):
                total_cost += self.orig_distance_matrix[stores[i]['store_id']][stores[i+1]['store_id']]
            total_cost += self.orig_distance_matrix[stores[-1]['store_id']][self.dc['store_id']]

        if self.is_solomon:
            return (len(solution), total_cost)
            
        total_cost += len(solution) * self.vehicle_cost
        return total_cost

    def _initial_pheromone(self, cost):
        cost_val = cost[0] * self.vehicle_cost + cost[1] if self.is_solomon else cost
        initial_pheromone = self.q / cost_val
        self.tau0 = initial_pheromone
        self.pheromone_matrix.fill(initial_pheromone)
        for i in range(self.pheromone_matrix.shape[0]):
            self.pheromone_matrix[i, i] = 0.0

    def _log_iteration(self, i, ant_costs, iter_best_cost, iter_worst_cost=None):
        if iter_worst_cost is None:
            iter_worst_cost = float(np.max(ant_costs))

        self.log.append({
            'iteration': i,
            'iter_worst_cost': iter_worst_cost,
            'iter_best_cost': iter_best_cost,
            'iter_avg_cost': float(sum(ant_costs) / len(ant_costs)),
            'std_cost': float(np.std(ant_costs)),
            'best_cost': self.best_cost,
        })

    def _solution_construction(self, start_store_idx=None):
        if not self.remaining_stores:
            return {}

        ant_solution = {}
        vehicle_num = 101
        unvisited_idx = set(range(1, self.store_count + 1))
        route_manager = RouteManager(ant_solution, self.orig_distance_matrix, self.orig_time_matrix, is_solomon=self.is_solomon)

        while unvisited_idx:
            vehicle_id = f'{vehicle_num}'
            route = self._initial_route(vehicle_id)
            ant_solution[vehicle_id] = route

            # Try to pick the first store for this route
            feasible_arr = self._feasible_stores_idx(route, unvisited_idx)
            if len(feasible_arr) == 0:
                # No feasible stores from DC: force-assign one store and continue
                forced_idx = next(iter(unvisited_idx))
                forced_store = self.i2s[forced_idx]
                route_manager.add_store(vehicle_id, forced_store)
                unvisited_idx.remove(forced_idx)
                vehicle_num += 1
                continue

            if start_store_idx is not None and start_store_idx in unvisited_idx and vehicle_num == 101:
                if start_store_idx in feasible_arr:
                    current_idx = start_store_idx
                else:
                    current_idx = feasible_arr[0]
            else:
                rand_q = random.uniform(0.0, 1.0)
                rand_r = random.uniform(0.0, 1.0)
                current_idx = _njit_roulette_selection(0, feasible_arr, self.np_dist, self.pheromone_matrix, self.alpha, self.beta, self.q0, rand_q, rand_r)
            
            if current_idx == -1:
                break

            current_store = self.i2s[current_idx]
            route_manager.add_store(vehicle_id, current_store)
            unvisited_idx.remove(current_idx)

            # Pick subsequent stores
            while unvisited_idx:
                feasible_arr = self._feasible_stores_idx(route, unvisited_idx)
                if len(feasible_arr) == 0:
                    break

                rand_q = random.uniform(0.0, 1.0)
                rand_r = random.uniform(0.0, 1.0)
                next_idx = _njit_roulette_selection(current_idx, feasible_arr, self.np_dist, self.pheromone_matrix, self.alpha, self.beta, self.q0, rand_q, rand_r)
                
                if next_idx == -1:
                    break

                next_store = self.i2s[next_idx]
                route_manager.add_store(vehicle_id, next_store)
                current_idx = next_idx
                unvisited_idx.remove(next_idx)
            
            vehicle_num += 1

        return ant_solution


    def _evaporate_pheromone(self):
        self.pheromone_matrix *= (1 - self.rho)

    def _calculate_tau_bounds(self):
        p_decisive = 0.05
        avg = 2.5
        root_n = p_decisive ** (1.0 / max(1, self.store_count))
        best_val = self.best_cost[0] * self.vehicle_cost + self.best_cost[1] if self.is_solomon else self.best_cost
        tau_max = self.q / (self.rho * best_val)
        tau_min = (tau_max * (1 - root_n)) / ((avg - 1) * root_n)
        return tau_max, tau_min

    def _deposit_global_pheromone(self, solution, cost):
        cost_val = cost[0] * self.vehicle_cost + cost[1] if self.is_solomon else cost
        if cost_val == 0:
            return
        delta_pheromone = self.q / cost_val
        tau_max, tau_min = self._calculate_tau_bounds()
        
        for route in solution.values():
            stores = route['stores']
            if not stores:
                continue
            path_idx = [0] + [self.s2i[s['store_id']] for s in stores] + [0]
            for i in range(len(path_idx) - 1):
                u, v = path_idx[i], path_idx[i+1]
                self.pheromone_matrix[u, v] += delta_pheromone
                self.pheromone_matrix[v, u] += delta_pheromone

        np.clip(self.pheromone_matrix, tau_min, tau_max, out=self.pheromone_matrix)

    def run(self):
        if not self.remaining_stores:
            return 0, {}

        self._initial_pheromone((1, 1) if self.is_solomon else 1)
        greedy_solution = self._greedy_solution()
        greedy_cost = self._cost_function(greedy_solution)
        self.best_cost = greedy_cost
        self.best_solution = greedy_solution

        if self.iterations == 0:
            return self.best_cost, self.best_solution

        self._log_iteration(0, [greedy_cost[1] if self.is_solomon else greedy_cost], greedy_cost[1] if self.is_solomon else greedy_cost, greedy_cost[1] if self.is_solomon else greedy_cost)
        self._initial_pheromone(greedy_cost)
        early_stopper = EarlyStopper(patience=self.early_stop_patience)

        print(f'Support Line: iteration{0} -> best_cost: {self.best_cost[1] if self.is_solomon else self.best_cost:.2f}')

        for i in range(self.iterations):
            ant_costs = []
            distances = []
            iter_best_cost = (float('inf'), float('inf')) if self.is_solomon else float('inf')
            iter_best_solution = None
            for ant_idx in range(self.num_ants):
                start_store_idx = (ant_idx % self.store_count) + 1
                ant_solution = self._solution_construction(start_store_idx=start_store_idx)
                ant_cost = self._cost_function(ant_solution)
                ant_costs.append(ant_cost)
                distances.append(ant_cost[1] if self.is_solomon else ant_cost)

                if ant_cost < iter_best_cost:
                    iter_best_cost = ant_cost
                    iter_best_solution = ant_solution

            optimized_routes, optimized_cost = self.vnd.optimize(iter_best_solution)

            if optimized_cost < self.best_cost:
                self.best_cost = optimized_cost
                self.best_solution = optimized_routes

            if self.is_solomon:
                print(f'Support Line: iteration{i + 1} -> best_cost: {self.best_cost[0]} vehicles, {self.best_cost[1]:.2f} distance')
            else:
                print(f'Support Line: iteration{i + 1} -> best_cost: {self.best_cost:.2f}')

            self._evaporate_pheromone()
            self._deposit_global_pheromone(optimized_routes, optimized_cost)
            self._log_iteration(i + 1, distances, optimized_cost[1] if self.is_solomon else optimized_cost)

            if early_stopper.check(self.best_cost):
                break
                
            if self.target_cost is not None:
                if self.is_solomon:
                    if self.best_cost[0] < self.target_cost[0] or (self.best_cost[0] == self.target_cost[0] and self.best_cost[1] <= self.target_cost[1] + 1e-4):
                        print(f"ACO Early Stop: Reached target cost NV={self.target_cost[0]}, Dist={self.target_cost[1]:.2f} (Best Known)")
                        break
                else:
                    if self.best_cost <= self.target_cost + 1e-4:
                        print(f"ACO Early Stop: Reached target cost {self.target_cost:.2f} (Best Known)")
                        break

        return self.best_cost, self.best_solution
    pass

# ----------------------------------------------------------------------------
# NUMBA WARMUP (PRE-COMPILE) TO PREVENT MULTIPROCESSING DEADLOCKS
# ----------------------------------------------------------------------------
try:
    _njit_dist_heuristic(0, 0, np.zeros((1, 1)))
    _njit_transition_value(0, 0, np.zeros((1, 1)), np.zeros((1, 1)), 1.0, 1.0)
    _njit_greedy_selection(0, np.array([0], dtype=np.int64), np.zeros((1, 1)), np.zeros((1, 1)), 1.0, 1.0)
    _njit_roulette_selection(0, np.array([0], dtype=np.int64), np.zeros((1, 1)), np.zeros((1, 1)), 1.0, 1.0, 0.5, 0.5, 0.5)
    _njit_get_feasible_stores(
        np.array([0], dtype=np.int64), 0, 0.0, 0.0, 0, 0, 10.0, 10.0,
        np.array([0], dtype=np.int64), np.array([0], dtype=np.int64), np.array([0.0]),
        np.zeros((1, 1)), np.array([0], dtype=np.int64), np.array([0], dtype=np.int64), np.array([0], dtype=np.int64), False
    )
except Exception as e:
    pass
