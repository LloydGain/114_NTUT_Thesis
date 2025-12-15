import random
import hashlib
import numpy as np
from datetime import datetime, timedelta
from route.route import RouteManager
from route.utils import EarlyStopper
from route.support_line_aco import SupportLinePlanningACO

class StoreAllocationACO:
    """
    Notes:
        Store Allocation using Ant Colony Optimization (ACO).
    """
    def __init__(self, main_routes, remaining_stores, distance_matrix, time_matrix, num_ants=1, iterations=1, alpha=1, beta=1, rho=0.1, tau_ratio=50, q=1, q0=0.9, early_stop_patience=10):
        self.main_routes = main_routes
        self.remaining_stores = remaining_stores
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix
        self.dc = {'store_id': 'dc', 'longitude': 121.40712, 'latitude': 25.083282}
        self.time_limit_per_route = 5 * 60 * 60
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.tau_ratio = tau_ratio
        self.q = q
        self.q0 = q0
        self.num_ants = num_ants
        self.iterations = iterations
        self.early_stop_patience = early_stop_patience
        self.pheromone_matrix = dict()
        self.best_cost = float('inf')
        self.best_solution = None
        self.best_ant_solution = None
        self.best_remaining_solution = None
        self.cost_cache = {}
        self.log = []


    def _copy_routes_info(self, routes):
        """
        Notes:
            Create a shallow copy of routes info.
        
        Args:
            routes (dict): Original routes info.

        Returns:
            dict: Routes Info.
        """
        return {
            route_id: {
                "dc": route_data["dc"].copy(),
                "stores": [store.copy() for store in route_data["stores"]]
            } for route_id, route_data in routes.items()
        }                   


    def _heuristic(self, cost):
        """
        Notes:
            Get heuristic information between store and route.

        Args:
            cost (float): The store cost.

        Returns:
            float: Heuristic value (1 / cost).
        """
        return 1 / cost
        

    def _pheromone(self, store, route_id):
        """
        Notes:
            Get pheromone value between store and route.
        
        Args:
            store (dict): The current store.
            route_id (str): The candidate route.
        
        Returns:
            float: The pheromone value.
        """
        return self.pheromone_matrix[store['store_id']][route_id]


    def _transition_value(self, store, route_id, cost):
        """
        Notes:
            Calculate the transition value for moving from current_store to next_store.
        
        Args:
            store (dict): The current store.
            route_id (str): The candidate route.

        Returns:
            float: The transition value.
        """
        tau = self._pheromone(store, route_id)
        eta = self._heuristic(cost)
        return (tau ** self.alpha) * (eta ** self.beta)


    def _check_region_constraint(self, store, dc):
        """
        Notes:
            Check if a store is not within the opposite region.
            
        Args:
            store (dict): Store information.
            dc (dict): DC information.
        
        Returns:
            bool: True if region constraint is satisfied, False otherwise.
        """
        opposite = {'north': 'south', 'south': 'north', 'east': 'west', 'west': 'east'}

        if store['region'] == opposite[dc['region']]:
            return False
        
        return True


    def _check_volume_constraint(self, store, route_volume, max_capacity):
        """
        Notes:
            Check if adding a store violates the capacity constraint.

        Args:
            store (dict): Store information.
            route_volume (float): Current route volume.
            max_capacity (float): Route max capacity, 

        Returns:
            bool: True if capacity constraint is satisfied, False otherwise.
        """
        return (route_volume + store['volume']) <= max_capacity


    def _is_angle_valid(self, prev_store, next_store, new_store):
        """
        Notes:
            Check if the angle formed by (Prev -> New) and (Prev -> Next) is valid.

        Args:
            prev_store (dict): Previous store information.
            next_store (dict): Next store information.
            new_store (dict): New store information.

        Returns:
            bool: True if angle is valid, False otherwise.
        """
        prev_id = prev_store['store_id']
        new_id = new_store['store_id']
        next_id = next_store['store_id']
        max_deviation = 30

        dist_to_new = self.distance_matrix[prev_id][new_id] + self.distance_matrix[new_id][next_id] - self.distance_matrix[prev_id][next_id]
        if dist_to_new < 1: 
            max_deviation = 60

        vec_AC = np.array([
            new_store['longitude'] - prev_store['longitude'], 
            new_store['latitude'] - prev_store['latitude']
        ])

        vec_CB = np.array([
            new_store['longitude'] - next_store['longitude'], 
            new_store['latitude'] - next_store['latitude']
        ])

        norm_AC = np.linalg.norm(vec_AC)
        norm_CB = np.linalg.norm(vec_CB)

        if norm_AC == 0 or norm_CB == 0:
            return True

        cos_theta = np.dot(vec_AC, vec_CB) / (norm_AC * norm_CB)
        cos_threshold = np.cos(np.radians(180 - max_deviation))

        return cos_theta <= cos_threshold


    def _is_within_time_window(self, arrival_time, earliest_time, latest_time):
        """
        Check if a given arrival time within store time window.

        Args:
            arrival_time (datetime): Arrival time.
            earliest_time (datetime): Earliest time (start of time window).
            latest_time (datetime): Latest time (end of time window).

        Returns:
            bool: True if arrival_time is within the [earliest_time, latest_time] window, False otherwise.
        """
        return earliest_time <= arrival_time <= latest_time


    def _is_within_time_limit(self, duration):
        """
        Notes:
            Check route duration is within time limit per route.

        Args:
            duration (int): The total duration of the route in seconds.

        Returns:
            bool: True if duration is within the route time limit, False otherwise.
        """
        return duration <= self.time_limit_per_route


    def _check_time_constraint(self, route, pos, store):
        """
        Notes:
            Check if adding a store violates the time window constraint.
        
        Args:
            route (list): Current route (list of stores).
            pos (int): Position to insert the store.
            store (dict): Store information.
        
        Returns:
            bool: True if time window constraint is satisfied, False otherwise.
        """
        new_route = route[:pos] + [store] + route[pos:]
        duration = self.time_matrix['dc'][new_route[1]['store_id']] + self.time_matrix[new_route[-2]['store_id']]['dc'] 

        for i in range(2, len(new_route) - 1):
            prev_store, cur_store = new_route[i - 1], new_route[i]
            prev_id, curr_id = prev_store['store_id'], cur_store['store_id']
            travel_time = self.time_matrix[prev_id][curr_id]
            pre_dwell = prev_store['dwell_time']
            pre_pred_time = datetime.fromisoformat(prev_store['pred_time'])
            arrival_time = pre_pred_time + timedelta(seconds=travel_time + pre_dwell)
            arrival_time = arrival_time.replace(microsecond=0)
            earliest_time = datetime.fromisoformat(cur_store['earliest_time'])
            latest_time = datetime.fromisoformat(cur_store['latest_time'])
            duration += travel_time + pre_dwell

            if not self._is_within_time_window(arrival_time, earliest_time, latest_time):
                return False
        
        if not self._is_within_time_limit(duration):
            return False
            
        return True


    def _get_store_insertion_cost_and_pos(self, route_info, store):
        """
        Notes:
            Calculate the insertion cost of adding a store to a route.
        
        Args:
            route_info (dict): Route information.
            store (dict): Store information.
            
        Returns:
            min_cost (float): Insertion cost.
            best_pos (int): Best position to insert the store.
        """
        dc = route_info['dc']
        stores = route_info['stores']
        route_volume = dc['total_volume']
        route_max_capacity = dc['max_capacity']

        if not self._check_region_constraint(store, dc):
            return -1, -1

        if not self._check_volume_constraint(store, route_volume, route_max_capacity):
            return -1, -1

        best_pos = -1
        min_cost = float('inf')

        prev_store = self.dc
        route = [self.dc] + stores + [self.dc]
        for pos, (prev_store, next_store) in enumerate(zip(route, route[1:]), start=1):
            prev_id, next_id, store_id = prev_store['store_id'], next_store['store_id'], store['store_id']
            insert_cost = (self.distance_matrix[prev_id][store_id] + self.distance_matrix[store_id][next_id] - self.distance_matrix[prev_id][next_id])

            if not self._is_angle_valid(prev_store, next_store, store):
                continue

            if insert_cost < min_cost and insert_cost > 0 and self._check_time_constraint(route, pos, store):
                best_pos = pos - 1
                min_cost = insert_cost

        return min_cost, best_pos
        

    def _greedy_selection(self, current_store, feasible_routes):
        """
        Notes:
            Select the next route based on greedy approach.
        
        Args:
            current_store (dict): The current store.
            feasible_routes (list): List of feasible routes.

        Returns:
            best_route_id (str): The selected next route.
            best_pos (int): The position to insert the store.
        """
        best_route_id = None
        best_pos = -1
        best_value = -1
        for route_id, cost, pos in feasible_routes:
            value = self._transition_value(current_store, route_id, cost)
            if value > best_value:
                best_value = value
                best_route_id = route_id
                best_pos = pos

        return best_route_id, best_pos


    def _greedy_solution(self):
        """
        Notes:
            Generate an initial greedy solution for store allocation.

        Args:
            None.

        Returns:
            solution (dict): {route_id: [store, ...]}.
        """
        solutions = {}
        unassigned_stores = []
        copy_routes = self._copy_routes_info(self.main_routes)
        route_manager = RouteManager(copy_routes, self.distance_matrix, self.time_matrix)

        for store in self.remaining_stores:
            feasible_routes = []
            for route_id, route_info in route_manager.routes_info.items():
                cost, pos = self._get_store_insertion_cost_and_pos(route_info, store)
                if pos != -1:
                    feasible_routes.append((route_id, cost, pos))

            if not feasible_routes:
                unassigned_stores.append(store)
                continue

            selected_route, selected_pos = self._greedy_selection(store, feasible_routes)
            route_manager.insert_store(store, selected_route, selected_pos)
            solutions[store['store_id']] = selected_route

        return route_manager.routes_info, solutions, unassigned_stores


    def _encode_stores(self, stores):
        """
        Notes:
            Encode a list of stores.

        Args:
            stores (list): List of stores.

        Returns:
            str: Encoded stores.
        """
        store_set = sorted([store['store_id'] for store in stores])
        s = ','.join(map(str, store_set))
        return hashlib.md5(s.encode()).hexdigest()


    def _cost_function(self, routes, remaining_stores):
        """
        Notes:
            Calculate the cost of a solution.
        
        Args:
            routes (dict): { dc: {...}, stores: [...] }.
            remaining_stores (list): [store1, store2, ...].
        
        Returns:
            cost (float): Total cost of the solution.
        """
        greedy_cost = 0        
        for _, route_info in routes.items():
            prev_store = self.dc
            for store in route_info['stores']:
                greedy_cost += self.distance_matrix[prev_store['store_id']][store['store_id']]
                prev_store = store
            greedy_cost += self.distance_matrix[prev_store['store_id']][self.dc['store_id']]

        key = self._encode_stores(remaining_stores)

        if key not in self.cost_cache:
            support_cost, _ = SupportLinePlanningACO(remaining_stores, self.distance_matrix, self.time_matrix, num_ants=0, iterations=0).run()
            self.cost_cache[key] = support_cost
        else:
            support_cost = self.cost_cache[key]
        
        return greedy_cost + support_cost


    def _initial_pheromone(self, cost):
        """
        Notes:
            Initialize pheromone levels based on initial cost.

        Args:
            cost (float): Initial cost.

        Returns:
            None.
        """
        initial_pheromone = self.q / cost
        for store in self.remaining_stores:
            store_id = store['store_id']
            self.pheromone_matrix[store_id] = {}
        
            for route_id in self.main_routes.keys():
                self.pheromone_matrix[store_id][route_id] = initial_pheromone


    def _roulette_wheel_selection(self, current_store, feasible_routes):
        """
        Notes:
            Select the next route based on roulette wheel selection.
        
        Args:
            current_store (dict): The current store.
            feasible_routes (list): List of feasible routes.
        
        Returns:
            next_route (dict): The selected next route.
        """
        q = random.uniform(0, 1)
        if q < self.q0:
            return self._greedy_selection(current_store, feasible_routes)

        probabilities = []
        for route_id, cost, pos in feasible_routes:
            prob = self._transition_value(current_store, route_id, cost)
            probabilities.append(prob)
        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()
        next_route = random.choices(feasible_routes, weights=probabilities, k=1)[0]
        route_id, _, pos = next_route
        return route_id, pos

    def _solution_construct(self):
        """
        Construct a solution for store allocation using ACO.

        Returns:
            route_manager.routes_info: dict of routes with assigned stores
            solution: dict {store_id: route_id} for assigned stores
            unassigned_stores: list of stores that couldn't be assigned
        """
        solutions = {}
        unassigned_stores = []
        copy_routes = self._copy_routes_info(self.main_routes)
        route_manager = RouteManager(copy_routes, self.distance_matrix, self.time_matrix)

        stores_to_assign = self.remaining_stores[:]
        random.shuffle(stores_to_assign)
        stores_to_assign.sort(key=lambda x: x['volume'], reverse=True)

        for store in stores_to_assign:
            feasible_routes = []
            for route_id, route_info in route_manager.routes_info.items():
                cost, pos = self._get_store_insertion_cost_and_pos(route_info, store)
                if pos != -1:
                    feasible_routes.append((route_id, cost, pos))

            if not feasible_routes:
                unassigned_stores.append(store)
                continue

            selected_route, selected_pos = self._roulette_wheel_selection(store, feasible_routes)
            route_manager.insert_store(store, selected_route, selected_pos)
            solutions[store['store_id']] = selected_route

        return route_manager.routes_info, solutions, unassigned_stores


    def _evaporate_pheromone(self):
        """
        Notes:
            Evaporate Pheromone.
        
        Args:
            None.

        Return:
            None.
        """
        for store_id in self.pheromone_matrix:
            for route_id in self.pheromone_matrix[store_id]:
                self.pheromone_matrix[store_id][route_id] *= (1 - self.rho)


    def _calculate_tau_bounds(self):
        """
        Notes:
            Calculate bounds of tau.
        
        Args:
            None.

        Returns:
            tau_max (float): The upper bound of tau .
            tau_min (float): The low bound of tau.
        """
        tau_max = self.q / (self.rho * self.best_cost)
        tau_min = tau_max / self.tau_ratio
        return tau_max, tau_min


    def _deposit_pheromone(self, solution, cost):
        """
        Notes:
            Deposit pheromone based on the solution.

        Args:
            solution (dict): The solution found by an ant.
            cost (float): The cost of the solution.

        Returns:
            None.
        """
        delta = self.q / cost
        tau_max, tau_min = self._calculate_tau_bounds()
        for store_id, route_id in solution.items():
            self.pheromone_matrix[store_id][route_id] += delta
            
            if self.pheromone_matrix[store_id][route_id] > tau_max:
                self.pheromone_matrix[store_id][route_id] = tau_max
            elif self.pheromone_matrix[store_id][route_id] < tau_min:
                self.pheromone_matrix[store_id][route_id] = tau_min
        

    def run(self):
        """
        Notes:
            Execute the ACO algorithm for store allocation.
        """
        self._initial_pheromone(1)
        greedy_routes, greedy_solution, remaining_stores = self._greedy_solution()
        greedy_cost = self._cost_function(greedy_routes, remaining_stores)
        self._initial_pheromone(greedy_cost)
        self.best_cost = greedy_cost
        self.best_solution = greedy_routes
        self.best_ant_solution = greedy_solution
        self.best_remaining_solution = remaining_stores

        self.log.append({
            'iteration': 0,
            'iter_worst_cost': greedy_cost,
            'iter_best_cost': greedy_cost,
            'iter_avg_cost': greedy_cost,
            'std_cost': 0,
            'best_cost': self.best_cost,
        })

        early_stopper = EarlyStopper(patience=self.early_stop_patience)
        for i in range(self.iterations):
            ant_costs = []
            iter_best_cost = float("inf")
            iter_best_ant_solution = None
            for _ in range(self.num_ants):
                ant_routes, ant_solution, ant_remaining_stores = self._solution_construct()
                ant_cost = self._cost_function(ant_routes, ant_remaining_stores)
                ant_costs.append(ant_cost)

                if ant_cost < iter_best_cost:
                    iter_best_cost = ant_cost
                    iter_best_ant_solution = ant_solution

                if ant_cost < self.best_cost:
                    self.best_cost = ant_cost
                    self.best_solution = ant_routes
                    self.best_ant_solution = ant_solution
                    self.best_remaining_solution = ant_remaining_stores
            
            self._evaporate_pheromone()
            self._deposit_pheromone(iter_best_ant_solution, iter_best_cost)
            # self._deposit_pheromone(self.best_ant_solution, self.best_cost)

            print(f'Store Allocation: iteration{i + 1} -> best_cost: {self.best_cost:.4f}')

            self.log.append({
                'iteration': i + 1,
                'iter_worst_cost': float(np.max(ant_costs)),
                'iter_best_cost': float(min(ant_costs)),
                'iter_avg_cost': float(sum(ant_costs) / len(ant_costs)),
                'std_cost': float(np.std(ant_costs)),
                'best_cost': self.best_cost,
            })

            if early_stopper.check(self.best_cost):
                break

        return self.best_cost, self.best_solution, self.best_remaining_solution