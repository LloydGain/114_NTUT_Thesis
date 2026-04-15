import random
import numpy as np
from datetime import datetime, timedelta
from config import config
from models.route_manager import RouteManager
from solvers.vnd import VND
from utils.early_stopper import EarlyStopper

class SupportLinePlanningACO:
    """
    Notes:
        Ant Colony Optimization for Support Line Planning.
    """
    def __init__(self, remaining_stores, distance_matrix, time_matrix, num_ants=None, iterations=1, alpha=1, beta=1, rho=0.5, q=1, q0=0.8, early_stop_patience=10, support_capacity=7.2, vehicle_cost=2000):
        self.dc = config.DC_CONFIG
        self.remaining_stores = remaining_stores
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix
        self.num_ants = num_ants if num_ants is not None else len(remaining_stores)
        self.store_count = len(remaining_stores)
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.tau0 = 0
        self.q0 = q0
        self.early_stop_patience = early_stop_patience
        self.pheromone_matrix = {}
        self.support_capacity = support_capacity
        self.vehicle_cost = vehicle_cost
        self.time_limit_per_route = 5 * 60 * 60
        self.log = []
        self.best_cost = float('inf')
        self.best_solution = None

        self.vnd = VND(self.distance_matrix, self.time_matrix)


    def _dist_heuristic(self, current_store, next_store):
        """
        Notes:
            Get distance heuristic information between current_store and next_store.

        Args:
            current_store (dict): The current store.
            next_store (dict): The candidate next store.

        Returns:
            float: Heuristic value (1 / distance).
        """
        if current_store['store_id'] == next_store['store_id']:
            return 0

        return 1 / (self.distance_matrix[current_store['store_id']][next_store['store_id']] + 1e-12)


    def _pheromone(self, current_store, next_store):
        """
        Notes:
            Get pheromone value between current_store and next_store.

        Args:
            current_store (dict): The current store.
            next_store (dict): The candidate next store.

        Returns:
            float: The pheromone value.
        """
        return self.pheromone_matrix[current_store['store_id']][next_store['store_id']]


    def _transition_value(self, current_store, next_store):
        """
        Notes:
            Calculate the transition value for moving from current_store to next_store.

        Args:
            current_store (dict): The current store.
            next_store (dict): The candidate next store.

        Returns:
            float: The transition value.
        """
        tau = self._pheromone(current_store, next_store)
        eta = self._dist_heuristic(current_store, next_store)
        return (tau ** self.alpha) * (eta ** self.beta)


    def _initial_route(self, vehicle_id):
        """
        Create an initial route structure for a given vehicle.

        Args:
            vehicle_id (int): Vehicle identifier.

        Returns:
            dict: Initial route structure.
        """
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


    def _feasible_stores(self, route, unvisited_stores):
        """
        Notes:
            Determine which stores are feasible to visit next from the current route.

        Args:
            route (dict): Current route.
            unvisited_stores (list): List of stores unvisited.

        Returns:
            feasible (list): List of feasible next stores.
        """
        if len(route['stores']) == 0:
            return unvisited_stores

        last_store = route['stores'][-1]
        last_region = last_store['region']
        last_group = last_store['dist_group']

        opposite = {
            'north': 'south',
            'south': 'north',
            'east': 'west',
            'west': 'east'
        }

        feasible = []
        for store in unvisited_stores:
            store_region = store['region']
            store_group = store['dist_group']

            if last_group == 'far' and store_region == opposite[last_region]:
                continue

            if last_group == 'far':
                allowed_groups = ['far', 'mid', 'near']
            elif last_group == 'mid':
                allowed_groups = ['mid', 'near']
            else:
                allowed_groups = ['near']

            if store_group not in allowed_groups:
                continue

            if not self._capacity_and_time_constraints(route, store):
                continue

            feasible.append(store)

        return feasible


    def _greedy_selection(self, current_store, feasible_stores):
        """
        Notes:
            Select the next store based on greedy approach.

        Args:
            current_store (dict): The current store.
            feasible_stores (list): List of feasible stores.

        Returns:
            best_store (dict): The selected next store.
        """
        best_store = None
        best_value = float("-inf")
        for store in feasible_stores:
            value = self._transition_value(current_store, store)
            if value > best_value:
                best_value = value
                best_store = store
        return best_store


    def _greedy_solution(self):
        """
        Notes:
            Generate a greedy solution for support line planning.

        Args:
            None.

        Returns:
            dict: { dc: {...}, stores: [...] }.
        """
        if not self.remaining_stores:
            return {}

        solution = {}
        vehicle_num = 101
        unvisited_stores = [store.copy() for store in self.remaining_stores]
        route_manager = RouteManager(solution, self.distance_matrix, self.time_matrix)

        while unvisited_stores:
            vehicle_id = f'{vehicle_num}'
            route = self._initial_route(vehicle_id)
            solution[vehicle_id] = route

            # current_store = max(unvisited_stores, key=lambda store: store['volume'])
            # current_store = max(unvisited_stores, key=lambda store: self.distance_matrix['dc'][store['store_id']])
            # current_store = random.choice(unvisited_stores)
            current_store = self._greedy_selection(self.dc, unvisited_stores)

            route_manager.add_store(vehicle_id, current_store)
            unvisited_stores.remove(current_store)

            while unvisited_stores:
                feasible_stores = self._feasible_stores(route, unvisited_stores)
                if not feasible_stores:
                    vehicle_num += 1
                    break

                next_store = self._greedy_selection(current_store, feasible_stores)
                route_manager.add_store(vehicle_id, next_store)
                current_store = next_store
                unvisited_stores.remove(next_store)

        return solution


    def _cost_function(self, solution):
        """
        Notes:
            Calculate the total cost of a solution.

        Args:
            solution (dict): { dc: {...}, stores: [...] }.

        Returns:
            float: The total cost of the solution.
        """
        total_cost = 0
        for vehicle_id in solution:
            stores = solution[vehicle_id]['stores']
            if not stores:
                continue

            total_cost += self.distance_matrix[self.dc['store_id']][stores[0]['store_id']]
            for i in range(len(stores) - 1):
                total_cost += self.distance_matrix[stores[i]['store_id']][stores[i+1]['store_id']]
            total_cost += self.distance_matrix[stores[-1]['store_id']][self.dc['store_id']]

        total_cost += len(solution) * self.vehicle_cost

        return total_cost


    def _initial_pheromone(self, cost):
        """
        Notes:
            Initialize pheromone based on the cost of a greedy solution.

        Args:
            cost (float): Cost of the greedy solution.

        Returns:
            None.
        """
        initial_pheromone = self.q / cost
        self.tau0 = initial_pheromone
        self.pheromone_matrix[self.dc['store_id']] = {
            store['store_id']: initial_pheromone for store in self.remaining_stores
        }
        for s in self.remaining_stores:
            self.pheromone_matrix[s['store_id']] = {
                store['store_id']: initial_pheromone for store in self.remaining_stores if store['store_id'] != s['store_id']
            }
            self.pheromone_matrix[s['store_id']][self.dc['store_id']] = initial_pheromone


    def _is_within_time_window(self, arrival_time, earliest_time, latest_time):
        """
        Notes:
            Check if a given arrival time within store time window.

        Args:
            arrival_time (datetime): Arrival time.
            earliest_time (datetime): Earliest time.
            latest_time (datetime): Latest time.

        Returns:
            bool: True if arrival time within time window, False otherwise.
        """
        return earliest_time <= arrival_time <= latest_time


    def _is_within_time_limit(self, duration):
        """
        Notes:
            Check route duration is within time limit per route.

        Args:
            duration (int): Route duration.

        Returns:
            bool: True if route duration within time limit, False otherwise.
        """
        return duration <= self.time_limit_per_route


    def _log_iteration(self, i, ant_costs, iter_best_cost, iter_worst_cost=None):
        """
        Notes:
            Log iteration results.

        Args:
            i (int): Iteration number.
            ant_costs (list): Ant costs.
            iter_best_cost (float): Iteration best cost.
            iter_worst_cost (float, optional): Iteration worst cost. Defaults to None.

        Returns:
            None.
        """
        if iter_worst_cost is None:
            iter_worst_cost = float(np.max(ant_costs))

        self.log.append({
            'iteration': i + 1,
            'iter_worst_cost': iter_worst_cost,
            'iter_best_cost': iter_best_cost,
            'iter_avg_cost': float(sum(ant_costs) / len(ant_costs)),
            'std_cost': float(np.std(ant_costs)),
            'best_cost': self.best_cost,
        })


    def _check_volume_constraint(self, route_volumn, store):
        """
        Notes:
            Check if adding a store violates the capacity constraint.

        Args:
            route_volumn (float): Current route volume.
            store (dict): Store information.

        Returns:
            bool: True if capacity constraint is satisfied, False otherwise.
        """
        return (route_volumn + store['volume']) <= self.support_capacity


    def _check_time_constraint(self, stores, store, duration):
        """
        Notes:
            Check if adding a store violates the time window constraint.

        Args:
            route (list): Current route (list of stores).
            store (dict): Store information.
            duration (float): Current total duration of the route in seconds.

        Returns:
            bool: True if time window constraint is satisfied, False otherwise.
        """
        prev_store = stores[-1]
        prev_id = prev_store['store_id']
        cur_id = store['store_id']
        prev_dwell_time = prev_store['dwell_time']
        pre_to_cur_time = self.time_matrix[prev_id][cur_id]
        pre_pred_time = datetime.fromisoformat(prev_store['pred_time'])
        arrival_time = pre_pred_time + timedelta(seconds=pre_to_cur_time + prev_dwell_time)
        arrival_time = arrival_time.replace(microsecond=0)
        earliest_time = datetime.fromisoformat(store['earliest_time'])
        latest_time = datetime.fromisoformat(store['latest_time'])

        pre_to_dc_time = self.time_matrix[prev_id][self.dc['store_id']]
        cur_to_dc_time = self.time_matrix[cur_id][self.dc['store_id']]
        cur_dwell_time = store['dwell_time']
        new_duration = duration + (pre_to_cur_time + cur_to_dc_time - pre_to_dc_time) + cur_dwell_time

        if not self._is_within_time_window(arrival_time, earliest_time, latest_time):
            return False

        if not self._is_within_time_limit(new_duration):
            return False

        return True


    def _capacity_and_time_constraints(self, route, store):
        """
        Notes:
            Check if adding a store to a route violates capacity and time constraints.

        Args:
            route (dict): The current route.
            store (dict): The candidate store to add.

        Returns:
            bool: True if constraints are satisfied, False otherwise.
        """
        dc = route['dc']
        stores = route['stores']
        duration = dc['duration']

        if not self._check_volume_constraint(dc['total_volume'], store):
            return False

        if len(stores) == 0:
            return True

        if self._check_time_constraint(stores, store, duration):
            return True

        return False


    def _roulette_wheel_selection(self, current_store, feasible_stores):
        """
        Notes:
            Select the next store based on roulette wheel selection.

        Args:
            current_store (dict): The current store.
            feasible_stores (list): List of feasible stores.

        Returns:
            next_store (dict): The selected next store.
        """
        if len(feasible_stores) == 1:
            return feasible_stores[0]

        q = random.uniform(0, 1)

        if q < self.q0:
            return self._greedy_selection(current_store, feasible_stores)

        probabilities = []
        for next_store in feasible_stores:
            prob = self._transition_value(current_store, next_store)
            probabilities.append(prob)
        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()
        next_store = random.choices(feasible_stores, weights=probabilities, k=1)[0]

        return next_store


    def _solution_construction(self):
        """
        Notes:
            Construct a solution for an ant.

        Args:
            None.

        Returns:
            solution (dict): { dc: {...}, stores: [...] }.
        """
        if not self.remaining_stores:
            return {}

        ant_solution = {}
        vehicle_num = 101
        unvisited_stores = [store.copy() for store in self.remaining_stores]
        route_manager = RouteManager(ant_solution, self.distance_matrix, self.time_matrix)

        while unvisited_stores:
            vehicle_id = f'{vehicle_num}'
            route = self._initial_route(vehicle_id)
            ant_solution[vehicle_id] = route

            # current_store = max(unvisited_stores, key=lambda store: store['volume'])
            # current_store = max(unvisited_stores, key=lambda store: self.distance_matrix['dc'][store['store_id']])
            # current_store = random.choice(unvisited_stores)
            current_store = self._roulette_wheel_selection(self.dc, unvisited_stores)

            route_manager.add_store(vehicle_id, current_store)
            unvisited_stores.remove(current_store)

            while unvisited_stores:
                feasible_stores = self._feasible_stores(route, unvisited_stores)
                if not feasible_stores:
                    vehicle_num += 1
                    break

                next_store = self._roulette_wheel_selection(current_store, feasible_stores)
                route_manager.add_store(vehicle_id, next_store)
                current_store = next_store
                unvisited_stores.remove(next_store)

        return ant_solution


    def _evaporate_pheromone(self):
        """
        Notes:
            Evaporate pheromone.

        Args:
            None.

        Returns:
            None.
        """
        for s in self.pheromone_matrix:
            for t in self.pheromone_matrix[s]:
                self.pheromone_matrix[s][t] *= (1 - self.rho)


    def _calculate_tau_bounds(self):
        """
        Notes:
            Calculate bounds of tau.

        Args:
            None.

        Returns:
            tuple: (tau_max, tau_min).
        """
        p_decisive = 0.05
        avg = 2.5
        root_n = p_decisive ** (1.0 / self.store_count)
        tau_max = self.q / (self.rho * self.best_cost)
        tau_min = (tau_max * (1 - root_n)) / ((avg - 1) * root_n)
        return tau_max, tau_min


    def _deposit_global_pheromone(self, solution, cost):
        """
        Notes:
            Update pheromone based on the solution.

        Args:
            solution (dict): The solution found by an ant.
            cost (float): The cost of the solution.

        Returns:
            None.
        """
        delta_pheromone = self.q / cost
        tau_max, tau_min = self._calculate_tau_bounds()
        for route in solution:
            stores = solution[route]['stores']
            path_nodes = [self.dc['store_id']] + [s['store_id'] for s in stores] + [self.dc['store_id']]
            for i in range(len(path_nodes) - 1):
                u, v = path_nodes[i], path_nodes[i+1]
                self.pheromone_matrix[u][v] += delta_pheromone
                self.pheromone_matrix[v][u] += delta_pheromone

        for s in self.pheromone_matrix:
            for e in self.pheromone_matrix[s]:
                val = self.pheromone_matrix[s][e]
                if val > tau_max:
                    self.pheromone_matrix[s][e] = tau_max
                elif val < tau_min:
                    self.pheromone_matrix[s][e] = tau_min


    def run(self):
        """
        Notes:
            Runs the ACO algorithm for support line planning.
        """
        if not self.remaining_stores:
            return 0, {}

        # print(f'Support Line stores: {len(self.remaining_stores)}')
        self._initial_pheromone(1)
        greedy_solution = self._greedy_solution()
        greedy_cost = self._cost_function(greedy_solution)
        self.best_cost = greedy_cost
        self.best_solution = greedy_solution

        if self.iterations == 0:
            return self.best_cost, self.best_solution

        self._log_iteration(0, [greedy_cost], greedy_cost, greedy_cost)

        self._initial_pheromone(greedy_cost)
        early_stopper = EarlyStopper(patience=self.early_stop_patience)
        
        for i in range(self.iterations):
            ant_costs = []
            iter_best_cost = float('inf')
            iter_best_solution = None
            for _ in range(self.num_ants):
                ant_solution = self._solution_construction()
                ant_cost = self._cost_function(ant_solution)
                ant_costs.append(ant_cost)

                if ant_cost < iter_best_cost:
                    iter_best_cost = ant_cost
                    iter_best_solution = ant_solution

            optimized_routes, optimized_cost = self.vnd.optimize(iter_best_solution)

            if optimized_cost < self.best_cost:
                self.best_cost = optimized_cost
                self.best_solution = optimized_routes

            print(f'Support Line: iteration{i + 1} -> best_cost: {self.best_cost:.4f}')

            self._evaporate_pheromone()
            self._deposit_global_pheromone(optimized_routes, optimized_cost)
            self._log_iteration(i, ant_costs, optimized_cost)

            if early_stopper.check(self.best_cost):
                break

        return self.best_cost, self.best_solution
