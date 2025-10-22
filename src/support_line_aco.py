import copy
import random
import numpy as np
from datetime import datetime, timedelta
from route import RouteManager
from google_maps import GoogleRoutesAPI

class SupportLinePlanningACO:
    """
    Notes:
        Ant Colony Optimization for Support Line Planning.
    """
    def __init__(self, remaining_stores, alpha=1, beta=2, rho=0.5, q=1, ants=20, iteration=100, support_capacity=7.2):
        self.remaining_stores = remaining_stores
        self.dc = {'store_id': 'dc', 'longitude': 121.40712, 'latitude': 25.083282}
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.ants = ants
        self.iteration = iteration
        self.support_capacity = support_capacity
        self.pheromone_matrix = dict()
        self.distance_matrix, self.time_matrix = self._calculate_distance_and_time_matrix()
        self.best_cost = float('inf')
        self.best_solution = None
        self.time_limit_per_route = 5 * 3600


    def _calculate_distance_and_time_matrix(self):
        """
        Notes:
            Calculate cost matrix based on distances between stores.

        Args:
            None.

        Returns:
            tuple: Distance matrix & Time matrix .
        """
        if not self.remaining_stores:
            return {}

        routes_api = GoogleRoutesAPI()
        origins, destinations = self.remaining_stores, self.remaining_stores
        distance_matrix, time_matrix = routes_api.batch_compute_route_matrix(origins, destinations)
        
        return distance_matrix, time_matrix


    def _heuristic(self, current_store, next_store):
        """
        Notes:
            Get heuristic information between current_store and next_store.

        Args:
            current_store (dict): The current store.
            next_store (dict): The candidate next store.

        Returns:
            float: Heuristic value (1 / distance).
        """
        return 1 / (self.distance_matrix[current_store['store_id']][next_store['store_id']] + 1e-6)
        

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
        eta = self._heuristic(current_store, next_store)
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


    def _greedy_selection(self, current_store, unvisited_stores):
        """
        Notes:
            Select the next store based on greedy approach.

        Args:
            current_store (dict): The current store.
            unvisited_stores (list): List of unvisited stores.

        Returns:
            best_store (dict): The selected next store.
        """
        best_store = None
        best_value = -1
        for store in unvisited_stores:
            value = self._heuristic(current_store, store)
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
        
        vehicle_num = 101
        solution = dict()
        unvisited_stores = copy.deepcopy(self.remaining_stores)
        route_manager = RouteManager(solution, self.distance_matrix, self.time_matrix)

        while unvisited_stores:
            vehicle_id = f'{vehicle_num}'
            route = self._initial_route(vehicle_id)
            solution[vehicle_id] = route

            current_store = None
            min_distance = float('inf')
            for store in unvisited_stores:
                store_id = store['store_id']
                distance = self.distance_matrix['dc'][store_id]
                if distance < min_distance:
                    current_store = store

            route_manager.add_store(vehicle_id, current_store)
            unvisited_stores.remove(current_store)

            while unvisited_stores:
                next_store = self._greedy_selection(current_store, unvisited_stores)

                if self._capacity_and_time_constraints(route, next_store):
                    route_manager.add_store(vehicle_id, next_store)
                    current_store = next_store
                    unvisited_stores.remove(next_store)
                else:
                    vehicle_num += 1
                    break
    
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

            total_cost += self.distance_matrix['dc'][stores[0]['store_id']]

            for i in range(len(stores) - 1):
                total_cost += self.distance_matrix[stores[i]['store_id']][stores[i+1]['store_id']]

            total_cost += self.distance_matrix[stores[-1]['store_id']]['dc']

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
        for s in self.remaining_stores:
            self.pheromone_matrix[s['store_id']] = {
                store['store_id']: initial_pheromone for store in self.remaining_stores if store['store_id'] != s['store_id']
            }


    def _is_within_time_window(self, arrival_time, sched_time, before=30, after=60):
        """
        Notes:
            Check if arrival_time is within the time window of sched_time.

        Args:
            arrival_time (datetime): The arrival time.
            sched_time (datetime): The scheduled time.
            before (int): Minutes before scheduled time.
            after (int): Minutes after scheduled time.

        Returns:
            bool: True if within time window, False otherwise.
        """
        window_start = sched_time - timedelta(minutes=before)
        window_end = sched_time + timedelta(minutes=after)
        return window_start <= arrival_time <= window_end


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
        if route['dc']['total_volume'] + store['volume'] > self.support_capacity:
            return False

        if len(route['stores']) == 0:
            return True

        pre_id = route['stores'][-1]['store_id']
        next_id = store['store_id']
        duration = route['dc']['duration']
        dwell_time = store['dwell_time']
        time_pre_to_dc = self.time_matrix[pre_id]['dc']
        time_pre_to_next = self.time_matrix[pre_id][next_id]
        time_next_to_dc = self.time_matrix[next_id]['dc']

        new_duration = duration - time_pre_to_dc + time_pre_to_next + time_next_to_dc + dwell_time

        if new_duration > self.time_limit_per_route:
            return False
        
        sched_time = datetime.fromisoformat(store['sched_time'])
        pre_arrival_time = datetime.fromisoformat(route['stores'][-1]['pred_time'])
        arrival_time = pre_arrival_time + timedelta(seconds=time_pre_to_next)

        if not self._is_within_time_window(arrival_time, sched_time):
            return False

        return True


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

        vehicle_num = 101
        ant_solution = dict()
        unvisited_stores = copy.deepcopy(self.remaining_stores)
        route_manager = RouteManager(ant_solution, self.distance_matrix, self.time_matrix)

        while unvisited_stores:
            vehicle_id = f'{vehicle_num}'
            route = self._initial_route(vehicle_id)
            ant_solution[vehicle_id] = route

            current_store = random.choice(unvisited_stores)
            route_manager.add_store(vehicle_id, current_store)
            unvisited_stores.remove(current_store)

            while unvisited_stores:
                probabilities = []
                for next_store in unvisited_stores:
                    prob = self._transition_value(current_store, next_store)
                    probabilities.append(prob)
                probabilities = np.array(probabilities)
                probabilities[probabilities < 1e-12] = 1e-12
                probabilities /= probabilities.sum()

                next_store = random.choices(unvisited_stores, weights=probabilities, k=1)[0]

                if self._capacity_and_time_constraints(route, next_store):
                    route_manager.add_store(vehicle_id, next_store)
                    current_store = next_store
                    unvisited_stores.remove(next_store)
                else:
                    vehicle_num += 1
                    break
        
        return ant_solution


    def _update_pheromone(self, solution, cost):
        """
        Notes:
            Update pheromone based on the solution.

        Args:
            solution (dict): The solution found by an ant.
            cost (float): The cost of the solution.

        Returns:
            None.
        """
        for s in self.pheromone_matrix:
            for t in self.pheromone_matrix[s]:
                self.pheromone_matrix[s][t] *= (1 - self.rho)
        
        delta_pheromone = self.q / cost
        for route in solution:
            stores = solution[route]['stores']
            for i in range(len(stores) - 1):
                store_1 = stores[i]['store_id']
                store_2 = stores[i+1]['store_id']
                self.pheromone_matrix[store_1][store_2] += delta_pheromone
                self.pheromone_matrix[store_2][store_1] += delta_pheromone


    def run(self):
        """
        Notes:
            Runs the ACO algorithm for support line planning.
        """
        greedy_solution = self._greedy_solution()
        greedy_cost = self._cost_function(greedy_solution)
        self._initial_pheromone(greedy_cost)
        for _ in range(self.iteration):
            for _ in range(self.ants):
                ant_solution = self._solution_construction()
                ant_cost = self._cost_function(ant_solution)

                if ant_cost < self.best_cost:
                    self.best_cost = ant_cost
                    self.best_solution = ant_solution
                
                self._update_pheromone(ant_solution, ant_cost)

        return self.best_cost, self.best_solution