import copy
import random
import numpy as np
from datetime import datetime, timedelta
from route import RouteManager
from support_line_aco import SupportLinePlanningACO

class StoreAllocationACO:
    """
    Notes:
        Store Allocation using Ant Colony Optimization (ACO).
    """
    def __init__(self, main_routes, remaining_stores, distance_matrix, time_matrix, alpha=1, beta=1, rho=0.5, q=100, num_ants=20, iterations=50):
        self.main_routes = main_routes
        self.remaining_stores = remaining_stores
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix
        self.dc = {'store_id': 'dc', 'longitude': 121.40712, 'latitude': 25.083282}
        self.time_limit_per_route = 5 * 60 * 60
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.num_ants = num_ants
        self.iterations = iterations
        self.pheromone_matrix = dict()
        self.best_cost = float('inf')
        self.best_solution = None


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


    def _get_all_stores_in_route(self):
        """
        Notes:
            Get all stores in the main routes.

        Args:
            None.

        Returns:
            list: All stores in the main routes.
        """
        all_stores = []
        for route_info in self.main_routes.values():
            all_stores.extend(route_info['stores'])
        return all_stores


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

        if not self._check_volume_constraint(store, route_volume, route_max_capacity):
            return -1, -1

        min_cost = float('inf')
        best_pos = -1

        prev_store = self.dc
        route = [self.dc] + stores + [self.dc]
        for pos in range(1, len(route)):
            prev_store, next_store = route[pos - 1], route[pos]
            prev_id, next_id, store_id = prev_store['store_id'], next_store['store_id'], store['store_id']
            insert_cost = self.distance_matrix[prev_id][store_id] + self.distance_matrix[store_id][next_id] - self.distance_matrix[prev_id][next_id]
            
            if insert_cost < min_cost and self._check_time_constraint(route, pos, store):
                min_cost, best_pos = insert_cost, (pos - 1)

        return min_cost, best_pos
        

    def _greedy_solution(self):
        """
        Notes:
            Generate an initial greedy solution for store allocation.

        Args:
            None.

        Returns:
            solution (dict): {route_id: [store, ...]}.
        """
        unassigned_stores = []
        copy_routes = self._copy_routes_info(self.main_routes)
        route_manager = RouteManager(copy_routes, self.distance_matrix, self.time_matrix)

        for store in self.remaining_stores:
            assigned = False
            min_cost = float('inf')
            best_route_id = None
            best_pos = -1
            for route_id, route_info in route_manager.routes_info.items():
                cost, pos = self._get_store_insertion_cost_and_pos(route_info, store)
                if pos != -1 and cost < min_cost:
                    assigned = True
                    min_cost = cost
                    best_route_id = route_id
                    best_pos = pos
            if assigned:
                route_manager.insert_store(store, best_route_id, best_pos)
            else:
                unassigned_stores.append(store)

        return route_manager.routes_info, unassigned_stores


    def _cost_function(self, routes):
        """
        Notes:
            Calculate the cost of a solution.
        
        Args:
            solution (dict): { dc: {...}, stores: [...] }.
        
        Returns:
            cost (float): Total cost of the solution.
        """
        total_cost = 0

        for _, route_info in routes.items():
            prev_store = self.dc
            for store in route_info['stores']:
                total_cost += self.distance_matrix[prev_store['store_id']][store['store_id']]
                prev_store = store
            total_cost += self.distance_matrix[prev_store['store_id']][self.dc['store_id']]

        return total_cost


    # def _initial_pheromone(self, cost):
    #     """
    #     Notes:
    #         Initialize pheromone levels based on initial cost.

    #     Args:
    #         cost (float): Initial cost.

    #     Returns:
    #         None.
    #     """
    #     initial_pheromone = self.q / cost
    #     all_stores = self._get_all_stores_in_route() + self.remaining_stores
    #     for s in all_stores:
    #         self.pheromone_matrix[s['store_id']] = {
    #             store['store_id']: initial_pheromone for store in all_stores if store['store_id'] != s['store_id']
    #         }


    # def _construct_solution(self):
    #     """
    #     Notes:
    #         Construct a solution for store allocation.

    #     Args:
    #         None.
        
    #     Returns:
    #         solution (dict): { dc: {...}, stores: [...] }.
    #     """
    #     unvisited_stores = copy.deepcopy(self.remaining_stores)
    #     route_manager = RouteManager(copy.deepcopy(self.main_routes))


    def _combine_solutions(self, main_routes, support_solution):
        """
        Notes:
            Combine main routes and support line solutions.

        Args:
            main_routes (dict): { dc: {...}, stores: [...] }.
            support_solution (dict): { dc: {...}, stores: [...] }.

        Returns:
            combined_solution (dict): Combined solution { dc: {...}, stores: [...] }.
        """
        return {**support_solution, **main_routes}


    # def _update_pheromone(self, cost):


    def run(self):
        """
        Notes:
            Execute the ACO algorithm for store allocation.
        """
        greedy_routes, remaining_stores = self._greedy_solution()
        greedy_cost = self._cost_function(greedy_routes)
        
        support_line = SupportLinePlanningACO(remaining_stores, self.distance_matrix, self.time_matrix)
        support_cost, support_solution = support_line.run()

        initial_cost = greedy_cost + support_cost
        # self._initial_pheromone(initial_cost)

        return initial_cost, self._combine_solutions(greedy_routes, support_solution)

        # for _ in range(self.iterations):
        #     for _ in range(self.num_ants):
        #         ant_routes, ant_remaining_stores = self._construct_solution()
        #         support_line = SupportLinePlanningACO(ant_remaining_stores)
        #         support_cost, support_solution = support_line.run()
        #         ant_cost = self._cost_function(ant_routes)
        #         ant_cost = ant_cost + support_cost

        #         ant_solution = self._combine_solutions(ant_routes, support_solution)

        #         if ant_cost < self.best_cost:
        #             self.best_cost = ant_cost
        #             self.best_solution = ant_solution
        #         self._update_pheromone(ant_cost)

        # return self.best_cost, self.best_solution