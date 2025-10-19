import copy
import random
import numpy as np
import haversine as hs
from route import RouteManager
from support_line_aco import SupportLinePlanningACO

class StoreAllocationACO:
    """
    Notes:
        Store Allocation using Ant Colony Optimization (ACO).
    """
    def __init__(self, main_routes, remaining_stores, alpha=1, beta=2, rho=0.5, q=100, num_ants=20, iterations=50):
        self.main_routes = main_routes
        self.remaining_stores = remaining_stores
        self.dc = {'longitude': 121.40712, 'latitude': 25.083282}
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.num_ants = num_ants
        self.iterations = iterations
        self.pheromone_matrix = dict()
        self.cost_matrix = self._calculate_cost_matrix()
        self.best_cost = float('inf')
        self.best_solution = None


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


    def _calculate_cost_matrix(self):
        """
        Notes:
            Calculate cost matrix based on distances between stores.

        Args:
            None.

        Returns:
            cost_matrix (dict): store_id : {store_id: distance, ...}.
        """
        if not self.remaining_stores:
            return {}
        
        all_stores = self._get_all_stores_in_route() + self.remaining_stores
        store_ids = ['dc'] + [store['store_id'] for store in all_stores]

        coords = np.array([(self.dc['latitude'], self.dc['longitude'])] + [(store['latitude'], store['longitude']) for store in all_stores])
        dist_matrix = hs.haversine_vector(coords, coords, comb=True, unit=hs.Unit.KILOMETERS)
        
        cost_matrix = {
            store_id: {
                store_id_j: dist_matrix[i][j] for j, store_id_j in enumerate(store_ids)
            } for i, store_id in enumerate(store_ids)
        }
        
        return cost_matrix


    def _get_store_insertion_cost(self, route_info, store):
        """
        Notes:
            Calculate the insertion cost of adding a store to a route.
        
        Args:
            route_info (dict): Route information.
            store (dict): Store information.
            
        Returns:
            cost (float): Insertion cost.
        """
        min_insertion_cost = float('inf')
        dc = {'store_id': 'dc', 'longitude': self.dc['longitude'], 'latitude': self.dc['latitude']}
        prev_store = dc
        for next_store in route_info['stores'] + [dc]:
            cost = (self.cost_matrix[prev_store['store_id']][store['store_id']] +
                    self.cost_matrix[store['store_id']][next_store['store_id']] -
                    self.cost_matrix[prev_store['store_id']][next_store['store_id']])
            min_insertion_cost = min(min_insertion_cost, cost)
            prev_store = next_store
        return min_insertion_cost
        

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
        copy_remaining_stores = copy.deepcopy(self.remaining_stores)
        route_manager = RouteManager(copy.deepcopy(self.main_routes))

        for store in self.remaining_stores:
            assigned = False
            insertion_cost = float('inf')
            best_route_id = None
            for route_id, route_info in route_manager.routes_info.items():
                if route_info['dc']['total_volume'] + store['volume'] < route_info['dc']['max_capacity']:
                    insertion_cost = self._get_store_insertion_cost(route_info, store)
                    if insertion_cost < float('inf'):
                        assigned = True
                        best_route_id = route_id
            if assigned:
                route_manager.add_store(best_route_id, store)
                copy_remaining_stores.remove(store)
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
        dc = {'store_id': 'dc', 'longitude': self.dc['longitude'], 'latitude': self.dc['latitude']}

        for _, route_info in routes.items():
            prev_store = dc
            for store in route_info['stores']:
                total_cost += self.cost_matrix[prev_store['store_id']][store['store_id']]
                prev_store = store
            total_cost += self.cost_matrix[prev_store['store_id']][dc['store_id']]

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
        return {**main_routes, **support_solution}


    # def _update_pheromone(self, cost):


    def run(self):
        """
        Notes:
            Execute the ACO algorithm for store allocation.
        """
        greedy_routes, remaining_stores = self._greedy_solution()
        greedy_cost = self._cost_function(greedy_routes)
        
        support_line = SupportLinePlanningACO(remaining_stores)
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