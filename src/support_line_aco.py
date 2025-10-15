import numpy as np
import copy
import random
import haversine as hs

class SupportLinePlanningACO:
    """
    Notes:
        Ant Colony Optimization for Support Line Planning.
    """
    def __init__(self, remaining_stores, alpha=1, beta=2, rho=0.5, q=1, ants=20, iteration=100, support_capacity=7.2):
        self.remaining_stores = remaining_stores
        self.dc = {'longitude': 121.40712, 'latitude': 25.083282}
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.ants = ants
        self.iteration = iteration
        self.support_capacity = support_capacity
        self.pheromone_matrix = dict()
        self.cost_matrix = self._calculate_cost_matrix()
        self.best_cost = float('inf')
        self.best_solution = None


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
        
        store_ids = ['dc'] + [store['store_id'] for store in self.remaining_stores]
        coords = np.array([(self.dc['latitude'], self.dc['longitude'])] + [(store['latitude'], store['longitude']) for store in self.remaining_stores])
        
        dist_matrix = hs.haversine_vector(coords, coords, comb=True, unit=hs.Unit.KILOMETERS)
        
        cost_matrix = {
            store_id: {
                store_id_j: dist_matrix[i][j] for j, store_id_j in enumerate(store_ids)
            } for i, store_id in enumerate(store_ids)
        }
        
        return cost_matrix


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
        return 1 / (self.cost_matrix[current_store['store_id']][next_store['store_id']] + 1e-6)
        

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
            dict: {vehicle_id: [store1, store2, ...], ...}.
        """
        if not self.remaining_stores:
            return {}
        
        unvisited_stores = copy.deepcopy(self.remaining_stores)
        solution = dict()
        vehicle_num = 0
        while unvisited_stores:
            current_store = random.choice(unvisited_stores)
            solution[vehicle_num] = [current_store]
            total_volume = current_store['volume']
            unvisited_stores.remove(current_store)

            while unvisited_stores:
                next_store = self._greedy_selection(current_store, unvisited_stores)
                if total_volume + next_store['volume'] <= self.support_capacity:
                    solution[vehicle_num].append(next_store)
                    total_volume += next_store['volume']
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
            solution (dict): {vehicle_id: [store1, store2, ...], ...}.

        Returns:
            float: The total cost of the solution.
        """
        total_cost = 0

        for vehicle_id in solution:
            route = solution[vehicle_id]
            if not route:
                continue

            total_cost += self.cost_matrix['dc'][route[0]['store_id']]

            for i in range(len(route) - 1):
                total_cost += self.cost_matrix[route[i]['store_id']][route[i+1]['store_id']]

            total_cost += self.cost_matrix[route[-1]['store_id']]['dc']

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
        initial_pheromone = 1 / cost
        for s in self.remaining_stores:
            self.pheromone_matrix[s['store_id']] = {
                store['store_id']: initial_pheromone for store in self.remaining_stores if store['store_id'] != s['store_id']
            }


    def _pheromone_update(self, solution, cost):
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
            for i in range(len(solution[route]) - 1):
                store_1 = solution[route][i]['store_id']
                store_2 = solution[route][i+1]['store_id']
                self.pheromone_matrix[store_1][store_2] += delta_pheromone
                self.pheromone_matrix[store_2][store_1] += delta_pheromone


    # def _capacity_and_time_constraints(self, store, ):


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
                unvisited_stores = copy.deepcopy(self.remaining_stores)
                ant_solution = dict()
                vehicle_num = 0

                while unvisited_stores:
                    current_store = random.choice(unvisited_stores)
                    ant_solution[vehicle_num] = [current_store]
                    total_volume = current_store['volume']
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

                        if total_volume + next_store['volume'] <= self.support_capacity:
                            ant_solution[vehicle_num].append(next_store)
                            total_volume += next_store['volume']
                            current_store = next_store
                            unvisited_stores.remove(next_store)
                        else:
                            vehicle_num += 1
                            break
                
                ant_cost = self._cost_function(ant_solution)
                if ant_cost < self.best_cost:
                    self.best_cost = ant_cost
                    self.best_solution = ant_solution
                
                self._pheromone_update(ant_solution, ant_cost)

        return self.best_cost, self.best_solution