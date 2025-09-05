import numpy as np
import copy
from support_line_aco import SupportLinePlanningACO
from route import RouteManager

class StoreAllocationACO:
    """
    將 remaining_stores 插入到 main_routes 或支援線
    """
    def __init__(self, main_routes, remaining_stores, alpha=1, beta=2, rho=0.5, q=1, num_ants=10, iterations=50, support_capacity=7.2):
        self.main_routes = main_routes
        self.remaining_stores = remaining_stores
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.num_ants = num_ants
        self.iterations = iterations
        self.support_capacity = support_capacity
        self.order_pheromone = {store['store_id']: 1.0 for store in remaining_stores}
        self.route_pheromone = {store['store_id']: {route: 1.0 for route in main_routes} for store in remaining_stores}


    def _order_pheromone(self, store):
        return self.order_pheromone[store['store_id']]


    def _order_heuristic(self, store):
        return 1.0


    def _route_pheromone(self, store, route):
        return self.route_pheromone[store['store_id']][route]
    

    def _route_heuristic(self, store, route):
        return 1.0
    

    def _pheromone_evaporation(self):
        for store in self.order_pheromone:
            self.order_pheromone[store] *= (1 - self.rho)
        for store in self.route_pheromone:
            for route in self.route_pheromone[store]:
                self.route_pheromone[store][route] *= (1 - self.rho)


    def _pheromone_update(self, ant_solutions, ant_costs):
        for solution, cost in zip(ant_solutions, ant_costs):
            delta_pheromone = self.q / cost
            for route in solution:
                for store in solution[route]['stores']:
                    if store in self.remaining_stores:
                        self.order_pheromone[store['store_id']] += delta_pheromone
                        self.route_pheromone[store['store_id']][route] += delta_pheromone



    def _feasible_routes(self, store, routes):
        return [route for route in routes if (routes[route]['dc']['max_capacity'] - routes[route]['dc']['total_volume']) >= store['volume']]


    def _cost_evaluation(self):
        return 1.0
        # costs = []
        # for i in range(len(self.main_routes)):


    def _merged_routes(self, route1, route2):
        route1_copy = copy.deepcopy(route1)
        route2_copy = copy.deepcopy(route2)
        route2_copy.update(route1_copy)
        return route2_copy
    

    def run(self):
        best_solution = None
        best_cost = float('inf')
        for _ in range(self.iterations):
            ant_costs = []
            ant_solutions = []

            for _ in range(self.num_ants):
                support_stores = []
                main_routes_copy = copy.deepcopy(self.main_routes)
                remaining_stores_copy = copy.deepcopy(self.remaining_stores)
                route_manager = RouteManager(main_routes_copy)

                while remaining_stores_copy:
                    store_probs = []
                    for store in remaining_stores_copy:
                        tau = self._order_pheromone(store)
                        eta = self._order_heuristic(store)
                        prob = (tau ** self.alpha) * (eta ** self.beta)
                        store_probs.append(prob)
                    store_probs = np.array(store_probs)
                    store_probs = store_probs / store_probs.sum()
                    selected_store = np.random.choice(remaining_stores_copy, p=store_probs)

                    feasible_routes = self._feasible_routes(selected_store, main_routes_copy)
                    if feasible_routes:
                        route_probs = []
                        for route in feasible_routes:
                            tau = self._route_pheromone(selected_store, route)
                            eta = self._route_heuristic(selected_store, route)
                            prob = (tau ** self.alpha) * (eta ** self.beta)
                            route_probs.append(prob)
                        route_probs = np.array(route_probs)
                        route_probs = route_probs / route_probs.sum()
                        selected_route = np.random.choice(feasible_routes, p=route_probs)

                        route_manager.add_store(selected_route, store)
                    else:
                        support_stores.append(selected_store)

                    remaining_stores_copy.remove(selected_store)
                
                # support = SupportLinePlanningACO(support_stores)
                # support_line = support.run()

                support_line = {}

                routes = self._merged_routes(main_routes_copy, support_line)

                # evaluator = RouteEvaluator(routes)
                total_cost = self._cost_evaluation()

                ant_solutions.append(routes)
                ant_costs.append(total_cost)

            min_idx = np.argmin(ant_costs)
            if ant_costs[min_idx] < best_cost:
                best_cost = ant_costs[min_idx]
                best_solution = ant_solutions[min_idx]
        
            self._pheromone_evaporation()
            self._pheromone_update(ant_solutions, ant_costs)
        
        print(best_solution)