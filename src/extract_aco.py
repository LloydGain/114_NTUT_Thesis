import numpy as np
import random
from haversine import haversine_vector, Unit
from route import RouteManager


class StoreExtractionACO:
    """
    Notes:
        Store extraction for overloaded routes.
    """
    def __init__(self, main_routes, alpha=1, beta=2, rho=0.5):
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.main_routes = main_routes
        self.route_manager = RouteManager(main_routes)
        self.overloaded_routes = self._get_overloaded_routes()
        self.cost_matrix = self._calculate_cost_matrix()
        self.pheromones = {route_id: np.ones((len(route_info['stores']), len(route_info['stores']))) for route_id, route_info in self.overloaded_routes.items()}
        self.extracted_stores = []


    def _get_overloaded_routes(self):
        """
        Notes:
            Get overloaded routes from main_routes.

        Args:
            None.

        Returns:
            dict: Loaded routes {route_id: route_info}.
        """
        return {route_id: route_info for route_id, route_info in self.main_routes.items() if route_info['dc'].get('load_rate', 0) > 1.0}


    def _calculate_cost_matrix(self):
        """
        Notes:
            Calculate cost matrix for overloaded routes.

        Args:
            None.

        Returns:
            dict: Cost matrices {route_id: cost_matrix}.
        """
        cost_matrix = {}
        for route_id, route_info in self.overloaded_routes.items():
            stores = route_info['stores']
            if not stores:
                cost_matrix[route_id] = np.zeros((0, 0))
                continue

            coords = np.array([(s['latitude'], s['longitude']) for s in stores])
            dist_matrix = haversine_vector(coords, coords, comb=True, unit=Unit.KILOMETERS)
            np.fill_diagonal(dist_matrix, 0)
            cost_matrix[route_id] = dist_matrix

        return cost_matrix


    def _heuristic(self, route_id, current_idx, next_idx):
        """
        Notes:
            Get heuristic information value.

        Args:
            route_id (str): Route ID.
            current_idx (int): Index of current store.
            next_idx (int): Index of next store.

        Returns:
            float: Heuristic value (1 / distance)
        """
        route_matrix = self.cost_matrix.get(route_id)
        if route_matrix is None:
            return 0
        return 1 / (route_matrix[current_idx, next_idx] + 1e-6)


    def _calculate_probabilities(self, route_id, current_idx):
        """
        Notes:
           Calculate transition probabilities for candidate stores.

        Args:
            route_id (str): Route ID.
            current_idx (int): Index of current store.

        Returns:
            np.ndarray: Probability for candidate stores
        """
        probabilities = []
        stores = self.main_routes[route_id]['stores']
        for idx in range(len(stores)):
            tau = self.pheromones[route_id][current_idx, idx]
            eta = self._heuristic(route_id, current_idx, idx)
            prob = (tau ** self.alpha) * (eta ** self.beta)
            probabilities.append(prob)

        probabilities = np.array(probabilities)
        total = probabilities.sum()
        if total > 0:
            probabilities /= probabilities.sum()
        return probabilities


    def _extract_store(self, route_id, store):
        """
        Notes:
            Remove the store from route and append to extracted_stores.

        Args:
            route_id (str): Route ID.
            store (dict): Store information.

        Returns:
            None
        """
        self.extracted_stores.append(store)
        self.route_manager.remove_store(route_id, store)

    def run(self):
        """
        Notes:
            Perform store extraction using ACO.
        
        Args:
            None.

        Returns:
            list: Extracted stores.
        """
        self.extracted_stores = []

        for route_id in self.overloaded_routes.keys():
            stores = self.main_routes[route_id]['stores']
            if not stores:
                continue

            current_idx = random.randint(0, len(stores) - 1)
            extracted_store = stores[current_idx]
            self._extract_store(route_id, extracted_store)

            while True:
                stores = self.main_routes[route_id]['stores']
                if not stores:
                    break

                probs = self._calculate_probabilities(route_id, current_idx)
                if probs.sum() == 0:
                    break

                next_idx = np.random.choice(len(stores), p=probs)
                extracted_store = stores[next_idx]
                self._extract_store(route_id, extracted_store)
                current_idx = next_idx

                if self.route_manager.get_route_info(route_id, field='load_rate') <= 1.0:
                    break

        return self.extracted_stores


        # def update_pheromone(self, best_solution):
        #     for route_id, route_info in best_solution.items():
        #         stores = route_info['stores']
        #         for j, store in enumerate(stores):
        #             self.pheromones[route_id][j] *= (1 - self.rho)
        #             self.pheromones[route_id][j] += 1.0