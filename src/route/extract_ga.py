import hashlib
import numpy as np
from route.route import RouteManager
from route.allocate_aco import StoreAllocationACO
from route.support_line_aco import SupportLinePlanningACO

# np.random.seed(1234)

class StoreExtractionGA:
    """
    Notes: 
        Genetic Algorithm for Store Extraction.
    """
    def __init__(self, main_routes, distance_matrix, time_matrix, population_size=10, generations=50, cross_rate=0.8, mutation_rate=0.2):
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix
        self.main_routes = self._routes(main_routes)
        self.population_size = population_size
        self.generations = generations
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.overloaded_routes = self._get_overloaded_routes()
        self.best_cost = float('inf')
        self.best_individual = None
        self.fitness_cache = {}
    

    def _routes(self, routes):
        """
        Notes:
            Update route info.

        Args:
            routes (dict): Original routes info.

        Returns:
            routes (dict): Updated routes info.
        """

        route_manager = RouteManager(routes, self.distance_matrix, self.time_matrix)
        for route_id in routes:
            route_manager._update_route_info(routes[route_id])
        
        return routes


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


    def _copy_individual(self, individual):
        """
        Notes:
            Create a shallow copy of individual.
        
        Args:
            routes (dict): Original individual.

        Returns:
            dict: Individual.
        """
        return {
            route_id: [store.copy() for store in stores] for route_id, stores in individual.items()
        }


    def _get_overloaded_routes(self):
        """
        Notes:
            Get overloaded routes from main_routes.

        Args:
            None.

        Returns:
            dict: Loaded routes {route_id: route_info}.
        """
        return {route_id: route_info for route_id, route_info in self.main_routes.items() if route_info['dc']['load_rate'] > 1.0}


    def _extract_stores(self, route_id):
        """
        Notes:
            Randomly extracts stores from a route until load constraint is satisfied.

        Args:
            route_id (str): Route ID.

        Returns:
            list: Selected stores for the route.
        """
        copy_routes = self._copy_routes_info(self.main_routes)
        route_manager = RouteManager(copy_routes, self.distance_matrix, self.time_matrix)
        selected_idxs = []
        selected_stores = []
        stores = self.overloaded_routes[route_id]['stores']

        while route_manager.get_route_info(route_id, field='load_rate') > 1.0:
            idx = np.random.randint(len(stores))
            if idx not in selected_idxs:
                selected_idxs.append(idx)
                selected_store = stores[idx]
                selected_stores.append(selected_store)
                route_manager.remove_store(route_id, selected_store)

        return selected_stores


    def _generate_individual(self):
        """
        Notes:
            Generate an individual for the population.

        Args:
            None.

        Returns:
            dict: {route_id: list of selected store dicts}.
        """
        individual = {}
        for route_id in self.overloaded_routes:
            individual[route_id] = self._extract_stores(route_id)
        return individual


    def _init_population(self):
        """
        Notes:
            Initialize GA population with valid individuals.

        Args:
            None.

        Returns:
            list: List of individuals (dicts)
        """
        population = []
        for _ in range(self.population_size):
            individual = self._generate_individual()
            population.append(individual)
        return population


    def _get_individual_routes(self, indiviudal):
        """
        Notes:
            Remove the extracted stores from main routes.
        
        Args:
            individual (dict): { route_id: [store1, store2, ...] }
        
        Returns:
            dict: route info { dc: {...}, stores: [...] }.
        """
        copy_routes = self._copy_routes_info(self.main_routes)
        route_manager = RouteManager(copy_routes, self.distance_matrix, self.time_matrix)
        for route_id in indiviudal:
            route_manager.remove_stores(route_id, indiviudal[route_id])
        
        return route_manager.routes_info

    
    def _encode_individual(self, individual):
        """
        Notes:
            Generates a unique string key for an individual.

        Args:
            individual (dict): { route_id: [store1, store2, ...] }

        Returns:
            str: Unique key representing the individual
        """
        store_set = sorted([s['store_id'] for stores in individual.values() for s in stores])
        s = ','.join(map(str, store_set))
        return hashlib.md5(s.encode()).hexdigest()


    def _fitness(self, individual):
        """
        Notes:
            Calculates the fitness value for an individual solution
        
        Args:
            individual (dict): { route_id: [store1, store2, ...] }
        
        Returns:
            float: The fitness value
        """
        key = self._encode_individual(individual)
        if key in self.fitness_cache:
            return self.fitness_cache[key]
        
        routes, stores = self._get_individual_routes(individual), self._individual_to_list(individual)
        allocate_cost, _, remaining_stores = StoreAllocationACO(routes, stores, self.distance_matrix, self.time_matrix, num_ants=0, iterations=0).run()
        support_cost, _ = SupportLinePlanningACO(remaining_stores, self.distance_matrix, self.time_matrix, num_ants=0, iterations=0).run()

        fitness = allocate_cost + support_cost
        self.fitness_cache[key] = fitness
        return fitness

    
    def _roulette_wheel_selection(self, population, fitnesses):
        """
        Notes:
            Selects two parents from the population using roulette wheel selection.

        Args:
            population (list): List of individuals.
            fitnesses (list): Fitness values for each individual.

        Returns:
            tuple: Two selected parents (individuals).
        """
        inverse_fitnesses = 1 / np.array(fitnesses)
        total_inverse_fitness = sum(inverse_fitnesses)
        probabilities = np.array(inverse_fitnesses) / total_inverse_fitness
        parent1_idx = np.random.choice(len(fitnesses), p=probabilities)
        parent2_idx = np.random.choice(len(fitnesses), p=probabilities)
        return population[parent1_idx], population[parent2_idx]


    def _uniform_crossover(self, parent1, parent2):
        """
        Notes:
            Performs uniform crossover on two parents. (50%)

        Args:
            parent1 (dict): First parent individual.
            parent2 (dict): Second parent individual.

        Returns:
            tuple: Two children individuals (child1, child2).
        """
        child1 = self._copy_individual(parent1)
        child2 = self._copy_individual(parent2)
        
        route_ids = list(self.overloaded_routes.keys())
        
        for r in route_ids:
            if np.random.rand() < 0.5:
                child1[r], child2[r] = child2[r], child1[r]
        
        return child1, child2


    def _crossover(self, parent1, parent2):
        """
        Notes:
            Performs crossover on two parents.

        Args:
            parent1 (dict): First parent individual.
            parent2 (dict): Second parent individual.

        Returns:
            tuple: Two children individuals.
        """
        cross = np.random.rand()
        if cross < self.cross_rate:
            return self._uniform_crossover(parent1, parent2)
        return self._copy_individual(parent1), self._copy_individual(parent2)
    

    def _mutate(self, individual):
        """
        Notes:
            Mutates an individual.

        Args:
            individual (dict): The individual to mutate.

        Returns:
            None.
        """
        mutate = np.random.rand()
        if mutate < self.mutation_rate:
            route_id = np.random.choice(list(self.overloaded_routes.keys()))
            selected_stores = self._extract_stores(route_id)
            individual[route_id] = selected_stores


    def _best_main_routes(self, individual):
        """
        Notes:
            Get the main routes after extraction based on the individual.

        Args:
            individual (dict): The individual representing extracted stores.

        Returns:
            dict: Updated main routes after extraction.
        """
        copy_routes = self._copy_routes_info(self.main_routes)
        route_manager = RouteManager(copy_routes, self.distance_matrix, self.time_matrix)
        for route_id, stores in individual.items():
            for store in stores:
                route_manager.remove_store(route_id, store)
        return route_manager.routes_info


    def _individual_to_list(self, individual):
        """
        Notes:
            Converts an individual to a set of store IDs.
        
        Args:
            individual (dict): The individual to convert.
        
        Returns:
            extracted_stores (list): List of extracted stores.
        """
        extracted_stores = []
        for route_id in individual:
            for store in individual[route_id]:
                extracted_stores.append(store)
        return extracted_stores


    def run(self):
        """
        Notes:
            Runs the genetic algorithm for store extraction.
        """
        population = self._init_population()
        for i in range(self.generations):
            fitnesses = [self._fitness(individual) for individual in population]
            # print(len(self.fitness_cache))
            # for idx, fitness in enumerate(fitnesses):
            #     print(f'individual{idx+1} -> cost = {fitness}')

            current_best_index = np.argmin(fitnesses)
            current_best_cost = fitnesses[current_best_index]
            current_best_individual = population[current_best_index]

            if current_best_cost < self.best_cost:
                self.best_cost = current_best_cost
                self.best_individual = current_best_individual

            print(f'Store Extraction: iteration{i+1} -> best cost = {self.best_cost}')

            new_population = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = self._roulette_wheel_selection(population, fitnesses)
                child1, child2 = self._crossover(parent1, parent2)
                self._mutate(child1)
                self._mutate(child2)
                new_population.extend([child1, child2])
            population = new_population

        return self._best_main_routes(self.best_individual), self._individual_to_list(self.best_individual)