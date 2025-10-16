import numpy as np
import copy
from route import RouteManager

# np.random.seed(1234)

class StoreExtractionGA:
    """
    Notes: 
        Genetic Algorithm for Store Extraction.
    """
    def __init__(self, main_routes, population_size=10, generations=10, cross_rate=0.6, mutation_rate=1):
        self.main_routes = main_routes
        self.population_size = population_size
        self.generations = generations
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.overloaded_routes = self._get_overloaded_routes()
        self.best_individual = None
    

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
        route_manager = RouteManager(copy.deepcopy(self.main_routes))
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

    
    def _fitness(self, individual):
        # Fitness function (Not done yet)
        return 1

    
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
        total_fitness = sum(fitnesses)
        probabilities = np.array(fitnesses) / total_fitness
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
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        
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
        return copy.deepcopy(parent1), copy.deepcopy(parent2)
    

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
        updated_routes = copy.deepcopy(self.main_routes)
        route_manager = RouteManager(updated_routes)
        for route_id, stores in individual.items():
            for store in stores:
                route_manager.remove_store(route_id, store)
        return route_manager.routes_info


    def _best_individual_to_set(self, individual):
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
        for _ in range(self.generations):
            fitnesses = [self._fitness(population) for _ in population]
            new_population = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = self._roulette_wheel_selection(population, fitnesses)
                child1, child2 = self._crossover(parent1, parent2)
                self._mutate(child1)
                self._mutate(child2)
                new_population.extend([child1, child2])
            population = new_population
        # Select the best individual from the final population
        self.best_individual = population[0]
        return self._best_main_routes(self.best_individual), self._best_individual_to_set(self.best_individual)