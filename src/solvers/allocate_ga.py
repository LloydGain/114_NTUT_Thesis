import random
import hashlib
import numpy as np
from config import config
from datetime import datetime, timedelta
from models.route_manager import RouteManager
from utils.early_stopper import EarlyStopper
from solvers.support_line_aco import SupportLinePlanningACO

class StoreAllocationGA:
    """
    Store Allocation using Genetic Algorithm (GA) with Strict Time Window & Capacity Constraints.
    """
    def __init__(self, main_routes, remaining_stores, distance_matrix, time_matrix, pop_size=50, elite_rate=0.1, generations=50, cross_rate=0.8, mutation_rate=0.2, early_stop_patience=100):
        self.main_routes = main_routes
        self.remaining_stores = remaining_stores
        self.distance_matrix = distance_matrix
        self.time_matrix = time_matrix
        self.dc = config.DC_CONFIG
        self.pop_size = pop_size
        self.elite_size = int(pop_size * elite_rate)
        self.generations = generations
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        self.early_stop_patience = early_stop_patience
        self.route_choices = list(self.main_routes.keys()) + ['SUPPORT']
        self.time_limit_per_route = 5 * 60 * 60
        self.best_cost = float('inf')
        self.best_solution = None
        self.best_remaining_solution = None
        self.cost_cache = {}
        self.log = []

        self.remaining_stores = self._sort_stores_by_insertion_cost(remaining_stores)


    def _sort_stores_by_insertion_cost(self, stores):
        """
        Notes:
            Sorts the remaining stores based on their minimum insertion cost into any main route.
        
        Args:
            stores (list): List of store dictionaries.
            
        Returns:
            list: Sorted list of store dictionaries.
        """
        temp_routes = self._copy_routes_info(self.main_routes)
        route_manager = RouteManager(temp_routes, self.distance_matrix, self.time_matrix)

        store_with_costs = []
        for store in stores:
            min_cost = float('inf')
            for r_id in self.main_routes.keys():
                cost, pos = self._get_store_insertion_cost_and_pos(route_manager.routes_info[r_id], store)
                if pos != -1 and cost < min_cost:
                    min_cost = cost
            store_with_costs.append((store, min_cost))

        return [s[0] for s in sorted(store_with_costs, key=lambda x: x[1])]


    def _generate_greedy_individual(self):
        """
        Notes:
            Generates a greedy individual for initial population seed.
        
        Args:
            None.
            
        Returns:
            list: A chromosome representing the greedy solution.
        """
        temp_routes = self._copy_routes_info(self.main_routes)
        route_manager = RouteManager(temp_routes, self.distance_matrix, self.time_matrix)
        greedy_chromo = []

        for store in self.remaining_stores:
            best_r = 'SUPPORT'
            min_cost = float('inf')
            best_pos = -1

            for r_id in self.main_routes.keys():
                cost, pos = self._get_store_insertion_cost_and_pos(route_manager.routes_info[r_id], store)
                if pos != -1 and cost < min_cost:
                    min_cost = cost
                    best_r = r_id
                    best_pos = pos

            greedy_chromo.append(best_r)
            if best_r != 'SUPPORT':
                route_manager.insert_store(store, best_r, best_pos)

        return greedy_chromo


    def _copy_routes_info(self, routes):
        """
        Notes:
            Creates a shallow copy of routes information.
        
        Args:
            routes (dict): Original routes data.
            
        Returns:
            dict: Copied routes data.
        """
        return {
            route_id: {
                "dc": route_data["dc"].copy(),
                "stores": [store.copy() for store in route_data["stores"]]
            } for route_id, route_data in routes.items()
        }


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
            insert_cost = self.distance_matrix[prev_id][store_id] + self.distance_matrix[store_id][next_id] - self.distance_matrix[prev_id][next_id]

            # if not self._is_angle_valid(prev_store, next_store, store):
            #     continue

            if 0 < insert_cost < min_cost and self._check_time_constraint(route, pos, store):
                best_pos = pos - 1
                min_cost = insert_cost

        return min_cost, best_pos


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


    def _calculate_total_distance(self, routes):
        """
        Notes:
            Calculates total distance of all main routes.
        
        Args:
            routes (dict): Routes Information.

        Returns:
            total_cost (float): total distance of all main routes.
        """
        total_cost = 0
        for _, route_info in routes.items():
            prev = self.dc
            for store in route_info['stores']:
                total_cost += self.distance_matrix[prev['store_id']][store['store_id']]
                prev = store
            total_cost += self.distance_matrix[prev['store_id']][self.dc['store_id']]
        return total_cost


    def _evaluate_individual(self, individual):
        """
        Notes:
            Decodes chromosome and calculates total fitness cost.

        Args:
            individual (list): individual

        Returns:
            costs (float): Main Routes & Support Line Cost
            routes (dict): Routes Information.
            support_pool (list): list of store
            individual (list): 
        """
        temp_routes = self._copy_routes_info(self.main_routes)
        route_manager = RouteManager(temp_routes, self.distance_matrix, self.time_matrix)
        support_pool = []

        for i, target in enumerate(individual):
            store = self.remaining_stores[i]
            if target == 'SUPPORT':
                support_pool.append(store)
                continue

            _, pos = self._get_store_insertion_cost_and_pos(route_manager.routes_info[target], store)
            if pos != -1:
                route_manager.insert_store(store, target, pos)
            else:
                support_pool.append(store)
                individual[i] = 'SUPPORT'

        main_cost = self._calculate_total_distance(route_manager.routes_info)
        cache_key = self._encode_stores(support_pool)
        if cache_key not in self.cost_cache:
            support_cost = SupportLinePlanningACO(support_pool, self.distance_matrix, self.time_matrix, num_ants=0, iterations=0).run()[0] if support_pool else 0
            self.cost_cache[cache_key] = support_cost
        else:
            support_cost = self.cost_cache[cache_key]

        return main_cost + support_cost, route_manager.routes_info, support_pool, individual


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
        child1 = parent1.copy()
        child2 = parent2.copy()
        for i, (g1, g2) in enumerate(zip(parent1, parent2)):
            if random.random() < 0.5:
                child1[i], child2[i] = g2, g1

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
        if random.random() < self.cross_rate:
            return self._uniform_crossover(parent1, parent2)

        return parent1.copy(), parent2.copy()


    def _mutate(self, individual):
        """
        Notes:
            Mutates an individual.

        Args:
            individual (dict): The individual to mutate.

        Returns:
            None.
        """
        for i, _ in enumerate(individual):
            if random.random() < self.mutation_rate:
                individual[i] = random.choice(self.route_choices)

        return individual


    def run(self):
        """
        Notes:
            Runs the genetic algorithm for store allocation.
        """
        greedy_chromo = self._generate_greedy_individual()
        g_cost, g_routes, g_support, g_repaired = self._evaluate_individual(greedy_chromo)
        self.best_cost = g_cost
        self.best_solution = g_routes
        self.best_remaining_solution = g_support

        population = [g_repaired]
        num_stores = len(self.remaining_stores)
        early_stopper = EarlyStopper(patience=self.early_stop_patience)

        while len(population) < self.pop_size:
            population.append([random.choice(self.route_choices) for _ in range(num_stores)])

        for gen_idx in range(self.generations):
            fitnesses = []
            evaluated_pop = []

            for chromo in population:
                cost, routes, support, repaired = self._evaluate_individual(chromo)
                evaluated_pop.append({
                    'individual': repaired, 
                    'cost': cost, 
                    'routes': routes, 
                    'support': support
                })
                fitnesses.append(cost)

            evaluated_pop.sort(key=lambda x: x['cost'])
            current_best = evaluated_pop[0]

            if current_best['cost'] < self.best_cost:
                self.best_cost = current_best['cost']
                self.best_solution = current_best['routes']
                self.best_remaining_solution = current_best['support']

            self.log.append({
                'generation': gen_idx + 1,
                'iter_worst_cost': float(np.max(fitnesses)),
                'iter_best_cost': float(current_best['cost']),
                'iter_avg_cost': float(np.mean(fitnesses)),
                'std_cost': float(np.std(fitnesses)),
                'best_cost': self.best_cost,
            })

            print(f'Store Allocation: iteration{gen_idx+1} -> best cost = {self.best_cost}')

            if early_stopper.check(self.best_cost):
                break

            next_gen = [ind['individual'] for ind in evaluated_pop[:self.elite_size]]
            weights = [1.0 / (ind['cost'] + 1e-6) for ind in evaluated_pop]

            while len(next_gen) < self.pop_size:
                p1, p2 = random.choices(evaluated_pop, weights=weights, k=2)
                c1, c2 = self._crossover(p1['individual'], p2['individual'])
                next_gen.extend([self._mutate(c1), self._mutate(c2)])

            population = next_gen[:self.pop_size]

        return self.best_cost, self.best_solution, self.best_remaining_solution