import numpy as np
import config

class BaseACO:
    """
    Notes:
        Base class for Ant Colony Optimization algorithms.
    """
    def __init__(self, num_ants=1, iterations=1, alpha=1, beta=1, rho=0.1, q=1, early_stop_patience=1):
        self.dc = config.DC_CONFIG
        self.num_ants = num_ants
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.early_stop_patience = early_stop_patience
        self.pheromone_matrix = {}
        self.best_cost = float('inf')
        self.best_solution = None
        self.log = []
        self.time_limit_per_route = 5 * 60 * 60


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
