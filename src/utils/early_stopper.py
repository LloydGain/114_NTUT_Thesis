class EarlyStopper:
    """
    Notes:
        Early stopping when no improvement is observed for a certain number of iterations.
    """
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_cost = float('inf')
        self.early_stop = False

    def check(self, current_cost):
        """
        Notes:
            Check if early stopping condition is met.

        Args:
            current_cost (float or tuple): The current cost to evaluate.

        Returns:
            bool: True if early stopping condition is met, False otherwise.
        """
        is_inf = False
        if isinstance(self.best_cost, tuple):
            is_inf = (self.best_cost == (float('inf'), float('inf')))
        else:
            is_inf = (self.best_cost == float('inf'))
            
        if is_inf:
            self.best_cost = current_cost
            self.counter = 0
            return False
            
        is_improved = False
        if isinstance(current_cost, tuple):
            if current_cost[0] < self.best_cost[0] or (current_cost[0] == self.best_cost[0] and current_cost[1] < self.best_cost[1] - self.min_delta):
                is_improved = True
        else:
            if current_cost < (self.best_cost - self.min_delta):
                is_improved = True
                
        if is_improved:
            self.best_cost = current_cost
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False
