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
            current_cost (float): The current cost to evaluate.

        Returns:
            bool: True if early stopping condition is met, False otherwise.
        """
        if current_cost < (self.best_cost - self.min_delta):
            self.best_cost = current_cost
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False
