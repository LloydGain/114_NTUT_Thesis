import requests
import config

class OSRM:
    """
    Notes:
        Initialize the OSRM with a server URL.
    """
    def __init__(self):
        self.url = config.OSRM_HOST
        self.dc = config.DC_CONFIG
        self.timeout = 60


    def get_distance_and_time_matrix(self, stores):
        """
        Notes:
            Calculate the distance and duration matrices between a list of stores.

        Args:
            stores (list): List of store.

        Returns:
           distances (list): Distance matrix in km.
           durations (list): Duration marix in seconds.
        """
        if len(stores) == 0:
            raise ValueError('Store list is empty. Cannot calculate matrices.')

        coords = ';'.join([f"{store['longitude']},{store['latitude']}" for store in stores])
        table_url = f'{self.url}/table/v1/driving/{coords}?annotations=distance,duration'
        response = requests.get(table_url, timeout=self.timeout)
        data = response.json()

        if 'distances' in data and 'durations' in data:
            distances = [[d / 1000 for d in row] for row in data['distances']]
            durations = [[t * 1.75 for t in row] for row in data['durations']]
        else:
            raise ValueError('OSRM table query failed.')

        return distances, durations


    def compute_cost_matrices(self, stores):
        """
        Notes:
            Compute distance and time matrices.

        Args:
            Stores (list): List of store.
            dist_file (str): File path to save distance matrix.
            time_file (str): File path to save time matrix.

        Returns:
            tuple: (distance_matrix, time_matrix)
        """
        stores = [self.dc] + stores
        stores_id = [store['store_id'] for store in stores]

        dist, time = self.get_distance_and_time_matrix(stores)

        dist_matrix = {
            store_id: {
                store_id_j: dist[i][j] for j, store_id_j in enumerate(stores_id)
            } for i, store_id in enumerate(stores_id)
        }

        time_matrix = {
            store_id: {
                store_id_j: time[i][j] for j, store_id_j in enumerate(stores_id)
            } for i, store_id in enumerate(stores_id)
        }

        return dist_matrix, time_matrix
