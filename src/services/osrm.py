import json
import requests

class OSRM:
    def __init__(self, url='http://localhost:5000'):
        """
        Notes:
            Initialize the OSRM with a server URL.
        """
        self.OSRM_URL = url
        self.dc = {'store_id': 'dc', 'longitude': 121.40712, 'latitude': 25.083282}
    

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
            raise Exception('Store list is empty. Cannot calculate matrices.')

        coords = ';'.join([f"{store['longitude']},{store['latitude']}" for store in stores])
        table_url = f'{self.OSRM_URL}/table/v1/driving/{coords}?annotations=distance,duration'
        response = requests.get(table_url)
        data = response.json()

        if 'distances' in data and 'durations' in data:
            distances = [[d / 1000 for d in row] for row in data['distances']]
            durations = [[t * 1.75 for t in row] for row in data['durations']]
            return distances, durations
        else:
            raise ValueError(f'OSRM table query failed.')
    

    def _compute_cost_matrices(self, stores, dist_file=None, time_file=None):
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

        if dist_file and time_file:
            self.save_matrices_to_file(dist_matrix, time_matrix, dist_file, time_file)

        return dist_matrix, time_matrix
    

    def save_matrices_to_file(self, distance_matrix, time_matrix, dist_file, time_file):
        """
        Notes:
            Save distance and time matrices to files.

        Args:
            distance_matrix (dict): Distance matrix.
            time_matrix (dict): Time matrix.
            dist_file (str): File path to save distance matrix.
            time_file (str): File path to save time matrix.
        
        Returns:
            None.
        """
        with open(dist_file, 'w') as df:
            json.dump(distance_matrix, df, indent=4)

        with open(time_file, 'w') as tf:
            json.dump(time_matrix, tf, indent=4)