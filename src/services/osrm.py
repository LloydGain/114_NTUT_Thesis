import sys
import requests
from config import config

class OSRM:
    """
    Notes:
        Initialize the OSRM with a server URL.
    """
    def __init__(self):
        self.url = config.OSRM_HOST
        self.dc = config.DC_CONFIG
        self.timeout = 60
        self.batch_size = 750
        self.check_osrm()


    def check_osrm(self):
        """
        Notes:
            Check if OSRM is running.
        
        Args:
            None
        
        Returns:
            None
        """
        try:
            requests.get(self.url + "/health", timeout=2)
        except requests.exceptions.RequestException:
            print("[ERROR] OSRM not running. Please start it first.")
            sys.exit(1)


    # def get_distance_and_time_matrix(self, stores):
    #     """
    #     Notes:
    #         Calculate the distance and duration matrices between a list of stores.

    #     Args:
    #         stores (list): List of store.

    #     Returns:
    #        distances (list): Distance matrix in km.
    #        durations (list): Duration marix in seconds.
    #     """
    #     if len(stores) == 0:
    #         raise ValueError('Store list is empty. Cannot calculate matrices.')

    #     coords = ';'.join([f"{store['longitude']},{store['latitude']}" for store in stores])
    #     table_url = f'{self.url}/table/v1/driving/{coords}?annotations=distance,duration'
    #     response = requests.get(table_url, timeout=self.timeout)
    #     data = response.json()

    #     if 'distances' in data and 'durations' in data:
    #         distances = [[d / 1000 for d in row] for row in data['distances']]
    #         durations = [[t * 1.75 for t in row] for row in data['durations']]
    #     else:
    #         raise ValueError('OSRM table query failed.')

    #     return distances, durations


    # def compute_cost_matrices(self, stores):
    #     """
    #     Notes:
    #         Compute distance and time matrices.

    #     Args:
    #         Stores (list): List of store.
    #         dist_file (str): File path to save distance matrix.
    #         time_file (str): File path to save time matrix.

    #     Returns:
    #         tuple: (distance_matrix, time_matrix)
    #     """
    #     stores = [self.dc] + stores
    #     stores_id = [store['store_id'] for store in stores]

    #     dist, time = self.get_distance_and_time_matrix(stores)

    #     dist_matrix = {
    #         store_id: {
    #             store_id_j: dist[i][j] for j, store_id_j in enumerate(stores_id)
    #         } for i, store_id in enumerate(stores_id)
    #     }

    #     time_matrix = {
    #         store_id: {
    #             store_id_j: time[i][j] for j, store_id_j in enumerate(stores_id)
    #         } for i, store_id in enumerate(stores_id)
    #     }

    #     return dist_matrix, time_matrix


    def get_block_matrix(self, sources, destinations):
        """
        Notes:
            Compute OSRM table for a block: sources x destinations.

        Args:
            sources (list): List of source store.
            destinations (list): List of destination store.

        Returns:
            tuple: (distances, durations)
        """
        coords = sources + destinations
        coord_str = ';'.join(f"{s['longitude']},{s['latitude']}" for s in coords)

        src_idx = ';'.join(map(str, range(len(sources))))
        dst_idx = ';'.join(map(str, range(len(sources), len(coords))))

        url = f"{self.url}/table/v1/driving/{coord_str}?sources={src_idx}&destinations={dst_idx}&annotations=distance,duration"

        response = requests.get(url, timeout=self.timeout)
        if response.status_code != 200:
            raise RuntimeError(f"OSRM error {response.status_code}: {response.text}")

        data = response.json()
        return data['distances'], data['durations']


    def compute_cost_matrices_batched(self, stores):
        """
        Notes:
            Compute full distance/time matrices using batch blocks.

        Args:
            stores (list): List of store.

        Returns:
            tuple: (distance_matrix, time_matrix)
        """
        all_stores = [self.dc] + stores
        ids = [s['store_id'] for s in all_stores]

        dist_matrix = {sid: {} for sid in ids}
        time_matrix = {sid: {} for sid in ids}

        N = len(all_stores)

        for i in range(0, N, self.batch_size):
            src_batch = all_stores[i:i+self.batch_size]

            for j in range(0, N, self.batch_size):
                dst_batch = all_stores[j:j+self.batch_size]

                distances, durations = self.get_block_matrix(src_batch, dst_batch)

                for si, s in enumerate(src_batch):
                    for di, d in enumerate(dst_batch):
                        dist_matrix[s['store_id']][d['store_id']] = distances[si][di] / 1000
                        time_matrix[s['store_id']][d['store_id']] = durations[si][di] * 1.75

        return dist_matrix, time_matrix