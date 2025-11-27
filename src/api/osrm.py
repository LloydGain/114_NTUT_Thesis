import requests

class OSRM:
    def __init__(self, url='http://localhost:5000'):
        """
        Notes:
            Initialize the OSRM with a server URL.
        """
        self.OSRM_URL = url
    

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
            durations = [[t * 2 for t in row] for row in data['durations']]
            return distances, durations
        else:
            raise ValueError(f'OSRM table query failed.')
    

    def _compute_cost_matrices(self, routes_info):
        """
        Notes:
            Compute distance and time matrices.

        Args:
            routes_info (dict): Route information.
        
        Returns:
            tuple: (distance_matrix, time_matrix)
        """
        routes = routes_info.values()
        stores_id = [self.dc['store_id']] + [store['store_id'] for route in routes for store in route['stores']]
        coordinates = [self.dc] + [store for route in routes for store in route['stores']]

        dist, time = self.get_distance_and_time_matrix(coordinates)

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