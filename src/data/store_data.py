import json
import pandas as pd
from services.osrm import OSRM

class StoreData:
    """
    Notes:
        Load store-related data from source files.
    """
    def __init__(self, excel_file):
        self.file = excel_file
        self.stores = self._load_stores_info()


    def _load_stores_info(self):
        """
        Notes:
            Load stores information.

        Args:
            None.

        Returns:
            stores (list): list of store's information.
        """
        stores_df = pd.read_excel(self.file, sheet_name=0)

        stores = []
        for _, row in stores_df.iterrows():
            value = row['店鋪編號']
            store_id = value.strip() if isinstance(value, str) else str(int(value))
            long = row['經度']
            lat = row['緯度']
            dc = row['DC別']
            if dc != '林口DC':
                continue
            if long > 125 or long < 115 or lat < 20 or lat > 30:
                continue
            store = {'store_id': store_id, "longitude": long, "latitude": lat}
            stores.append(store)

        return stores


    def get_cost_matrices(self, dist_file=None, time_file=None):
        """
        Notes:
            Get the distance and time matrix.

        Args:
            None.

        Returns:
            distance_matrix, time_matrix (tuple): Store distance matrix and time matrix.
        """
        osrm = OSRM()
        dist_matrix, time_matrix = osrm.compute_cost_matrices_batched(self.stores)

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
        with open(dist_file, 'w', encoding='utf-8') as df:
            json.dump(distance_matrix, df, indent=4)

        with open(time_file, 'w', encoding='utf-8') as tf:
            json.dump(time_matrix, tf, indent=4)


    def load_matrices_from_file(self, dist_file, time_file):

        """
        Notes:
            Load distance and time matrices from files.

        Args:
            dist_file (str): File path of distance matrix.
            time_file (str): File path of time matrix.

        Returns:
            distance_matrix (dict): Distance matrix.
            time_matrix (dict): Time matrix.
        """
        with open(dist_file, 'r', encoding='utf-8') as df:
            dist_matrix = json.load(df)

        with open(time_file, 'r', encoding='utf-8') as tf:
            time_matrix = json.load(tf)

        return dist_matrix, time_matrix
