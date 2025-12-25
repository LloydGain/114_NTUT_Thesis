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
            store_id = str(int(row['店鋪編號']))
            long = row['經度']
            lat = row['緯度']
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
        return osrm._compute_cost_matrices(self.stores, dist_file, time_file)
    

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
        with open(dist_file, 'r') as df:
            dist_matrix = json.load(df)
        
        with open(time_file, 'r') as tf:
            time_matrix = json.load(tf)
        
        return dist_matrix, time_matrix