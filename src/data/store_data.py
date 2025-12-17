import pandas as pd
from api.osrm import OSRM

class StoreData:
    """
    Notes:
        Load store-related data from source files.
    """
    def __init__(self, excel_file):
        self.file = excel_file
        self.stores = self._load_stores_info() 
        self.distance_matrix, self.time_matrix = self._get_cost_matrices()


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


    def _get_cost_matrices(self):
        """
        Notes:
            Get the distance and time matrix.
        
        Args:
            None.
        
        Returns:
            distance_matrix, time_matrix (tuple): Store distance matrix and time matrix. 
        """
        osrm = OSRM()
        return osrm._compute_cost_matrices(self.stores)