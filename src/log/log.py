import os
import json
import pandas as pd
from datetime import datetime

class Log:
    def __init__(self, log_dir, parameters):
        """
        Notes:
            Log for recording experiment params and results.
        """
        self.log_dir = log_dir
        self.params_file = f'{log_dir}/params.json'
        self.parameters = parameters
        os.makedirs(log_dir, exist_ok=True)

    
    def log_parameters(self):
        """
        Notes:
            Logs the parameters used in the execution to a JSON file with a timestamp.

        Args:
            None.
        
        Returns:
            None.
        """
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")

        record = {"time": current_time, "parameters": self.parameters}

        with open(self.params_file, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=4)
    

    def log_route(self, log_file, routes1, routes2, routes3):
        """
        Notes:
            Log route stores count.

        Args:
            log_file (str): log file destination.
            routes1 (dict): routes info.
            routes2 (dict): routes info.
            routes3 (dict): routes info.
        
        Returns:
            None.
        """
        rows = []

        for route_id in routes1:
            route1_data = routes1.get(route_id)
            route2_data = routes2.get(route_id)
            route3_data = routes3.get(route_id)
            stores1 = routes1[route_id]['stores'] if route1_data else []
            stores2 = routes2[route_id]['stores'] if route2_data else []
            stores3 = routes3[route_id]['stores'] if route3_data else []
            original_load_rate = stores1[route_id]['dc']['load_rate']
            maunal_same_code_count = sum(1 for store in stores2 if store['route_code'].startswith(store['route_id']))
            optimized_same_code_count = sum(1 for store in stores3 if store['route_code'].startswith(store['route_id']))

            rows.append({
                "route_id": route_id,
                "original_store_count": f'{len(stores1)}',
                "original_load_rate": f'{original_load_rate}',
                "manual_store_count": f'{len(stores2)}({maunal_same_code_count})',
                "optimized_store_count": f'{len(stores3)}({optimized_same_code_count})',
            })

        df = pd.DataFrame(rows)

        log_dest = f'{self.log_dir}/{log_file}'
        df.to_excel(log_dest, index=False)


    def log_execution(self, log_file, log_data):
        """
        Notes:
            Logs execution data to a specified JSON file.

        Args:
            log_file (str): The path to the log file.
            log_data (list): The data to be logged.
        
        Returns:
            None.
        """
        log_dest = f'{self.log_dir}/{log_file}'
        df = pd.DataFrame(log_data)
        df.to_excel(log_dest, index=False)