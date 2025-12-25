import os
import json
import pandas as pd
from datetime import datetime

class Log:
    def __init__(self, log_dir, parameters, times):
        """
        Notes:
            Log for recording experiment params and results.
        """
        self.log_dir = log_dir
        self.parameters = parameters
        self.times = times
        self.params_file = f'{log_dir}/params.json'
        self.times_file = f'{log_dir}/times.json'
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


    def log_times(self):
        """
        Notes:
            Logs the time consume in the execution.

        Args:
            None.
        
        Returns:
            None.
        """
        with open(self.times_file, 'w', encoding='utf-8') as f:
            json.dump(self.times, f, indent=4)
    

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

        route1_total = 0
        route2_total = 0
        route3_total = 0
        route2_same_code_total = 0
        route3_same_code_total = 0
        route2_support_line_total = 0
        route3_support_line_total = 0

        for route_id in routes1:
            route1_data = routes1.get(route_id)
            route2_data = routes2.get(route_id)
            route3_data = routes3.get(route_id)
            stores1 = routes1[route_id]['stores'] if route1_data else []
            stores2 = routes2[route_id]['stores'] if route2_data else []
            stores3 = routes3[route_id]['stores'] if route3_data else []
            original_load_rate = round(route1_data['dc']['load_rate'], 3)
            maunal_same_code_count = sum(1 for store in stores2 if store['route_code'].startswith(store['route_id']))
            optimized_same_code_count = sum(1 for store in stores3 if store['route_code'].startswith(store['route_id']))

            rows.append({
                "route_id": route_id,
                "original_store_count": f'{len(stores1)}',
                "original_load_rate": f'{original_load_rate}',
                "manual_store_count": f'{len(stores2)}({maunal_same_code_count})',
                "optimized_store_count": f'{len(stores3)}({optimized_same_code_count})',
            })

            route1_total += len(stores1)
            route2_total += len(stores2)
            route3_total += len(stores3)
            route2_same_code_total += maunal_same_code_count
            route3_same_code_total += optimized_same_code_count

        for route_id in routes2:
            if route_id.isdigit():
                route2_data = routes2.get(route_id)
                stores2 = routes2[route_id]['stores'] if route2_data else []
                route2_support_line_total += len(stores2)
        route2_total += route2_support_line_total

        for route_id in routes3:
            if route_id.isdigit():
                route3_data = routes3.get(route_id)
                stores3 = routes3[route_id]['stores'] if route3_data else []
                route3_support_line_total += len(stores3)
        route3_total += route3_support_line_total

        rows.append({
            "route_id": "Support Line",
            "original_store_count": '0',
            "original_load_rate": 'X',
            "manual_store_count": f'{route2_support_line_total}',
            "optimized_store_count": f'{route3_support_line_total}',
        })

        rows.append({
            "route_id": "Total",
            "original_store_count": f'{route1_total}',
            "original_load_rate": 'X',
            "manual_store_count": f'{route2_total}({route2_same_code_total})',
            "optimized_store_count": f'{route3_total}({route3_same_code_total})',
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