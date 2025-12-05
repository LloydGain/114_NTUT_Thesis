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