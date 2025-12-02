import json
from datetime import datetime

class Log:
    def __init__(self, log_dir, parameters):
        """
        Notes:
            Log for recording experiment params and results.
        """
        self.params_file = f'{log_dir}/params.json'
        self.parameters = parameters
    
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

        print(json.dumps(record, indent=4))