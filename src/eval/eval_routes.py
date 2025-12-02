import os
import json
import pandas as pd


class EvalRoutes:
    """
    Notes:
        Evaluate and compare manual and optimized routes.
    """
    def __init__(self, manual_routes_file, program_routes_file, optimized_routes_file):
        self.manual_routes = self._load_routes(manual_routes_file)
        self.program_routes = self._load_routes(program_routes_file)
        self.optimized_routes = self._load_routes(optimized_routes_file)
        self.manual_summary = self._summarize_routes(self.manual_routes)
        self.program_summary = self._summarize_routes(self.program_routes)
        self.optimized_summary = self._summarize_routes(self.optimized_routes)
        self.comparison = self._compare_routes()
    
    def _load_routes(self, file_path):
        """
        Notes:
            Load routes data from a JSON file.

        Args:
            file_path: Path to the JSON file.

        Returns:
            dict: Routes information.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            routes_info = json.load(f)
        
        return routes_info
    

    def _summarize_routes(self, routes):
        """
        Notes:
            Generate a summary of the routes.

        Args:
            routes (dict): Routes information.

        Returns:
            dict: Summary statistics.
        """
        total_vehicles = len(routes)
        total_stores = sum(len(route['stores']) for route in routes.values())
        
        total_distance = sum(route['dc']['distance'] for route in routes.values())
        total_duration = sum(route['dc']['duration'] for route in routes.values()) / (60 * 60) 
        
        average_distance = total_distance / total_vehicles
        average_duration = total_duration / total_vehicles

        summary = {
            'total_vehicles': total_vehicles,
            'total_stores': total_stores,
            'total_distance': total_distance,
            'total_duration': total_duration,
            'average_distance': average_distance,
            'average_duration': average_duration
        }

        return summary


    def _compare_routes(self):
        """
        Notes:
            Compare manual and optimized routes.
        
        Args:
            None.
        
        Returns:
            dict: Comparison results.
        """
        comparison = {
            'vehicle_num' : [
                self.manual_summary['total_vehicles'], 
                self.program_summary['total_vehicles'],
                self.optimized_summary['total_vehicles']
            ],
            'store_num' : [
                self.manual_summary['total_stores'], 
                self.program_summary['total_stores'],
                self.optimized_summary['total_stores']
            ],
            'total_dist(km)' : [
                self.manual_summary['total_distance'], 
                self.program_summary['total_distance'],
                self.optimized_summary['total_distance']
            ],
            'total_time(hr)' : [
                self.manual_summary['total_duration'], 
                self.program_summary['total_duration'],
                self.optimized_summary['total_duration']
            ],
            'avg_dist(km)' : [
                self.manual_summary['average_distance'],
                self.program_summary['average_distance'],
                self.optimized_summary['average_distance']
            ],
            'avg_time(hr)' : [
                self.manual_summary['average_duration'], 
                self.program_summary['average_duration'],
                self.optimized_summary['average_duration']
            ]
        }

        df = pd.DataFrame(comparison, index=['manual', 'program', 'optimized']).T
        return df


    def export_to_excel(self, file_path):
        """
        Notes:
            Export the comparison results to an Excel file.

        Args:
            file_path (str): Path to the output Excel file.
        
        Returns:
            None.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.comparison.to_excel(file_path, engine="openpyxl")