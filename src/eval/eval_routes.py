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
        main_vehicles = sum(1 for route in routes.values() if route['dc']['route_id'][0] != '1')
        support_vehicles = sum(1 for route in routes.values() if route['dc']['route_id'][0] == '1')

        total_stores = sum(len(route['stores']) for route in routes.values())
        main_stores = sum(len(route['stores']) for route in routes.values() if route['dc']['route_id'][0] != '1')
        support_stores = sum(len(route['stores']) for route in routes.values() if route['dc']['route_id'][0] == '1')

        total_distance = sum(route['dc']['distance'] for route in routes.values())
        total_duration = sum(route['dc']['duration'] for route in routes.values()) / (60 * 60) 
        total_main_distance = sum(route['dc']['distance'] for route in routes.values() if route['dc']['route_id'][0] != '1')
        total_main_duration = sum(route['dc']['duration'] for route in routes.values() if route['dc']['route_id'][0] != '1') / (60 * 60)
        total_support_distance = sum(route['dc']['distance'] for route in routes.values() if route['dc']['route_id'][0] == '1')
        total_support_duration = sum(route['dc']['duration'] for route in routes.values() if route['dc']['route_id'][0] == '1') / (60 * 60)

        average_distance = total_distance / total_vehicles
        average_duration = total_duration / total_vehicles
        average_main_distance = total_main_distance / main_vehicles
        average_main_duration = total_main_duration / main_vehicles
        average_support_distance = total_support_distance / support_vehicles
        average_support_duration = total_support_duration / support_vehicles

        average_load_rate = sum(route['dc']['load_rate'] for route in routes.values()) / total_vehicles
        average_main_load_rate = sum(route['dc']['load_rate'] for route in routes.values() if route['dc']['route_id'][0] != '1') / main_vehicles
        average_support_load_rate = sum(route['dc']['load_rate'] for route in routes.values() if route['dc']['route_id'][0] == '1') / support_vehicles

        summary = {
            'total_vehicles': total_vehicles,
            'main_vehicles': main_vehicles,
            'support_vehicles': support_vehicles, 
            'total_stores': total_stores,
            'main_stores': main_stores,
            'support_stores': support_stores,
            'total_distance': total_distance,
            'total_duration': total_duration,
            'total_main_distance': total_main_distance,
            'total_main_duration': total_main_duration,
            'total_support_distance': total_support_distance,
            'total_support_duration': total_support_duration,
            'average_distance': average_distance,
            'average_duration': average_duration,
            'average_main_distance': average_main_distance,
            'average_main_duration': average_main_duration,
            'average_support_distance': average_support_distance,
            'average_support_duration': average_support_duration,
            'average_load_rate': average_load_rate,
            'average_main_load_rate': average_main_load_rate,
            'average_support_load_rate': average_support_load_rate
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
            'vehicle_num(main)' : [
                self.manual_summary['main_vehicles'], 
                self.program_summary['main_vehicles'],
                self.optimized_summary['main_vehicles']
            ],
            'vehicle_num(sup)' : [
                self.manual_summary['support_vehicles'], 
                self.program_summary['support_vehicles'],
                self.optimized_summary['support_vehicles']
            ],
            'total_store_num' : [
                self.manual_summary['total_stores'], 
                self.program_summary['total_stores'],
                self.optimized_summary['total_stores']
            ],
            'store_num(main)' : [
                self.manual_summary['main_stores'], 
                self.program_summary['main_stores'],
                self.optimized_summary['main_stores']
            ],
            'store_num(sup)' : [
                self.manual_summary['support_stores'], 
                self.program_summary['support_stores'],
                self.optimized_summary['support_stores']
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
            'total_main_dist(km)' : [
                self.manual_summary['total_main_distance'], 
                self.program_summary['total_main_distance'],
                self.optimized_summary['total_main_distance']
            ],
            'total_main_time(hr)' : [
                self.manual_summary['total_main_duration'], 
                self.program_summary['total_main_duration'],
                self.optimized_summary['total_main_duration']
            ],
            'total_sup_dist(km)' : [
                self.manual_summary['total_support_distance'], 
                self.program_summary['total_support_distance'],
                self.optimized_summary['total_support_distance']
            ],
            'total_sup_time(hr)' : [
                self.manual_summary['total_support_duration'], 
                self.program_summary['total_support_duration'],
                self.optimized_summary['total_support_duration']
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
            ],
            'avg_main_dist(km)' : [
                self.manual_summary['average_main_distance'],
                self.program_summary['average_main_distance'],
                self.optimized_summary['average_main_distance']
            ],
            'avg_main_time(hr)' : [
                self.manual_summary['average_main_duration'], 
                self.program_summary['average_main_duration'],
                self.optimized_summary['average_main_duration']
            ],
            'avg_sup_dist(km)' : [
                self.manual_summary['average_support_distance'],
                self.program_summary['average_support_distance'],
                self.optimized_summary['average_support_distance']
            ],
            'avg_sup_time(hr)' : [
                self.manual_summary['average_support_duration'], 
                self.program_summary['average_support_duration'],
                self.optimized_summary['average_support_duration']
            ],
            'avg_load_rate' : [
                self.manual_summary['average_load_rate'], 
                self.program_summary['average_load_rate'],
                self.optimized_summary['average_load_rate']
            ],
            'avg_main_load_rate' : [
                self.manual_summary['average_main_load_rate'], 
                self.program_summary['average_main_load_rate'],
                self.optimized_summary['average_main_load_rate']
            ],
            'avg_sup_load_rate' : [
                self.manual_summary['average_support_load_rate'], 
                self.program_summary['average_support_load_rate'],
                self.optimized_summary['average_support_load_rate']
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