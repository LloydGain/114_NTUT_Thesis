import os
import json
import pandas as pd


class EvalRoutes:
    """
    Notes:
        Evaluate and compare manual and optimized routes.
    """
    def __init__(self, manual_routes_file, optimized_routes_file, program_routes_file=None):
        self.manual_routes = self._load_routes(manual_routes_file)
        self.manual_summary = self._summarize_routes(self.manual_routes)
        self.manual_simple_summary = self._summarize_routes(self.manual_routes, simple=True)
        self.optimized_routes = self._load_routes(optimized_routes_file)
        self.optimized_summary = self._summarize_routes(self.optimized_routes)
        self.optimized_simple_summary = self._summarize_routes(self.optimized_routes, simple=True)

        if program_routes_file is not None:
            self.program_routes = self._load_routes(program_routes_file)
            self.program_summary = self._summarize_routes(self.program_routes)
            self.program_simple_summary = self._summarize_routes(self.program_routes, simple=True)
            self.comparison = self._compare_routes()
            self.comparison_simple = self._compare_routes(simple=True)
        else:
            self.comparison = self._compare_manual_and_optim()
            self.comparison_simple = self._compare_manual_and_optim(simple=True)

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


    def _is_within_time_window(self, arrival_time, earliest_time, latest_time):
        """
        Notes:
            Check if a given arrival time within store time window.

        Args:
            arrival_time (datetime): Arrival time.
            earliest_time (datetime): Earliest time (start of time window).
            latest_time (datetime): Latest time (end of time window).

        Returns:
            bool: True if within time window, False otherwise.
        """
        return earliest_time <= arrival_time <= latest_time


    def _summarize_routes(self, routes, simple=False):
        """
        Notes:
            Generate a summary of the routes.

        Args:
            routes (dict): Routes information.
            simple (bool): Whether to use simple summary.

        Returns:
            dict: Summary statistics.
        """
        total_vehicles = len(routes)
        main_vehicles = sum(1 for route in routes.values() if not route['dc']['route_id'].isdigit())
        support_vehicles = sum(1 for route in routes.values() if route['dc']['route_id'].isdigit())

        total_stores = sum(len(route['stores']) for route in routes.values())
        main_stores = sum(len(route['stores']) for route in routes.values() if not route['dc']['route_id'].isdigit())
        support_stores = sum(len(route['stores']) for route in routes.values() if route['dc']['route_id'].isdigit())

        total_distance = sum(route['dc']['distance'] for route in routes.values())
        total_duration = sum(route['dc']['duration'] for route in routes.values()) / (60 * 60)
        total_main_distance = sum(route['dc']['distance'] for route in routes.values() if not route['dc']['route_id'].isdigit())
        total_main_duration = sum(route['dc']['duration'] for route in routes.values() if not route['dc']['route_id'].isdigit()) / (60 * 60)
        total_support_distance = sum(route['dc']['distance'] for route in routes.values() if route['dc']['route_id'].isdigit())
        total_support_duration = sum(route['dc']['duration'] for route in routes.values() if route['dc']['route_id'].isdigit()) / (60 * 60)

        average_distance = total_distance / total_vehicles
        average_duration = total_duration / total_vehicles
        average_main_distance = total_main_distance / main_vehicles
        average_main_duration = total_main_duration / main_vehicles
        average_support_distance = total_support_distance / support_vehicles if not support_vehicles == 0 else 0
        average_support_duration = total_support_duration / support_vehicles if not support_vehicles == 0 else 0

        average_load_rate = sum(route['dc']['load_rate'] for route in routes.values()) / total_vehicles
        average_main_load_rate = sum(route['dc']['load_rate'] for route in routes.values() if not route['dc']['route_id'].isdigit()) / main_vehicles
        average_support_load_rate = sum(route['dc']['load_rate'] for route in routes.values() if route['dc']['route_id'].isdigit()) / support_vehicles if not support_vehicles == 0 else 0

        on_time = 0
        main_on_time = 0
        support_on_time = 0
        for route in routes.values():
            stores = route['stores']
            if route['dc']['route_id'].isdigit():
                for store in stores:
                    if self._is_within_time_window(store['pred_time'], store['earliest_time'], store['latest_time']):
                        support_on_time += 1
                        on_time += 1
            else:
                for store in stores:
                    if self._is_within_time_window(store['pred_time'], store['earliest_time'], store['latest_time']):
                        main_on_time += 1
                        on_time += 1
        on_time_rate = on_time / total_stores
        main_on_time_rate = main_on_time / main_stores
        support_on_time_rate = support_on_time / support_stores if not support_stores == 0 else 0

        if not simple:
            summary = {
                'total_vehicles': f'{total_vehicles} ({main_vehicles} / {support_vehicles})',
                'total_stores': f'{total_stores} ({main_stores} / {support_stores})',
                'total_distance': f'{total_distance:.2f} ({total_main_distance:.2f} / {total_support_distance:.2f})',
                'total_duration': f'{total_duration:.2f} ({total_main_duration:.2f} / {total_support_duration:.2f})',
                'average_distance': f'{average_distance:.2f} ({average_main_distance:.2f} / {average_support_distance:.2f})',
                'average_duration': f'{average_duration:.2f} ({average_main_duration:.2f} / {average_support_duration:.2f})',
                'average_load_rate': f'{average_load_rate:.2f} ({average_main_load_rate:.2f} / {average_support_load_rate:.2f})',
                'on_time_rate': f'{on_time_rate:.2f} ({main_on_time_rate:.2f} / {support_on_time_rate:.2f})'
            }
        else:
            summary = {
                'total_vehicles': f'{total_vehicles}',
                'total_distance': f'{total_distance:.2f}',
                'total_duration': f'{total_duration:.2f}',
                'average_load_rate': f'{average_load_rate:.2f}',
                'on_time_rate': f'{on_time_rate:.2f}'
            }

        return summary


    def _compare_routes(self, simple=False):
        """
        Notes:
            Compare manual, program and optimized routes.

        Args:
            simple (bool): Whether to use simple summary.

        Returns:
            dict: Comparison results.
        """
        if not simple:
            comparison = {
                'vehicle_num' : [
                    self.manual_summary['total_vehicles'],
                    self.program_summary['total_vehicles'],
                    self.optimized_summary['total_vehicles']
                ],
                'total_store_num' : [
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
                ],
                'avg_load_rate' : [
                    self.manual_summary['average_load_rate'],
                    self.program_summary['average_load_rate'],
                    self.optimized_summary['average_load_rate']
                ],
                'on_time_rate' : [
                    self.manual_summary['on_time_rate'],
                    self.program_summary['on_time_rate'],
                    self.optimized_summary['on_time_rate']
                ]
            }
        else:
            comparison = {
                'vehicle_num' : [
                    self.manual_simple_summary['total_vehicles'],
                    self.program_simple_summary['total_vehicles'],
                    self.optimized_simple_summary['total_vehicles']
                ],
                'total_dist(km)' : [
                    self.manual_simple_summary['total_distance'],
                    self.program_simple_summary['total_distance'],
                    self.optimized_simple_summary['total_distance']
                ],
                'total_time(hr)' : [
                    self.manual_simple_summary['total_duration'],
                    self.program_simple_summary['total_duration'],
                    self.optimized_simple_summary['total_duration']
                ],
                'avg_load_rate' : [
                    self.manual_simple_summary['average_load_rate'],
                    self.program_simple_summary['average_load_rate'],
                    self.optimized_simple_summary['average_load_rate']
                ],
                'on_time_rate' : [
                    self.manual_simple_summary['on_time_rate'],
                    self.program_simple_summary['on_time_rate'],
                    self.optimized_simple_summary['on_time_rate']
                ]
            }

        df = pd.DataFrame(comparison, index=['manual', 'program', 'optimized']).T
        return df


    def _compare_manual_and_optim(self, simple=False):
        """
        Notes:
            Compare manual and optimized routes.

        Args:
            simple (bool): Whether to use simple summary.

        Returns:
            dict: Comparison results.
        """
        if not simple:
            comparison = {
                'vehicle_num' : [
                    self.manual_summary['total_vehicles'],
                    self.optimized_summary['total_vehicles']
                ],
                'total_store_num' : [
                    self.manual_summary['total_stores'],
                    self.optimized_summary['total_stores']
                ],
                'total_dist(km)' : [
                    self.manual_summary['total_distance'],
                    self.optimized_summary['total_distance']
                ],
                'total_time(hr)' : [
                    self.manual_summary['total_duration'],
                    self.optimized_summary['total_duration']
                ],
                'avg_dist(km)' : [
                    self.manual_summary['average_distance'],
                    self.optimized_summary['average_distance']
                ],
                'avg_time(hr)' : [
                    self.manual_summary['average_duration'],
                    self.optimized_summary['average_duration']
                ],
                'avg_load_rate' : [
                    self.manual_summary['average_load_rate'],
                    self.optimized_summary['average_load_rate']
                ],
                'on_time_rate' : [
                    self.manual_summary['on_time_rate'],
                    self.optimized_summary['on_time_rate']
                ]
            }
        else:
            comparison = {
                'vehicle_num' : [
                    self.manual_simple_summary['total_vehicles'],
                    self.optimized_simple_summary['total_vehicles']
                ],
                'total_dist(km)' : [
                    self.manual_simple_summary['total_distance'],
                    self.optimized_simple_summary['total_distance']
                ],
                'total_time(hr)' : [
                    self.manual_simple_summary['total_duration'],
                    self.optimized_simple_summary['total_duration']
                ],
                'avg_load_rate' : [
                    self.manual_simple_summary['average_load_rate'],
                    self.optimized_simple_summary['average_load_rate']
                ],
                'on_time_rate' : [
                    self.manual_simple_summary['on_time_rate'],
                    self.optimized_simple_summary['on_time_rate']
                ]
            }

        df = pd.DataFrame(comparison, index=['manual', 'optimized']).T
        return df


    def export_to_excel(self, file_path, simple=False):
        """
        Notes:
            Export the comparison results to an Excel file.

        Args:
            file_path (str): Path to the output Excel file.
            simple (bool): Whether to use simple summary.

        Returns:
            None.
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if not simple:
            self.comparison.to_excel(file_path, engine="openpyxl")
        else:
            self.comparison_simple.to_excel(file_path, engine="openpyxl")
