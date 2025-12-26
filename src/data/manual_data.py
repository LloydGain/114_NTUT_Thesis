import os
import copy
import shutil
import pandas as pd
from models.route_manager import RouteManager
from data.base_data import BaseDataManager

class MDataManager(BaseDataManager):
    """
    Notes:
        Load route-related data from source files (manual).
    """
    def __init__(self, excel_files, distance_matrix, time_matrix):
        super().__init__(excel_files, distance_matrix, time_matrix)
        self.routes_info = self._load_manual_routes()
        self._update_routes_info()


    def _load_manual_routes(self):
        """
        Notes:
            Load routes data from excel & transfer to dict.

        Args:
            file (str): Path to the Excel File.

        Returns:
            dict: routes information, with route code as key.
        """
        routes_info = {}

        update_main_route = True
        main_route_code = None
        for _, row in self.routes_df.iterrows():
            route_code = str(row['車次'])

            if not route_code:
                continue

            store_name = row['店名']
            store_id = self._get_store_id_by_name(store_name)
            lng, lat = self._get_coordinates(store_id)
            dwell_time = self._get_dwell_time(store_id)
            sched_time = pd.to_datetime(row['表定時間']).isoformat()
            earliest_time = (pd.to_datetime(row['表定時間']) - pd.Timedelta(minutes=1)).isoformat()
            latest_time = (pd.to_datetime(row['表定時間']) + pd.Timedelta(hours=1)).isoformat()
            pred_time = pd.to_datetime(row['預定時間']).isoformat()
            volume = row['貨量']
            load_rate = row['裝載率']

            if main_route_code is None:
                main_route_code = route_code[:3] if route_code.isdigit() else route_code[:2]

            if update_main_route:
                main_route_code = route_code[:3] if route_code.isdigit() else route_code[:2]
                update_main_route = False

            if self._DC_CENTER in route_code:
                update_main_route = True

            max_capacity = self._get_max_capacity_by_route_code(main_route_code)

            if main_route_code not in routes_info:
                routes_info[main_route_code] = {"dc" : None, "stores": []}

            if self._DC_CENTER not in route_code and len(route_code) < 4:
                routes_info[main_route_code]["dc"] = {
                    "route_id": main_route_code,
                    "route_code": route_code,
                    "store_id": "DC",
                    "store_name": store_name,
                    "total_volume": volume,
                    "load_rate": load_rate,
                    "max_capacity": max_capacity,
                    "distance": 0,
                    "duration": 0
                }
            elif self._DC_CENTER not in route_code:
                routes_info[main_route_code]["stores"].append({
                    "route_id": main_route_code,
                    "route_code": route_code,
                    "store_id": store_id,
                    "store_name": store_name,
                    "longitude": lng,
                    "latitude": lat,
                    "sched_time": sched_time,
                    "earliest_time": earliest_time,
                    "latest_time": latest_time,
                    "pred_time": pred_time,
                    "dwell_time": dwell_time,
                    "volume": volume
                })

        return routes_info


    def create_data_folder(self, dst_dir, source_manual_file, dest_manual_file):
        """
        Notes:
            Create the data foler.

        Args:
            dst_dir (str): folder.
            source_manual_file (str): Source Manual file.
            dest_manual_file (str): Destination Manual file.

        Returns:
            None.
        """
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(source_manual_file, dest_manual_file)


    def _get_origin_routes(self):
        """
        Note:
            Transfer the manual route to the original route.

        Args:
            None.

        Returns:
            data (list): data row with routes info.
        """
        data = []
        routes = copy.deepcopy(self.routes_info)
        route_manager = RouteManager(routes, self.distance_matrix, self.time_matrix)
        route_manager.move_store_to_original_route()
        routes = route_manager.routes_info

        for route_id in routes:
            dc = routes[route_id]['dc']
            stores = routes[route_id]['stores']
            data.append({
                "不可大車": "",
                "車次": dc["route_code"],
                "ID": "",
                "店名": '林口DC',
                "表定時間": "",
                "預定時間": "",
                "貨量": dc["total_volume"],
                "裝載率": dc["load_rate"]
            })
            for s in stores:
                data.append({
                    "不可大車": False,
                    "車次": s["route_code"],
                    "ID": s["store_id"],
                    "店名": s["store_name"],
                    "表定時間": s["sched_time"],
                    "預定時間": s["sched_time"],
                    "貨量": s["volume"],
                    "裝載率": ""
                })
            data.append({
                "不可大車": False,
                "車次": f'{dc["route_code"]}DC',
                "ID": "",
                "店名": '林口DC',
                "表定時間": "",
                "預定時間": "",
                "貨量": 0,
                "裝載率": ""
            })

        return data


    def export_origin_excel_file(self, dest_file):
        """
        Notes:
            Export origin excel.

        Args:
            dest_file (str): original file.

        Returns:
            None.
        """
        data = self._get_origin_routes()
        columns = [
            "不可大車",
            "車次",
            "ID",
            "店名",
            "表定時間",
            "預定時間",
            "貨量",
            "裝載率"
        ]
        df = pd.DataFrame(data, columns=columns)
        with pd.ExcelWriter(dest_file, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Sheet1", index=False, startrow=3)
