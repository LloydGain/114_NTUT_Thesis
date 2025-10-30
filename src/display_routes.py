import json
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

class DisplayRoutes():
    def __init__(self, file_path='../output/optimized_routes_info.json'):
        self.file_path = file_path
        self.dc = {'store_id': 'dc', 'longitude': 121.40712, 'latitude': 25.083282}
        self.routes = self._load_routes()
    

    def _load_routes(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data


    def show_optimized_result(self):
        avg_duration = 0
        avg_distance = 0
        num_vehicle = len(self.routes)
        
        for route in self.routes:
            avg_duration += self.routes[route]['dc']['duration']
            avg_distance += self.routes[route]['dc']['distance']
        
        avg_duration /= num_vehicle * 60 * 60
        avg_distance /= num_vehicle
        print(f'Total Vehicle: {num_vehicle}')
        print(f'Avg Time: {avg_duration}hr')
        print(f'Avg Distance: {avg_distance}km')


    def plot_routes(self):
        cmap = get_cmap("tab10")

        for idx, route_id in enumerate(self.routes):
            route_info = self.routes[route_id]
            load_rate = round(route_info['dc']['load_rate'], 2)
            stores = route_info['stores']
            if not stores:
                continue

            lons = [self.dc['longitude']] + [s['longitude'] for s in stores] + [self.dc['longitude']]
            lats = [self.dc['latitude']] + [s['latitude'] for s in stores] + [self.dc['latitude']]
            labels = ["DC"] + [f'({idx+1}){store['route_code']}' for idx, store in enumerate(stores)] + ["DC"]

            plt.figure(figsize=(8, 6))
            plt.scatter(lons, lats, color=cmap(idx % 10), s=50)

            for lon, lat, label in zip(lons, lats, labels):
                plt.text(lon, lat + 0.0005, label, fontsize=8, alpha=0.7, ha='center', va='bottom')

            plt.plot(lons, lats, color=cmap(idx % 10), linestyle='--', alpha=0.5)

            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.title(f"Route {route_id}(load_rate: {load_rate})")
            plt.grid(True)
            plt.show()