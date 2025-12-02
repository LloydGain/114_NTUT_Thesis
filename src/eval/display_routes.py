import os
import json
import shutil
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

class DisplayRoutes():
    """
    Notes:
        Visualize and display routes.
    """
    def __init__(self, source_file, dest_file):
        self.dc = {'store_id': 'dc', 'longitude': 121.40712, 'latitude': 25.083282}
        self.source_file = source_file
        self.dest_file = dest_file
        self.routes = self._load_routes()
    

    def _load_routes(self):
        """
        Notes:
            Load routes from a JSON file.

        Args:
            None.

        Returns:
            dict: Routes data.
        """
        with open(self.source_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data


    def plot_routes(self):
        if os.path.exists(self.dest_file):
            shutil.rmtree(self.dest_file)
        os.makedirs(self.dest_file)

        cmap = get_cmap("tab10")
        cmap_alt = get_cmap('Set1')

        for idx, route_id in enumerate(self.routes):
            route_info = self.routes[route_id]
            load_rate = round(route_info['dc']['load_rate'], 2)
            stores = route_info['stores']
            if not stores:
                continue

            lons = [self.dc['longitude']] + [s['longitude'] for s in stores] + [self.dc['longitude']]
            lats = [self.dc['latitude']] + [s['latitude'] for s in stores] + [self.dc['latitude']]
            labels = ["DC"] + [f"({idx+1}){store['route_code']}" for idx, store in enumerate(stores)] + ["DC"]

            plt.figure(figsize=(8, 6))

            colors = []
            for store in stores:
                if str(route_id) in store['route_code']:
                    colors.append(cmap(idx % 10))
                else:
                    colors.append(cmap_alt(idx % 10))

            point_colors = [cmap(idx % 10)] + colors + [cmap(idx % 10)]

            plt.scatter(lons, lats, color=point_colors, s=50)

            for lon, lat, label in zip(lons, lats, labels):
                plt.text(lon, lat + 0.0005, label, fontsize=8, alpha=0.7,
                         ha='center', va='bottom')

            plt.plot(lons, lats, color=cmap(idx % 10), linestyle='--', alpha=0.5)

            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.title(f"Route {route_id} (load_rate: {load_rate})")
            plt.grid(True)

            save_path = os.path.join(self.dest_file, f"route_{route_id}.png")
            plt.savefig(save_path, dpi=200, bbox_inches='tight')

            plt.close()