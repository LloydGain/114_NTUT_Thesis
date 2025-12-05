import os
import json
import shutil
import requests
import polyline
import folium
from folium.features import DivIcon
from folium.plugins import AntPath
import matplotlib.pyplot as plt

class DisplayRoutes:
    """
    Notes:
        Display delivery routes using Folium for HTML output and Matplotlib for PNG output.
    """
    def __init__(self, source_file):
        self.dc = {'store_id': 'dc', 'longitude': 121.40712, 'latitude': 25.083282}
        self.source_file = source_file
        self.routes = self._load_routes()
        self.main_colors = ["blue", "green", "purple", "orange", "cadetblue", "pink", "darkgreen", "darkblue", "darkorange"]
        self.alt_color = "darkred"


    def _load_routes(self):
        """
        Notes:
            Load routes from JSON file.

        Args:
            None.

        Returns:
            dict: Routes data.
        """
        with open(self.source_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data


    def plot_routes_png(self, dest_dir):
        """
        Notes:
            Plot routes and save as PNG files.

        Args:
            dest_dir (str): Directory to save PNG files.

        Returns:
            None.
        """

        os.makedirs(dest_dir, exist_ok=True)

        for idx, (route_id, route_info) in enumerate(self.routes.items()):
            dc = route_info.get("dc", {})
            stores = route_info.get("stores", [])
            if not stores:
                continue

            main_color = self.main_colors[idx % len(self.main_colors)]

            waypoints = [(s["longitude"], s["latitude"]) for s in stores]
            lons = [self.dc['longitude']] + [lon for lon, _ in waypoints] + [self.dc['longitude']]
            lats = [self.dc['latitude']] + [lat for _, lat in waypoints] + [self.dc['latitude']]
            labels = ["DC"] + [f"({i+1}){store['route_code']}" for i, store in enumerate(stores)] + ["DC"]

            plt.figure(figsize=(8, 6))
            plt.plot(lons, lats, color=main_color, linestyle='--', alpha=0.7)
            plt.scatter(lons[0], lats[0], color=main_color, s=50)

            for _, store in enumerate(stores):
                lon, lat = store["longitude"], store["latitude"]
                if store["route_code"][:2] == store['route_id']:
                    color = main_color
                else:
                    color = self.alt_color
                plt.scatter(lon, lat, color=color, s=50)

            plt.scatter(lons[-1], lats[-1], color=main_color, s=50)

            for lon, lat, label in zip(lons, lats, labels):
                plt.text(lon, lat + 0.0005, label, fontsize=8, alpha=0.7,
                        ha='center', va='bottom')

            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.title(f"Route {route_id} ({dc['load_rate']})")
            plt.grid(True)

            save_path = os.path.join(dest_dir, f"{route_id}.png")
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close()



    def plot_routes_html(self, dest_html):
        """
        Notes:
            Plot routes and save as an interactive HTML file.
        
        Args:
            dest_html (str): Path to save the HTML file.

        Returns:
            None.
        """
        m = folium.Map(location=(self.dc["latitude"], self.dc["longitude"]), zoom_start=12)

        folium.Marker(
            location=(self.dc["latitude"], self.dc["longitude"]),
            popup="Depot",
            icon=folium.Icon(color="red", icon="dc")
        ).add_to(m)

        for idx, (route_id, route_info) in enumerate(self.routes.items()):
            stores = route_info["stores"]

            main_color = self.main_colors[idx % len(self.main_colors)]
            waypoints = [(s["longitude"], s["latitude"]) for s in stores]

            coords = [f"{self.dc['longitude']},{self.dc['latitude']}"]
            coords += [f"{lon},{lat}" for lon, lat in waypoints]
            coords.append(f"{self.dc['longitude']},{self.dc['latitude']}")
            coord_str = ";".join(coords)

            url = f"http://localhost:5000/route/v1/driving/{coord_str}?overview=full&geometries=polyline"
            res = requests.get(url).json()

            encoded = res["routes"][0]["geometry"]
            decoded_points = polyline.decode(encoded)

            fg = folium.FeatureGroup(name=f"Route {route_id}", show=False)

            AntPath(
                locations=decoded_points,
                color=main_color,
                weight=6,
                opacity=0.85,
                delay=3000,
                dash_array=[10, 20],
                pulse_color="white"
            ).add_to(fg)

            for i, store in enumerate(stores, start=1):
                lon, lat = store["longitude"], store["latitude"]
                marker_color = main_color if route_id in store["route_code"] else self.alt_color
                popup_text = f"{i}. {store['store_name']} ({store['route_code']}) ({store['volume']} ({store['pred_time']})"

                folium.Marker(
                    location=(lat, lon),
                    popup=popup_text,
                    icon=DivIcon(
                        icon_size=(30, 30),
                        icon_anchor=(15, 15),
                        html=f'''
                        <div style="
                            font-size:12pt;
                            color:white;
                            background-color:{marker_color};
                            border-radius:50%;
                            width:26px;
                            height:26px;
                            text-align:center;
                            line-height:26px;">
                            {i}
                        </div>'''
                    )
                ).add_to(fg)

            fg.add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)
        m.save(dest_html)