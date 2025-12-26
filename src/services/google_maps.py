import os
import requests
import config

class GoogleRoutesAPI:
    """
    Notes:
        Routes API (Google Maps).
    """
    def __init__(self):
        self.api_key = config.GOOGLE_API_KEY
        self.routes_base_url = "https://routes.googleapis.com/directions/v2:computeRoutes"
        self.dist_matrix_base_url = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"
        self.dc = config.DC_CONFIG
        self.max_elements_per_request = 625
        self.timeout = 10


    def parse_duration(self, duration_str):
        """
        Notes:
            Parse duration string to seconds.

        Args:
            duration_str (str): Duration string (e.g., '730s').

        Returns:
            int: Duration in seconds.
        """
        if duration_str.endswith('s'):
            return int(duration_str[:-1])
        return 0


    def distance_meter_to_km(self, distance_meters):
        """
        Notes:
            Convert distance from meters to kilometers.

        Args:
            distance_meters (int): Distance in meters.

        Returns:
            float: Distance in kilometers.
        """
        return distance_meters / 1000.0


    def _compute_route_matrix(self, origins, destinations, travel_mode="DRIVE"):
        """
        Notes:
            Compute route matrix using Google Maps Routes API.

        Args:
            origins (list): List of origin coordinates.
            destinations (list): List of destination coordinates.
            travel_mode (str): Mode of travel.

        Returns:
            dict: Route matrix response.
        """
        body = {
            "origins": [{'waypoint': {'location': {'latLng' : {'latitude': origin['latitude'], 'longitude': origin['longitude']}}}} for origin in origins],
            "destinations": [{'waypoint': {'location': {'latLng' : {'latitude': des['latitude'], 'longitude': des['longitude']}}}} for des in destinations],
            "travelMode": travel_mode
        }

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": "originIndex,destinationIndex,duration,distanceMeters"
        }

        response = requests.post(self.dist_matrix_base_url, json=body, headers=headers, timeout=self.timeout)

        if response.status_code != 200:
            raise RuntimeError(f"Error in route matrix request: {response.status_code}, {response.text}")

        return response.json()


    def batch_compute_route_matrix(self, origins, destinations, travel_mode="DRIVE"):
        """
        Notes:
            Batch compute route matrix to handle large number of origins/destinations.

        Args:
            origins (list): List of origin coordinates.
            destinations (list): List of destination coordinates.
            travel_mode (str): Mode of travel.

        Returns:
            dict: List of route matrix responses.
        """

        origins = [self.dc] + origins
        destinations = [self.dc] + destinations

        n_origin = len(origins)
        n_destination = len(destinations)
        distance_matrix = {store['store_id']: {s['store_id']: 0 for s in destinations} for store in origins}
        time_matrix = {store['store_id']: {s['store_id']: 0 for s in destinations} for store in origins}
        batch_size = int(self.max_elements_per_request ** 0.5)

        for i in range(0, n_origin, batch_size):
            for j in range(0, n_destination, batch_size):
                origin_batch = origins[i : i + batch_size]
                destination_batch = destinations[j : j + batch_size]
                data = self._compute_route_matrix(origin_batch, destination_batch, travel_mode)

                for element in data:
                    origins_idx = element['originIndex']
                    destinations_idx = element['destinationIndex']
                    origin_store_id = origin_batch[origins_idx]['store_id']
                    dest_store_id = destination_batch[destinations_idx]['store_id']

                    if origin_store_id == dest_store_id:
                        continue

                    distance_matrix[origin_store_id][dest_store_id] = self.distance_meter_to_km(element['distanceMeters'])
                    time_matrix[origin_store_id][dest_store_id] = self.parse_duration(element['duration'])

        return distance_matrix, time_matrix


    def compute_route(self, waypoints, travel_mode="DRIVE"):
        """
        Notes:
            Compute route using Google Maps Directions API.

        Args:
            waypoints (list): List of waypoint coordinates.
            travel_mode (str): Mode of travel.

        Returns:
            tuple: (distance in km, duration in seconds).
        """
        dc_latlng = { "latitude": self.dc["latitude"], "longitude": self.dc["longitude"]}

        body = {
            "origin": {'location': {'latLng' : dc_latlng}},
            "destination": {'location': {'latLng' : dc_latlng}},
            "intermediates": [{'location': {'latLng' : {'latitude': wp['latitude'], 'longitude': wp['longitude']}}} for wp in waypoints],
            "travelMode": travel_mode
        }

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": "routes.distanceMeters,routes.duration,routes.polyline,routes.legs.startLocation,routes.legs.endLocation,routes.legs.distanceMeters,routes.legs.staticDuration"
        }

        response = requests.post(self.routes_base_url, json=body, headers=headers, timeout=self.timeout)

        if response.status_code != 200:
            raise RuntimeError(f"Error in route request: {response.status_code}, {response.text}")

        data = response.json()
        distance, duration = self.distance_meter_to_km(data['routes'][0]['distanceMeters']), self.parse_duration(data['routes'][0]['duration'])
        duration += sum(wp['dwell_time'] for wp in waypoints)

        # encoded_polyline = data['routes'][0]['polyline']['encodedPolyline']

        durations = []
        for leg in data['routes'][0]['legs']:
            leg_static_duration = self.parse_duration(leg['staticDuration'])
            durations.append(leg_static_duration)

        durations = durations[1:-1]

        return distance, duration, durations
