import csv
from operator import attrgetter

import numpy as np

MAP_BOUNDARIES_OFFSET = 0.01


class FlightPoint:
    def __init__(self, data_row):
        self.latitude = float(data_row[0])
        self.longitude = float(data_row[1])

        self.speed = float(data_row[4]) * (10. / 36.)  # to m/s
        self.heading = float(data_row[9])
        self.time_step = int(data_row[11])

    def __str__(self):
        return (f'[latitude: {self.latitude}, '
                f'longitude: {self.longitude}, '
                f'speed: {self.speed}, '
                f'heading: {self.heading}, '
                f'time step: {self.time_step}]')

    def __repr__(self):
        return self.__str__()


class Flight:
    def __init__(self, path: str):
        self.flight_data = self._load_data(path)
        self.map_boundaries = ()

        self.calculate_map_boundaries()

    def calculate_map_boundaries(self):
        min_lat = min(self.flight_data, key=attrgetter("latitude")).latitude
        max_lat = max(self.flight_data, key=attrgetter("latitude")).latitude
        min_lon = min(self.flight_data, key=attrgetter("longitude")).longitude
        max_lon = max(self.flight_data, key=attrgetter("latitude")).longitude

        map_size_lat = max_lat - min_lat
        map_size_lon = max_lon - min_lon

        min_lat -= map_size_lat * MAP_BOUNDARIES_OFFSET
        max_lat += map_size_lat * MAP_BOUNDARIES_OFFSET
        min_lon -= map_size_lon * MAP_BOUNDARIES_OFFSET
        max_lon += map_size_lon * MAP_BOUNDARIES_OFFSET

        self.map_boundaries = (min_lat, min_lon, max_lat, max_lon)

    def get_gps_points(self) -> np.ndarray:
        points = np.zeros((len(self.flight_data), 2))

        for i, flight_point in enumerate(self.flight_data):
            points[i][0] = flight_point.latitude
            points[i][1] = flight_point.longitude

        return points


    @staticmethod
    def _load_data(path: str) -> list[FlightPoint]:
        with open(path, newline='') as flight_raw_data:
            flight_data_reader = csv.reader(flight_raw_data, delimiter='\t')
            result = []

            for data_row in flight_data_reader:
                try:
                    result.append(FlightPoint(data_row))
                except ValueError:
                    continue

            return result
