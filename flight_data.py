import csv
import math
import random
from copy import copy
from operator import attrgetter

import smopy

import geographic_utils

import numpy as np

import kalman_filter

MAP_BOUNDARIES_OFFSET = 0.01


class FlightPoint:
    def __init__(self, data_row):
        self.latitude = float(data_row[0])
        self.longitude = float(data_row[1])
        self.cart = geographic_utils.to_plane_pos(np.array([self.latitude, self.longitude]))

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

    def get_plane_points(self) -> np.ndarray:
        gps_points = self.get_gps_points()

        result = []
        for point in gps_points:
            result.append(geographic_utils.to_plane_pos(point))

        return np.array(result)

    def plane_to_map_points(self, points: np.ndarray, sm_map: smopy.Map) -> np.ndarray:
        geo_points = []

        for i, point in enumerate(points):
            point = geographic_utils.to_geo_pos(
                np.array([point[0], point[1], self.flight_data[i].cart[2]]))
            geo_points.append(point)

        result = np.zeros((len(points), 2))
        for i, geo_point in enumerate(geo_points):
            result[i][0], result[i][1] = sm_map.to_pixels(geo_point[0], geo_point[1])

        return result

    def cart_to_map_points(self, points: np.ndarray, sm_map: smopy.Map) -> np.ndarray:
        geo_points = []

        for i, point in enumerate(points):
            point = geographic_utils.to_geo_pos(
                np.array([point[0], point[1], point[2]]))
            geo_points.append(point)

        result = np.zeros((len(points), 2))
        for i, geo_point in enumerate(geo_points):
            result[i][0], result[i][1] = sm_map.to_pixels(geo_point[0], geo_point[1])

        return result

    def predict_points(self) -> np.ndarray:
        points = []
        points.append(np.array([self.flight_data[0].latitude, self.flight_data[0].longitude]))

        last_flight_point = self.flight_data[0]
        for flight_point in self.flight_data[1:-1]:
            delta_time = flight_point.time_step - last_flight_point.time_step
            predicted_point = self.predict_position(flight_point, delta_time)
            points.append(predicted_point)

            last_flight_point = flight_point

        return np.array(points)

    def predict_cart(self):
        gps = self.get_plane_points()
        points = []
        points.append(np.array([gps[0][0], gps[0][1]]))

        flight_points = self.flight_data

        last_point = np.array([gps[0][0], gps[0][1]])
        for i, gps_point in enumerate(gps):
            if i == 0:
                continue

            speed = flight_points[i].speed
            heading = math.radians(flight_points[i].heading - 110)
            velocity = np.array([speed * math.sin(heading), speed * math.cos(heading)])

            delta_time = flight_points[i].time_step - flight_points[i - 1].time_step
            predicted_point = last_point + velocity * delta_time
            points.append(np.array([predicted_point[0], predicted_point[1]]))

            last_point = predicted_point
            # last_point = np.array([gps_point[0], gps_point[1]])

        return np.array(points)

    def predict_kalman(self, delta_time=50, observation_noise=30, prediction_noise=1):
        points = []

        flight_points = self.flight_data
        points.append(np.array([
            flight_points[0].cart[0],
            flight_points[0].cart[1],
            flight_points[0].cart[2]]))

        point = np.array([flight_points[0].cart[0], flight_points[0].cart[1]])
        speed = flight_points[0].speed
        heading = math.radians(flight_points[0].heading - 110)
        velocity = np.array([speed * math.sin(heading), speed * math.cos(heading)])

        filter = kalman_filter.KalmanFilter(point, velocity, observation_noise, prediction_noise)

        time = flight_points[0].time_step

        while time < flight_points[-1].time_step:
            time += delta_time

            flight_point = self.get_flight_point_by_time_step(time)

            observation = np.array([flight_point.cart[0], flight_point.cart[1]])

            speed = flight_point.speed
            heading = math.radians(flight_point.heading - 110)
            velocity = np.array([speed * math.sin(heading), speed * math.cos(heading)])

            state, covariance = filter.predict(delta_time, velocity)
            filter.update(observation, delta_time)
            points.append(np.array([state[0], state[2], flight_point.cart[2]]))

        # for i in range(len(flight_points) - 1):
        #     if i == 0:
        #         continue
        #
        #     time += delta_time
        #
        #     # delta_time = flight_points[i].time_step - flight_points[i - 1].time_step
        #
        #     speed = flight_points[i].speed
        #     heading = math.radians(flight_points[i].heading - 110)
        #     velocity = np.array([speed * math.sin(heading), speed * math.cos(heading)])
        #
        #     observation = np.array([gps[i][0], gps[i][1]])
        #     state, covariance = filter.predict(delta_time, velocity)
        #     filter.update(observation, delta_time)
        #     points.append(np.array([state[0], state[2]]))

        return np.array(points)

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

    @staticmethod
    def predict_position(flight_point: FlightPoint, delta_time: float):
        return geographic_utils.predict_position(
            flight_point.latitude,
            flight_point.longitude,
            flight_point.speed,
            flight_point.heading,
            delta_time
        )

    def get_flight_point_by_time_step(self, time_step: int):
        previous_point = min(self.flight_data, key=lambda x: abs(x.time_step - time_step))
        previous_point_index = self.flight_data.index(previous_point)

        assert (previous_point == self.flight_data[previous_point_index])

        if previous_point_index - 1 < 0 or previous_point_index + 1 >= len(self.flight_data):
            return previous_point

        if previous_point.time_step > time_step:
            next_point = self.flight_data[previous_point_index - 1]
            next_point, previous_point = previous_point, next_point
        else:
            next_point = self.flight_data[previous_point_index + 1]

        assert (next_point.time_step > previous_point.time_step)

        result = copy(previous_point)

        points_delta_time = next_point.time_step - previous_point.time_step
        lerp_time = (time_step - previous_point.time_step) / points_delta_time

        result.latitude = 0
        result.longitude = 0
        result.cart = lerp(previous_point.cart, next_point.cart, lerp_time)

        return result


def lerp(a, b, t: float):
    return b * t + a * (1 - t)
