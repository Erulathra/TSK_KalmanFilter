import math
from math import cos, sin, pi

import numpy as np

EARTH_RADIUS = 6378137


def to_plane_pos(point: np.ndarray):
    point_radians = point * pi / 180
    x = EARTH_RADIUS * cos(point_radians[0]) * cos(point_radians[1])
    y = EARTH_RADIUS * cos(point_radians[0]) * sin(point_radians[1])
    z = EARTH_RADIUS * sin(point_radians[0])

    return np.array([x, y, z])


def to_geo_pos(point: np.ndarray):
    r = np.sqrt(np.sum(np.power(point, 2)))
    lon = math.degrees(np.arctan(point[1] / point[0]))
    lat = math.degrees(np.arcsin(point[2] / r))

    return np.array([lat, lon])


def predict_position(lat_0: float, long_0: float, speed: float, heading: float, delta_time: float):
    delta_x = speed * math.sin(math.radians(heading)) * delta_time
    delta_y = speed * math.cos(math.radians(heading)) * delta_time

    lat = lat_0 + 180 / math.pi * delta_y / EARTH_RADIUS
    long = long_0 + 180 / math.pi / math.sin(math.radians(lat_0)) * delta_x / EARTH_RADIUS

    return np.array([lat, long])
