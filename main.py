import math

import numpy as np
import smopy
import matplotlib.pyplot as plt

from flight_data import Flight


def to_map_points(points: np.ndarray, map: smopy.Map):
    result = np.zeros((len(points), 2))
    for i, goe_point in enumerate(points):
        result[i][0], result[i][1] = map.to_pixels(goe_point[0], goe_point[1])

    return result


def draw_plot(flight: Flight):
    sm_map = smopy.Map(flight.map_boundaries, z=8)
    ax = sm_map.show_mpl()

    geo_gps = flight.get_gps_points()
    gps = to_map_points(geo_gps, sm_map)
    ax.plot(gps[:, 0], gps[:, 1], "r")
    plt.show()

    cart_points = flight.get_plane_points()
    plt.plot(cart_points[:, 0], cart_points[:, 1], "g")
    pred_points = flight.predict_cart()
    plt.plot(pred_points[:, 0], pred_points[:, 1], "y")
    plt.show()

    plane = flight.predict_cart()
    map_points = flight.plane_to_map_points(plane, sm_map)
    ax = sm_map.show_mpl()
    ax.plot(gps[:, 0], gps[:, 1], "r")
    ax.plot(map_points[:, 0], map_points[:, 1], "y")
    plt.show()


def main():
    flight = Flight('data/326f29ca_piotrkow_trybunalski_airport_to_unknown.tsv')
    draw_plot(flight)


if __name__ == "__main__":
    main()
