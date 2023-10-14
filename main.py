import math

import numpy as np
import smopy
import matplotlib.pyplot as plt

from flight_data import Flight


def draw_plot(flight: Flight):
    sm_map = smopy.Map(flight.map_boundaries)
    ax = sm_map.show_mpl()

    geo_points = flight.get_gps_points()
    cart_points = np.zeros((len(geo_points), 2))
    for i, goe_point in enumerate(geo_points):
        cart_points[i][0], cart_points[i][1] = sm_map.to_pixels(goe_point[0], goe_point[1])

    ax.plot(cart_points[:, 0], cart_points[:, 1], "r")
    # plt.plot(cart_points[:, 0], cart_points[:, 1], "r")

    plt.show()


def main():
    flight = Flight('data/326f39e9_mielec_airport_to_unknown.tsv')
    draw_plot(flight)


if __name__ == "__main__":
    main()
