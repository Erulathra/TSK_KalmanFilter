import math
import sys

import numpy as np
import smopy
import matplotlib.pyplot as plt

from flight_data import Flight
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets


class Window(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        self.figure = plt.figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.button = QtWidgets.QPushButton('Plot')
        self.button.clicked.connect(self.plot)
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.button)
        self.setLayout(layout)
        self.resize(1200, 800)


    def plot(self, flight: Flight):
        self.figure.clear()
        
        flight = Flight('data/326f29ca_piotrkow_trybunalski_airport_to_unknown.tsv')

        sm_map = smopy.Map(flight.map_boundaries, z=8)

        ax = self.figure.add_subplot(111)
        ax = sm_map.show_mpl(ax=ax, figsize=(8,6))

        geo_gps = flight.get_gps_points()
        gps = to_map_points(geo_gps, sm_map)
        ax.plot(gps[:, 0], gps[:, 1], "r")

        plane = flight.predict_cart()
        map_points = flight.plane_to_map_points(plane, sm_map)
        ax.plot(map_points[:, 0], map_points[:, 1], "y")

        kalman_points = flight.predict_kalman()
        kalman_map = flight.plane_to_map_points(kalman_points, sm_map)
        ax.plot(kalman_map[:, 0], kalman_map[:, 1], "g")

        self.canvas.draw()


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

    plane = flight.predict_cart()
    map_points = flight.plane_to_map_points(plane, sm_map)
    ax.plot(map_points[:, 0], map_points[:, 1], "y")

    kalman_points = flight.predict_kalman()
    kalman_map = flight.plane_to_map_points(kalman_points, sm_map)
    ax.plot(kalman_map[:, 0], kalman_map[:, 1], "g")

    plt.show()


def main():
    app = QtWidgets.QApplication(sys.argv)

    # flight = Flight('data/326f29ca_piotrkow_trybunalski_airport_to_unknown.tsv')
    # draw_plot(flight)

    main = Window()
    main.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
