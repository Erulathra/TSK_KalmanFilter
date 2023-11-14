import math
import os
import sys

import numpy as np
import smopy
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QMessageBox

from flight_data import Flight
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from PyQt5 import QtCore, QtWidgets


class Window(QtWidgets.QDialog):
    flight: Flight


    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        self.files = list()
        self.directory = 'data'
        for filename in os.listdir(self.directory):
            self.files.append(filename)

        self.flight = Flight('data/326f29ca_piotrkow_trybunalski_airport_to_unknown.tsv')
        self.figure = plt.figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        self.slider_group = QtWidgets.QGroupBox()

        # Delta time widget
        delta_time_widget = QtWidgets.QWidget()
        self.delta_time_slider = self.create_slider(30, 100, 50)
        delta_time_hbox = self.create_slider_hbox(self.delta_time_slider, 0.1)
        delta_time_widget.setLayout(delta_time_hbox)

        # Prediction noise widget
        prediction_noise_widget = QtWidgets.QWidget()
        self.prediction_noise_slider = self.create_slider(1, 100, 1)
        prediction_noise_hbox = self.create_slider_hbox(self.prediction_noise_slider, 0.1)
        prediction_noise_widget.setLayout(prediction_noise_hbox)

        # Observation noise widget
        observation_noise_widget = QtWidgets.QWidget()
        self.observation_noise_slider = self.create_slider(1, 500, 30)
        observation_noise_hbox = self.create_slider_hbox(self.observation_noise_slider, 0.1)
        observation_noise_widget.setLayout(observation_noise_hbox)

        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(QtWidgets.QLabel('Time step slider'))
        vbox.addWidget(delta_time_widget)
        vbox.addWidget(QtWidgets.QLabel('Prediction noise slider'))
        vbox.addWidget(prediction_noise_widget)
        vbox.addWidget(QtWidgets.QLabel('Observation noise slider'))
        vbox.addWidget(observation_noise_widget)
        vbox.addStretch(1)
        self.slider_group.setLayout(vbox)

        self.help_button = QtWidgets.QPushButton('Help')
        self.help_button.clicked.connect(self.show_help)
        self.button = QtWidgets.QPushButton('Draw Map')
        self.button.clicked.connect(self.plot)

        # Combobox
        self.combobox = QtWidgets.QComboBox()
        for filename in self.files:
            self.combobox.addItem(filename)

        self.combobox.activated.connect(self.choose_flight)

        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.slider_group)
        layout.addWidget(self.combobox)
        layout.addWidget(self.button)
        layout.addWidget(self.help_button)

        self.setLayout(layout)
        self.resize(1200, 800)


    def choose_flight(self, id):
        self.flight = Flight(self.directory + '/' + self.files[id])


    def create_slider(self, min, max, value) -> QtWidgets.QSlider:
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setFocusPolicy(QtCore.Qt.StrongFocus)
        slider.setTickPosition(QtWidgets.QSlider.NoTicks)
        slider.setRange(min, max)
        slider.setValue(value)
        slider.setSingleStep(1)

        return slider
    

    def create_slider_hbox(self, slider: QtWidgets.QSlider, step=1) -> QtWidgets.QHBoxLayout:

        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(slider)
        label = QtWidgets.QLabel(str(float(slider.value()) / (1 / step)))
        hbox.addWidget(label)

        slider.valueChanged.connect(lambda value: label.setNum(float(value) / (1 / step)))
        return hbox


    def plot(self):
        
        self.figure.clear()

        sm_map = smopy.Map(self.flight.map_boundaries, z=8)

        ax = self.figure.add_subplot(111)
        ax = sm_map.show_mpl(ax=ax, figsize=(8,6))

        geo_gps = self.flight.get_gps_points()
        gps = to_map_points(geo_gps, sm_map)
        ax.plot(gps[:, 0], gps[:, 1], "r")

        plane = self.flight.predict_cart()
        map_points = self.flight.plane_to_map_points(plane, sm_map)
        ax.plot(map_points[:, 0], map_points[:, 1], "y")

        kalman_points = self.flight.predict_kalman(self.delta_time_slider.value(), self.observation_noise_slider.value(), self.prediction_noise_slider.value())
        kalman_map = self.flight.cart_to_map_points(kalman_points, sm_map)
        ax.plot(kalman_map[:, 0], kalman_map[:, 1], "g")

        self.canvas.draw()

    def show_help(self):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setText(
            'Red - Observation\n'
            'Yellow - Prediction\n'
            'Green - Position filtered using Kalman Filter\n')
        msg_box.exec_()


def to_map_points(points: np.ndarray, map: smopy.Map):
    result = np.zeros((len(points), 2))
    for i, goe_point in enumerate(points):
        result[i][0], result[i][1] = map.to_pixels(goe_point[0], goe_point[1])

    return result


def main():
    app = QtWidgets.QApplication(sys.argv)
    main = Window()
    main.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
