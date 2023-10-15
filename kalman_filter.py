import numpy as np


class KalmanFilter:

    def __init__(self, position, velocity, observation_noise, prediction_noise):
        self.state = np.array([
            position[0],
            velocity[0],
            position[1],
            velocity[1]]).T

        self.observation_model = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        self.observation_noise = np.array([
            [1, 0],
            [0, 1]
        ]) * observation_noise

        self.estimate_covariance = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        self.prediction_noise = np.array([
            [0, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1]
        ]) * prediction_noise

    def get_state_transition(self, delta_time: float):
        return np.array([
            [1, 0, delta_time, 0],
            [0, 1, 0, delta_time],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def predict(self, delta_time: float):
        f = self.get_state_transition(delta_time)
        self.state = np.matmul(f, self.state)
        self.estimate_covariance = np.matmul(f, self.prediction_noise)
        self.estimate_covariance = np.matmul(self.estimate_covariance, f.T)
        self.estimate_covariance += self.prediction_noise

        return self.state, self.estimate_covariance

    def update(self, observation: np.ndarray, delta_time: float):
        kalman_gain = np.matmul(self.observation_model, self.estimate_covariance)
        kalman_gain = np.matmul(kalman_gain, self.observation_model.T)
        kalman_gain += self.observation_noise
        kalman_gain = np.matmul(np.matmul(self.estimate_covariance, self.observation_noise), kalman_gain)

        self.estimate_covariance = self.estimate_covariance - np.matmul(np.matmul(kalman_gain, self.observation_model),
                                                                        self.estimate_covariance)

        observation_delta = observation - np.matmul(self.observation_model, self.state)
        self.state = self.state + np.matmul(kalman_gain, np.linalg.inv(observation_delta))
