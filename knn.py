import numpy as np

from typing import Iterable
from sklearn.neighbors import KNeighborsRegressor
from matplotlib import pyplot as plt
from landmark_analysis import greatest_distance_pair_index



class KNNRegressor:

    """
The K-Nearest Neighbors Regressor that will be used to determine
the correctness and pacing of a specific exercise execution.

Parameters:
- angles: list containing the landmark angles from the reference executions. \n
The landmark angles are a list of the stored angles for each exercise execution.
- time: list of times at which the landmark angles of the previous argument
were observed during a reference execution. \n
The time should be normalized to the range [0, 1]
    """

    def __init__(self, angles: Iterable[Iterable[float]], time: Iterable[float], n_neighbors=3, weights='uniform', name=None):
        self.name = name
        self.reference_angles = angles

        self.model = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=weights,
        ).fit(angles, time)

        self.model: KNeighborsRegressor

    # One-by-one
    def progress(self, landmark_angles: Iterable[float]) -> float:
        landmark_angles = np.array(landmark_angles).reshape(1, -1)
        return self.model.predict(landmark_angles)

    def correctness(self, landmark_angles: Iterable[float]) -> float:
        landmark_angles = np.array(landmark_angles).reshape(1, -1)
        distances, kneighbors = self.model.kneighbors(landmark_angles, n_neighbors=1, return_distance=True)
        
        correctness = 10 / np.average(distances[0])
        idx = greatest_distance_pair_index(landmark_angles, self.reference_angles[ kneighbors[0] ])
        return correctness, idx



if __name__ == '__main__':

    X_train = np.array([
        0, 10, 20, 23.5, 14, 12.136, 30.1, 26.674
    ]).reshape(-1, 1)

    y_train = np.array([
        0.0, 0.3, 0.6, 0.69, 0.36, 0.33, 1.0, 0.81
    ])

    knn = KNeighborsRegressor(
        n_neighbors=1,
        weights='uniform'
    )

    model = knn.fit(X_train, y_train)

    T = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)

    plt.scatter(X_train, y_train)
    plt.plot(T, model.predict(T))
    plt.show()

