import numpy as np

from typing import Iterable, List, Tuple
from sklearn.neighbors import KNeighborsRegressor
from matplotlib import pyplot as plt
from .utils import greatest_difference_pair, landmark_list_angles
from .reference_store import ExerciseReference



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
        return self.model.predict(landmark_angles)[0]

    def correctness(self, landmark_angles: Iterable[float]) -> Tuple[float, float, int]:
        distances, kneighbors = self.model.kneighbors(
            np.array(landmark_angles).reshape(1, -1), n_neighbors=1, return_distance=True)
        
        correctness = 10 / distances[0][0]
        greatest_distance, greatest_distance_idx = greatest_difference_pair(landmark_angles, self.reference_angles[ kneighbors[0][0] ])
        return correctness, greatest_distance, greatest_distance_idx

    @classmethod
    def from_exercise_references(cls, exercise_references: List[ExerciseReference], d2=True, **kwargs) -> 'KNNRegressor':
        if not exercise_references:
            return None

        angles = landmark_list_angles([ref.landmarks for ref in exercise_references], d2=d2)
        time = [ref.progress for ref in exercise_references]

        return KNNRegressor(angles, time, **kwargs)



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

