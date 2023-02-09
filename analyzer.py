import numpy as np

from typing import Iterable, List, Tuple
from sklearn.neighbors import KNeighborsRegressor
from matplotlib import pyplot as plt
from .utils import greatest_difference_pair, landmark_list_angles, obtain_angles
from .reference_store import ExerciseReference



class Analyzer:

    """
    Analyzer module that is determines the correctness and pacing of a specific exercise execution.

    Internally uses a K-Nearest Neighbors regressor.

    Parameters
    ----------

    angles : List[List[float]]
        List containing the landmark angles for each of the poses in the training data, associated with one exercise.
        The landmark angles are a list of the stored angles for each landmark, at a given pose.
    time : List[float]
        List of times at which the landmark angles of the previous argument were observed during a reference execution.
        The time should be normalized to the range [0, 1].
    """

    def __init__(self, angles: List[List[float]], time: List[float], **model_params):
        self.reference_angles = angles

        self.model: KNeighborsRegressor = KNeighborsRegressor(
            n_neighbors=3,
            weights='uniform',
            **model_params,
        ).fit(angles, time)


    def progress(self, pose: List[float]) -> float:
        """Predict the progress of the given pose within the exercise. The pose is characterized by the angles of the different landmarks."""
        landmark_angles = np.array(pose).reshape(1, -1)
        return self.model.predict(landmark_angles)[0]


    def correctness(self, pose: List[float]) -> Tuple[float, int, float]:
        """Predict the correctness of the given pose within the exercise. The pose is characterized by the angles of the different landmarks."""
        closest_poses, closest_poses_distances = self.closest_reference_poses(pose, n_poses=1)
        closest_pose = closest_poses[0]
        closest_pose_distance = closest_poses_distances[0]
        
        # Model's maximum distance for any given point
        # (assumes euclidean distance, watch out for the model's parameters if another distance is used)
        max_distance = np.sqrt(len(pose) * (360**2))
        
        correctness = max(0, (1 / max_distance) * (max_distance - closest_pose_distance**2))
        greatest_distance_idx = greatest_difference_pair(pose, closest_pose)
        return correctness, greatest_distance_idx, closest_pose[greatest_distance_idx]


    @classmethod
    def from_exercise_references(cls, exercise_references: List[ExerciseReference], exercise_angles: str=None, d2=True, **kwargs) -> 'Analyzer':
        """Create an Analyzer instance from `ExerciseReference` data."""
        if not exercise_references:
            return None

        angles_to_use = obtain_angles(exercise_angles)
        angles = [landmark_list_angles(ref.landmarks, angles=angles_to_use, d2=d2) for ref in exercise_references]
        time = [ref.progress for ref in exercise_references]

        return Analyzer(angles, time, **kwargs)


    def closest_reference_poses(self, pose: Iterable[float], n_poses: int) -> Tuple[List[List[float]], List[int]]:
        """Return the N most similar reference poses to the given pose."""
        distances, kneighbors = self.model.kneighbors(
            np.array(pose).reshape(1, -1), n_neighbors=n_poses, return_distance=True)

        return [self.reference_angles[pose_idx] for pose_idx in kneighbors[0]], list(distances[0])



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

