import math
import mediapipe as mp

from typing import Iterable


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pl = mp_pose.PoseLandmark

# IntelliSense
mp_drawing: mp.solutions.mediapipe.python.solutions.drawing_utils
mp_drawing_styles = mp.solutions.mediapipe.python.solutions.drawing_styles
mp_pose: mp.solutions.mediapipe.python.solutions.pose


LANDMARKS_OF_INTEREST = [
    pl.NOSE,

    pl.RIGHT_WRIST,
    pl.RIGHT_ELBOW,
    pl.RIGHT_SHOULDER,
    pl.RIGHT_HIP,
    pl.RIGHT_KNEE,
    pl.RIGHT_ANKLE,
    pl.RIGHT_HEEL,
    pl.RIGHT_FOOT_INDEX,

    pl.LEFT_WRIST,
    pl.LEFT_ELBOW,
    pl.LEFT_SHOULDER,
    pl.LEFT_HIP,
    pl.LEFT_KNEE,
    pl.LEFT_ANKLE,
    pl.LEFT_HEEL,
    pl.LEFT_FOOT_INDEX,
]

ANGLES_OF_INTEREST = [
    (pl.NOSE, pl.RIGHT_SHOULDER, pl.LEFT_SHOULDER),
    (pl.NOSE, pl.LEFT_SHOULDER, pl.RIGHT_SHOULDER),

    (pl.RIGHT_WRIST, pl.RIGHT_ELBOW, pl.RIGHT_SHOULDER),
    (pl.RIGHT_ELBOW, pl.RIGHT_SHOULDER, pl.RIGHT_HIP),
    (pl.RIGHT_SHOULDER, pl.RIGHT_HIP, pl.RIGHT_KNEE),
    (pl.RIGHT_HIP, pl.RIGHT_KNEE, pl.RIGHT_ANKLE),
    (pl.RIGHT_KNEE, pl.RIGHT_ANKLE, pl.RIGHT_HEEL),
    (pl.RIGHT_ANKLE, pl.RIGHT_HEEL, pl.RIGHT_KNEE),

    (pl.LEFT_WRIST, pl.LEFT_ELBOW, pl.LEFT_SHOULDER),
    (pl.LEFT_ELBOW, pl.LEFT_SHOULDER, pl.LEFT_HIP),
    (pl.LEFT_SHOULDER, pl.LEFT_HIP, pl.LEFT_KNEE),
    (pl.LEFT_HIP, pl.LEFT_KNEE, pl.LEFT_ANKLE),
    (pl.LEFT_KNEE, pl.LEFT_ANKLE, pl.LEFT_HEEL),
    (pl.LEFT_ANKLE, pl.LEFT_HEEL, pl.LEFT_KNEE),
]

def landmark_angle_2d(landmark_first, landmark_middle, landmark_last):
    angle = math.degrees(abs(\
        math.atan2(landmark_first.y - landmark_middle.y, landmark_first.x - landmark_middle.x) \
        - math.atan2(landmark_last.y - landmark_middle.y, landmark_last.x - landmark_middle.x)))
    
    return angle if angle < 180 else 360 - angle

# TODO
def landmark_angle_3d(landmark_first, landmark_middle, landmark_last):
    pass

def landmark_list_angles(landmark_list, d2: bool=True):
    angle_method = landmark_angle_2d if d2 else landmark_angle_3d

    return [angle_method(landmark_list[l1], landmark_list[l2], landmark_list[l3])
            for l1, l2, l3 in ANGLES_OF_INTEREST]

def euclidean_distance(ps1: Iterable[float], ps2: Iterable[float]) -> float:
    return math.sqrt((p1 - p2)**2 for p1, p2 in zip(ps1, ps2))

def greatest_distance_pair_index(ps1: Iterable[float], ps2: Iterable[float]) -> int:
    greatest_distance = 0
    greatest_idx = 0
    for idx, (p1, p2) in enumerate(zip(ps1, ps2)):
        distance = math.sqrt((p1 - p2)**2)
        if distance > greatest_distance:
            greatest_distance = distance
            greatest_idx = idx
    return greatest_idx