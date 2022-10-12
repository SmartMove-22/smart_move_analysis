from enum import Enum
import math
import mediapipe as mp

from typing import Iterable, Tuple


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

class ANGLES_OF_INTEREST_IDX(Enum):
    Nose_RightShoulder_LeftShoulder = 0
    Nose_LeftShoulder_RightShoulder = 1

    RightWrist_RightElbow_RightShoulder = 2
    RightElbow_RightShoulder_RightHip = 3
    RightShoulder_RightHip_RightKnee = 4
    RightHip_RightKnee_RightAnkle = 5
    RightKnee_RightAnkle_RightHeel = 6
    RightAnkle_RightHeel_RightKnee = 7

    LeftWrist_LeftElbow_LeftShoulder = 8
    LeftElbow_LeftShoulder_LeftHip = 9
    LeftShoulder_LeftHip_LeftKnee = 10
    LeftHip_LeftKnee_LeftAnkle = 11
    LeftKnee_LeftAnkle_LeftHeel = 12
    LeftAnkle_LeftHeel_LeftKnee = 13


def get_landmarks_from_angle(landmark_angle: int) -> Tuple[int, int, int]:
    '''From a landmark angle index, return the correspondent 3 landmarks' indices.'''

    return ANGLES_OF_INTEREST[landmark_angle]

def landmark_angle_2d(landmark_first, landmark_middle, landmark_last):
    '''Convert 3 positional landmarks into the angle defined by them. Done in 2D'''

    angle = math.degrees(abs(\
        math.atan2(landmark_first.y - landmark_middle.y, landmark_first.x - landmark_middle.x) \
        - math.atan2(landmark_last.y - landmark_middle.y, landmark_last.x - landmark_middle.x)))
    
    return angle if angle < 180 else 360 - angle

# TODO
def landmark_angle_3d(landmark_first, landmark_middle, landmark_last):
    '''Convert 3 positional landmarks into the angle defined by them. Done in 3D'''
    pass

def landmark_list_angles(landmark_list, d2: bool=True):
    '''Convert a list of positional landmarks into a list of angles defined by those landmarks. \n
    The order of `ANGLES_OF_INTEREST` is followed. `d2` is `True` if the angle calculation is 2D
    or `False` if it's 3D.'''

    angle_method = landmark_angle_2d if d2 else landmark_angle_3d

    return [angle_method(landmark_list[l1], landmark_list[l2], landmark_list[l3])
            for l1, l2, l3 in ANGLES_OF_INTEREST]

def euclidean_distance(ps1: Iterable[float], ps2: Iterable[float]) -> float:
    return math.sqrt((p1 - p2)**2 for p1, p2 in zip(ps1, ps2))

def greatest_difference_pair(ps1: Iterable[float], ps2: Iterable[float]) -> Tuple[float, int]:
    '''Find the pair of values from `ps1` and `ps2` that have the greatest difference. \n
    Returns that difference and the index of that pair.'''
    greatest_distance = 0
    greatest_idx = 0
    for idx, (p1, p2) in enumerate(zip(ps1, ps2)):
        distance = abs(p1 - p2)
        if distance > greatest_distance:
            greatest_distance = distance
            greatest_idx = idx
    return greatest_distance, greatest_idx