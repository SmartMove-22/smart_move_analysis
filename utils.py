import math
import mediapipe as mp

from typing import Iterable, Tuple, List


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

class ANGLES_OF_INTEREST_IDX:
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


# Pre-defined sets of interest angles for each exercise.
# They will be the only ones accounted for when evaluating an exercise.
EXERCISE_ANGLES = {
    'squat': [
            ANGLES_OF_INTEREST_IDX.RightHip_RightKnee_RightAnkle,
            ANGLES_OF_INTEREST_IDX.LeftHip_LeftKnee_LeftAnkle,

            ANGLES_OF_INTEREST_IDX.RightShoulder_RightHip_RightKnee,
            ANGLES_OF_INTEREST_IDX.LeftShoulder_LeftHip_LeftKnee,
    ]
}


def get_landmarks_from_angle(landmark_angle: int, exercise_angles: str=None) -> Tuple[int, int, int]:
    '''From a landmark angle index, return the 3 correspondent landmarks.
    If an exercise is specified, then the index is relative to that exercise's angles.'''

    return ANGLES_OF_INTEREST[landmark_angle] if not exercise_angles \
        else ANGLES_OF_INTEREST[ EXERCISE_ANGLES[exercise_angles][landmark_angle] ]


def landmark_angle_2d(landmark_first, landmark_middle, landmark_last):
    '''Convert 3 positional landmarks into the angle defined by them. Done in 2D'''

    angle = math.degrees(abs(
        math.atan2(landmark_first.y - landmark_middle.y, landmark_first.x - landmark_middle.x)
        - math.atan2(landmark_last.y - landmark_middle.y, landmark_last.x - landmark_middle.x)))
    
    return angle if angle < 180 else 360 - angle

# TODO
def landmark_angle_3d(landmark_first, landmark_middle, landmark_last):
    '''Convert 3 positional landmarks into the angle defined by them. Done in 3D'''
    pass


def landmark_list_angles(landmark_list, angles: list=ANGLES_OF_INTEREST, d2: bool=True):
    '''Convert a list of positional landmarks into a list of angles defined by those landmarks. \n
    The order of `angles` is followed. `d2` is `True` if the angle calculation is 2D
    or `False` if it's 3D.'''

    angle_method = landmark_angle_2d if d2 else landmark_angle_3d

    return [angle_method(landmark_list[l1], landmark_list[l2], landmark_list[l3])
            for l1, l2, l3 in angles]


def obtain_angles(exercise_angles: str) -> List[Tuple[int, int, int]]:
    '''For a given exercise name from `EXERCISE_ANGLES`, return the list of correspondent angles, each defined by 3 landmark indices.'''
    if exercise_angles is None:
        return ANGLES_OF_INTEREST
    
    if exercise_angles not in EXERCISE_ANGLES:
        raise ValueError(f'The exercise \'{exercise_angles}\' doesn\'t have landmarks set! Available exercises: {EXERCISE_ANGLES.keys()}')
    
    return [ANGLES_OF_INTEREST[angle_idx] for angle_idx in EXERCISE_ANGLES[exercise_angles]]


''' TEST CODE
pacings = calculate_pacing(
    progress_list=[
        [0.2, 0.4, 0.6, 0.8, 0.95],  # first rep, first half
        [0.2, 0.4, 0.6, 0.8, 0.95],  # first rep, second half
        [0.2, 0.4, 0.6, 0.8, 0.95],  # second rep, first half
        [0.2, 0.4, 0.6, 0.8, 0.95]   # second rep, second half
    ],
    time_list=[
        [100, 200, 275, 350, 400],
        [100, 200, 500, 800, 900],
        [100, 220, 270, 340, 410],
        [100, 200, 450, 790, 950],
    ]
)

plt.plot(pacings, np.linspace(0, 1, len(pacings)))
plt.title('Pacing')
plt.xlabel('Time (ms)')
plt.ylabel('Progress')
plt.show()
'''
def calculate_pacing(progress_list: List[List[float]], time_list: List[List[int]], nbins=20) -> List[float]:
    pacing = [[] for _ in range(nbins)]
    pacing_idx = lambda progress: min(int(progress*nbins), 19)

    for i in range(len(progress_list)//2):
        first_half_idx = i*2
        second_half_idx = i*2 + 1

        first_half_progress = progress_list[first_half_idx]
        first_half_time = time_list[first_half_idx]
        second_half_progress = progress_list[second_half_idx]
        second_half_time = time_list[second_half_idx]
        
        for progress, time in zip(first_half_progress, first_half_time):
            progress = progress/2
            pacing[pacing_idx(progress)].append(time)
        last_time = time
        for progress, time in zip(second_half_progress, second_half_time):
            progress = 0.5 + progress/2
            pacing[pacing_idx(progress)].append(last_time + time)

    pacing = [sum(times)/len(times) for times in pacing if times]

    return pacing


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
