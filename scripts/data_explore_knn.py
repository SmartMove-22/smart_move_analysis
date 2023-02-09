import argparse
import json
import os
from typing import List, Tuple

import cv2
import mediapipe as mp

from ..knn import KNNRegressor
from ..reference_store import LandmarkData
from ..utils import get_landmarks_from_angle, landmark_list_angles, obtain_angles

description = '''
Script to quickly and easily explore the KNN model trained on captured data
'''

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# IntelliSense
# mp_drawing: mp.solutions.mediapipe.python.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.mediapipe.python.solutions.drawing_styles
# mp_pose: mp.solutions.mediapipe.python.solutions.pose


def image_status(image: cv2.Mat, string: str, repetitions: int, first_half: bool, at_starting_position: bool=False):
    text_style = {
        'fontFace': cv2.FONT_HERSHEY_SIMPLEX,
        'fontScale': 0.70,
        'color': (80, 80, 255),
        'thickness': 2
    }
    outline_style = {
        'fontFace': cv2.FONT_HERSHEY_SIMPLEX,
        'fontScale': 0.70,
        'color': (0, 0, 0),
        'thickness': 6
    }
    
    cv2.putText(image, string, (0, 25), **outline_style)
    cv2.putText(image, string, (0, 25), **text_style)
    cv2.putText(image, f'Repetitions: {repetitions:>3} | First half: {first_half}', (0, 50), **outline_style)
    cv2.putText(image, f'Repetitions: {repetitions:>3} | First half: {first_half}', (0, 50), **text_style)
    cv2.putText(image, f'At starting position? {at_starting_position}', (0, 75), **outline_style)
    cv2.putText(image, f'At starting position? {at_starting_position}', (0, 75), **text_style)

    
def model_results(landmarks, model: KNNRegressor, angles_to_use: List[Tuple[int, int, int]]):
    angles = landmark_list_angles(landmarks, angles=angles_to_use, d2=True)
    return model.progress(angles), model.correctness(angles)


def camera_loop(
        model_fh: KNNRegressor,
        model_sh: KNNRegressor,
        angles_to_use: List[Tuple[int, int, int]],
        exercise_angles: str):
    
    # For webcam input
    cap = cv2.VideoCapture(0)
    most_diverging_angle_value = None
    most_diverging_angle_idx = None
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        
        repetitions = 0
        first_half = True
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print ('Ignoring empty camera frame.')
                # If loading a video, use 'break' instead of 'continue'
                continue
        
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Draw the status text and pose annotation on the image
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Analysis
            # The flips at the beginning and end are done to not interfere with the
            # final image flip for the selfie-view, or else the text will be flipped
            image = cv2.flip(image, 1)

            if first_half:
                if results.pose_landmarks:
                    progress, (correctness, most_diverging_angle_value, most_diverging_angle_idx) = \
                        model_results(results.pose_landmarks.landmark, model_fh, angles_to_use)

                    image_status(image, f'C: {correctness:.4%} | P: {progress:.4}', repetitions, first_half, correctness > 0.5 and progress < 0.2)

                    if progress > 0.90 and correctness > 0.5:
                        first_half = not first_half
                else:
                    image_status(image, f'ERROR: no results', repetitions, first_half)

            else:
                if results.pose_landmarks:
                    progress, (correctness, most_diverging_angle_value, most_diverging_angle_idx) = \
                        model_results(results.pose_landmarks.landmark, model_sh, angles_to_use)
                    
                    image_status(image, f'C: {correctness:.4%} | P: {progress:.4}', repetitions, first_half)

                    if progress > 0.90 and correctness > 0.5:
                        first_half = not first_half
                        repetitions += 1
                else:
                    image_status(image, f'ERROR: no results', repetitions, first_half)

            image = cv2.flip(image, 1)

            # Draw the landmarks, as well as the most diverging angle
            pose_landmarks_style = mp_drawing_styles.get_default_pose_landmarks_style()
            for angle_landmarks in angles_to_use:
                for landmark in angle_landmarks:
                    pose_landmarks_style[landmark] = \
                        mp_drawing_styles.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=2)
            if most_diverging_angle_idx and most_diverging_angle_value > 10:
                pose_landmarks_style[ get_landmarks_from_angle(most_diverging_angle_idx, exercise_angles)[1] ] = \
                    mp_drawing_styles.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=3)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=pose_landmarks_style)

            # Flip the image horizontally for a selfie-view display
            image = cv2.flip(image, 1)

            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        prog='data_explore_knn',
        description=description
    )

    parser.add_argument('exercise',
        type=str,
        help='The name of the exercise of which reference data will be used.'
    )

    parser.add_argument('-a', '--angles',
        type=str,
        help='''The name of the exercise specifying the set of angles that should be used for testing. By default, use all angles.
        Should ideally match the name of the exercise of which data will be created, if possible.''')
    
    parser.add_argument('-i', '--input-folder',
        type=str,
        required=True,
        help='The folder from which the created reference data will be obtained.')

    args = parser.parse_args()
    exercise = args.exercise
    exercise_angles = args.angles
    data_folder = args.input_folder

    angles_to_use = obtain_angles(exercise_angles)

    # Train model with captured data
    reference_data_fh = []
    reference_data_sh = []
    progress_data_fh = []
    progress_data_sh = []
    for file in os.scandir(data_folder):
        if file.is_file():
            with open(file.path, 'rt') as reference_file:
                for reference in json.load(reference_file):
                    if reference['exercise'] == exercise:
                        if reference['first_half']:
                            reference_data_fh.append(
                                    landmark_list_angles([LandmarkData(**landmark) for landmark in reference['landmarks']], angles=angles_to_use))
                            progress_data_fh.append(reference['progress'])
                        else:
                            reference_data_sh.append(
                                    landmark_list_angles([LandmarkData(**landmark) for landmark in reference['landmarks']], angles=angles_to_use))
                            progress_data_sh.append(reference['progress'])

    # One model for each half of the exercise 
    model_fh = KNNRegressor(reference_data_fh, progress_data_fh)
    model_sh = KNNRegressor(reference_data_sh, progress_data_sh)

    camera_loop(
            model_fh=model_fh,
            model_sh=model_sh,
            angles_to_use=angles_to_use,
            exercise_angles=exercise_angles,
    )