import argparse
import json
import os
import time
from enum import Enum

import cv2
import mediapipe as mp

from .utils import ANGLES_OF_INTEREST_IDX, landmark_list_angles

description = '''
Script for creating exercise reference data
'''

class DataStage(Enum):
    WAITING = 0
    ABOUT_TO_CAPTURE = 1
    CAPTURE_FIRST_HALF = 2
    FIRST_HALF_WAIT = 3
    CAPTURE_SECOND_HALF = 4
    SECOND_HALF_WAIT = 5

# The number of frames to capture for each relevant stage
# Each exercise's half should be done throughout all of its respective frames
NFRAMES = {
    'ABOUT_TO_CAPTURE': 50,
    'FIRST_HALF': 50,
    'FIRST_HALF_WAIT': 1,
    'SECOND_HALF': 50,
    'SECOND_HALF_WAIT': 1,
}

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# IntelliSense
mp_drawing: mp.solutions.mediapipe.python.solutions.drawing_utils
mp_drawing_styles = mp.solutions.mediapipe.python.solutions.drawing_styles
mp_pose: mp.solutions.mediapipe.python.solutions.pose


def image_status(image: cv2.Mat, string: str, counter: int, cap: int, exercise: str):
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
    cv2.putText(image, f'Exercise: {exercise} | Counter: {counter:>3}/{cap:>3}', (0, 50), **outline_style)
    cv2.putText(image, f'Exercise: {exercise} | Counter: {counter:>3}/{cap:>3}', (0, 50), **text_style)

    
def landmark_to_dict(landmark):
    return {
        "x": landmark.x,
        "y": landmark.y,
        "z": landmark.z,
        "visibility": landmark.visibility
    }


def camera_loop(
        exercise: str,
        data_folder: str):
    # For webcam input
    cap = cv2.VideoCapture(0)
    reference_data = []
    reference_data_temp = []
    data_stage = DataStage.WAITING
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        
        counter = 0
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

            # Draw the pose annotation on the image
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # Flip the image horizontally for a selfie-view display
            image = cv2.flip(image, 1)

            # Analysis
            if data_stage == DataStage.WAITING:
                if results.pose_landmarks:
                    image_status(image, f'Flex your elbows to start capturing data', counter, 0, exercise)
                    landmark_angles = landmark_list_angles(results.pose_landmarks.landmark, d2=True)
                    if landmark_angles[ANGLES_OF_INTEREST_IDX.LeftWrist_LeftElbow_LeftShoulder] < 20 \
                            and landmark_angles[ANGLES_OF_INTEREST_IDX.RightWrist_RightElbow_RightShoulder] < 20:
                        data_stage = DataStage.ABOUT_TO_CAPTURE
                else:
                    image_status(image, f'ERROR: no results', counter, 0, exercise)
            
            elif data_stage == DataStage.ABOUT_TO_CAPTURE:
                image_status(image, f'About to capture data, prepare yourself', counter, NFRAMES['ABOUT_TO_CAPTURE'], exercise)
                
                counter += 1
                if counter >= NFRAMES['ABOUT_TO_CAPTURE']:
                    data_stage = DataStage.CAPTURE_FIRST_HALF
                    counter = 0
            
            elif data_stage == DataStage.CAPTURE_FIRST_HALF:
                if results.pose_landmarks:
                    image_status(image, f'Capturing first half of exercise', counter, NFRAMES['FIRST_HALF'], exercise)

                    reference_data_temp.append({
                        "first_half": True,
                        "exercise": exercise,
                        "progress": counter/NFRAMES['FIRST_HALF'],
                        "landmarks": [landmark_to_dict(landmark) for landmark in results.pose_landmarks.landmark],
                        "landmarks_world": [landmark_to_dict(landmark) for landmark in results.pose_world_landmarks.landmark]
                    })

                    counter += 1
                    if counter >= NFRAMES['FIRST_HALF']:
                        reference_data.extend(reference_data_temp)
                        reference_data_temp.clear()
                        data_stage = DataStage.FIRST_HALF_WAIT
                        counter = 0

                else:
                    image_status(image, f'ERROR: no landmarks detected, resetting to previous stage', counter, NFRAMES['FIRST_HALF'], exercise)
                    reference_data_temp.clear()
                    data_stage = DataStage.ABOUT_TO_CAPTURE
                    counter = 0
            
            elif data_stage == DataStage.FIRST_HALF_WAIT:
                image_status(image, f'First half captured, about to capture second', counter, NFRAMES['FIRST_HALF_WAIT'], exercise)

                counter += 1
                if counter >= NFRAMES['FIRST_HALF_WAIT']:
                    data_stage = DataStage.CAPTURE_SECOND_HALF
                    counter = 0
            
            elif data_stage == DataStage.CAPTURE_SECOND_HALF:
                if results.pose_landmarks:
                    image_status(image, f'Capturing second half of exercise', counter, NFRAMES['SECOND_HALF'], exercise)

                    reference_data_temp.append({
                        "first_half": False,
                        "exercise": exercise,
                        "progress": counter/NFRAMES['SECOND_HALF'],
                        "landmarks": [landmark_to_dict(landmark) for landmark in results.pose_landmarks.landmark],
                        "landmarks_world": [landmark_to_dict(landmark) for landmark in results.pose_world_landmarks.landmark]
                    })

                    counter += 1
                    if counter >= NFRAMES['SECOND_HALF']:
                        reference_data.extend(reference_data_temp)
                        reference_data_temp.clear()
                        data_stage = DataStage.SECOND_HALF_WAIT
                        counter = 0
                
                else:
                    image_status(image, f'ERROR: no landmarks detected, resetting to previous stage', counter, NFRAMES['SECOND_HALF'], exercise)
                    reference_data_temp.clear()
                    data_stage = DataStage.FIRST_HALF_WAIT
                    counter = 0
            
            elif data_stage == DataStage.SECOND_HALF_WAIT:
                image_status(image, f'Second half captured', counter, NFRAMES['SECOND_HALF_WAIT'], exercise)

                counter += 1
                if counter >= NFRAMES['SECOND_HALF_WAIT']:
                    data_stage = DataStage.WAITING
                    counter = 0

            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(5) & 0xFF == 27:
                if reference_data:
                    with open(os.path.join(data_folder, f'capture_{int(time.time())}.json'), 'wt') as reference_data_file:
                        json.dump(reference_data, reference_data_file)
                break
            

    cap.release()



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        prog='stream_data_creator',
        description=description
    )

    parser.add_argument('exercise',
        type=str,
        help='The name of the exercise of which reference data will be created.'
    )

    parser.add_argument('-o', '--output-folder',
        type=str,
        required=True,
        help='The folder to which the created reference data will be saved.')

    args = parser.parse_args()
    exercise = args.exercise
    data_folder = args.output_folder
    
    if not os.path.isdir(data_folder):
        os.mkdir(data_folder)

    camera_loop(
            exercise=exercise,
            data_folder=data_folder)
