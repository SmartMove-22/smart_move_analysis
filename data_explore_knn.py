import json
import os
import cv2
import mediapipe as mp

from numpy import linspace
from knn import KNNRegressor
from utils import landmark_list_angles, get_landmarks_from_angle
from reference_store import LandmarkData

'''
Script to quickly and easily explore the KNN model
'''

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# IntelliSense
mp_drawing: mp.solutions.mediapipe.python.solutions.drawing_utils
mp_drawing_styles = mp.solutions.mediapipe.python.solutions.drawing_styles
mp_pose: mp.solutions.mediapipe.python.solutions.pose

def image_status(image, string, counter):
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
    cv2.putText(image, f'Counter: {counter:>3}', (0, 50), **outline_style)
    cv2.putText(image, f'Counter: {counter:>3}', (0, 50), **text_style)

# Train model with data from the 'data' folder
EXERCISE = 'squat'
for file in os.scandir('data'):
    reference_data = []
    progress_data = []
    if file.is_file():
        with open(file.path, 'rt') as reference_file:
            for reference in json.load(reference_file):
                reference_data.append([LandmarkData(**landmark) for landmark in reference['landmarks']])
                progress_data.append(reference['progress'])
            
model = KNNRegressor(reference_data, progress_data)

# For webcam input
cap = cv2.VideoCapture(0)
reference_data = []
most_diverging_angle_value = None
most_diverging_angle_idx = None
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

        # Draw the status text and pose annotation on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Analysis
        # The flips at the beginning and end are done to not interfere with the
        # final image flip for the selfie-view, or else the text will be flipped
        image = cv2.flip(image, 1)
        if results.pose_landmarks:
            angles = landmark_list_angles(results.pose_landmarks.landmark, d2=True)
            correctness, most_diverging_angle_value, most_diverging_angle_idx = model.correctness(angles)

            image_status(image, f'C: {correctness:.4} | P: {model.progress(angles):.4}', counter)
        else:
            image_status(image, f'ERROR: no results', counter)
            counter -= 1
        image = cv2.flip(image, 1)

        # Draw the landmarks, as well as the most diverging angle
        pose_landmarks_style = mp_drawing_styles.get_default_pose_landmarks_style()
        if most_diverging_angle_idx and most_diverging_angle_value > 10:
            pose_landmarks_style[ get_landmarks_from_angle(most_diverging_angle_idx)[1] ] = \
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
        
        counter += 1

cap.release()