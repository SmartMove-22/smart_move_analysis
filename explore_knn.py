import cv2
import mediapipe as mp

from smart_move_analysis.knn import KNNRegressor
from smart_move_analysis.landmark_analysis import landmark_list_angles
from numpy import linspace

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# IntelliSense
mp_drawing: mp.solutions.mediapipe.python.solutions.drawing_utils
mp_drawing_styles = mp.solutions.mediapipe.python.solutions.drawing_styles
mp_pose: mp.solutions.mediapipe.python.solutions.pose

def image_status(image, string, counter):
    cv2.putText(image, f'Counter: {counter:>3} | ' + string, (0, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(247, 224, 47), thickness=2)

# For webcam input
cap = cv2.VideoCapture(0)
reference_data = []
model = None
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
        if counter < 100:
            image_status(image, f'Prepare for training', counter)
        elif counter < 200:
            if results.pose_landmarks:
                image_status(image, f'Collecting data', counter)
                reference_data.append( landmark_list_angles(results.pose_landmarks.landmark, d2=True) )
            else:
                image_status(image, f'ERROR: no results', counter)
                counter -= 1
        elif counter < 201:
            image_status(image, f'Training model', counter)
            model = KNNRegressor(reference_data, linspace(0.0, 1.0, len(reference_data)))
        elif counter < 400:
            if results.pose_landmarks:
                angles = landmark_list_angles(results.pose_landmarks.landmark, d2=True)
                image_status(image, f'C: {model.correctness(angles):.4} | P: {model.progress(angles)}', counter)
            else:
                image_status(image, f'ERROR: no results', counter)
                counter -= 1
        else:
            counter = 0
            reference_data.clear()
            model = None

        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
        
        counter += 1

cap.release()