import csv
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

# Initialize columns for CSV
columns = ["class"]
for val in range(1, 34):
    columns.extend([f"x{val}", f"y{val}", f"z{val}", f"v{val}"])

# Create the CSV file and write headers
with open('coords.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(columns)

def export_landmark(results, action, csv_writer):
    if results.pose_landmarks:
        # Flatten the list of keypoints
        keypoints = [action]  # Start with the action label
        for res in results.pose_landmarks.landmark:
            keypoints.extend([res.x, res.y, res.z, res.visibility])
        csv_writer.writerow(keypoints)
    else:
        print("No landmarks to export.")

# Setup mediapipe instance
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Setup video capture
cap = cv2.VideoCapture('/Users/soheilsaneei/Desktop/Exercise Library/Deadlift_1.MOV')

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, open('coords.csv', mode='a', newline="") as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    frames_to_capture = 20  # Number of frames to capture after each keypress
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            print("Empty frame received. End of video?")
            break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))   
        
        cv2.imshow('Pose Estimation', image)

        k = cv2.waitKey(1)
        if k in [117, 100]:  # ASCII for 'u' and 'd'
            action = "up" if k == 117 else "down"
            for _ in range(frames_to_capture):
                ret, image = cap.read()
                if not ret:
                    break
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                export_landmark(results, action, csv_writer)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Load the CSV data into a DataFrame
df = pd.read_csv('coords.csv')
excel_path = '/Users/soheilsaneei/Desktop/Fitness App/Deadlift1.xlsx'
df.to_excel(excel_path, index=False)

print(f"Landmarks have been saved to Excel at {excel_path}")