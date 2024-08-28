import tkinter as tk 
import customtkinter as ck 
import pandas as pd 
import numpy as np 
import pickle 
import mediapipe as mp
import cv2
from PIL import Image, ImageTk 
from landmarks import landmarks  # Ensure this is correctly imported
from collections import deque

# Initialize Tkinter window
window = tk.Tk()
window.geometry("800x700")  # Increased width to ensure full frame is visible
window.title("Rep Counter") 
ck.set_appearance_mode("dark")

# Labels
classLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10)
classLabel.place(x=10, y=1)
classLabel.configure(text='STAGE') 
counterLabel = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10)
counterLabel.place(x=160, y=1)
counterLabel.configure(text='REPS') 
probLabel  = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10)
probLabel.place(x=300, y=1)
probLabel.configure(text='PROB') 
classBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
classBox.place(x=10, y=41)
classBox.configure(text='0') 
counterBox = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
counterBox.place(x=160, y=41)
counterBox.configure(text='0') 
probBox = ck.CTkLabel(window, height=40, width=120, text_color="white", fg_color="blue")
probBox.place(x=300, y=41)
probBox.configure(text='0') 

# Reset counter
def reset_counter(): 
    global counter
    counter = 0 

button = ck.CTkButton(window, text='RESET', command=reset_counter, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
button.place(x=10, y=600)

# Frame for video feed
frame = tk.Frame(height=480, width=640)  # Adjusted width for full view
frame.place(x=10, y=90) 
lmain = tk.Label(frame) 
lmain.place(x=0, y=0) 

# Mediapipe initialization
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.7, min_detection_confidence=0.7)  # Increased confidence thresholds

# Load the new scaler and model
with open('scaler_deadlift.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
with open('random_forest_model_deadlift.pkl', 'rb') as model_file:
    loaded_rf = pickle.load(model_file)

# Initialize video capture from video file
video_path = '/Users/soheilsaneei/Desktop/Fitness App/Me_doing_deadlifts.mov'  # Change this to the path of your video file
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height

if not cap.isOpened():
    print("Error: Could not open video capture.")
else:
    print("Video capture opened successfully.")

bounding_box = {
    'x_min': 100,   # Adjusted x_min
    'y_min': 50,   # Adjusted y_min
    'x_max': 540,   # Adjusted x_max
    'y_max': 430    # Adjusted y_max
}

current_stage = ''
counter = 0 
bodylang_prob = np.array([0,0]) 
bodylang_class = '' 

# Buffer for state transitions
state_buffer = deque(maxlen=5)  # Buffer size of 5 frames

def detect(): 
    global current_stage
    global counter
    global bodylang_class
    global bodylang_prob 
    global state_buffer

    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image or end of video reached")
        window.after(10, detect)
        return

    # Check if frame is valid
    if frame is None or frame.size == 0:
        print("Empty frame received")
        window.after(10, detect)
        return

    # Maintain aspect ratio
    h, w, _ = frame.shape
    aspect_ratio = w / h
    if aspect_ratio > 1:
        new_width = 640
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = 480
        new_width = int(new_height * aspect_ratio)
    frame = cv2.resize(frame, (new_width, new_height))

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    results = pose.process(image)

    if results.pose_landmarks:
        is_within_bbox = all(
            bounding_box['x_min'] <= int(lm.x * frame.shape[1]) <= bounding_box['x_max'] and
            bounding_box['y_min'] <= int(lm.y * frame.shape[0]) <= bounding_box['y_max']
            for lm in results.pose_landmarks.landmark
        )

        if is_within_bbox:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                mp_drawing.DrawingSpec(color=(106,13,173), thickness=4, circle_radius = 5), 
                mp_drawing.DrawingSpec(color=(255,102,0), thickness=5, circle_radius = 10)) 

            try: 
                row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
                X = pd.DataFrame([row], columns = landmarks) 
                X = scaler.transform(X)  # Standardize the real-time data
                bodylang_prob = loaded_rf.predict_proba(X)[0]
                bodylang_class = loaded_rf.predict(X)[0] 

                print(f"Class: {bodylang_class}, Probability: {bodylang_prob[bodylang_prob.argmax()]}")

                # Add current classification to buffer
                state_buffer.append(bodylang_class)

                # Use the most frequent state in the buffer
                if len(state_buffer) == state_buffer.maxlen:
                    most_common_state = max(set(state_buffer), key=state_buffer.count)

                    if most_common_state == "down" and bodylang_prob[bodylang_prob.argmax()] > 0.7: 
                        current_stage = "down" 
                    elif current_stage == "down" and most_common_state == "up" and bodylang_prob[bodylang_prob.argmax()] > 0.6:
                        current_stage = "up" 
                        counter += 1 

            except Exception as e: 
                print(f"Error during detection: {e}") 

    img = image[:, :, :]  # Ensure full image is used
    imgarr = Image.fromarray(img) 
    imgtk = ImageTk.PhotoImage(imgarr) 
    lmain.imgtk = imgtk 
    lmain.configure(image=imgtk)
    lmain.after(10, detect)  

    counterBox.configure(text=counter) 
    probBox.configure(text=f"{bodylang_prob[bodylang_prob.argmax()]:.2f}") 
    classBox.configure(text=current_stage) 

print("Starting detection...")
detect() 
window.mainloop()
