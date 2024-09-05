import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
import tkinter as tk 
import customtkinter as ck 
import pandas as pd 
import numpy as np 
import pickle 
import mediapipe as mp
import cv2
from PIL import Image, ImageTk 
from landmarks import landmarks
from collections import deque
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

# Initialize Tkinter window
window = tk.Tk()
window.geometry("800x720")  # Increase height to fit all elements
window.title("Rep Counter") 
ck.set_appearance_mode("dark")
window.configure(bg="lightgray")

# Initialize labels and buttons
def init_ui():
    labels = [
        ("STAGE", 10), ("REPS", 160), ("PROB", 300)
    ]
    for text, x in labels:
        label = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="black", padx=10)
        label.place(x=x, y=1)
        label.configure(text=text)
        
        box = ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
        box.place(x=x, y=41)
        box.configure(text='0')
        
    global classBox, counterBox, probBox, feedback_text, feedback_frame
    classBox, counterBox, probBox = [ck.CTkLabel(window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue") for _ in range(3)]
    classBox.place(x=10, y=41)
    counterBox.place(x=160, y=41)
    probBox.place(x=300, y=41)
    
    reset_button = ck.CTkButton(window, text='RESET', command=reset_counter, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue")
    reset_button.place(x=10, y=630)
    
    replay_button = ck.CTkButton(window, text='REPLAY ERRORS', command=replay_incorrect_frames, height=40, width=180, font=("Arial", 20), text_color="white", fg_color="red")
    replay_button.place(x=160, y=630)

    global feedback_text
    feedback_frame = ck.CTkFrame(window, width=780, height=120, fg_color="red")  # Change to a distinctive color
    feedback_frame.place(x=10, y=580)  # Adjust y-coordinate to be below the video frame
    feedback_frame.pack_propagate(False)  # Prevent the frame from shrinking

    feedback_text = ck.CTkTextbox(feedback_frame, width=760, height=110, font=("Arial", 14), text_color="white", fg_color="blue", wrap="word")
    feedback_text.pack(fill="both", expand=True, padx=10, pady=5)
    feedback_text.insert("1.0", "LLM feedback will appear here")
    feedback_text.configure(state="disabled")  # Make it read-only

    # Force update the window to ensure all elements are drawn
    window.update_idletasks()

    # Print debug information
    print(f"Window size: {window.winfo_width()}x{window.winfo_height()}")
    print(f"Feedback frame: x={feedback_frame.winfo_x()}, y={feedback_frame.winfo_y()}, width={feedback_frame.winfo_width()}, height={feedback_frame.winfo_height()}")
    print(f"Feedback text: x={feedback_text.winfo_x()}, y={feedback_text.winfo_y()}, width={feedback_text.winfo_width()}, height={feedback_text.winfo_height()}")

    # Print information about all widgets
    for widget in window.winfo_children():
        print(f"Widget: {widget}, Position: x={widget.winfo_x()}, y={widget.winfo_y()}, width={widget.winfo_width()}, height={widget.winfo_height()}")

# Function to update feedback text
def update_feedback(text):
    feedback_text.configure(state="normal")
    feedback_text.delete("1.0", tk.END)
    feedback_text.insert("1.0", text)
    feedback_text.configure(state="disabled")

# Initialize video frame
def init_video_frame():
    frame = tk.Frame(height=480, width=640)
    frame.place(x=10, y=90) 
    global lmain
    lmain = tk.Label(frame) 
    lmain.place(x=0, y=0)

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.7, min_detection_confidence=0.7)

# Load model and scaler
with open('scaler_deadlift.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('random_forest_model_deadlift.pkl', 'rb') as f:
    loaded_rf = pickle.load(f)

# Initialize video capture
video_path = '/Users/soheilsaneei/Desktop/Fitness App/Me_doing_deadlifts.mov'
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize Anthropic client
anthropic = Anthropic(api_key='EnterYourAPIKEY')

# Global variables
current_stage = ''
counter = 0 
bodylang_prob = np.array([0,0]) 
bodylang_class = '' 
state_buffer = deque(maxlen=5)
incorrect_frames = []
form_issues = {"back_straight": 0}
bounding_box = {'x_min': 100, 'y_min': 50, 'x_max': 540, 'y_max': 430}

def reset_counter(): 
    global counter
    counter = 0 

def replay_incorrect_frames():
    for frame in incorrect_frames:
        cv2.imshow("Incorrect Rep", frame)
        cv2.waitKey(500)

def get_llm_feedback(rep_count, form_issues, current_stage):
    print(f"Generating feedback for rep {rep_count}")  # Debug print
    try:
        prompt = f"{HUMAN_PROMPT}I'm analyzing a deadlift exercise. Here's the current status:\n" \
                 f"- Completed reps: {rep_count}\n" \
                 f"- Current stage: {current_stage}\n" \
                 f"- Form issues: {form_issues}\n\n" \
                 f"Based on this information, can you provide:\n" \
                 f"1. A brief assessment of the performance\n" \
                 f"2. Specific feedback on form\n" \
                 f"3. A suggestion for improvement\n\n" \
                 f"Please keep your response concise, within 2-3 sentences for each point.{AI_PROMPT}"

        response = anthropic.completions.create(
            model="claude-2.0",
            max_tokens_to_sample=300,
            prompt=prompt
        )
        return response.completion.strip()
    except Exception as e:
        print(f"Error in get_llm_feedback: {e}")
        return f"Test feedback for rep {rep_count}. Form issues: {form_issues}. Current stage: {current_stage}"

def detect():
    print("Detect function called")  # Debug print
    global current_stage, counter, bodylang_class, bodylang_prob, state_buffer, incorrect_frames, form_issues

    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        print("End of video reached. Closing application.")
        window.quit()  # This will close the Tkinter window
        return

    # Resize frame
    h, w, _ = frame.shape
    aspect_ratio = w / h
    new_width = 640 if aspect_ratio > 1 else int(480 * aspect_ratio)
    new_height = int(new_width / aspect_ratio) if aspect_ratio > 1 else 480
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
                X = scaler.transform(X)
                bodylang_prob = loaded_rf.predict_proba(X)[0]
                bodylang_class = loaded_rf.predict(X)[0] 

                state_buffer.append(bodylang_class)

                if len(state_buffer) == state_buffer.maxlen:
                    most_common_state = max(set(state_buffer), key=state_buffer.count)

                    if most_common_state == "down" and bodylang_prob[bodylang_prob.argmax()] > 0.7: 
                        current_stage = "down" 
                    elif current_stage == "down" and most_common_state == "up" and bodylang_prob[bodylang_prob.argmax()] > 0.6:
                        current_stage = "up" 
                        counter += 1
                        print(f"Rep completed. Counter: {counter}")  # Debug print
                        
                        # Get LLM feedback every 5 reps or when form issues are detected
                        if counter % 5 == 0 or form_issues["back_straight"] > 0:
                            print(f"Attempting to generate feedback. Counter: {counter}, Form issues: {form_issues}")  # Debug print
                            llm_feedback = get_llm_feedback(counter, form_issues, current_stage)
                            
                            # Print LLM feedback to console
                            print("LLM Feedback:")
                            print(llm_feedback)
                            print("-" * 50)  # Separator for readability

                            # Display feedback in GUI
                            update_feedback(llm_feedback)
                            window.update_idletasks()  # Force update
                            
                            # Reset form issues after getting feedback
                            form_issues = {"back_straight": 0}

                # Form Analysis
                shoulder_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
                hip_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y
                if shoulder_y > hip_y:
                    feedback_text = "Keep your back straight!"
                    incorrect_frames.append(frame.copy())
                    form_issues["back_straight"] += 1
                    update_feedback(feedback_text)
                    window.update_idletasks()

            except Exception as e: 
                print(f"Error during detection: {e}") 

    img = Image.fromarray(image)
    imgtk = ImageTk.PhotoImage(img) 
    lmain.imgtk = imgtk 
    lmain.configure(image=imgtk)
    lmain.after(10, detect)  

    counterBox.configure(text=counter) 
    probBox.configure(text=f"{bodylang_prob[bodylang_prob.argmax()]:.2f}") 
    classBox.configure(text=current_stage) 

if __name__ == "__main__":
    init_ui()
    init_video_frame()
    detect()
    window.mainloop()
