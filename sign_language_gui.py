import tkinter as tk
from tkinter import ttk, font
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
import time
import threading

# Load the model
model = tf.keras.models.load_model('model3.h5')

# Setup MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Define the actions
DATA_PATH = os.path.join('Action_Dataset')
actions = np.array([folder for folder in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, folder))])


# MediaPipe detection function
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False 
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Drawing landmarks function
def draw_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
    
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                              mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
    
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                              mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
    
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                              mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))

# Extract keypoints function
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, left_hand, right_hand])

# Probability visualization function
def prob_viz(res, actions, input_frame):
    output_frame = input_frame.copy()
    top_idx = np.argmax(res)
    prob = res[top_idx]
    color = (245, 117, 16)  # Single color for all classes
    cv2.rectangle(output_frame, (0, 60), (int(prob * 100), 90), color, -1)
    cv2.putText(output_frame, actions[top_idx], (0, 85), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame

# Function to run the sign language detection
def detection_thread():
    global running
    
    # Set capture device
    cap = cv2.VideoCapture(0)
    
    # Detection variables
    sequence = []     # collect frames to generate prediction
    sentence = []     # prediction history
    predictions = []
    threshold = 0.95  # confidence threshold
    
    # Start detection loop
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while running and cap.isOpened():
            # Read feed
            ret, frame = cap.read()
        
            # Make detection
            image, results = mediapipe_detection(frame, holistic)
            print(results)

            # Draw landmarks
            draw_landmarks(image, results)

            #make prediction
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))

                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:            
                            if actions[np.argmax(res)] != sentence[-1]:    #check if not same action 
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])
        
                if len(sentence) > 5:
                    sentence = sentence[-5:]
        
                #visualization probability
                image = prob_viz(res, actions, image)
            
            #rendering text
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Show to screen
            cv2.imshow('BIM Detection', image)
            
            # Break loop
            if cv2.waitKey(10) & 0xFF == ord('q'):
                running = False
                break
                
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    # Update UI
    root.after(0, update_ui_after_stop)

# Function to start detection in a separate thread
def run_detection():
    global running, detection_thread_obj
    
    if running:
        return
    
    # Update UI
    start_button.config(state="disabled")
    status_label.config(text="Running... (Press 'Q' to stop)")
    
    # Start detection in a separate thread
    running = True
    detection_thread_obj = threading.Thread(target=detection_thread)
    detection_thread_obj.daemon = True
    detection_thread_obj.start()

# Function to stop detection
def stop_detection():
    global running
    running = False
    status_label.config(text="Stopping...")

# Function to update UI after stopping
def update_ui_after_stop():
    start_button.config(state="normal")
    status_label.config(text="Ready")

# Function to handle key press events
def on_key_press(event):
    # Check if Q or q was pressed
    if event.char.lower() == 'q' and running:
        stop_detection()

# Create the main application window
root = tk.Tk()
root.title("BIM Recognition System")
root.geometry("450x300")
root.configure(bg="#f0f0f0")

# Bind key press event to the root window
root.bind('<Key>', on_key_press)

# Set global style
style = ttk.Style()
style.configure("TButton", font=("Arial", 14), padding=10)

# Create header
header_font = font.Font(family="Arial", size=18, weight="bold")
header = tk.Label(root, text="BIM Recognition System", font=header_font, bg="#f0f0f0")
header.pack(pady=20)

# Create instruction text
instructions = tk.Label(
    root, 
    text="Press 'Start' to begin sign language detection.\nPress the 'Q' key to stop detection.\nPress 'Quit' to exit the application.",
    font=("Arial", 12),
    bg="#f0f0f0",
    justify="center"
)
instructions.pack(pady=10)

# Create button frame
button_frame = tk.Frame(root, bg="#f0f0f0")
button_frame.pack(pady=20)

# Create Start button
start_button = ttk.Button(button_frame, text="Start", command=run_detection)
start_button.pack(side="left", padx=10)

# Create Quit button
quit_button = ttk.Button(button_frame, text="Quit", command=root.destroy)
quit_button.pack(side="left", padx=10)

# Create status label
status_label = tk.Label(root, text="Ready", font=("Arial", 10), bg="#f0f0f0")
status_label.pack(pady=10)

# Initialize global variables
running = False
detection_thread_obj = None

if __name__ == "__main__":
    root.mainloop() 

