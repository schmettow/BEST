import cv2 # OpenCV for video capture and face detection
import datetime # For timestamping emotion recordings
import pandas as pd # For handling CSV files and storing emotions
import os # For file hangling
import signal # For handling Ctrl+C
import sys # For exiting the program
from deepface import DeepFace # For facial emotion detection
from collections import deque # For batching emotion data

# Ask for Participant ID
participant_id = input("Enter Participant ID: ")

# Loads the pre-trained Haar Cascade model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Defines CSV filename for storing emotion data
csv_filename = "facial_emotions.csv"
if not csv_filename.endswith(".csv"):
    csv_filename += ".csv"

# Checks if CSV file exists, if not, creates it
if not os.path.exists(csv_filename):
    df = pd.DataFrame(columns=["Timestamp", "Participant_ID", "Emotion"])
    df.to_csv(csv_filename, index=False, encoding='utf-8')

# Loads existing CSV (prevents overwriting)
df = pd.read_csv(csv_filename)

# Initialises video capture from webcam (0 = default webcam)
cap = cv2.VideoCapture(0)
if not cap.isOpened(): # Checks if webcam is available, if not exits the program
    print("Error: Could not open webcam.")
    sys.exit()

# Creates a queue with a maximum length of 5 for batching emotion data before writing to CSV
video_data_queue = deque(maxlen=5)

# Tracks the last timestamp an emotion was recorded (for 1-second interval)
last_recorded_time = datetime.datetime.now()

# Defines conotation of emotions for visualisation (color coding)
positive_emotions = ["happy", "surprise"] # Positive emotions
stress_related_emotions = ["angry", "fear", "sad", "disgust"]  # Stress-related emotions
neutral_emotions = ["neutral"] # Neutral emotions

# Initialises a variable to store the currently detected emotion
current_emotion = None

# Handles program termination (Ctrl+C)
def signal_handler(sig, frame):
    print("\nShutting down...")
    cap.release()
    cv2.destroyAllWindows()
    # Saves any remaining data in the queue before exiting
    if video_data_queue:
        df_batch = pd.DataFrame(video_data_queue, columns=["Timestamp", "Participant_ID", "Emotion"])
        df_batch.to_csv(csv_filename, mode='a', header=False, index=False)
    sys.exit(0)

# Links the signal handler to the SIGINT signal (Ctrl+C) for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)

print("Press 'q' to quit.")

while True: # Starts an infinite loop to process video frames continuously
    ret, frame = cap.read() # Reads a frame from the webcam
    if not ret: # Checks if the frame was read successfully, if not, it exits the loop
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Converts the frame to grayscale for better face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)) # Detects faces in the frame

    if len(faces) > 0: # Checks if at least one face is detected
        (x, y, w, h) = faces[0] # Gets the coordinates of the first detected face
        face_region = frame[y:y+h, x:x+w] # Extracts the face region from the frame

        # Check if it's time to record emotion
        current_time = datetime.datetime.now()
        if (current_time - last_recorded_time).total_seconds() >= 1: # Ensures emotions are recorded only once per second
            last_recorded_time = current_time

            try:
                analysis = DeepFace.analyze(face_region, actions=['emotion'], enforce_detection=False) # Uses DeepFace to analyse the emotions in the detected face
                current_emotion = analysis[0]['dominant_emotion'] if analysis else "Unknown" # Extracts the dominant emotion detected by DeepFace
            except Exception as e:
                print(f"DeepFace Error: {e}")
                current_emotion = "Error"

            # Stores the detected emotio along with the timestamp and participant ID in the queue
            video_data_queue.append([current_time.strftime("%Y-%m-%d %H:%M:%S"), participant_id, current_emotion])

            # Writes the stored emotions to the CSV file if the queue is full (5 emotions)
            if len(video_data_queue) == 5:
                df_batch = pd.DataFrame(video_data_queue, columns=["Timestamp", "Participant_ID", "Emotion"])
                df_batch.to_csv(csv_filename, mode='a', header=False, index=False)
                video_data_queue.clear()

        # Sets rectangle color based on emotion detected
        if current_emotion in positive_emotions:
            rectangle_color = (0, 255, 0)  # Green for happiness and surprise
        elif current_emotion in stress_related_emotions:
            rectangle_color = (0, 0, 255)  # Red for stress-related emotions (anger, fear, sadness, disgust)
        else:
            rectangle_color = (0, 255, 255)  # Yellow for neutral or others

        # Draws rectangle around the face continuously
        cv2.rectangle(frame, (x, y), (x + w, y + h), rectangle_color, 2)

        # Displays the detected emotion above the face
        if current_emotion:
            cv2.putText(frame, current_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Shows the video feed with the detected face and emotion
    cv2.imshow("Video Feed", frame)

    # Allows the user to quit the program by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nThe model has quit.")
        break

# Ensures proper shutdown when the program exits
signal_handler(None, None)

