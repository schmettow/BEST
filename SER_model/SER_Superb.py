import pyaudio # PyAudio for capturing audio from the microphone
import torch # PyTorch for deep learning 
import datetime # For timestamping emotion recordings
import pandas as pd # For handling CSV files and storing emotions
import os # For file hangling
import signal # For handling Ctrl+C
import sys # For exiting the program
import numpy as np # Numpy for numerical operations on audio data
import cv2 # OpenCV for visualisation
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification # For importing the Superb model from the Hugging Face library
from collections import deque # For batching emotion data

# Asks for Participant ID
participant_id = input("Enter Participant ID: ")

# Initialises PyAudio for audio capture
p = pyaudio.PyAudio()
RATE = 16000
CHUNK = 8000
CHANNELS = 1
FORMAT = pyaudio.paFloat32
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK) # Opens an audio stream for capturing audio data from the microphone 

# Defines CSV filename for storing emotion data
csv_filename = "speech_emotions.csv"
if not os.path.exists(csv_filename):
    df = pd.DataFrame(columns=["Timestamp", "Participant_ID", "Emotion"])
    df.to_csv(csv_filename, index=False, encoding='utf-8')

# Loads the Superb model for speech emotion recognition
model = Wav2Vec2ForSequenceClassification.from_pretrained("Superb/wav2vec2-base-superb-er")
processor = Wav2Vec2FeatureExtractor.from_pretrained("Superb/wav2vec2-base-superb-er") # Loads the feature extractor for preprocessing audio input
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Uses GPU if available, if not uses CPU
model = model.to(device) # Moves the model to the selected device

# Creates a queue to store detected emotions before writing to CSV
audio_data_queue = deque(maxlen=5)

# Tracks the last timestamp an emotion was recorded (for 1-second interval)
last_recorded_time = datetime.datetime.now()

# Initialises a variable to store the currently detected emotion
current_emotion = None

# Handles program termination (Ctrl+C)
def signal_handler(sig, frame):
    print("\nShutting down...")
    stream.stop_stream()
    stream.close()
    p.terminate()
    if audio_data_queue:
        df_batch = pd.DataFrame(audio_data_queue, columns=["Timestamp", "Participant_ID", "Emotion"])
        df_batch.to_csv(csv_filename, mode='a', header=False, index=False)
    sys.exit(0)

# Links the signal handler to the SIGINT signal (Ctrl+C) for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)

print("Press 'q' to quit.")

# Create a visualization window
window_height = 400
window_width = 800
vis_window = np.zeros((window_height, window_width, 3), dtype=np.uint8)

while True: # Starts an infinite loop to process audio data continuously
    try:
        # Capture audio data
        audio_data = stream.read(CHUNK, exception_on_overflow=False) # Captures a chunk of audio data from the microphone
        audio_np = np.frombuffer(audio_data, dtype=np.float32).copy() # Converts raw audio data into a NumPy array

        # Process every second
        if (datetime.datetime.now() - last_recorded_time).total_seconds() >= 1:
            last_recorded_time = datetime.datetime.now()

            # Process audio for emotion prediction
            inputs = processor([audio_np], sampling_rate=RATE, return_tensors="pt").to(device) # Prepares audio data for model input
            with torch.no_grad():
                outputs = model(**inputs) # Feeds the audio data to the model
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1) # Converts model output to probabilities
                predicted_label = torch.argmax(predictions, dim=-1).item() # Determines the most likely emotion label
                current_emotion = model.config.id2label[predicted_label] # Retrieves the emotion label corresponding to the model's prediction

            # Save detected emotion
            current_time = datetime.datetime.now()
            audio_data_queue.append([current_time.strftime("%Y-%m-%d %H:%M:%S"), participant_id, current_emotion]) # Stores the detected emotion in the queue

            # Writes the stored emotions to the CSV file if the queue is full (5 emotions)
            if len(audio_data_queue) == 5:
                df_batch = pd.DataFrame(audio_data_queue, columns=["Timestamp", "Participant_ID", "Emotion"])
                df_batch.to_csv(csv_filename, mode='a', header=False, index=False)
                audio_data_queue.clear()

        # Visualises audio waveform
        waveform = (audio_np * 128 + 128).astype(np.uint8)
        for i in range(min(len(waveform), window_width)):
            cv2.line(vis_window, (i, window_height // 2), (i, int(window_height // 2 + waveform[i])), (0, 255, 0), 1)

        # Adds recording indicator
        cv2.circle(vis_window, (30, 30), 10, (0, 0, 255), -1)  # Red circle when recording

        # Displays current emotion
        if current_emotion:
            cv2.putText(vis_window, f"Emotion: {current_emotion}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Shows audio level meter
        audio_level = np.abs(audio_np).mean() * 500  # Scale factor for visibility
        cv2.rectangle(vis_window, (700, 350), (750, 350 - int(audio_level)), (0, 255, 0), -1)

        # Displays the visualization
        cv2.imshow("Audio Emotion Analysis", vis_window)

        # Allows the user to quit the program by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nThe model has quit.")
            break

    except Exception as e:
        print(f"Error in speech emotion processing: {e}")
        continue

# Ensures proper shutdown when the program exits
signal_handler(None, None)

# The code snippet above demonstrates a real-time speech emotion recognition system using the Superb model. The system captures audio data from the microphone, processes it every second, predicts the emotion, and visualizes the audio waveform and emotion in a window. The detected emotions are saved to a CSV file for analysis.

# PENDING REVISIONS:
## The quit function is not working properly. 
# Takes some time to start the analysis (not immediate) 
# Is only identifying HAPPY and NEUTRAL emotions. 
# The visualisaiton is not good, the audio level meter is not working properly and the emotions written down are not clear (overlapping with each other)
