import cv2 # Version 4.8.0.76 
import datetime 
import pandas as pd # Version 2.12.0
import os
import signal 
import sys 
import numpy as np # Version 1.23.5 # 
import pyaudio # Version 0.2.14
import torch
import matplotlib.pyplot as plt
from deepface import DeepFace # Version 0.0.79
from collections import deque
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from scipy.io import wavfile


# Configuration constants
CONFIG = {
   'AUDIO_BUFFER': 3,  # 3 seconds of audio processed at a time
   'VIDEO_BUFFER': 1,  # 1 frame per second processed at a time
   'CSV_BUFFER': 20     # Save every 5 entries
}


# Initialise PyAudio for audio capture
p = pyaudio.PyAudio()
RATE = 16000
CHUNK = 8000 # Change this for chunk size (e.g., 8000 for 0.5 second chunks) 
CHANNELS = 1
FORMAT = pyaudio.paInt16
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK) # Opens the audio stream for capturing audio data from the microphone


# Ask for Participant ID
participant_id = input("Enter Participant ID: ")


# Load pre-trained Haar Cascade model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# Define CSV filename for storing emotion data
csv_filename = "FER_SER_emotions.csv"
if not os.path.exists(csv_filename): # Check if CSV file exists, if not, create it
   df = pd.DataFrame(columns=["Timestamp", "Participant ID", "Facial Emotion", "Speech Emotion"])
   df.to_csv(csv_filename, index=False, encoding='utf-8')

# Loads existing CSV (to prevent overwriting)
df = pd.read_csv(csv_filename)


# Initialise video capture from webcam and reduce buffer size to prevent lag
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened(): # If webcam is not available, exit the program
   print("Error: Could not open webcam.")
   sys.exit()


# Load the Superb/WavVec2 model for speech emotion recognition
model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-er").to("cpu")
processor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er")


# Create queues for data batching
emotion_data_queue = deque(maxlen=CONFIG['CSV_BUFFER'])


# Track last recorded time (for 1-second interval)
last_recorded_time = datetime.datetime.now()


# Enable interactive mode for Matplotlib
plt.ion()


# Initialise list to store audio chunks for batch processing
audio_chunks = []


# Track last facial emotion timestamp and detected face
last_facial_emotion_time = datetime.datetime.now()
first_face_detected = False  # Track if a face has been detected
last_facial_emotion = "Unknown"


def process_audio():
    # Capture audio chunk
    audio_data = stream.read(CHUNK, exception_on_overflow=False)
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0  # Normalize audio
    audio_chunks.append(audio_np)

    # If the buffer has enough data for 3 seconds, process it
    if len(audio_chunks) * CHUNK / RATE >= CONFIG['AUDIO_BUFFER']:
        # Concatenate all audio chunks collected so far
        audio_data_concat = np.concatenate(audio_chunks)
        audio_chunks.clear()  # Clear buffer after processing

        # Process the concatenated audio and get speech emotion
        inputs = processor([audio_data_concat], sampling_rate=RATE, return_tensors="pt").to("cpu")
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_label = torch.argmax(predictions, dim=-1).item()
            speech_emotion = model.config.id2label[predicted_label]

        return speech_emotion, audio_data_concat

    return "Processing", None


# Plot audio waveform as visual feedback
def plot_audio_waveform(audio_np, speech_emotion):
   plt.clf()
   plt.plot(audio_np, color="blue")
   plt.title(f"Audio Waveform - Emotion: {speech_emotion}")  # Show emotion in title
   plt.xlabel("Time")
   plt.ylabel("Amplitude")
   plt.draw()
   plt.pause(0.01)



# Handle program termination (Ctrl+C)
def signal_handler(sig, frame):
   print("\nShutting down...")
   cap.release()
   stream.stop_stream()
   stream.close()
   p.terminate()
   cv2.destroyAllWindows()
   cleanup_resources()
   sys.exit(0)



# Link signal handler to SIGINT signal (Ctrl+C) for smoother shutdown
signal.signal(signal.SIGINT, signal_handler)
print("Press 'q' to quit.")


# Cleanup resources before exiting
def cleanup_resources():
   audio_chunks.clear()
   emotion_data_queue.clear()
   plt.close('all')



# Process facial emotion using DeepFace
def process_facial_emotion(frame, current_time):
   global last_facial_emotion_time, last_facial_emotion


   if (current_time - last_facial_emotion_time).total_seconds() >= CONFIG['VIDEO_BUFFER']:
       last_facial_emotion_time = current_time
       try:
           analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
           last_facial_emotion = analysis[0]['dominant_emotion'] if analysis else "Unknown"
       except Exception as e:
           print(f"DeepFace Error: {e}")
           last_facial_emotion = "Error"


   return last_facial_emotion



emotion_writer_counter=0 # Count the number of speech emotion entries labeled as "Processing"
while True:
   ret, frame = cap.read()
   if not ret:
       print("Error: Could not read frame.")
       break

   # Format frame for face detection 
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))


   facial_emotion = "Unknown"


   if len(faces) > 0: # Check if a face is detected
       (x, y, w, h) = faces[0]  # Gets the coordinates of the first detected face


       if not first_face_detected:
           first_face_detected = True  # Mark that we started tracking a face
      
       # Extract face region for analysis
       face_region = frame[y:y+h, x:x+w]

       # Check if it is time to record emotion 
       current_time = datetime.datetime.now()
       facial_emotion = process_facial_emotion(face_region, current_time)
        

       # Define rectangle color based on emotion
       positive_emotions = ['happy', 'surprise', 'calm']
       negative_emotions = ['sad', 'angry', 'fear']


       if facial_emotion in positive_emotions:
           rectangle_color = (0, 255, 0)  # Green for positive emotions
       elif facial_emotion in negative_emotions:
           rectangle_color = (0, 0, 255)  # Red for negative emotions
       else:
           rectangle_color = (0, 255, 255)  # Yellow for neutral or unknown emotions


       # Draw rectangle around the detected face
       cv2.rectangle(frame, (x, y), (x + w, y + h), rectangle_color, 2)


       # Display detected emotion
       cv2.putText(frame, facial_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


   else:
       first_face_detected = False  # Reset tracking if the face disappears


   # Process audio and get speech emotion
   speech_emotion, audio_waveform = process_audio()


   # Plot audio waveform for visual feedback
   if audio_waveform is not None:
       plot_audio_waveform(audio_waveform, speech_emotion)


   current_time = datetime.datetime.now()


   # Save emotions to CSV 
   if (current_time - last_recorded_time).total_seconds() >= 1:
       last_recorded_time = current_time
       if(speech_emotion == "Processing"):  
           emotion_writer_counter+=1 #If speech emotion is "Processing", increment the counter
           emotion_data_queue.append([current_time.strftime("%Y-%m-%d %H:%M:%S"), participant_id, facial_emotion, speech_emotion])
       else:
           emotion_data_queue.append([current_time.strftime("%Y-%m-%d %H:%M:%S"), participant_id, facial_emotion, speech_emotion])
           
       if len(emotion_data_queue) >= CONFIG['CSV_BUFFER'] or speech_emotion != "Processing" : # If speech emotion is not "Processing", write the batch to CSV
           
           # Replace the speech emotion of the last "Processing" entries with the last detected speech emotion and rewrite the last detected emotion ## (pop() deletes the last element)
           rewriter_batch=[]
           for i in range(emotion_writer_counter+1):
               temp_emotion_rewriter=emotion_data_queue.pop()
               temp_emotion_rewriter[-1]=speech_emotion
               rewriter_batch.append(temp_emotion_rewriter)

           # Reverse the rewriter batch to maintain the temporal sequence
           rewriter_batch.reverse()
           emotion_data_queue.extend(rewriter_batch)
           df_batch = pd.DataFrame(emotion_data_queue, columns=["Timestamp", "Participant ID", "Facial Emotion", "Speech Emotion"])
           df_batch.to_csv(csv_filename, mode='a', header=False, index=False)
           emotion_data_queue.clear()
           emotion_writer_counter=0

   # Visualise the video feed with the detected face and emotion 
   cv2.imshow("Face & Emotion Detection", frame)

   # Allow the user to quit the program by pressing 'q' 
   if cv2.waitKey(1) & 0xFF == ord('q'):
       print("\nThe model has quit.")
       break


signal_handler(None, None)

