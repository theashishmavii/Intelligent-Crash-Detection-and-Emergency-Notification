# ------------------------------- Import Necessary Libraries -------------------------------

import os
import cv2     # OpenCV for capturing video and processing frames
import math
import geocoder   # To obtain geographical location
import requests
import pandas as pd   # For handling CSV files
import numpy as np    # NumPy for mathematical operations
from keras.models import load_model   # Load trained model
from keras.applications.vgg16 import VGG16, preprocess_input  # Pretrained VGG16 model & preprocessing
from skimage.transform import resize   # For resizing images
from geopy.geocoders import Nominatim  # For reverse geolocation (lat/lng to address)
from twilio.rest import Client   # Twilio API for sending SMS alerts
from matplotlib import pyplot as plt   # For image visualization
import time   # For time-based operations
from playsound import playsound   # For playing alert sound

# ------------------------------- Load the Trained Model -------------------------------

model = load_model('model/Accident_detection_model.keras')  # Load trained accident detection model

# ------------------------------- Load Pretrained VGG16 Model -------------------------------

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))  
# Load VGG16 without the top classification layer

# ------------------------------- Video Frame Extraction -------------------------------

count = 0
videoFile = "datasets/Accident-1.mp4"  # Input accident video file
cap = cv2.VideoCapture(videoFile)   # Open the video file
frameRate = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video
x = 1

while(cap.isOpened()):
    frameId = cap.get(cv2.CAP_PROP_POS_FRAMES)  # Get the current frame number
    ret, frame = cap.read()  # Read the next frame

    if not ret:  # If video ends, exit loop
        break

    if frameId % math.floor(frameRate) == 0:  # Save frames at frame rate interval
        filename = "test%d.jpg" % count  # Name frames as test0.jpg, test1.jpg, ...
        count += 1
        cv2.imwrite(filename, frame)  # Save extracted frames

cap.release()  # Release the video capture object
print("Done!")  # Confirmation message

# ------------------------------- Read Test CSV File -------------------------------

test = pd.read_csv('datasets/test.csv')  # Load test dataset containing image filenames

# ------------------------------- Load & Preprocess Test Images -------------------------------

test_image = []
for img_name in test.Image_ID:  
    img = plt.imread(img_name)  # Read each test image
    test_image.append(img)  # Append image to list

test_img = np.array(test_image)  # Convert list to NumPy array

# ------------------------------- Resize Test Images -------------------------------

test_image = []
for i in range(0, test_img.shape[0]):  
    a = resize(test_img[i], preserve_range=True, output_shape=(224, 224, 3)).astype(int)  # Resize to 224x224x3
    test_image.append(a)  

test_image = np.array(test_image)  # Convert list to NumPy array

# ------------------------------- Preprocess Test Images -------------------------------

test_image = preprocess_input(test_image, data_format=None)  # Apply VGG16 preprocessing

# ------------------------------- Extract Features from Images -------------------------------

test_image = base_model.predict(test_image)  # Extract features using VGG16
print(test_image.shape)  # Print shape of extracted features

# ------------------------------- Flatten the Extracted Features -------------------------------

test_image = test_image.reshape(9, 7*7*512)  # Flatten feature maps to 1D

# ------------------------------- Predict Accident vs No Accident -------------------------------

predictions = model.predict(test_image)  # Get model predictions
print(predictions)  # Print raw prediction values

for i in range(0, 9):  # Iterate over predictions
    if predictions[i][0] < predictions[i][1]:  # If first class probability is lower
        print("No Accident")
    else:
        print("Accident")

# ------------------------------- Get Location of the Accident -------------------------------

geoLoc = Nominatim(user_agent="GetLoc")  # Initialize geolocator
g = geocoder.ip('me')  # Get current location from IP
locname = geoLoc.reverse(g.latlng)  # Convert latitude/longitude to address

# ------------------------------- Twilio API for Sending Alert -------------------------------

account_sid = 'your twillio account sid'  # Twilio account SID
auth_token = 'twillio authorised token'   # Twilio authentication token
client = Client(account_sid, auth_token)  # Initialize Twilio client

# ------------------------------- Display Video with Predictions -------------------------------

cap = cv2.VideoCapture('datasets/Accident-1.mp4')  # Open the test video
i = 0
flag = 0  # Flag to check if accident detected

while True:
    ret, frame = cap.read()  # Read frames from video

    if ret:  # If frame is available
        if predictions[int(i/15) % 9][0] < predictions[int(i/15) % 9][1]:  # Check model prediction
            predict = "No Accident"
        else:
            predict = "Accident"
            flag = 1  # Set flag if accident detected

            # Send SMS Alert
            client.messages.create(
                body="🚨 Accident detected at " + locname.address + f"\n📍 Coordinates: {g.latlng[0]}, {g.latlng[1]}",
                from_='number generated by twillio',  # Twilio sender number
                to='mobile number on which you want'  # Recipient number
            )

            playsound('datasets/alert_sound.mp3')  # you can add any alert sound here

            # Save accident image for reference
            cv2.imwrite('accident_image.jpg', frame)

        # ------------------------------- Display Video with Prediction Overlay -------------------------------

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, predict, (50, 50), font, 1, (0, 255, 255), 3, cv2.LINE_4)  # Overlay prediction text
        cv2.imshow('Frame', frame)  # Show the video frame with predictions

        i += 1  # Increment frame index
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit video playback
            break
    else:
        break  # Exit loop if no more frames are available

# ------------------------------- Play Sound Alert if Accident is Detected -------------------------------

if flag == 1:
    playsound('datasets/alert_sound.mp3')  # you can add any alert sound here

# ------------------------------- Release Resources & Close Windows -------------------------------

cap.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows