# ------------------------------- Import Necessary Libraries -------------------------------

import os
import cv2     # OpenCV for real-time video capture and processing
import math
import geocoder   # To obtain geographical location
import requests
import pandas as pd   # For handling CSV files
from twilio.rest import Client   # Twilio API for sending SMS alerts
from geopy.geocoders import Nominatim  # For reverse geolocation (latitude/longitude to address)
from keras.preprocessing import image   # For preprocessing images
import numpy as np    # NumPy for mathematical operations
from keras.utils import to_categorical  # For converting labels to one-hot encoding
from matplotlib import pyplot as plt   # For displaying images
from skimage.transform import resize   # For resizing images
import tensorflow as tf   # TensorFlow for loading the trained model
import time   # Time module for handling time-related operations
from playsound import playsound   # For playing alert sound

from keras.models import Sequential   # Keras Sequential model
from keras.applications.vgg16 import VGG16  # Pretrained VGG16 model
from keras.layers import Dense, InputLayer, Dropout   # Neural network layers
from keras.applications.vgg16 import preprocess_input  # Preprocessing function for VGG16

# ------------------------------- Load the Trained Model -------------------------------

model = tf.keras.models.load_model('model/Accident_detection_model.keras')  # Load trained model from "model/" directory

# ------------------------------- Load Pretrained VGG16 Model -------------------------------

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))  
# Load VGG16 without the top classification layer

# ------------------------------- Get System Location -------------------------------

geoLoc = Nominatim(user_agent="GetLoc")  # Initialize geolocator
g = geocoder.ip('me')  # Get current location using IP address
locname = geoLoc.reverse(g.latlng)  # Convert latitude/longitude to an address
location = g.latlng  # Store latitude and longitude values

# ------------------------------- Twilio API for Sending Alert -------------------------------

account_sid = 'your twillio account sid'  # Twilio account SID
auth_token = 'twillio authorised token'   # Twilio authentication token
client = Client(account_sid, auth_token)  # Initialize Twilio client

# ------------------------------- Real-Time Video Capture & Accident Detection -------------------------------

cap = cv2.VideoCapture(0)  # Capture video from webcam (change to video file path if needed)
frameRate = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the camera
i = 0  # Frame index counter
flag = 0  # Flag to check if accident detected

while True:
    frameId = cap.get(cv2.CAP_PROP_POS_FRAMES)  # Get the current frame number
    ret, frame = cap.read()  # Read frames from the webcam

    if ret:  # If frame is available
        cv2.imwrite('captured_image.jpg', frame)  # Save the captured frame
        print("Image successfully captured")

        # ------------------------------- Preprocess Frame for Model -------------------------------

        # Resize the frame to 224x224 for VGG16 model
        resized_frame = resize(frame, preserve_range=True, output_shape=(224,224,3)).astype(int)
        
        # Expand dimensions to match model input format
        test_image = np.expand_dims(resized_frame, axis=0)  
        test_image = preprocess_input(test_image)  # Apply VGG16 preprocessing

        # Extract features from the frame using VGG16
        test_image = base_model.predict(test_image)

        # Flatten the extracted features
        test_image = test_image.reshape(1, -1)

        # Get prediction from the trained model
        predictions = model.predict(test_image)
        print(predictions)  # Print model output

        # ------------------------------- Determine If Accident Occurred -------------------------------

        if predictions[int(i/15) % 9][0] < predictions[int(i/15) % 9][1]:  # Compare prediction probabilities
            predict = "No Accident"
        else:
            predict = "Accident"

            # Send SMS Alert
            client.messages.create(
                body="🚨 Accident detected at " + locname.address + f"\n📍 Coordinates: {location[0]}, {location[1]}",
                from_='number generated by twillio',  # Twilio sender number
                to='mobile number on which you want'  # Recipient number
            )

            playsound('datasets/alert_sound.mp3')  # you can add any alert sound here

            # Save accident image for future reference
            cv2.imwrite('accident_image.jpg', frame)

            flag = 1  # Set flag indicating an accident has occurred

        # ------------------------------- Display Video with Prediction Overlay -------------------------------

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, predict, (50, 50), font, 1, (0, 255, 255), 3, cv2.LINE_4)  # Overlay prediction text
        cv2.imshow('Frame', frame)  # Show the video frame with predictions

        i += 1  # Increment frame index
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit video playback
            break
    else:
        break  # Exit loop if no more frames are available

# ------------------------------- Release Resources & Close Windows -------------------------------

cap.release()  # Release the webcam/video file object
cv2.destroyAllWindows()  # Close all OpenCV windows