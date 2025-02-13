import os
import cv2     # OpenCV for capturing and processing video frames
import math
import geocoder   # For obtaining geographical location
import requests
%matplotlib inline
import pandas as pd   # For handling CSV files and dataframes
from twilio.rest import Client   # Twilio API for sending SMS alerts
from geopy.geocoders import Nominatim  # For reverse geolocation (latitude/longitude to address)
from keras.preprocessing import image   # For preprocessing image inputs
import numpy as np    # NumPy for mathematical and array operations
from keras.utils import to_categorical  # For converting labels to one-hot encoding
from matplotlib import pyplot as plt   # For plotting images
from skimage.transform import resize   # For resizing images
from keras.models import Sequential   # Keras Sequential model
from keras.applications.vgg16 import VGG16  # Pretrained VGG16 model
from keras.layers import Dense, InputLayer, Dropout   # Neural network layers
from keras.applications.vgg16 import preprocess_input  # Preprocessing function for VGG16

# ------------------------------- Video Frame Extraction -------------------------------

count = 0
videoFile = "datasets/Accidents.mp4"  # Path to input accident video
cap = cv2.VideoCapture(videoFile)   # Capturing the video from the given path
frameRate = cap.get(5)  # Getting the frame rate of the video
x = 1

while(cap.isOpened()):
    frameId = cap.get(1)  # Get the current frame number
    ret, frame = cap.read()  # Read the next frame

    if (ret != True):  # If the video ends, break the loop
        break

    if (frameId % math.floor(frameRate) == 0):  # Save frames at the frame rate interval
        filename = "%d.jpg" % count  # Naming frames as 0.jpg, 1.jpg, ...
        count += 1
        cv2.imwrite(filename, frame)  # Save extracted frames

cap.release()  # Release the video capture object
print("Done!")  # Confirmation message

# ------------------------------- Display Sample Frame -------------------------------

img = plt.imread('0.jpg')  # Reading the first extracted frame
plt.imshow(img)  # Displaying the image

# ------------------------------- Read CSV Mapping File -------------------------------

data = pd.read_csv('datasets/mapping.csv')  # Load the CSV file containing image-to-class mapping
data.head()  # Display first few rows of the CSV file

# ------------------------------- Load & Preprocess Images -------------------------------

X = []  # Empty list to store image data
for img_name in data.Image_ID:  
    img = plt.imread(img_name)  # Read each image file
    X.append(img)  # Append image to the list

X = np.array(X)  # Convert list to NumPy array

y = data.Class  # Extract class labels
dummy_y = to_categorical(y)  # Convert labels to one-hot encoding format

# ------------------------------- Resize Images for VGG16 -------------------------------

image = []
for i in range(0, X.shape[0]):  
    a = resize(X[i], preserve_range=True, output_shape=(224, 224, 3)).astype(int)  # Resize image to 224x224x3
    image.append(a)  # Append resized image to list

X = np.array(image)  # Convert list to NumPy array

# ------------------------------- Preprocess Images for VGG16 -------------------------------

X = preprocess_input(X, data_format=None)  # Apply VGG16 preprocessing

# ------------------------------- Split Data into Train & Validation Sets -------------------------------

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, dummy_y, test_size=0.3, random_state=42)  
# 70% training, 30% validation

# ------------------------------- Load Pretrained VGG16 Model -------------------------------

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))  
# Load VGG16 without the top classification layer

# Extract Features Using VGG16
X_train = base_model.predict(X_train)
X_valid = base_model.predict(X_valid)

print(X_train.shape, X_valid.shape)  # Print shape of extracted features

# ------------------------------- Flatten the Features -------------------------------

X_train = X_train.reshape(X_train.shape[0], 7 * 7 * 512)  # Flatten to 1D
X_valid = X_valid.reshape(X_valid.shape[0], 7 * 7 * 512)  

# Normalize Data
X_train = X_train / X_train.max()  # Scale values between 0 and 1
X_valid = X_valid / X_train.max()

# ------------------------------- Build Neural Network Model -------------------------------

model = Sequential()  # Initialize a sequential model
model.add(InputLayer((7 * 7 * 512,)))  # Input layer
model.add(Dense(units=1024, activation='sigmoid'))  # Hidden layer
model.add(Dense(2, activation='softmax'))  # Output layer with 2 classes (Accident / No Accident)

model.summary()  # Print model architecture

# ------------------------------- Compile the Model -------------------------------

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
# Compile model using categorical cross-entropy loss and Adam optimizer

# ------------------------------- Train the Model -------------------------------

model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid))  
# Train the model for 100 epochs

# ------------------------------- Save the Model -------------------------------

model.save('models/Accident_detection_model.keras')  # Save trained model in "models" directory