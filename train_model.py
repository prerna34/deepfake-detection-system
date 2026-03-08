import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

face_folder = "faces"

data = []
labels = []

for img in os.listdir(face_folder):

    img_path = os.path.join(face_folder, img)

    image = cv2.imread(img_path)
    image = cv2.resize(image, (224,224))
    image = image / 255.0

    data.append(image)

    # abhi testing ke liye sabko real label dete hain
    labels.append(0)

data = np.array(data)
labels = np.array(labels)

print("Dataset loaded:", data.shape)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train,
    y_train,
    epochs=5,
    validation_data=(X_test, y_test)
)

model.save("deepfake_model.h5")

print("Model training complete")