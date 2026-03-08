import os
import cv2
import numpy as np

face_folder = "faces"

data = []

for img in os.listdir(face_folder):

    img_path = os.path.join(face_folder, img)

    image = cv2.imread(img_path)
    image = cv2.resize(image, (224,224))

    image = image / 255.0

    data.append(image)

data = np.array(data)

print("Dataset shape:", data.shape)