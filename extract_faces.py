import cv2
import os
from mtcnn import MTCNN

detector = MTCNN()

frame_folder = "frames"
face_folder = "faces"

os.makedirs(face_folder, exist_ok=True)

for img in os.listdir(frame_folder):

    img_path = os.path.join(frame_folder, img)
    image = cv2.imread(img_path)

    faces = detector.detect_faces(image)

    for i, face in enumerate(faces):

        x, y, w, h = face['box']
        face_img = image[y:y+h, x:x+w]

        face_name = f"{img}_face{i}.jpg"
        cv2.imwrite(os.path.join(face_folder, face_name), face_img)

print("Faces extracted successfully")