import cv2
import os

video_folder = "dataset/real"
output_folder = "frames"

os.makedirs(output_folder, exist_ok=True)

for video in os.listdir(video_folder):

    video_path = os.path.join(video_folder, video)
    cap = cv2.VideoCapture(video_path)

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.resize(frame, (224,224))

        frame_name = f"{video}_{frame_count}.jpg"
        cv2.imwrite(os.path.join(output_folder, frame_name), frame)

        frame_count += 1

    cap.release()

print("Frames extracted successfully")