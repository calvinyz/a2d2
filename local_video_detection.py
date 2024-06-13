import cv2
import os
from PIL import Image
import numpy as np

from detection.YOLOv8.model import YOLOv8
from utils.image import display_img, save_img

def extract_frames(video_path, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Read the first frame
    success, frame = video.read()
    frame_count = 0

    while success:
        # Save the frame as a png file
        frame_path = os.path.join(output_dir, f"frame_{frame_count}.png")
        cv2.imwrite(frame_path, frame)

        # Read the next frame
        success, frame = video.read()
        frame_count += 1

    # Release the video file
    video.release()

    print(f"Frames extracted: {frame_count}")

if __name__ == "__main__":

    weights_file = 'yolov8n_wildfire_local_e200.pt'

    detector = YOLOv8(weights_file)

    # Path to the mp4 video file
    video_path = ".\\fire_video.mp4"

    # Output directory to save the frames
    output_dir = ".\\images\\fire_frames"

    # Call the function to extract frames
    # extract_frames(video_path, output_dir)

    fire_img = cv2.imread(".\\images\\fire_frames\\frame_111_1.png")
    fire_img_disp = cv2.cvtColor(fire_img, cv2.COLOR_BGR2RGB)
    fire_img_disp = Image.fromarray(fire_img_disp)    
    fire_img_disp = np.asarray(fire_img_disp)
    # display_img(fire_img_disp)

    fire_img_copy = fire_img.copy()

    res = detector.detect(fire_img, [1088, 1920])
    fire_img_res = detector.visualize_detections(fire_img_copy, res)
    display_img(fire_img_res)    

