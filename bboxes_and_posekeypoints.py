import os
import cv2
import xml.etree.ElementTree as ET
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# Path to JAAD dataset and output directory
jaad_path = './JAAD_dataset/JAAD_clips'
annotation_dir = './JAAD_dataset/annotations'
output_dir = os.path.join(jaad_path, 'frames_with_bboxes')
pose_output_dir = os.path.join(jaad_path, 'pose_keypoints')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(pose_output_dir, exist_ok=True)




def extract_and_save_frames_with_bboxes_and_pose_keypoints(video_path, annotation_path, video_output_dir, pose_output_dir):
    # Load video
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.basename(video_path).split('.')[0]

    # Create subdirectory for the video frames
    os.makedirs(video_output_dir, exist_ok=True)
    os.makedirs(pose_output_dir, exist_ok=True)

    # Parse XML annotation file
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    # Get the total number of frames in the video for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_id = 0
    with tqdm(total=total_frames, desc=f"Processing {video_name}", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            for track in root.findall('.//track'):
                for box in track.findall('.//box'):
                    if int(box.get('frame')) == frame_id:
                        xtl = int(float(box.get('xtl')))
                        ytl = int(float(box.get('ytl')))
                        xbr = int(float(box.get('xbr')))
                        ybr = int(float(box.get('ybr')))
                        cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0, 255, 0), 2)

            # Resize the frame
            frame = cv2.resize(frame, (640, 480))

            # Extract Keypoints with Mediapipe
            r = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if r.pose_landmarks:
                keypoints = []
                for landmark in r.pose_landmarks.landmark:
                    keypoints.append([landmark.x, landmark.y, landmark.z])
                keypoints = np.array(keypoints)
                # Disegna i keypoints sull'immagine
                
for video_file in video_files[:100]:
    video_path = os.path.join(jaad_path, video_file)
    annotation_file = video_file.replace('.mp4', '.xml')
    annotation_path = os.path.join(annotation_dir, annotation_file)

    # Create subdirectory for the video frames and pose keypoints
    video_output_dir = os.path.join(output_dir, video_file.split('.')[0])
    video_pose_output_dir = os.path.join(pose_output_dir, video_file.split('.')[0])

    extract_and_save_frames_with_bboxes_and_pose_keypoints(video_path, annotation_path, video_output_dir, video_pose_output_dir)

print("\nFrame extraction with bounding boxes and pose keypoints complete!")
