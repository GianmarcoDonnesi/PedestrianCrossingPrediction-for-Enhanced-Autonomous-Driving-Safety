import os
import cv2
import xml.etree.ElementTree as ET
import mediapipe as mp
import numpy as np
from tqdm import tqdm
import warnings


# Path to JAAD dataset and output directory
jaad_path = '/content/drive/My Drive/CV_Project/JAAD_dataset/JAAD_clips'
annotation_dir = '/content/drive/My Drive/CV_Project/JAAD_dataset/annotations'
output_dir = os.path.join(jaad_path, 'frames_with_bboxes')
pose_output_dir = os.path.join(jaad_path, 'pose_keypoints')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(pose_output_dir, exist_ok=True)

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode = False, model_complexity = 2, enable_segmentation = False, min_detection_confidence = 0.4)

#
def preprocess_frames(frame):
    # Resize the frame
    frame = cv2.resize(frame, (640, 480))

    # Convert to grayscale and enhance contrast
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize = (8, 8))
    grayscale = clahe.apply(grayscale)

    # Denoise
    frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

    # Convert back to BGR
    frame = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)

    return frame


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

            # Preprocess
            frame = preprocess_frames(frame)

            for track in root.findall('.//track'):
                for box in track.findall('.//box'):
                    if int(box.get('frame')) == frame_id:
                        xtl = int(float(box.get('xtl')))
                        ytl = int(float(box.get('ytl')))
                        xbr = int(float(box.get('xbr')))
                        ybr = int(float(box.get('ybr')))
                        cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0, 255, 0), 2)

            frame_filename = os.path.join(video_output_dir, f"{video_name}_frame_{frame_id:05d}.jpg")
            cv2.imwrite(frame_filename, frame)

            # Extract Keypoints with Mediapipe
            r = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if r.pose_landmarks:
                keypoints = []
                for landmark in r.pose_landmarks.landmark:
                    keypoints.append([landmark.x, landmark.y, landmark.z])
                keypoints = np.array(keypoints)
            else:
                # Create an empty array
                keypoints = np.empty((0, 3))

            # Save keypoints as .npy file
            keypoints_filename = os.path.join(pose_output_dir, f"{video_name}_frame_{frame_id:05d}.npy")
            np.save(keypoints_filename, keypoints)

            frame_id += 1
            pbar.update(1)

        cap.release()

# Extract frames with bounding boxes and pose keypoints for the first 10 videos and annotation files
video_files = sorted([file for file in os.listdir(jaad_path) if file.endswith('.mp4')])

for video_file in video_files[:346]:
    video_path = os.path.join(jaad_path, video_file)
    annotation_file = video_file.replace('.mp4', '.xml')
    annotation_path = os.path.join(annotation_dir, annotation_file)

    # Create subdirectory for the video frames and pose keypoints
    video_output_dir = os.path.join(output_dir, video_file.split('.')[0])
    video_pose_output_dir = os.path.join(pose_output_dir, video_file.split('.')[0])

    extract_and_save_frames_with_bboxes_and_pose_keypoints(video_path, annotation_path, video_output_dir, video_pose_output_dir)

print("\n Frame extraction with bounding boxes and pose keypoints complete!")
