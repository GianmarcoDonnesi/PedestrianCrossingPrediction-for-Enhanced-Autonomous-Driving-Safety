import os
import requests
from zipfile import ZipFile
import cv2

# Funzione per scaricare il file dal URL
def download_file(url, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    file_name = url.split('/')[-1]
    file_path = os.path.join(dest_folder, file_name)
    if not os.path.exists(file_path):
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    f.write(chunk)
    return file_path

# Funzione per estrarre un file zip
def extract_zip(file_path, dest_folder):
    with ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(dest_folder)

# Funzione per estrarre i frame da un video
def extract_frames_from_video(video_path, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    cap = cv2.VideoCapture(video_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(dest_folder, f"frame_{count:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        count += 1
    cap.release()

# Scarica e estrai il dataset JAAD
dataset_url = 'http://data.nvision2.eecs.yorku.ca/JAAD_dataset/data/JAAD_clips.zip'
dataset_folder = 'JAAD_dataset'
video_folder = os.path.join(dataset_folder, 'JAAD_clips')

zip_file_path = download_file(dataset_url, dataset_folder)
extract_zip(zip_file_path, dataset_folder)

# Estrai i frame da ogni video nel dataset
for video_file in os.listdir(video_folder):
    if video_file.endswith('.mp4'):
        video_path = os.path.join(video_folder, video_file)
        frames_output_folder = os.path.join(dataset_folder, 'frames', os.path.splitext(video_file)[0])
        extract_frames_from_video(video_path, frames_output_folder)

print("Estrazione dei frame completata.")
