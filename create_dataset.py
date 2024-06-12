import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms
import pickle
from tqdm import tqdm
from PIL import Image
from multiprocessing import Value, cpu_count

# Create a global counter for tracking progress
counter = Value('i', 0)

# Custom Dataset class for the JAAD dataset
class JAADDataset(Dataset):
    def __init__(self, frames_dir, keypoints_dir, cache_dir, transform=None):
        self.frames_dir = frames_dir
        self.keypoints_dir = keypoints_dir
        self.cache_dir = cache_dir
        self.transform = transform
        self.video_names = sorted(os.listdir(frames_dir))[:100]  
        self.data = []

        # Load data for each video
        for video in self.video_names:
            video_id = video.split('_')[1]
            cache_file = os.path.join(cache_dir, f"video_{video_id}.pkl")
            if not os.path.exists(cache_file):
                continue
            with open(cache_file, 'rb') as f:
                video_data = pickle.load(f)
                for frame_id, label, traffic_info, vehicle_info, appearance_info, attributes_info in video_data:
                    frame_path = os.path.join(frames_dir, video, f"{video}_frame_{frame_id:05d}.jpg")
                    keypoint_file = os.path.join(keypoints_dir, video, f"{video}_frame_{frame_id:05d}.npy")
                    if os.path.exists(frame_path) and os.path.exists(keypoint_file):
                        self.data.append((frame_path, keypoint_file, label, traffic_info, vehicle_info, appearance_info, attributes_info))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        with counter.get_lock():
            counter.value += 1
            if counter.value % 128 == 0:
                tqdm.write(f"Processed {counter.value} frames")

        frame_path, keypoint_file, label, traffic_info, vehicle_info, appearance_info, attributes_info = self.data[idx]
        frame = Image.open(frame_path).convert('RGB')

        keypoints = np.load(keypoint_file)
        if keypoints.size == 0:  # Handle empty keypoints
            keypoints = np.zeros((33, 3), dtype=np.float32)

        keypoints = torch.tensor(keypoints, dtype=torch.float32)

        traffic_info = torch.tensor(list(traffic_info.values()), dtype=torch.float32)
        vehicle_info = torch.tensor(list(vehicle_info.values()), dtype=torch.float32)
        appearance_info = torch.tensor(list(appearance_info.values()), dtype=torch.float32)
        attributes_info = torch.tensor(list(attributes_info.values()), dtype=torch.float32)

        if self.transform:
            frame = self.transform(frame)

        return frame, keypoints, label, traffic_info, vehicle_info, appearance_info, attributes_info

# Data augmentation transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Directories for frames, keypoints, and cache
frames_dir = './JAAD_dataset/JAAD_clips/frames_with_bboxes'
keypoints_dir = './JAAD_dataset/JAAD_clips/pose_keypoints'
cache_dir = './JAAD_dataset/cache'
base_dataset = JAADDataset(frames_dir, keypoints_dir, cache_dir, transform)

# Split dataset into training and validation sets
train_indices, val_indices = train_test_split(list(range(len(base_dataset))), test_size=0.2, random_state=42)

train_set = Subset(base_dataset, train_indices)
val_set = Subset(base_dataset, val_indices)


# Directories for saving preprocessed data
train_save_dir = './training_data'
val_save_dir = './validation_data'
os.makedirs(train_save_dir, exist_ok=True)
os.makedirs(val_save_dir, exist_ok=True)


# Check if training data and validation data already exist
if not os.listdir(train_save_dir) or not os.listdir(val_save_dir):
    
    # Function to save preprocessed data
    def save_preprocessed_data(dataset, save_dir):
        for idx in tqdm(range(len(dataset))):
            frame, keypoints, label, traffic_info, vehicle_info, appearance_info, attributes_info = dataset[idx]
            save_path = os.path.join(save_dir, f'data_{idx}.pt')
            torch.save({
                'frame': frame,
                'keypoints': keypoints,
                'label': label,
                'traffic_info': traffic_info,
                'vehicle_info': vehicle_info,
                'appearance_info': appearance_info,
                'attributes_info': attributes_info
            }, save_path)

    # Save the preprocessed data
    save_preprocessed_data(train_set, train_save_dir)
    save_preprocessed_data(val_set, val_save_dir)
else:
  print("Training data and validation data already exist. Skipping preprocessing.")


# Custom Dataset class for loading preprocessed data
class PreprocessedDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.data_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pt')])
        self.transform = transform

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data = torch.load(self.data_files[idx])
        frame = data['frame']
        keypoints = data['keypoints']
        label = data['label']
        traffic_info = data['traffic_info']
        vehicle_info = data['vehicle_info']
        appearance_info = data['appearance_info']
        attributes_info = data['attributes_info']

        if self.transform:
            frame = self.transform(frame)

        return frame, keypoints, label, traffic_info, vehicle_info, appearance_info, attributes_info

# Create the dataset for training and validation
train_dataset = PreprocessedDataset(train_save_dir, transform=None)
val_dataset = PreprocessedDataset(val_save_dir, transform=None)


# Custom collate function to handle batches with different tensor sizes
def collate_fn(batch):
    frames, keypoints, labels, traffic_infos, vehicle_infos, appearance_infos, attributes_infos = zip(*batch)

    frames = torch.stack(frames)
    keypoints = torch.stack(keypoints)
    labels = torch.tensor(labels)
    traffic_infos = torch.stack(traffic_infos)
    vehicle_infos = torch.stack(vehicle_infos)
    appearance_infos = torch.stack(appearance_infos)
    attributes_infos = torch.stack(attributes_infos)

    return frames, keypoints, labels, traffic_infos, vehicle_infos, appearance_infos, attributes_infos


# Create DataLoader
num_workers = min(16, cpu_count())

# Check if DataLoader files already exist
train_loader_path = './train_loader.pkl'
val_loader_path = './val_loader.pkl'

if not os.path.exists(train_loader_path) or not os.path.exists(val_loader_path):
    with tqdm(total=len(train_set), desc="Creating train DataLoader", unit="sample") as pbar:
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
        for _ in train_loader:
            pbar.update(128)

    with tqdm(total=len(val_set), desc="Creating val DataLoader", unit="sample") as pbar:
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
        for _ in val_loader:
            pbar.update(128)

    # Save the DataLoader for later use
    with open('./train_loader.pkl', 'wb') as f:
        pickle.dump(train_loader, f)
    with open('./val_loader.pkl', 'wb') as f:
        pickle.dump(val_loader, f)

else:
  print("DataLoader files already exist. Skipping DataLoader creation.")


print("Datasets and DataLoader created.")
