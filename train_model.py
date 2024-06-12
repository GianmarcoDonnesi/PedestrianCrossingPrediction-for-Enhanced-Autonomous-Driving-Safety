# Codice di addestramento
import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from create_dataset import PreprocessedDataset, collate_fn
from model.model import PedestrianCrossingPredictor
import os

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the training function
def train(model, criterion, optimizer, scheduler, train_loader, num_epochs=10):
    model.train()
    print(f"Inizio addestramento modello:")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for frames, keypoints, labels, traffic_info, vehicle_info, appearance_info, attributes_info in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            frames, traffic_info, vehicle_info, appearance_info, attributes_info, labels = frames.to(device), traffic_info.to(device), vehicle_info.to(device), appearance_info.to(device), attributes_info.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(frames, traffic_info, vehicle_info, appearance_info, attributes_info)
            labels = labels.unsqueeze(1).float()  # Add a dimension and convert to float
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Clear CUDA cache to prevent OOM errors
            torch.cuda.empty_cache()

        print(f"Training Loss: {running_loss / len(train_loader):.4f}")

        model.train()

        # Step scheduler
        scheduler.step()

    return model

# Load the model, loss function, and optimizer
model = PedestrianCrossingPredictor()
model.load_state_dict(torch.load('/content/drive/My Drive/CV_Project/model.pth'))
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
optimizer.load_state_dict(torch.load('/content/drive/My Drive/CV_Project/optimizer.pth'))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# Directory dei dati preprocessati
train_data_dir = '/content/drive/My Drive/CV_Project/training_data'

# Carica il dataset preprocessato per il training dai file .pt
train_dataset_pt = PreprocessedDataset(train_data_dir, transform=None)

# Load the DataLoader
with open('/content/drive/My Drive/CV_Project/train_loader.pkl', 'rb') as f:
    train_loader_pkl = pickle.load(f)

# Create DataLoader per i file .pt
num_workers = min(16, cpu_count())
train_loader_pt = DataLoader(train_dataset_pt, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

# Concatenate DataLoaders (note that ConcatDataset concatenates datasets, not DataLoaders directly)
combined_train_dataset = ConcatDataset([train_loader_pt.dataset, train_loader_pkl.dataset])
combined_train_loader = DataLoader(combined_train_dataset, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)


# Train the model
trained_model = train(model, criterion, optimizer, scheduler, combined_train_loader, num_epochs=10)


torch.save(trained_model.state_dict(), '/content/drive/My Drive/CV_Project/trained_model.pth')

print(f"Addestramento completato. Modello addestrato salvato")
