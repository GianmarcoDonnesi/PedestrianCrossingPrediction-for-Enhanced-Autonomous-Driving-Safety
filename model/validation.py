import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score
from torch.utils.data import ConcatDataset, Subset
from create_dataset import PreprocessedDataset, collate_fn
from model import PedestrianCrossingPredictor
from multiprocessing import cpu_count
import numpy as np

def validate(model, criterion, val_loader, ablation=None):
    model.eval()
    val_running_loss = 0.0
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for frames, keypoints, labels, traffic_info, vehicle_info, appearance_info, attributes_info in tqdm(val_loader, desc="Validating", unit="batch"):
            frames, keypoints, traffic_info, vehicle_info, appearance_info, attributes_info, labels = (
                frames.to(device), keypoints.to(device), traffic_info.to(device), 
                vehicle_info.to(device), appearance_info.to(device), attributes_info.to(device), labels.to(device)
            )

            if ablation == 'traffic':
                traffic_info = torch.zeros_like(traffic_info)
            elif ablation == 'vehicle':
                vehicle_info = torch.zeros_like(vehicle_info)
            elif ablation == 'appearance':
                appearance_info = torch.zeros_like(appearance_info)
            elif ablation == 'attributes':
                attributes_info = torch.zeros_like(attributes_info)

            outputs = model(frames, keypoints, traffic_info, vehicle_info, appearance_info, attributes_info)
            labels = labels.unsqueeze(1).float()
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()

            preds = torch.sigmoid(outputs).cpu().numpy()
            val_preds.extend(preds)
            val_targets.extend(labels.cpu().numpy())

            torch.cuda.empty_cache()

    avg_val_loss = val_running_loss / len(val_loader)
    val_accuracy = accuracy_score(val_targets, (np.array(val_preds) > 0.5).astype(int))
    val_recall = recall_score(val_targets, (np.array(val_preds) > 0.5).astype(int))
    val_f1 = f1_score(val_targets, (np.array(val_preds) > 0.5).astype(int))

    return avg_val_loss, val_accuracy, val_recall, val_f1
