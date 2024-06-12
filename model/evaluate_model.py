import torch
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score
from torch.utils.data import DataLoader, ConcatDataset
import pickle
from multiprocessing import cpu_count
from create_dataset import PreprocessedDataset, collate_fn
from model import PedestrianCrossingPredictor

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model and load the trained weights
model = PedestrianCrossingPredictor()
model.load_state_dict(torch.load('./model/trained_model.pth'))
model.to(device)

# Load the DataLoader
with open('./val_loader.pkl', 'rb') as f:
    val_loader_pkl = pickle.load(f)

# Directory for validation data
validation_data_dir = './validation_data'

def evaluate(model, dataloader, ablation=None):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for batch in dataloader:
            frames, keypoints, labels, traffic_info, vehicle_info, appearance_info, attributes_info = batch
            frames = frames.to(device)
            keypoints = keypoints.to(device)
            labels = labels.to(device)
            traffic_info = traffic_info.float().to(device)
            vehicle_info = vehicle_info.float().to(device)
            appearance_info = appearance_info.float().to(device)
            attributes_info = attributes_info.float().to(device)

            # Apply ablation if specified
            if ablation == 'traffic':
                traffic_info = torch.zeros_like(traffic_info)
            elif ablation == 'vehicle':
                vehicle_info = torch.zeros_like(vehicle_info)
            elif ablation == 'appearance':
                appearance_info = torch.zeros_like(appearance_info)
            elif ablation == 'attributes':
                attributes_info = torch.zeros_like(attributes_info)

            # Forward pass
            outputs = model(frames, traffic_info, vehicle_info, appearance_info, attributes_info)
            
            # Apply sigmoid to get probabilities
            preds = torch.sigmoid(outputs).cpu().numpy()
            predictions.extend(preds)
            actuals.extend(labels.cpu().numpy())
    return predictions, actuals

# Function to evaluate the model with different ablations
def evaluate_ablation(model, test_loader):
    ablations = [None, 'traffic', 'vehicle', 'appearance', 'attributes']
    results = {}

    for ablation in ablations:
        print(f"Evaluating with ablation: {ablation if ablation else 'None'}")
        predictions, actuals = evaluate(model, test_loader, ablation)

        # Check for single class issue in labels
        if len(np.unique(actuals)) < 2:
            print(f"Warning: Single class in labels for ablation {ablation if ablation else 'None'}. Metrics may be less informative.")

        # Calculate metrics
        accuracy = accuracy_score(actuals, np.round(predictions))
        recall = recall_score(actuals, np.round(predictions), zero_division=0)
        f1 = f1_score(actuals, np.round(predictions), zero_division=0)
        
        
        # Store results
        results[ablation if ablation else 'None'] = {
            'accuracy': accuracy,
            'recall': recall,
            'f1_score': f1
        }
        
        # Print results
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

    return results

# Load the preprocessed validation dataset from .pt files
validation_dataset_pt = PreprocessedDataset(validation_data_dir, transform=None)

# Create DataLoader for the .pt files
num_workers = min(16, cpu_count())
validation_loader_pt = DataLoader(validation_dataset_pt, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

# Concatenate the DataLoaders
combined_val_dataset = ConcatDataset([validation_loader_pt.dataset, val_loader_pkl.dataset])
combined_val_loader = DataLoader(combined_val_dataset, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

# Evaluate the model performance with and without ablation
results = evaluate_ablation(model, combined_val_loader)

# Print the results
for ablation, metrics in results.items():
    print(f"Ablation: {ablation}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print()
