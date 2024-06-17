import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score
from multiprocessing import cpu_count
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from create_dataset import PreprocessedDataset, collate_fn
from model.model import PedestrianCrossingPredictor
from torch.cuda.amp import autocast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(model, criterion, test_loader, ablation = None):
    model.eval()
    test_running_loss = 0.0 # Running loss
    test_preds = [] # Predictions
    test_targets = [] # Targets

    with torch.no_grad():
        for frames, keypoints, labels, traffic_info, vehicle_info, appearance_info, attributes_info in tqdm(test_loader, desc="Validating", unit="batch"):
            frames, keypoints, traffic_info, vehicle_info, appearance_info, attributes_info, labels = (frames.to(device), keypoints.to(device), traffic_info.to(device), vehicle_info.to(device), appearance_info.to(device), attributes_info.to(device), labels.to(device))

            # Ablation
            if ablation == 'traffic':
                traffic_info = torch.zeros_like(traffic_info)
            elif ablation == 'vehicle':
                vehicle_info = torch.zeros_like(vehicle_info)
            elif ablation == 'appearance':
                appearance_info = torch.zeros_like(appearance_info)
            elif ablation == 'attributes':
                attributes_info = torch.zeros_like(attributes_info)

            with autocast():
                outputs = model(frames, keypoints, traffic_info, vehicle_info, appearance_info, attributes_info)
                labels = labels.unsqueeze(1).float()
                loss = criterion(outputs, labels)
            
            test_running_loss += loss.item() * frames.size(0)
            preds = torch.sigmoid(outputs).cpu().numpy()
            test_preds.extend(preds)
            test_targets.extend(labels.cpu().numpy())

    avg_test_loss = test_running_loss / len(test_loader.dataset)
    test_preds = (np.array(test_preds) > 0.5).astype(int)
    test_accuracy = accuracy_score(test_targets, test_preds)
    test_recall = recall_score(test_targets, test_preds)
    test_f1 = f1_score(test_targets, test_preds)

    return avg_test_loss, test_accuracy, test_recall, test_f1

def ablation(model, criterion, test_loader):
    ablations = [None, 'traffic', 'vehicle', 'appearance', 'attributes']
    results = []

    for ablation in ablations:
        print(f"Evaluating with ablation: {ablation if ablation else 'None'}")
        avg_test_loss, accuracy, recall, f1 = test(model, criterion, test_loader, ablation)
        
        result = {
            'ablation': ablation if ablation else 'None',
            'loss': avg_test_loss,
            'accuracy': accuracy,
            'recall': recall,
            'f1_score': f1
        }
        results.append(result)
        
        print(f"Ablation: {result['ablation']}, Loss: {result['loss']:.4f}, Accuracy: {result['accuracy']:.4f}, Recall: {result['recall']:.4f}, F1 Score: {result['f1_score']:.4f}")
    
    return results

# Initialize the model and load the trained weights
model = PedestrianCrossingPredictor().to(device)
model.load_state_dict(torch.load('./model/trained_model.pth'))
criterion = nn.BCEWithLogitsLoss()

# Load the DataLoader
with open('./test_loader.pkl', 'rb') as f:
    test_loader_pkl = pickle.load(f)

test_dir = './test_data'

# Load the preprocessed test dataset
test_pt = PreprocessedDataset(test_dir, transform=None)

# Concatenate the DataLoaders
comb_test = ConcatDataset([test_pt, test_loader_pkl.dataset])

n_w = min(16, cpu_count())
comb_test_loader = DataLoader(comb_test, batch_size=32, shuffle=True, num_workers=n_w, pin_memory=True, collate_fn=collate_fn)

# Evaluate the model performance with and without ablation
results = ablation(model, criterion, comb_test_loader)

# Save results for visualization
with open('./results.pkl', 'wb') as f:
    pickle.dump(results, f)

for result in results:
    print(f"Ablation: {result['ablation']}")
    print(f"Loss: {result['loss']:.4f}")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    print(f"F1 Score: {result['f1_score']:.4f}")
    print()
