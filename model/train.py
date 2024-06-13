import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score

def train(model, train_loader, optimizer, criterion, scheduler=None, num_epochs=10, device='cuda', verbose=True):
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0  # Running loss
        pred = []  # Predictions
        target = []  # Targets

        for batch_idx, (frames, keypoints, label, traffic_info, vehicle_info, appearance_info, attributes_info) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")):
            frames, keypoints, label, traffic_info, vehicle_info, appearance_info, attributes_info = frames.to(device), keypoints.to(device), label.to(device), traffic_info.to(device), vehicle_info.to(device), appearance_info.to(device), attributes_info.to(device)
            
            optimizer.zero_grad()  # Clear gradients
            output = model(frames, traffic_info, vehicle_info, appearance_info, attributes_info)  # Forward pass
            loss = criterion(output, label.float().view(-1, 1))  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update parameters

            running_loss += loss.item() * frames.size(0)
            pred_label = torch.sigmoid(output) > 0.5
            pred.extend(pred_label.cpu().numpy().flatten())
            target.extend(label.cpu().numpy().flatten())

            torch.cuda.empty_cache()

        # Epoch metrics
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = accuracy_score(target, pred)
        epoch_recall = recall_score(target, pred)
        epoch_f1_score = f1_score(target, pred)

        # Metrics
        if verbose:
            print("Epoch {}/{} - Training Loss: {:.4f}, Accuracy: {:.4f}, Recall: {:.4f}, F1-Score: {:.4f}".format(
                epoch + 1, num_epochs, epoch_loss, epoch_accuracy, epoch_recall, epoch_f1_score))

        # Step the scheduler
        if scheduler:
            scheduler.step()

    return model
