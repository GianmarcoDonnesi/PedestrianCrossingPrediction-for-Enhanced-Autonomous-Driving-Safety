# vgg19_and_lstm.py
import torch
import torch.nn as nn
import torchvision.models as models

# Attention mechanism
class SoftAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SoftAttention, self).__init__()
        self.hidden_dim = hidden_dim
        # Attention network: 2 linear layers + ReLU activation
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, lstm_output):
        # lstm_output: [batch_size, seq_len, hidden_dim]
        # Calculate attention weights
        attn_weights = self.attention(lstm_output)  #[batch_size, seq_len, 1]
        
        # Normalize the attention weights using softmax
        attn_weights = torch.softmax(attn_weights, dim=1)  #[batch_size, seq_len, 1]
        
        # Compute the context vector as a weighted sum of LSTM outputs
        context = torch.sum(attn_weights * lstm_output, dim=1)  #[batch_size, hidden_dim]
        return context, attn_weights

# Define the PedestrianCrossingPredictor class
class PedestrianCrossingPredictor(nn.Module):
    def __init__(self):
        super(PedestrianCrossingPredictor, self).__init__()
      

       
            
        # Define LSTM with input size of 4096 and hidden size of 256    
        self.lstm = nn.LSTM(input_size=4096, hidden_size=256, num_layers=1, batch_first=True)  

        # Attention module
        self.attention = SoftAttention(hidden_dim=256)

      

        self.fc3 = nn.Linear(64, 1)

        
        
        
        

    def forward(self, x, keypoints, traffic_info, vehicle_info, appearance_info, attributes_info, future_steps=10):
        if x.dim() == 4:  # If the input has 4 dimensions, expand it to 5
            x = x.unsqueeze(1)  # Add a sequence length dimension
        batch_size, seq_len, c, h, w = x.size()
        c_in = x.view(batch_size * seq_len, c, h, w)
        c_out = self.vgg19_features(c_in)
        c_out = self.avgpool(c_out)
        

        # Pass through LSTM
        lstm_out, _ = self.lstm(c_out)

        # Apply attention
        context, attn_weights = self.attention(lstm_out)

        # Apply BatchNorm and Dropout
        context = self.bn1(context)
        context = self.d1(context)
        
        # Flatten keypoints
        keypoints = keypoints.view(batch_size, -1)

        # Concatenate additional information
        additional_info = torch.cat([keypoints, traffic_info, vehicle_info, appearance_info, attributes_info], dim=1)

        goal_out = self.goal_module(context)

        combined = torch.cat((context, goal_out, additional_info), dim=1)

        

        return combined




model = PedestrianCrossingPredictor()


# Define loss function
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.1)

# Save the model and optimizer states
torch.save(model.state_dict(), './model/model.pth')
torch.save(optimizer.state_dict(), './model/optimizer.pth')

print("Model saved")