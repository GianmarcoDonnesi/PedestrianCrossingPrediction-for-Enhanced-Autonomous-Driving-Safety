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

class PedestrianCrossingPredictor(nn.Module):
    def __init__(self):
        super(PedestrianCrossingPredictor, self).__init__()
        vgg19 = models.vgg19(pretrained=True)

        # Extract features
        self.vgg19_features = vgg19.features
        self.vgg19_avgpool = vgg19.avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.vgg19_classifier = nn.Sequential(*list(vgg19.classifier.children())[:-1])

        # Freeze more layers
        for param in self.vgg19_features[:36].parameters():  # Freeze more layers
            param.requires_grad = False

        self.lstm = nn.LSTM(input_size=4096, hidden_size=256, num_layers=1, batch_first=True)  # Reduced LSTM

        # Attention module
        self.attention = SoftAttention(hidden_dim=256)

        # BatchNorm and Dropout
        self.bn1 = nn.BatchNorm1d(256)
        self.d1 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.d2 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.d3 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(64, 1)

        self.goal_module = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        self.com_fc1 = nn.Linear(256 + 256 + 11, 256)  # Update the size to 256
        self.com_bn1 = nn.BatchNorm1d(256)
        self.com_d1 = nn.Dropout(0.5)
        self.com_fc2 = nn.Linear(256, 128)
        self.com_bn2 = nn.BatchNorm1d(128)
        self.com_d2 = nn.Dropout(0.5)
        self.com_fc3 = nn.Linear(128, 1)

    def forward(self, x, traffic_info, vehicle_info, appearance_info, attributes_info, future_steps=10):
        if x.dim() == 4:  # If the input has 4 dimensions, expand it to 5
            x = x.unsqueeze(1)  # Add a sequence length dimension
        batch_size, seq_len, c, h, w = x.size()
        c_in = x.view(batch_size * seq_len, c, h, w)
        c_out = self.vgg19_features(c_in)
        c_out = self.avgpool(c_out)
        c_out = c_out.view(c_out.size(0), -1)
        c_out = self.vgg19_classifier(c_out)
        c_out = c_out.view(batch_size, seq_len, -1)

        lstm_out, _ = self.lstm(c_out)

        # Apply attention
        context, attn_weights = self.attention(lstm_out)

        context = self.bn1(context)
        context = self.d1(context)

        additional_info = torch.cat([traffic_info, vehicle_info, appearance_info, attributes_info], dim=1)

        goal_out = self.goal_module(context)

        combined = torch.cat((context, goal_out, additional_info), dim=1)

        combined = self.com_fc1(combined)
        combined = torch.relu(combined)
        combined = self.com_bn1(combined)
        combined = self.com_d1(combined)

        combined = self.com_fc2(combined)
        combined = torch.relu(combined)
        combined = self.com_bn2(combined)
        combined = self.com_d2(combined)

        combined = self.com_fc3(combined)

        return combined

# Inizializza i pesi, il modello, la loss function, l'optimizer e lo scheduler
def init_w(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_normal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

model = PedestrianCrossingPredictor()
model.apply(init_w)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.1)

# Salva il modello, la loss function e l'optimizer per uso successivo
torch.save(model.state_dict(), '/content/drive/My Drive/CV_Project/model.pth')
torch.save(optimizer.state_dict(), '/content/drive/My Drive/CV_Project/optimizer.pth')

print("Model saved")