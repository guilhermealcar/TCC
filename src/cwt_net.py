import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    from .config import DEVICE
except ImportError:
    from config import DEVICE

class CWTVisionNet(nn.Module):
    """
    A 2D Convolutional Neural Network designed for CWT Spectrograms.
    Treats the 10 EMG channels as depth (like a 10-channel image).
    """
    def __init__(self, num_classes: int):
        super(CWTVisionNet, self).__init__()
        
        # Input: (Batch, 10 Channels, 16 Freq Scales, 20 Time Steps)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # Output: (32, 8, 10)
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # Output: (64, 4, 5)
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Global Average Pooling -> (128, 1, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten the 1x1 spatial grid
        return self.classifier(x)

def train_cwt_vision_model(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    epochs: int = 40, batch_size: int = 64
) -> tuple:
    
    num_classes = int(max(y_train.max(), y_test.max()) + 1)
    model = CWTVisionNet(num_classes).to(DEVICE)
    
    weights = np.ones(num_classes, dtype=np.float32)
    classes, counts = np.unique(y_train, return_counts=True)
    weights[classes] = 1.0 / counts
    tensor_weights = torch.FloatTensor(weights / weights.sum()).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=tensor_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Reshape from (N, Freq, Time, Channels) to PyTorch standard (N, Channels, Freq, Time)
    X_tr_t = torch.FloatTensor(X_train).permute(0, 3, 1, 2)
    y_tr_t = torch.LongTensor(y_train)
    
    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)
    
    print(f"\nTraining CWT Deep Vision Network on {DEVICE}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")
            
    model.eval()
    with torch.no_grad():
        X_te_t = torch.FloatTensor(X_test).permute(0, 3, 1, 2).to(DEVICE)
        preds = torch.argmax(model(X_te_t), dim=1).cpu().numpy()
        
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    
    return acc, f1, model