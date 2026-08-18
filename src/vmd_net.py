import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    from .config import DEVICE
except ImportError:
    from config import DEVICE

class MultiScaleVMDNet(nn.Module):
    """
    Multi-Scale CNN that treats 3 IMFs as spatial channels (like RGB).
    Uses parallel branches with different kernel sizes to capture both 
    rapid spikes and slow macroscopic movements.
    """
    def __init__(self, num_classes: int):
        super(MultiScaleVMDNet, self).__init__()
        
        # Input shape: (Batch, 3 IMFs, 20 Time, 10 Electrodes)
        
        # Branch 1: Fine temporal resolution (3x3 kernel)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Branch 2: Medium temporal resolution (5x3 kernel)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Branch 3: Coarse temporal resolution (7x3 kernel)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(7, 3), padding=(3, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Fusion and Global Average Pooling (GAP)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # GAP flattens the spatial/temporal dimensions while retaining feature depth
            nn.AdaptiveAvgPool2d((1, 1)) 
        )
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass through all three multi-scale branches
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        
        # Concatenate along the feature channel dimension (dim=1)
        fused = torch.cat((out1, out2, out3), dim=1) 
        
        # Pool to a fixed shift-invariant representation
        pooled = self.fusion_conv(fused) 
        flattened = pooled.view(pooled.size(0), -1) 
        
        return self.classifier(flattened)

def train_vmd_model(
    model: nn.Module, 
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    epochs: int = 40, batch_size: int = 64, lr: float = 0.001
) -> nn.Module:
    """Trains the VMD network using safely mapped class-weighted Cross-Entropy Loss."""
    
    y_train = y_train.astype(np.int64)
    total_classes = int(max(y_train.max(), y_test.max()) + 1)
    
    weights = np.ones(total_classes, dtype=np.float32)
    classes, counts = np.unique(y_train, return_counts=True)
    weights[classes] = 1.0 / counts
    weights = weights / weights.sum()
    tensor_weights = torch.FloatTensor(weights).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=tensor_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # The VMD tensor is already perfectly shaped for Conv2d
    X_tr = torch.FloatTensor(X_train)
    y_tr = torch.LongTensor(y_train)
    
    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr), 
        batch_size=batch_size, 
        shuffle=True
    )
    
    model.to(DEVICE)
    print(f"Training VMD Multi-Scale Network on {DEVICE}...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")
            
    return model