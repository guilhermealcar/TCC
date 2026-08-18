import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple

try:
    from .config import DEVICE
except ImportError:
    from config import DEVICE

class DualBranchEMGNet(nn.Module):
    """
    Dual-Branch Neural Network for sEMG Classification.
    Branch 1: 1D-CNN for spatial-spectral FFT data.
    Branch 2: MLP for highly non-linear Sparse Dictionary Codes.
    """
    def __init__(self, num_channels: int, freq_bins: int, sparse_dim: int, num_classes: int):
        super(DualBranchEMGNet, self).__init__()
        
        # Branch 1: Frequency-Domain CNN
        self.branch1_cnn = nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Flattens to exactly 64 features regardless of freq_bins size
        )
        
        # Branch 2: Sparse Feature MLP
        self.branch2_mlp = nn.Sequential(
            nn.Linear(sparse_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # Fusion Head
        self.fusion_classifier = nn.Sequential(
            nn.Linear(64 + 128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_freq: torch.Tensor, x_sparse: torch.Tensor) -> torch.Tensor:
        # Pass through Branch 1
        out1 = self.branch1_cnn(x_freq)
        out1 = out1.view(out1.size(0), -1) # Shape: (Batch, 64)
        
        # Pass through Branch 2
        out2 = self.branch2_mlp(x_sparse) # Shape: (Batch, 128)
        
        # Concatenate representations
        fused = torch.cat((out1, out2), dim=1) # Shape: (Batch, 192)
        
        # Final classification
        return self.fusion_classifier(fused)

def train_dual_branch_model(
    model: nn.Module, 
    X_freq_train: np.ndarray, X_sparse_train: np.ndarray, y_train: np.ndarray,
    epochs: int = 30, batch_size: int = 64, lr: float = 0.001
) -> nn.Module:
    """Trains the Dual-Branch PyTorch network with class-weighted Cross-Entropy Loss."""
    
    y_train = y_train.astype(np.int64)
    total_classes = int(y_train.max() + 1)
    
    weights = np.ones(total_classes, dtype=np.float32)
    classes, counts = np.unique(y_train, return_counts=True)
    weights[classes] = 1.0 / counts
    weights = weights / weights.sum()
    tensor_weights = torch.FloatTensor(weights).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=tensor_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Reshape X_freq for PyTorch CNN (Batch, Channels, Freq_Bins)
    X_f_tr = torch.FloatTensor(X_freq_train).transpose(1, 2)
    X_s_tr = torch.FloatTensor(X_sparse_train)
    y_tr = torch.LongTensor(y_train)
    
    train_loader = DataLoader(
        TensorDataset(X_f_tr, X_s_tr, y_tr), 
        batch_size=batch_size, 
        shuffle=True
    )
    
    model.to(DEVICE)
    print(f"Training Dual-Branch Network on {DEVICE}...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_f, batch_s, batch_y in train_loader:
            batch_f = batch_f.to(DEVICE)
            batch_s = batch_s.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_f, batch_s)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")
            
    return model