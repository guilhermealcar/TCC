import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def train_svm_classifier(X_train: np.ndarray, y_train: np.ndarray, kernel: str = "rbf", **kwargs) -> Any:
    """Trains an SVM with class balancing to penalize majority class bias."""
    print(f"Training SVM (Kernel: {kernel}) on {X_train.shape[0]} samples...")
    clf = SVC(class_weight='balanced', kernel=kernel, random_state=42, **kwargs)
    clf.fit(X_train, y_train)
    print("Training complete.")
    return clf

def evaluate_classifier(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    """Evaluates the model and returns accuracy, Macro F1, and raw matrices."""
    print("Evaluating model on test set...")
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'macro_f1': f1_score(y_test, y_pred, average='macro'),
        'report': classification_report(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

# Set device automatically
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class sEMG1DCNN(nn.Module):
    """
    A 1D Convolutional Neural Network designed for short temporal windows (e.g., 20 samples).
    Input shape: (Batch, Channels, Window_Size)
    """
    def __init__(self, num_channels: int, num_classes: int):
        super(sEMG1DCNN, self).__init__()
        
        # Block 1: Feature Extraction
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2) # Halves the temporal dimension (20 -> 10)
        )
        
        # Block 2: Deep Feature Extraction
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2) # Halves the temporal dimension (10 -> 5)
        )
        
        # Block 3: Classification Head
        # If input window is 20 samples, output of Block 2 is 5 temporal steps
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # Prevents overfitting on the dominant classes
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x

def train_pytorch_model(model: nn.Module, X_train: np.ndarray, y_train: np.ndarray, 
                        X_test: np.ndarray, y_test: np.ndarray, 
                        epochs: int = 30, batch_size: int = 64, lr: float = 0.001) -> nn.Module:
    """Trains the PyTorch CNN using a safely mapped class-weighted Cross-Entropy Loss."""
    
    # 0. FORCE INTEGER TYPES: Eradicate any float64 contamination from preprocessing
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)
    
    # 1. Safely calculate class weights for the exact number of network outputs
    total_classes = int(max(y_train.max(), y_test.max()) + 1)
    
    # Initialize weights with 1.0 (prevents division by zero for missing classes)
    weights = np.ones(total_classes, dtype=np.float32)
    
    # Calculate inverse frequencies only for classes actually present in training
    classes, counts = np.unique(y_train, return_counts=True)
    
    # Because y_train is np.int64, classes is now guaranteed to be np.int64, making indexing safe
    weights[classes] = 1.0 / counts
    
    # Normalize weights so they sum to 1
    weights = weights / weights.sum()
    tensor_weights = torch.FloatTensor(weights).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=tensor_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 2. Reshape for PyTorch: (Batch, Channels, Window_Size)
    X_tr = torch.FloatTensor(X_train).transpose(1, 2)
    y_tr = torch.LongTensor(y_train)
    X_te = torch.FloatTensor(X_test).transpose(1, 2)
    y_te = torch.LongTensor(y_test)
    
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    
    model.to(DEVICE)
    print(f"Training on {DEVICE}...")
    
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