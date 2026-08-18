import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from .config import DEVICE
except ImportError:
    from config import DEVICE

class FrozenMultiScaleExtractor(nn.Module):
    """
    Extracts high-dimensional shift-invariant features using randomly 
    initialized, non-trainable (frozen) multi-scale convolutions.
    """
    def __init__(self):
        super(FrozenMultiScaleExtractor, self).__init__()
        
        self.branch1 = nn.Conv2d(3, 128, kernel_size=(3, 3), padding=(1, 1))
        self.branch2 = nn.Conv2d(3, 128, kernel_size=(5, 3), padding=(2, 1))
        self.branch3 = nn.Conv2d(3, 128, kernel_size=(7, 3), padding=(3, 1))
        
        for param in self.parameters():
            param.requires_grad = False
            
        # FIX: Pool the 20 time steps down to 1, but keep the 10 electrodes isolated
        self.gap = nn.AdaptiveAvgPool2d((1, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        
        fused = torch.cat((out1, out2, out3), dim=1)
        
        # Pooled shape will now be (Batch, 384 channels, 1 time, 10 electrodes)
        pooled = self.gap(fused)
        
        # Flattening this preserves the spatial channel separation
        # Resulting feature vector size: 384 * 10 = 3840 features
        return pooled.view(pooled.size(0), -1)

def extract_frozen_features(X_vmd: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Passes the VMD dataset through the frozen network to extract tabular features."""
    extractor = FrozenMultiScaleExtractor().to(DEVICE)
    extractor.eval()
    
    tensor_x = torch.FloatTensor(X_vmd)
    loader = DataLoader(TensorDataset(tensor_x), batch_size=batch_size, shuffle=False)
    
    features = []
    print(f"Extracting Frozen Features on {DEVICE}...")
    with torch.no_grad():
        for batch in loader:
            batch_x = batch[0].to(DEVICE)
            out = extractor(batch_x)
            features.append(out.cpu().numpy())
            
    return np.vstack(features)