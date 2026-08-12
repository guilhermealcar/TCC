import numpy as np

def reconstruct_signal(sparse_coefficients: np.ndarray, dictionary: np.ndarray) -> np.ndarray:
    """
    Reconstructs the original signal using the sparse coefficients and the learned dictionary.
    Mathematically: X_recon = Sparse_Coefs (alpha) * Dictionary (D)
    """
    return np.dot(sparse_coefficients, dictionary)

def calculate_reconstruction_metrics(original: np.ndarray, reconstructed: np.ndarray) -> tuple[float, float]:
    """
    Computes reconstruction quality metrics.
    Returns:
        rmse (float): Root Mean Square Error
        prd (float): Percent Root-Mean-Square Difference
    """
    rmse = np.sqrt(np.mean((original - reconstructed)**2))
    
    numerator = np.sum((original - reconstructed)**2)
    denominator = np.sum(original**2)
    prd = np.sqrt(numerator / (denominator + 1e-8)) * 100.0
    
    return rmse, prd