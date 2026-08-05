import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning
from ksvd import ApproximateKSVD

def train_minibatch_dictionary(X_train_windows, n_components=50, alpha=1.0, batch_size=32, max_iter=1000):
    """
    Treina o dicionário otimizado em lotes (MiniBatch). Rápido e ideal para bases grandes.
    """
    n_windows = X_train_windows.shape[0]
    X_flat = X_train_windows.reshape(n_windows, -1)
    
    print(f"Treinando MiniBatch Dictionary ({n_components} átomos)...")
    dict_learner = MiniBatchDictionaryLearning(
        n_components=n_components,
        alpha=alpha,
        batch_size=batch_size,
        max_iter=max_iter,
        random_state=42,
        transform_algorithm='omp'
    )
    dict_learner.fit(X_flat)
    return dict_learner, dict_learner.components_

def train_ksvd_dictionary(X_train_windows, n_components=50, transform_n_nonzero_coefs=5, max_iter=10):
    """
    Treina o dicionário usando a matemática real do K-SVD (via ApproximateKSVD).
    
    Parâmetros:
    -----------
    transform_n_nonzero_coefs : Define a esparsidade rígida (quantos átomos podem ser 
                                usados para reconstruir o sinal). No K-SVD clássico, 
                                controlamos o número de coeficientes (T0) em vez do erro (alpha).
    """
    n_windows = X_train_windows.shape[0]
    X_flat = X_train_windows.reshape(n_windows, -1)
    
    print(f"Treinando Approximate K-SVD ({n_components} átomos, {transform_n_nonzero_coefs} coeficientes ativos)...")
    dict_learner = ApproximateKSVD(
        n_components=n_components,
        transform_n_nonzero_coefs=transform_n_nonzero_coefs,
        max_iter=max_iter
    )
    
    dict_learner.fit(X_flat)
    return dict_learner, dict_learner.components_

def extract_sparse_codes(X_windows, dict_learner):
    """
    Usa o modelo treinado para encontrar o vetor esparso.
    Funciona tanto para o dict_learner do sklearn quanto para o ApproximateKSVD.
    """
    n_windows = X_windows.shape[0]
    X_flat = X_windows.reshape(n_windows, -1)
    return dict_learner.transform(X_flat)