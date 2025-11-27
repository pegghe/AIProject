import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from scipy.sparse import csr_matrix
import scipy as sp
from sklearn.metrics.pairwise import cosine_similarity

def my_cosine_similarity(X: sp.sparse.csr_matrix) -> sp.sparse.csr_matrix:
    # TODO: implement cosine similarity function
    # hint: should return a sparse (item x item)-matrix
    # hint: work with the entire matrix to benefit from highly optimized vectorized operations in numpy and scipy
    # important: don't forget to set the diagonal to 0 (no self-similarity)
    X = X.tocsc()
    norms = np.sqrt(X.power(2).sum(axis=0)) 
    norms = np.asarray(norms).flatten()
    norms[norms == 0] = 1e-10
    X_normalized = X / norms
    S = X_normalized.T @ X_normalized  
    S.setdiag(0)
    S.eliminate_zeros()
    
    return S
