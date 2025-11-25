import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from scipy.sparse import csr_matrix
import scipy as sp
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(train_sparse, batch_size=1000):
    '''
    Calculates the cosine similarity of the given sparse matrix in batches.

    Parameters:
        train_sparse (sparse matrix): Contains the user-item interactions in a sparse format
        batch_size (int): Size of each batch

    Yields:
        start (int): The last processed entry of the user-item interaction matrix
        user_similarity_batch (sparse matrix): The calculated cosine similarity of the batch across
        the entire interaction matrix
    '''
    num_users = train_sparse.shape[0]
    # Looping through the users in the interaction matrix until the batch size limit
    for start in tqdm(range(0, num_users, batch_size)):
        # Extracting the batch of the interaction matrix
        end = min(start + batch_size, num_users)
        user_batch = train_sparse[start:end]
        
        # Calculating the cosine similarity of each batch against the entire interaction matrix
        user_similarity_batch = cosine_similarity(user_batch, train_sparse, dense_output=False)

        # Yielding the last processed entry and the calculated similarity matrix of the batch
        yield start, user_similarity_batch

def my_cosine_similarity(X: sp.sparse.csr_matrix) -> sp.sparse.csr_matrix:
    '''
    Calculate the cosine similarity matrix for a given sparse matrix of user-item interactions.

    Parameters:
        X (sparse matrix): The user-item interaction matrix in sparse format.

    Returns:
        sparse matrix: A sparse matrix of cosine similarities.
    '''
    # Convert the input DataFrame or matrix to sparse CSR format if necessary
    if not sp.sparse.isspmatrix_csr(X):
        X = csr_matrix(X)

    num_users = X.shape[0]
    similarity_matrix = np.zeros((num_users, num_users), dtype='float32')

    # Calculate the cosine similarity for each batch
    for start, similarity_batch in calculate_cosine_similarity(X):
        batch_size = similarity_batch.shape[0]
        similarity_matrix[start:start + batch_size, :] = similarity_batch.toarray()

    # Create a sparse matrix for the similarity matrix (to save memory)
    similarity_matrix_sparse = sp.sparse.csr_matrix(similarity_matrix)

    return similarity_matrix_sparse
