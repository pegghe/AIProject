from recpack.util import get_top_K_values
import scipy as sp
import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from recpack.matrix import InteractionMatrix
from recpack.scenarios import StrongGeneralization
from recpack.util import get_top_K_ranks
from Components.my_cosine_similarity import my_cosine_similarity

def item_knn_scores(
    X_train: sp.sparse.csr_matrix, 
    X_test_in: sp.sparse.csr_matrix, 
    neighbor_count: int
) -> sp.sparse.csr_matrix:
    
    S = my_cosine_similarity(X_train)
    S_topk = get_top_K_values(S, neighbor_count) 
    scores = X_test_in @ S_topk.T
    scores = scores.tocsr()
    scores.eliminate_zeros()
    
    return scores
    
# helper function to turn a sparse matrix into a dataframe
def matrix2df(X) -> pd.DataFrame:
    coo = sp.sparse.coo_array(X)
    return pd.DataFrame({
        "user_id": coo.row,
        "item_id": coo.col,
        "value": coo.data
    })


# helper function to convert a score matrix into a dataframe of recommendations
def scores2recommendations(
    scores: sp.sparse.csr_matrix, 
    X_test_in: sp.sparse.csr_matrix, 
    recommendation_count: int,
    prevent_history_recos = True
) -> pd.DataFrame:
    # ensure you don't recommend fold-in items
    if prevent_history_recos:
        scores[(X_test_in > 0)] = 0
    # rank items
    ranks = get_top_K_ranks(scores, recommendation_count)
    # convert to a dataframe
    df_recos = matrix2df(ranks).rename(columns={"value": "rank"}).sort_values(["user_id", "rank"])
    return df_recos    

def save_user_item(df_recos, path="submission.csv"):
    """
    Save recommendations in user-item format for submission.

    Parameters:
        df_recos: DataFrame with columns ['user_id', 'item_id']
        path: file path for saving CSV
    """
    df = df_recos[["user_id", "item_id"]].copy()
    df.to_csv(path, index=False)
    print(f"User-item recommendation file saved to {path}")
