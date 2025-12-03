import torch
import numpy as np
import pandas as pd

def multivae_recommend(
    model,
    X_dense_test_in,
    index_to_user,
    known_items,
    top_k=20
):
    """
    Generate recommendations for test users using MultiVAE.
    Works with the mapping-based pipeline we built.

    Parameters:
        model               : trained MultiVAE
        X_dense_test_in     : dense test matrix (only test users)
        index_to_user       : dict mapping internal uid → real user_id
        known_items         : dict {internal_uid: list_of_seen_items}
        top_k               : number of recommended items

    Returns:
        DataFrame: columns [user_id, item_id]
    """

    model.eval()
    device = next(model.parameters()).device

    X_dense_test_in = X_dense_test_in.to(device)

    with torch.no_grad():
        logits, mu, logvar = model(X_dense_test_in)

    # convert to numpy
    scores = logits.cpu().numpy()  # shape [num_test_users, num_items]

    recommendations = []

    for uid_mapped in range(scores.shape[0]):
        user_scores = scores[uid_mapped]

        # mask known items
        seen = known_items.get(uid_mapped, [])
        user_scores[seen] = -np.inf

        # top-k items
        top_items = np.argpartition(user_scores, -top_k)[-top_k:]
        top_items = top_items[np.argsort(user_scores[top_items])[::-1]]

        # convert internal uid → real uid
        real_uid = index_to_user[uid_mapped]

        for item in top_items:
            recommendations.append([real_uid, int(item)])

    return pd.DataFrame(recommendations, columns=["user_id", "item_id"])



def save_submission(df, path="submission.csv"):
    df.to_csv(path, index=False)
    print(f"File saved to {path}")
