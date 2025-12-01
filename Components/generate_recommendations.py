import torch
import numpy as np
import pandas as pd


def multivae_recommend(
    model,
    X_dense,
    test_users,
    known_items,
    top_k=20,
):
    """
    Generate recommendations using MultiVAE following the original paper's logic.

    Parameters:
        model: trained MultiVAE model
        X_dense: torch.FloatTensor of shape (num_users, num_items)
        test_users: list/array of user ids for which to generate recommendations
        known_items: dict {user_id: list_of_seen_items}
        top_k: number of recommended items per user

    Returns:
        DataFrame with columns: user_id, item_id
    """

    model.eval()
    device = next(model.parameters()).device
    all_recs = []

    with torch.no_grad():
        for u in test_users:

            # 1) user vector
            x_u = X_dense[u].unsqueeze(0).to(device)  # shape (1, num_items)

            # 2) forward pass (VAE inference)
            logits, mu, logvar = model(x_u)

            # convert to numpy
            scores = logits.cpu().numpy().flatten()

            # 3) exclude known items (as the original implementation does)
            seen = known_items[u]
            scores[seen] = -np.inf

            # 4) take top-K
            top_items = np.argpartition(scores, -top_k)[-top_k:]
            top_items = top_items[np.argsort(scores[top_items])[::-1]]

            for item in top_items:
                all_recs.append([u, item])

    return pd.DataFrame(all_recs, columns=["user_id", "item_id"])


def save_submission(df, path="submission.csv"):
    df.to_csv(path, index=False)
    print(f"File saved to {path}")
