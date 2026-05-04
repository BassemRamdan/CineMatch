import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from collections import defaultdict

def evaluate_collaborative_model(model, test_df):
    """
    Evaluate the collaborative filtering model using RMSE and MAE.
    
    Args:
        model: Fitted CollaborativeRecommender instance
        test_df: Test dataframe with 'userId', 'movieId', 'rating'
    """
    predictions = []
    y_true = []
    y_pred = []
    
    for _, row in test_df.iterrows():
        true_r = row['rating']
        uid = row['userId']
        iid = row['movieId']
        
        est = model.predict(uid, iid)
        # Clip est to reasonable bounds for rating
        est = max(0.5, min(5.0, est))
        
        y_true.append(true_r)
        y_pred.append(est)
        
        predictions.append((uid, iid, true_r, est, None))
        
    if len(y_true) > 0:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
    else:
        rmse = 0
        mae = 0
    
    return {"RMSE": rmse, "MAE": mae, "predictions": predictions}

def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """
    Return precision and recall at k metrics for each user.
    """
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()

    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    mean_precision = sum(prec for prec in precisions.values()) / len(precisions) if precisions else 0
    mean_recall = sum(rec for rec in recalls.values()) / len(recalls) if recalls else 0
    
    f1_score = 0
    if (mean_precision + mean_recall) > 0:
        f1_score = 2 * (mean_precision * mean_recall) / (mean_precision + mean_recall)
        
    return {
        "Precision@K": mean_precision,
        "Recall@K": mean_recall,
        "F1-Score": f1_score
    }

def run_full_evaluation(model, test_df, k=10, threshold=3.5):
    """Run all evaluation metrics and return a report dictionary."""
    collab_eval = evaluate_collaborative_model(model, test_df)
    ir_metrics = precision_recall_at_k(collab_eval["predictions"], k=k, threshold=threshold)
    
    return {
        "RMSE": collab_eval["RMSE"],
        "MAE": collab_eval["MAE"],
        "Precision@K": ir_metrics["Precision@K"],
        "Recall@K": ir_metrics["Recall@K"],
        "F1-Score": ir_metrics["F1-Score"]
    }
