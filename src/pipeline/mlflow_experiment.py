import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path

import mlflow
import pandas as pd

from src.evaluation.recommender_metrics import (
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    ndcg_at_k,
)
from src.recommenders.collaborative_filtering import CollaborativeRecommender


BASE_DIR = Path(__file__).resolve().parents[2]
TEST_PATH = BASE_DIR / "data" / "processed" / "test_interactions.csv"
MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"
TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
EXPERIMENT_NAME = "recommendation-engine"


def evaluate_model(model, test, top_k=10):
    precision_scores = []
    recall_scores = []
    hit_rate_scores = []
    ndcg_scores = []

    for row in test.itertuples(index=False):
        user_id = int(row.user_id)
        actual_movie_id = int(row.movie_id)

        if user_id not in model.user_to_index:
            continue

        recommendations = model.recommend(
            user_id=user_id,
            top_k=top_k,
        )

        recommended_movies = (
            recommendations["movie_id"]
            .astype(int)
            .tolist()
        )

        relevant_movies = [actual_movie_id]

        precision_scores.append(
            precision_at_k(
                recommended_movies,
                relevant_movies,
                top_k,
            )
        )

        recall_scores.append(
            recall_at_k(
                recommended_movies,
                relevant_movies,
                top_k,
            )
        )

        hit_rate_scores.append(
            hit_rate_at_k(
                recommended_movies,
                relevant_movies,
                top_k,
            )
        )

        ndcg_scores.append(
            ndcg_at_k(
                recommended_movies,
                relevant_movies,
                top_k,
            )
        )

    if not precision_scores:
        raise RuntimeError("No users were evaluated.")

    return {
        "precision_at_10": sum(precision_scores) / len(precision_scores),
        "recall_at_10": sum(recall_scores) / len(recall_scores),
        "hit_rate_at_10": sum(hit_rate_scores) / len(hit_rate_scores),
        "ndcg_at_10": sum(ndcg_scores) / len(ndcg_scores),
        "users_evaluated": len(precision_scores),
    }


def run_experiment():
    test = pd.read_csv(TEST_PATH)

    mlflow.set_tracking_uri(TRACKING_URI)

    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    mlflow.set_experiment(EXPERIMENT_NAME)

    parameters = {
        "factors": 50,
        "regularization": 0.01,
        "iterations": 20,
        "random_state": 42,
        "top_k": 10,
    }

    print("Training collaborative filtering model...")

    model = CollaborativeRecommender(
        factors=parameters["factors"],
        regularization=parameters["regularization"],
        iterations=parameters["iterations"],
        random_state=parameters["random_state"],
    )

    model.fit()

    print("\nEvaluating model...")

    metrics = evaluate_model(
        model=model,
        test=test,
        top_k=parameters["top_k"],
    )

    with mlflow.start_run(run_name="als_baseline"):
        mlflow.log_params(parameters)
        mlflow.log_metrics(metrics)

        mlflow.set_tags(
            {
                "model_type": "ALS Collaborative Filtering",
                "dataset": "MovieLens 100K",
            }
        )

    print("\nMLflow experiment completed successfully.")
    print(f"Precision@10: {metrics['precision_at_10']:.4f}")
    print(f"Recall@10: {metrics['recall_at_10']:.4f}")
    print(f"Hit Rate@10: {metrics['hit_rate_at_10']:.4f}")
    print(f"NDCG@10: {metrics['ndcg_at_10']:.4f}")
    print(f"Users evaluated: {metrics['users_evaluated']}")
    print(f"Tracking database: {MLFLOW_DB_PATH}")


def main():
    run_experiment()


if __name__ == "__main__":
    main()