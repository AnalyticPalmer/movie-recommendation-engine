import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path

import pandas as pd

from src.evaluation.recommender_metrics import (
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    ndcg_at_k,
)
from src.recommenders.hybrid import HybridRecommender


BASE_DIR = Path(__file__).resolve().parents[2]

TEST_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "test_interactions.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "reports"
    / "metrics"
    / "hybrid_evaluation.csv"
)


def evaluate_hybrid(
    top_k=10,
    collaborative_weight=0.7,
    content_weight=0.3,
):
    test = pd.read_csv(TEST_PATH)

    print("Training hybrid recommender...\n")

    model = HybridRecommender(
        collaborative_weight=collaborative_weight,
        content_weight=content_weight,
    )

    model.fit()

    results = []
    skipped_users = 0

    for row in test.itertuples(index=False):
        user_id = int(row.user_id)
        actual_movie_id = int(row.movie_id)

        if user_id not in model.collaborative_model.user_to_index:
            skipped_users += 1
            continue

        try:
            recommendations = model.recommend(
                user_id=user_id,
                top_k=top_k,
            )
        except ValueError:
            skipped_users += 1
            continue

        recommended_movies = (
            recommendations["movie_id"]
            .astype(int)
            .tolist()
        )

        relevant_movies = [
            actual_movie_id
        ]

        results.append(
            {
                "user_id": user_id,
                "actual_movie_id": actual_movie_id,
                "precision_at_k": precision_at_k(
                    recommended_movies,
                    relevant_movies,
                    top_k,
                ),
                "recall_at_k": recall_at_k(
                    recommended_movies,
                    relevant_movies,
                    top_k,
                ),
                "hit_rate_at_k": hit_rate_at_k(
                    recommended_movies,
                    relevant_movies,
                    top_k,
                ),
                "ndcg_at_k": ndcg_at_k(
                    recommended_movies,
                    relevant_movies,
                    top_k,
                ),
            }
        )

    results_df = pd.DataFrame(results)

    if results_df.empty:
        raise RuntimeError(
            "Hybrid evaluation produced no results."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    precision = results_df[
        "precision_at_k"
    ].mean()

    recall = results_df[
        "recall_at_k"
    ].mean()

    hit_rate = results_df[
        "hit_rate_at_k"
    ].mean()

    ndcg = results_df[
        "ndcg_at_k"
    ].mean()

    print("\n=============================================")
    print("HYBRID RECOMMENDER EVALUATION")
    print("=============================================")
    print(f"Users evaluated: {len(results_df)}")
    print(f"Users skipped:   {skipped_users}")
    print(f"Top K:           {top_k}")
    print(
        f"Collaborative weight: "
        f"{collaborative_weight:.2f}"
    )
    print(
        f"Content weight:       "
        f"{content_weight:.2f}"
    )
    print("---------------------------------------------")
    print(f"Precision@{top_k}: {precision:.4f}")
    print(f"Recall@{top_k}:    {recall:.4f}")
    print(f"Hit Rate@{top_k}:  {hit_rate:.4f}")
    print(f"NDCG@{top_k}:      {ndcg:.4f}")
    print("=============================================")

    print("\nEvaluation saved to:")
    print(OUTPUT_PATH)

    return results_df


if __name__ == "__main__":
    evaluate_hybrid(
        top_k=10,
        collaborative_weight=0.7,
        content_weight=0.3,
    )