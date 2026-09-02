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
    / "hybrid_weight_tuning.csv"
)


def evaluate_weights(
    test,
    collaborative_weight,
    content_weight,
    top_k=10,
):
    model = HybridRecommender(
        collaborative_weight=collaborative_weight,
        content_weight=content_weight,
    )

    model.fit()

    precision_scores = []
    recall_scores = []
    hit_rate_scores = []
    ndcg_scores = []

    for row in test.itertuples(index=False):
        user_id = int(row.user_id)
        actual_movie_id = int(row.movie_id)

        if user_id not in model.collaborative_model.user_to_index:
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

        relevant_movies = [
            actual_movie_id
        ]

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

    return {
        "collaborative_weight": collaborative_weight,
        "content_weight": content_weight,
        "precision_at_10": sum(precision_scores) / len(precision_scores),
        "recall_at_10": sum(recall_scores) / len(recall_scores),
        "hit_rate_at_10": sum(hit_rate_scores) / len(hit_rate_scores),
        "ndcg_at_10": sum(ndcg_scores) / len(ndcg_scores),
    }


def main():
    test = pd.read_csv(TEST_PATH)

    weight_pairs = [
        (0.9, 0.1),
        (0.8, 0.2),
        (0.7, 0.3),
        (0.6, 0.4),
        (0.5, 0.5),
    ]

    results = []

    for collaborative_weight, content_weight in weight_pairs:
        print(
            f"\nTesting weights: "
            f"{collaborative_weight:.1f} / "
            f"{content_weight:.1f}"
        )

        result = evaluate_weights(
            test=test,
            collaborative_weight=collaborative_weight,
            content_weight=content_weight,
            top_k=10,
        )

        results.append(result)

        print(
            f"Hit Rate@10: "
            f"{result['hit_rate_at_10']:.4f}"
        )

        print(
            f"NDCG@10: "
            f"{result['ndcg_at_10']:.4f}"
        )

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        [
            "hit_rate_at_10",
            "ndcg_at_10",
        ],
        ascending=False,
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n=============================================")
    print("HYBRID WEIGHT TUNING RESULTS")
    print("=============================================")

    print(
        results_df.to_string(
            index=False
        )
    )

    best = results_df.iloc[0]

    print("\nBest weights:")
    print(
        f"Collaborative: "
        f"{best['collaborative_weight']:.1f}"
    )
    print(
        f"Content:       "
        f"{best['content_weight']:.1f}"
    )
    print(
        f"Hit Rate@10:   "
        f"{best['hit_rate_at_10']:.4f}"
    )
    print(
        f"NDCG@10:       "
        f"{best['ndcg_at_10']:.4f}"
    )

    print("\nResults saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()