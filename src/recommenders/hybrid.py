import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path

import pandas as pd

from src.recommenders.collaborative_filtering import (
    CollaborativeRecommender,
)
from src.recommenders.content_based import (
    ContentBasedRecommender,
)


BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "train_interactions.csv"
)


class HybridRecommender:
    def __init__(
        self,
        collaborative_weight=0.7,
        content_weight=0.3,
    ):
        if collaborative_weight < 0 or content_weight < 0:
            raise ValueError(
                "Model weights must be non-negative."
            )

        total_weight = (
            collaborative_weight
            + content_weight
        )

        if total_weight == 0:
            raise ValueError(
                "At least one model weight must be greater than zero."
            )

        self.collaborative_weight = (
            collaborative_weight / total_weight
        )
        self.content_weight = (
            content_weight / total_weight
        )

        self.collaborative_model = (
            CollaborativeRecommender()
        )

        self.content_model = (
            ContentBasedRecommender()
        )

        self.train = None

    def fit(self):
        self.train = pd.read_csv(
            TRAIN_PATH
        )

        self.collaborative_model.fit()
        self.content_model.fit()

        return self

    def recommend(
        self,
        user_id,
        top_k=10,
        candidate_k=50,
    ):
        if self.train is None:
            raise ValueError(
                "Model has not been fitted. Run fit() first."
            )

        if top_k < 1:
            raise ValueError(
                "top_k must be at least 1."
            )

        candidate_k = max(
            candidate_k,
            top_k,
        )

        collaborative_recs = (
            self.collaborative_model.recommend(
                user_id=user_id,
                top_k=candidate_k,
            )
            .copy()
        )

        if collaborative_recs.empty:
            return pd.DataFrame(
                columns=[
                    "movie_id",
                    "title",
                    "collaborative_score",
                    "content_score",
                    "hybrid_score",
                ]
            )

        min_score = (
            collaborative_recs["score"].min()
        )

        max_score = (
            collaborative_recs["score"].max()
        )

        if max_score > min_score:
            collaborative_recs[
                "collaborative_score"
            ] = (
                collaborative_recs["score"]
                - min_score
            ) / (
                max_score
                - min_score
            )
        else:
            collaborative_recs[
                "collaborative_score"
            ] = 1.0

        user_history = self.train[
            self.train["user_id"] == user_id
        ]

        liked_movies = user_history[
            user_history["rating"] >= 4
        ]

        content_scores = {}

        for movie_id in liked_movies[
            "movie_id"
        ]:
            try:
                similar_movies = (
                    self.content_model
                    .recommend_similar(
                        movie_id=int(movie_id),
                        top_k=candidate_k,
                    )
                )
            except ValueError:
                continue

            for row in similar_movies.itertuples():
                candidate_movie_id = int(
                    row.movie_id
                )

                similarity_score = float(
                    row.similarity_score
                )

                current_score = (
                    content_scores.get(
                        candidate_movie_id,
                        0.0,
                    )
                )

                content_scores[
                    candidate_movie_id
                ] = max(
                    current_score,
                    similarity_score,
                )

        collaborative_recs[
            "content_score"
        ] = (
            collaborative_recs["movie_id"]
            .map(content_scores)
            .fillna(0.0)
        )

        collaborative_recs[
            "hybrid_score"
        ] = (
            self.collaborative_weight
            * collaborative_recs[
                "collaborative_score"
            ]
            +
            self.content_weight
            * collaborative_recs[
                "content_score"
            ]
        )

        results = (
            collaborative_recs
            .sort_values(
                [
                    "hybrid_score",
                    "collaborative_score",
                ],
                ascending=False,
            )
            .head(top_k)
        )

        return results[
            [
                "movie_id",
                "title",
                "collaborative_score",
                "content_score",
                "hybrid_score",
            ]
        ].reset_index(drop=True)


def main():
    model = HybridRecommender(
        collaborative_weight=0.7,
        content_weight=0.3,
    )

    model.fit()

    user_id = 1

    recommendations = model.recommend(
        user_id=user_id,
        top_k=10,
    )

    print(
        f"\nHybrid recommendations for user {user_id}:\n"
    )

    print(
        recommendations.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()