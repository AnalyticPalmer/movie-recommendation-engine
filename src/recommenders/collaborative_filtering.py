import os

# Prevent OpenBLAS from creating multiple internal threads.
# This must be set BEFORE importing pandas, scipy, numpy, or implicit.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path

import pandas as pd
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares


BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "train_interactions.csv"
)

MOVIES_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "movies_processed.csv"
)


class CollaborativeRecommender:
    """
    Collaborative filtering recommender using
    Alternating Least Squares (ALS).
    """

    def __init__(
        self,
        factors=50,
        regularization=0.01,
        iterations=20,
        random_state=42,
    ):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.random_state = random_state

        self.model = None
        self.user_item_matrix = None

        self.user_to_index = {}
        self.index_to_user = {}

        self.movie_to_index = {}
        self.index_to_movie = {}

        self.movies = None

    def load_data(self):
        """Load training interactions and movie metadata."""

        ratings = pd.read_csv(TRAIN_PATH)
        movies = pd.read_csv(MOVIES_PATH)

        return ratings, movies

    def create_mappings(self, ratings):
        """Create user and movie ID mappings."""

        user_ids = sorted(
            ratings["user_id"].unique()
        )

        movie_ids = sorted(
            ratings["movie_id"].unique()
        )

        self.user_to_index = {
            user_id: index
            for index, user_id in enumerate(user_ids)
        }

        self.index_to_user = {
            index: user_id
            for user_id, index
            in self.user_to_index.items()
        }

        self.movie_to_index = {
            movie_id: index
            for index, movie_id in enumerate(movie_ids)
        }

        self.index_to_movie = {
            index: movie_id
            for movie_id, index
            in self.movie_to_index.items()
        }

    def build_user_item_matrix(self, ratings):
        """Convert ratings into a sparse user-item matrix."""

        rows = ratings["user_id"].map(
            self.user_to_index
        )

        columns = ratings["movie_id"].map(
            self.movie_to_index
        )

        values = ratings["rating"].astype(float)

        self.user_item_matrix = csr_matrix(
            (
                values,
                (
                    rows,
                    columns,
                ),
            ),
            shape=(
                len(self.user_to_index),
                len(self.movie_to_index),
            ),
        )

    def fit(self):
        """Train the ALS collaborative filtering model."""

        ratings, self.movies = self.load_data()

        self.create_mappings(
            ratings
        )

        self.build_user_item_matrix(
            ratings
        )

        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
        )

        self.model.fit(
            self.user_item_matrix
        )

        return self

    def recommend(
        self,
        user_id,
        top_k=10,
    ):
        """
        Generate personalized recommendations
        for an existing user.
        """

        if self.model is None:
            raise ValueError(
                "Model has not been fitted. Run fit() first."
            )

        if user_id not in self.user_to_index:
            raise ValueError(
                f"User ID {user_id} was not found."
            )

        user_index = self.user_to_index[
            user_id
        ]

        movie_indices, scores = (
            self.model.recommend(
                userid=user_index,
                user_items=self.user_item_matrix[
                    user_index
                ],
                N=top_k,
                filter_already_liked_items=True,
            )
        )

        movie_ids = [
            self.index_to_movie[
                int(movie_index)
            ]
            for movie_index in movie_indices
        ]

        recommendations = pd.DataFrame(
            {
                "movie_id": movie_ids,
                "score": scores,
            }
        )

        recommendations = recommendations.merge(
            self.movies[
                [
                    "movie_id",
                    "title",
                ]
            ],
            on="movie_id",
            how="left",
        )

        recommendations = recommendations[
            [
                "movie_id",
                "title",
                "score",
            ]
        ]

        return recommendations

    def get_user_history(
        self,
        user_id,
        top_n=None,
    ):
        """Return movies previously rated by a user."""

        if user_id not in self.user_to_index:
            raise ValueError(
                f"User ID {user_id} was not found."
            )

        ratings = pd.read_csv(
            TRAIN_PATH
        )

        history = ratings[
            ratings["user_id"] == user_id
        ].copy()

        history = history.merge(
            self.movies[
                [
                    "movie_id",
                    "title",
                ]
            ],
            on="movie_id",
            how="left",
        )

        history = history.sort_values(
            "rating",
            ascending=False,
        )

        columns = [
            "movie_id",
            "title",
            "rating",
        ]

        if top_n is not None:
            return history[
                columns
            ].head(top_n)

        return history[
            columns
        ]


def main():
    print(
        "Training collaborative filtering model..."
    )

    model = CollaborativeRecommender(
        factors=50,
        regularization=0.01,
        iterations=20,
        random_state=42,
    )

    model.fit()

    user_id = 1

    print(
        f"\nRecommendations for user {user_id}:\n"
    )

    recommendations = model.recommend(
        user_id=user_id,
        top_k=10,
    )

    print(
        recommendations.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()