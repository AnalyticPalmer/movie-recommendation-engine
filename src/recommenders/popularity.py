from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = BASE_DIR / "data" / "processed" / "train_interactions.csv"
MOVIES_PATH = BASE_DIR / "data" / "processed" / "movies_processed.csv"


class PopularityRecommender:
    def __init__(self, min_ratings=50):
        self.min_ratings = min_ratings
        self.ranked_movies = None

    def fit(self):
        ratings = pd.read_csv(TRAIN_PATH)
        movies = pd.read_csv(MOVIES_PATH)

        stats = (
            ratings.groupby("movie_id")
            .agg(
                rating_count=("rating", "count"),
                average_rating=("rating", "mean"),
            )
            .reset_index()
        )

        global_mean = ratings["rating"].mean()

        eligible = stats[
            stats["rating_count"] >= self.min_ratings
        ].copy()

        m = self.min_ratings

        eligible["weighted_score"] = (
            (
                eligible["rating_count"]
                / (eligible["rating_count"] + m)
            )
            * eligible["average_rating"]
            +
            (
                m
                / (eligible["rating_count"] + m)
            )
            * global_mean
        )

        self.ranked_movies = eligible.merge(
            movies[["movie_id", "title"]],
            on="movie_id",
            how="left",
        ).sort_values(
            "weighted_score",
            ascending=False,
        )

        return self

    def recommend(self, top_k=10):
        if self.ranked_movies is None:
            raise ValueError(
                "Model has not been fitted yet."
            )

        return self.ranked_movies.head(top_k)[
            [
                "movie_id",
                "title",
                "rating_count",
                "average_rating",
                "weighted_score",
            ]
        ]


def main():
    model = PopularityRecommender(
        min_ratings=50
    )

    model.fit()

    print(
        "\nTop Popularity-Based Recommendations:\n"
    )

    print(
        model.recommend(top_k=10).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()