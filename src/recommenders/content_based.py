from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parents[2]

MOVIES_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "movies_with_genres.csv"
)


class ContentBasedRecommender:

    def __init__(self):
        self.movies = None
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.movie_index = None

    def fit(self):

        self.movies = pd.read_csv(
            MOVIES_PATH
        )

        self.movies["genres"] = (
            self.movies["genres"]
            .fillna("")
        )

        vectorizer = TfidfVectorizer(
            token_pattern=r"(?u)\b[\w-]+\b"
        )

        self.tfidf_matrix = (
            vectorizer.fit_transform(
                self.movies["genres"]
            )
        )

        self.similarity_matrix = cosine_similarity(
            self.tfidf_matrix
        )

        self.movie_index = pd.Series(
            self.movies.index,
            index=self.movies["movie_id"],
        )

        return self

    def recommend_similar(
        self,
        movie_id,
        top_k=10,
    ):

        if self.similarity_matrix is None:
            raise ValueError(
                "Model has not been fitted."
            )

        if movie_id not in self.movie_index:
            raise ValueError(
                f"Movie ID {movie_id} not found."
            )

        index = self.movie_index[
            movie_id
        ]

        similarity_scores = list(
            enumerate(
                self.similarity_matrix[index]
            )
        )

        similarity_scores = sorted(
            similarity_scores,
            key=lambda x: x[1],
            reverse=True,
        )

        similarity_scores = (
            similarity_scores[1:top_k + 1]
        )

        movie_indices = [
            item[0]
            for item in similarity_scores
        ]

        scores = [
            item[1]
            for item in similarity_scores
        ]

        results = self.movies.iloc[
            movie_indices
        ][
            [
                "movie_id",
                "title",
                "genres",
            ]
        ].copy()

        results["similarity_score"] = scores

        return results


def main():

    model = ContentBasedRecommender()

    model.fit()

    movie_id = 50

    print(
        f"\nMovies similar to movie ID {movie_id}:\n"
    )

    print(
        model.recommend_similar(
            movie_id,
            top_k=10,
        ).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()