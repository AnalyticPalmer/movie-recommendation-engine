import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path

import joblib

from src.recommenders.hybrid import HybridRecommender


BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"


class RecommendationPipeline:
    def __init__(self):
        self.popularity_model = None
        self.content_model = None
        self.collaborative_model = None
        self.hybrid_model = None

    def load_popularity_model(self):
        if self.popularity_model is None:
            model_path = (
                MODELS_DIR
                / "popularity_model.pkl"
            )

            self._validate_model_path(
                model_path
            )

            self.popularity_model = joblib.load(
                model_path
            )

        return self.popularity_model

    def load_content_model(self):
        if self.content_model is None:
            model_path = (
                MODELS_DIR
                / "content_model.pkl"
            )

            self._validate_model_path(
                model_path
            )

            self.content_model = joblib.load(
                model_path
            )

        return self.content_model

    def load_collaborative_model(self):
        if self.collaborative_model is None:
            model_path = (
                MODELS_DIR
                / "collaborative_model.pkl"
            )

            self._validate_model_path(
                model_path
            )

            self.collaborative_model = joblib.load(
                model_path
            )

        return self.collaborative_model

    def load_hybrid_model(self):
        if self.hybrid_model is None:
            self.hybrid_model = HybridRecommender(
                collaborative_weight=0.7,
                content_weight=0.3,
            )

            self.hybrid_model.collaborative_model = (
                self.load_collaborative_model()
            )

            self.hybrid_model.content_model = (
                self.load_content_model()
            )

        return self.hybrid_model

    def recommend_popular(
        self,
        top_k=10,
    ):
        self._validate_top_k(
            top_k
        )

        model = self.load_popularity_model()

        return model.recommend(
            top_k=top_k
        )

    def recommend_similar(
        self,
        movie_id,
        top_k=10,
    ):
        self._validate_positive_integer(
            movie_id,
            "movie_id",
        )

        self._validate_top_k(
            top_k
        )

        model = self.load_content_model()

        return model.recommend_similar(
            movie_id=movie_id,
            top_k=top_k,
        )

    def recommend_for_user(
        self,
        user_id,
        top_k=10,
        model_type="collaborative",
    ):
        self._validate_positive_integer(
            user_id,
            "user_id",
        )

        self._validate_top_k(
            top_k
        )

        model_type = (
            model_type
            .lower()
            .strip()
        )

        if model_type == "collaborative":
            model = (
                self.load_collaborative_model()
            )

        elif model_type == "hybrid":
            model = (
                self.load_hybrid_model()
            )

        else:
            raise ValueError(
                "model_type must be "
                "'collaborative' or 'hybrid'."
            )

        return model.recommend(
            user_id=user_id,
            top_k=top_k,
        )

    @staticmethod
    def _validate_model_path(
        model_path,
    ):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: "
                f"{model_path}"
            )

    @staticmethod
    def _validate_top_k(
        top_k,
    ):
        if not isinstance(
            top_k,
            int,
        ):
            raise TypeError(
                "top_k must be an integer."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

    @staticmethod
    def _validate_positive_integer(
        value,
        name,
    ):
        if not isinstance(
            value,
            int,
        ):
            raise TypeError(
                f"{name} must be an integer."
            )

        if value <= 0:
            raise ValueError(
                f"{name} must be greater than 0."
            )


def main():
    pipeline = RecommendationPipeline()

    recommendations = (
        pipeline.recommend_for_user(
            user_id=1,
            top_k=10,
            model_type="collaborative",
        )
    )

    print(
        "\nRecommendations for user 1:\n"
    )

    print(
        recommendations.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()