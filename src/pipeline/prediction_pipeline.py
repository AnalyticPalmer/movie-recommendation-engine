import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path

import joblib


BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"


class RecommendationPipeline:
    def __init__(self):
        self.popularity_model = None
        self.content_model = None

    def load_popularity_model(self):
        if self.popularity_model is None:
            model_path = MODELS_DIR / "popularity_model.pkl"
            self._validate_model_path(model_path)
            self.popularity_model = joblib.load(model_path)

        return self.popularity_model

    def load_content_model(self):
        if self.content_model is None:
            model_path = MODELS_DIR / "content_model.pkl"
            self._validate_model_path(model_path)
            self.content_model = joblib.load(model_path)

        return self.content_model

    def recommend_popular(self, top_k=10):
        self._validate_top_k(top_k)

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

        self._validate_top_k(top_k)

        model = self.load_content_model()

        return model.recommend_similar(
            movie_id=movie_id,
            top_k=top_k,
        )

    @staticmethod
    def _validate_model_path(model_path):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

    @staticmethod
    def _validate_top_k(top_k):
        if not isinstance(top_k, int):
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
        if not isinstance(value, int):
            raise TypeError(
                f"{name} must be an integer."
            )

        if value <= 0:
            raise ValueError(
                f"{name} must be greater than 0."
            )