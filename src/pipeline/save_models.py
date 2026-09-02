import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path

import joblib

from src.recommenders.collaborative_filtering import CollaborativeRecommender
from src.recommenders.content_based import ContentBasedRecommender
from src.recommenders.popularity import PopularityRecommender


BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"


def save_models():
    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Training popularity recommender...")

    popularity_model = PopularityRecommender(
        min_ratings=50
    )
    popularity_model.fit()

    print("Saving popularity recommender...")

    joblib.dump(
        popularity_model,
        MODELS_DIR / "popularity_model.pkl",
    )

    print("Training content-based recommender...")

    content_model = ContentBasedRecommender()
    content_model.fit()

    print("Saving content-based recommender...")

    joblib.dump(
        content_model,
        MODELS_DIR / "content_model.pkl",
    )

    print("Training collaborative recommender...")

    collaborative_model = CollaborativeRecommender(
        factors=50,
        regularization=0.01,
        iterations=20,
        random_state=42,
    )
    collaborative_model.fit()

    print("Saving collaborative recommender...")

    joblib.dump(
        collaborative_model,
        MODELS_DIR / "collaborative_model.pkl",
    )

    print("\nModels saved successfully.")

    print(
        MODELS_DIR
        / "popularity_model.pkl"
    )

    print(
        MODELS_DIR
        / "content_model.pkl"
    )

    print(
        MODELS_DIR
        / "collaborative_model.pkl"
    )


def main():
    save_models()


if __name__ == "__main__":
    main()