from pathlib import Path

from src.recommenders.collaborative_filtering import (
    CollaborativeRecommender,
)
from src.recommenders.content_based import (
    ContentBasedRecommender,
)
from src.recommenders.popularity import (
    PopularityRecommender,
)


BASE_DIR = Path(__file__).resolve().parents[2]

REPORT_PATH = (
    BASE_DIR
    / "reports"
    / "metrics"
    / "training_summary.txt"
)


def train_models():
    print("Training popularity recommender...")
    popularity_model = PopularityRecommender(
        min_ratings=50
    )
    popularity_model.fit()

    print("Training content-based recommender...")
    content_model = ContentBasedRecommender()
    content_model.fit()

    print("Training collaborative recommender...")
    collaborative_model = CollaborativeRecommender(
        factors=50,
        regularization=0.01,
        iterations=20,
        random_state=42,
    )
    collaborative_model.fit()

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = (
        "Recommendation Engine Training Summary\n"
        "======================================\n"
        "Popularity model: trained\n"
        "Content-based model: trained\n"
        "Collaborative model: trained\n"
    )

    REPORT_PATH.write_text(
        summary,
        encoding="utf-8",
    )

    print("\nTraining completed successfully.")
    print(f"Summary saved to: {REPORT_PATH}")

    return {
        "popularity": popularity_model,
        "content_based": content_model,
        "collaborative": collaborative_model,
    }


def main():
    train_models()


if __name__ == "__main__":
    main()