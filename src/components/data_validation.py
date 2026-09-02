from pathlib import Path
import json

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

INTERIM_DIR = BASE_DIR / "data" / "interim"
REPORT_DIR = BASE_DIR / "reports" / "metrics"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    ratings = pd.read_csv(INTERIM_DIR / "ratings.csv")
    movies = pd.read_csv(INTERIM_DIR / "movies.csv")

    return ratings, movies


def validate_data(ratings, movies):

    report = {}

    # Basic dataset information
    report["ratings_rows"] = len(ratings)
    report["movies_rows"] = len(movies)

    report["unique_users"] = ratings["user_id"].nunique()
    report["unique_movies"] = ratings["movie_id"].nunique()

    # Missing values
    report["missing_user_ids"] = int(
        ratings["user_id"].isna().sum()
    )

    report["missing_movie_ids"] = int(
        ratings["movie_id"].isna().sum()
    )

    report["missing_ratings"] = int(
        ratings["rating"].isna().sum()
    )

    # Duplicate rows
    report["duplicate_ratings"] = int(
        ratings.duplicated().sum()
    )

    report["duplicate_movies"] = int(
        movies.duplicated().sum()
    )

    # Invalid ratings
    invalid_ratings = ratings[
        ~ratings["rating"].between(1, 5)
    ]

    report["invalid_ratings"] = len(invalid_ratings)

    # Interaction statistics
    interactions_per_user = ratings.groupby(
        "user_id"
    ).size()

    interactions_per_movie = ratings.groupby(
        "movie_id"
    ).size()

    report["average_interactions_per_user"] = float(
        interactions_per_user.mean()
    )

    report["median_interactions_per_user"] = float(
        interactions_per_user.median()
    )

    report["average_interactions_per_movie"] = float(
        interactions_per_movie.mean()
    )

    # User-item matrix sparsity
    total_possible_interactions = (
        ratings["user_id"].nunique()
        * ratings["movie_id"].nunique()
    )

    actual_interactions = len(ratings)

    sparsity = (
        1 - (actual_interactions / total_possible_interactions)
    ) * 100

    report["matrix_sparsity_percent"] = float(sparsity)

    # IDs appearing in ratings but missing from movie metadata
    rating_movie_ids = set(ratings["movie_id"])
    metadata_movie_ids = set(movies["movie_id"])

    missing_metadata = rating_movie_ids - metadata_movie_ids

    report["movies_missing_metadata"] = len(
        missing_metadata
    )

    # Cold-start style statistics
    report["users_with_less_than_5_interactions"] = int(
        (interactions_per_user < 5).sum()
    )

    report["movies_with_less_than_5_interactions"] = int(
        (interactions_per_movie < 5).sum()
    )

    # Overall validation
    critical_errors = (
        report["missing_user_ids"]
        + report["missing_movie_ids"]
        + report["missing_ratings"]
        + report["invalid_ratings"]
        + report["movies_missing_metadata"]
    )

    report["validation_passed"] = critical_errors == 0

    return report


def save_report(report):

    output_path = (
        REPORT_DIR / "data_validation.json"
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4
        )

    return output_path


def main():

    print("Starting data validation...\n")

    ratings, movies = load_data()

    report = validate_data(
        ratings,
        movies
    )

    for key, value in report.items():
        print(f"{key}: {value}")

    output_path = save_report(report)

    print(
        f"\nValidation report saved to:\n{output_path}"
    )

    if report["validation_passed"]:
        print("\nDATA VALIDATION PASSED")
    else:
        print("\nDATA VALIDATION FAILED")


if __name__ == "__main__":
    main()