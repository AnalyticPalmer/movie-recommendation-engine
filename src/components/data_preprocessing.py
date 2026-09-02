from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

INTERIM_DIR = BASE_DIR / "data" / "interim"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    ratings = pd.read_csv(
        INTERIM_DIR / "ratings.csv"
    )

    movies = pd.read_csv(
        INTERIM_DIR / "movies.csv"
    )

    return ratings, movies


def clean_ratings(ratings):
    ratings = ratings.copy()

    ratings = ratings.dropna(
        subset=["user_id", "movie_id", "rating"]
    )

    ratings = ratings.drop_duplicates()

    ratings = ratings[
        ratings["rating"].between(1, 5)
    ]

    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["movie_id"] = ratings["movie_id"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)

    ratings["datetime"] = pd.to_datetime(
        ratings["timestamp"],
        unit="s"
    )

    return ratings


def clean_movies(movies):
    movies = movies.copy()

    movies = movies.drop_duplicates(
        subset=["movie_id"]
    )

    movies["movie_id"] = movies[
        "movie_id"
    ].astype(int)

    movies["title"] = movies[
        "title"
    ].fillna("Unknown")

    return movies


def temporal_train_test_split(ratings):
    ratings = ratings.sort_values(
        ["user_id", "timestamp"]
    )

    test_indices = (
        ratings
        .groupby("user_id")
        .tail(1)
        .index
    )

    test = ratings.loc[test_indices]

    train = ratings.drop(test_indices)

    return train, test


def validate_split(train, test):
    overlap = set(train.index).intersection(
        set(test.index)
    )

    if overlap:
        raise ValueError(
            "Train/test data leakage detected."
        )

    print("Train interactions:", len(train))
    print("Test interactions:", len(test))
    print("Train users:", train["user_id"].nunique())
    print("Test users:", test["user_id"].nunique())


def save_processed_data(train, test, movies):
    train.to_csv(
        PROCESSED_DIR / "train_interactions.csv",
        index=False
    )

    test.to_csv(
        PROCESSED_DIR / "test_interactions.csv",
        index=False
    )

    movies.to_csv(
        PROCESSED_DIR / "movies_processed.csv",
        index=False
    )


def main():
    print("Starting data preprocessing...\n")

    ratings, movies = load_data()

    ratings = clean_ratings(ratings)
    movies = clean_movies(movies)

    train, test = temporal_train_test_split(
        ratings
    )

    validate_split(train, test)

    save_processed_data(
        train,
        test,
        movies
    )

    print(
        "\nData preprocessing completed successfully."
    )


if __name__ == "__main__":
    main()