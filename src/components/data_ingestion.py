from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "data" / "raw"
INTERIM_DIR = BASE_DIR / "data" / "interim"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)


def load_ratings():
    columns = [
        "user_id",
        "movie_id",
        "rating",
        "timestamp"
    ]

    ratings = pd.read_csv(
        RAW_DIR / "u.data",
        sep="\t",
        names=columns
    )

    return ratings


def load_movies():
    columns = [
        "movie_id",
        "title",
        "release_date",
        "video_release_date",
        "imdb_url"
    ]

    movies = pd.read_csv(
        RAW_DIR / "u.item",
        sep="|",
        encoding="latin-1",
        usecols=range(5),
        names=columns
    )

    return movies


def main():

    print("Starting data ingestion...")

    ratings = load_ratings()
    movies = load_movies()

    print("\nRatings shape:")
    print(ratings.shape)

    print("\nMovies shape:")
    print(movies.shape)

    print("\nRatings preview:")
    print(ratings.head())

    print("\nMovies preview:")
    print(movies.head())

    ratings.to_csv(
        INTERIM_DIR / "ratings.csv",
        index=False
    )

    movies.to_csv(
        INTERIM_DIR / "movies.csv",
        index=False
    )

    print("\nData ingestion completed successfully.")


if __name__ == "__main__":
    main()