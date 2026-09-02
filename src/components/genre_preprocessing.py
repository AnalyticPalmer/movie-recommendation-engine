from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

RAW_PATH = BASE_DIR / "data" / "raw" / "u.item"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "movies_with_genres.csv"


GENRES = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]


def main():

    columns = [
        "movie_id",
        "title",
        "release_date",
        "video_release_date",
        "imdb_url",
        *GENRES,
    ]

    movies = pd.read_csv(
        RAW_PATH,
        sep="|",
        names=columns,
        encoding="latin-1",
    )

    movies["genres"] = movies[GENRES].apply(
        lambda row: " ".join(
            genre
            for genre in GENRES
            if row[genre] == 1
        ),
        axis=1,
    )

    movies[
        [
            "movie_id",
            "title",
            "release_date",
            "imdb_url",
            "genres",
        ]
    ].to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("Genre preprocessing completed.")
    print("Movies:", len(movies))

    print(
        movies[
            [
                "movie_id",
                "title",
                "genres",
            ]
        ].head(10)
    )


if __name__ == "__main__":
    main()