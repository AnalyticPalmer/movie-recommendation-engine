from src.services.tmdb_service import TMDBService


class MovieEnrichmentService:
    def __init__(self):
        self.tmdb = TMDBService()

    def enrich_movie(self, movie):
        movie_data = dict(movie)
        title = movie_data.get("title")

        if not title:
            return movie_data

        try:
            tmdb_movie = self.tmdb.search_movie(title)

            if not tmdb_movie:
                return movie_data

            tmdb_id = tmdb_movie.get("tmdb_id")

            if tmdb_id:
                try:
                    details = self.tmdb.get_movie_details(tmdb_id)

                    if details:
                        tmdb_movie.update(details)

                except RuntimeError:
                    pass

            return {
                **movie_data,
                "tmdb_id": tmdb_movie.get("tmdb_id"),
                "tmdb_title": tmdb_movie.get("title"),
                "overview": tmdb_movie.get("overview"),
                "release_date": tmdb_movie.get("release_date"),
                "tmdb_rating": tmdb_movie.get("rating"),
                "vote_count": tmdb_movie.get("vote_count"),
                "popularity": tmdb_movie.get("popularity"),
                "tmdb_genres": tmdb_movie.get("genres", []),
                "poster_url": tmdb_movie.get("poster_url"),
                "backdrop_url": tmdb_movie.get("backdrop_url"),
            }

        except (RuntimeError, ValueError):
            return movie_data

    def enrich_movies(self, movies):
        if hasattr(movies, "to_dict"):
            records = movies.to_dict(orient="records")
        else:
            records = list(movies)

        return [
            self.enrich_movie(movie)
            for movie in records
        ]