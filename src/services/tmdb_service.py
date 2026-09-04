import os
import re

import httpx


class TMDBService:
    BASE_URL = "https://api.themoviedb.org/3"
    IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

    def __init__(self):
        self.access_token = os.getenv("TMDB_ACCESS_TOKEN")

        if not self.access_token:
            raise ValueError(
                "TMDB_ACCESS_TOKEN is not configured in the environment."
            )

        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "accept": "application/json",
        }

    def _request(self, endpoint, params=None):
        url = f"{self.BASE_URL}{endpoint}"

        try:
            response = httpx.get(
                url,
                headers=self.headers,
                params=params,
                timeout=10.0,
            )
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"TMDB returned HTTP {exc.response.status_code}."
            ) from exc

        except httpx.RequestError as exc:
            raise RuntimeError(
                "Unable to connect to TMDB."
            ) from exc

    @staticmethod
    def _extract_title_and_year(title):
        match = re.match(r"^(.*?)\s*\((\d{4})\)\s*$", title)

        if match:
            return match.group(1).strip(), int(match.group(2))

        return title.strip(), None

    def search_movie(self, title):
        movie_title, year = self._extract_title_and_year(title)

        params = {
            "query": movie_title,
            "include_adult": "false",
            "language": "en-US",
            "page": 1,
        }

        if year:
            params["year"] = year

        data = self._request("/search/movie", params=params)
        results = data.get("results", [])

        if not results and year:
            params.pop("year")
            data = self._request("/search/movie", params=params)
            results = data.get("results", [])

        if not results:
            return None

        return self._format_movie(results[0])

    def get_movie_details(self, tmdb_id):
        data = self._request(
            f"/movie/{tmdb_id}",
            params={"language": "en-US"},
        )

        return self._format_movie(data)

    def get_popular_movies(self, page=1):
        data = self._request(
            "/movie/popular",
            params={
                "language": "en-US",
                "page": page,
            },
        )

        return [
            self._format_movie(movie)
            for movie in data.get("results", [])
        ]

    def get_trending_movies(self):
        data = self._request(
            "/trending/movie/week",
            params={"language": "en-US"},
        )

        return [
            self._format_movie(movie)
            for movie in data.get("results", [])
        ]

    def _format_movie(self, movie):
        poster_path = movie.get("poster_path")
        backdrop_path = movie.get("backdrop_path")

        genres = movie.get("genres")

        if genres:
            genres = [
                genre.get("name")
                for genre in genres
                if genre.get("name")
            ]
        else:
            genres = []

        return {
            "tmdb_id": movie.get("id"),
            "title": movie.get("title"),
            "original_title": movie.get("original_title"),
            "overview": movie.get("overview"),
            "release_date": movie.get("release_date"),
            "rating": movie.get("vote_average"),
            "vote_count": movie.get("vote_count"),
            "popularity": movie.get("popularity"),
            "genres": genres,
            "poster_url": (
                f"{self.IMAGE_BASE_URL}{poster_path}"
                if poster_path
                else None
            ),
            "backdrop_url": (
                f"https://image.tmdb.org/t/p/original{backdrop_path}"
                if backdrop_path
                else None
            ),
        }