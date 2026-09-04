import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.pipeline.prediction_pipeline import RecommendationPipeline

from src.services.movie_enrichment_service import MovieEnrichmentService

app = FastAPI(
    title="Movie Recommendation Engine API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "https://movie-recommendation-engine-nu.vercel.app",
        "https://cinematch-ai-henna.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

pipeline = RecommendationPipeline()
enrichment_service = MovieEnrichmentService()

@app.get("/")
def root():
    return {
        "message": "Movie Recommendation Engine API"
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/api/movies/search")
def search_movies(
    query: str = Query(
        min_length=1,
    ),
    limit: int = Query(
        default=10,
        ge=1,
        le=50,
    ),
):
    try:
        movies = pipeline.search_movies(
            query=query,
            limit=limit,
        )

        return {
            "query": query,
            "count": len(movies),
            "movies": movies.to_dict(
                orient="records"
            ),
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@app.get("/api/recommendations/popular")
def popular_recommendations(
    top_k: int = Query(
        default=10,
        ge=1,
        le=100,
    )
):
    try:
        recommendations = pipeline.recommend_popular(
            top_k=top_k
        )

        enriched_recommendations = enrichment_service.enrich_movies(
            recommendations
        )

        return {
            "model": "popularity",
            "count": len(enriched_recommendations),
            "recommendations": enriched_recommendations,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

        return {
            "model": "popularity",
            "count": len(recommendations),
            "recommendations": recommendations.to_dict(
                orient="records"
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


@app.get("/api/recommendations/similar/{movie_id}")
def similar_recommendations(
    movie_id: int,
    top_k: int = Query(
        default=10,
        ge=1,
        le=100,
    ),
):
    try:
        recommendations = pipeline.recommend_similar(
            movie_id=movie_id,
            top_k=top_k,
        )

        return {
            "model": "content_based",
            "movie_id": movie_id,
            "count": len(recommendations),
            "recommendations": recommendations.to_dict(
                orient="records"
            ),
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc