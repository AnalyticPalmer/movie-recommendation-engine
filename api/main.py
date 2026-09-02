import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from fastapi import FastAPI, HTTPException, Query

from src.pipeline.prediction_pipeline import RecommendationPipeline


app = FastAPI(
    title="Recommendation Engine API",
    version="1.0.0",
    description="Movie recommendation API using popularity, content-based, collaborative, and hybrid models.",
)

pipeline = RecommendationPipeline()


@app.get("/")
def root():
    return {
        "message": "Recommendation Engine API is running."
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/recommendations/popular")
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


@app.get("/recommendations/similar/{movie_id}")
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


@app.get("/recommendations/user/{user_id}")
def user_recommendations(
    user_id: int,
    top_k: int = Query(
        default=10,
        ge=1,
        le=100,
    ),
    model_type: str = Query(
        default="collaborative"
    ),
):
    try:
        recommendations = pipeline.recommend_for_user(
            user_id=user_id,
            top_k=top_k,
            model_type=model_type,
        )

        return {
            "model": model_type,
            "user_id": user_id,
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