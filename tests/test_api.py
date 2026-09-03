from fastapi.testclient import TestClient

from api.index import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Movie Recommendation Engine API"
    }


def test_health():
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy"
    }


def test_popular_recommendations():
    response = client.get(
        "/api/recommendations/popular?top_k=5"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "popularity"
    assert data["count"] == 5
    assert len(data["recommendations"]) == 5


def test_similar_recommendations():
    response = client.get(
        "/api/recommendations/similar/50?top_k=5"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "content_based"
    assert data["movie_id"] == 50
    assert data["count"] == 5
    assert len(data["recommendations"]) == 5


def test_invalid_top_k():
    response = client.get(
        "/api/recommendations/popular?top_k=0"
    )

    assert response.status_code == 422


def test_top_k_above_limit():
    response = client.get(
        "/api/recommendations/popular?top_k=101"
    )

    assert response.status_code == 422


def test_invalid_movie_id():
    response = client.get(
        "/api/recommendations/similar/999999?top_k=5"
    )

    assert response.status_code in {
        400,
        500,
    }