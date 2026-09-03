# CineMatch AI — Movie Recommendation Engine

CineMatch AI is a production-oriented movie recommendation system built with Python, machine learning, FastAPI, and a responsive frontend.

The project started from the MovieLens 100K dataset and evolved into a deployed end-to-end recommendation application with multiple recommendation strategies, automated testing, continuous integration, and a live web interface.

## Live Application

Frontend:

https://cinematch-ai-henna.vercel.app

Backend API:

https://movie-recommendation-engine-nu.vercel.app

Health Check:

https://movie-recommendation-engine-nu.vercel.app/api/health

## Features

- Popularity-based movie recommendations
- Content-based movie similarity
- Collaborative filtering using Alternating Least Squares
- Hybrid recommendation system
- Temporal leave-one-out train/test split
- Recommender-system evaluation
- FastAPI inference API
- Responsive CineMatch AI frontend
- Vercel deployment
- Automated API testing
- GitHub Actions continuous integration
- Model serialization and reusable prediction pipeline

## Recommendation Models

### Popularity-Based Recommender

Ranks movies using a weighted rating approach that balances average rating and rating volume.

This prevents movies with very few ratings from dominating the recommendations.

### Content-Based Recommender

Uses movie genre information with TF-IDF feature extraction and cosine similarity.

Example recommendations for `Star Wars (1977)` include:

- Return of the Jedi
- The Empire Strikes Back
- Starship Troopers
- Independence Day

### Collaborative Filtering

Uses Alternating Least Squares to learn latent user and movie representations from user-item interactions.

Configuration:

- Factors: 50
- Regularization: 0.01
- Iterations: 20
- Random state: 42

### Hybrid Recommender

Combines collaborative filtering and content-based recommendation scores.

During evaluation, the collaborative filtering model slightly outperformed the hybrid model.

## Dataset

The project uses the MovieLens 100K dataset.

Dataset statistics:

- 100,000 ratings
- 943 users
- 1,682 movies
- 93.7% matrix sparsity
- No missing user IDs
- No missing movie IDs
- No invalid ratings
- No duplicate ratings

## Data Processing

The preprocessing pipeline:

1. Loads MovieLens ratings and movie metadata
2. Cleans and validates the data
3. Creates genre-based movie features
4. Performs a temporal leave-one-out split
5. Saves processed training and test datasets

Final split:

- Training interactions: 99,057
- Test interactions: 943

Each user contributes their latest interaction to the test set.

## Model Evaluation

Collaborative filtering evaluation at K=10:

| Metric | Score |
|---|---:|
| Precision@10 | 0.0155 |
| Recall@10 | 0.1548 |
| Hit Rate@10 | 0.1548 |
| NDCG@10 | 0.0813 |

Hybrid model evaluation:

| Metric | Score |
|---|---:|
| Precision@10 | 0.0146 |
| Recall@10 | 0.1463 |
| Hit Rate@10 | 0.1463 |
| NDCG@10 | 0.0802 |

The collaborative filtering model slightly outperformed the hybrid model.

## API

The application exposes a FastAPI inference service.

### Health Check

```http
GET /api/health
```

### Popular Recommendations

```http
GET /api/recommendations/popular?top_k=5
```

### Similar Movies

```http
GET /api/recommendations/similar/50?top_k=5
```

Movie ID `50` represents Star Wars in the MovieLens dataset.

## Frontend

The CineMatch AI frontend provides:

- Movie recommendation interface
- Popular movie exploration
- Similar-movie recommendations
- Result-count selection
- Loading states
- Error handling
- Responsive design
- Dark cinematic interface

The frontend communicates with the deployed FastAPI backend through CORS-enabled API requests.

## Project Structure

```text
recommendation-engine/
├── .github/
│   └── workflows/
│       └── tests.yml
├── api/
│   └── index.py
├── app/
├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── external/
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── models/
├── notebooks/
├── reports/
│   ├── figures/
│   └── metrics/
├── src/
│   ├── components/
│   ├── evaluation/
│   ├── pipeline/
│   └── recommenders/
├── tests/
│   └── test_api.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```powershell
git clone https://github.com/AnalyticPalmer/movie-recommendation-engine.git
cd movie-recommendation-engine
```

Create and activate a virtual environment:

```powershell
python -m venv recsys
.\recsys\Scripts\Activate.ps1
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

Install the project:

```powershell
pip install -e .
```

Install development dependencies:

```powershell
pip install pytest httpx
```

## Run the API Locally

```powershell
uvicorn api.index:app --reload
```

Then open:

```text
http://127.0.0.1:8000/api/health
```

## Run the Frontend Locally

From the project root:

```powershell
python -m http.server 5500 --directory frontend
```

Then open:

```text
http://127.0.0.1:5500
```

## Testing

Run the API test suite:

```powershell
pytest tests/test_api.py -v
```

Current result:

```text
7 passed
```

Tests cover:

- Root endpoint
- Health endpoint
- Popular recommendations
- Similar recommendations
- Invalid `top_k`
- Values above the allowed recommendation limit
- Invalid movie IDs

## Continuous Integration

GitHub Actions automatically runs the API tests whenever code is pushed to `main` or a pull request targets `main`.

Workflow:

```text
.github/workflows/tests.yml
```

The CI pipeline:

1. Checks out the repository
2. Installs Python 3.12
3. Installs application dependencies
4. Installs test dependencies
5. Runs the automated API test suite

Current GitHub Actions result:

```text
7 passed
```

## Deployment

The backend is deployed with Vercel using FastAPI.

The frontend is deployed separately as a static Vercel application.

The production frontend is explicitly allowed by the backend CORS configuration.

## Technology Stack

- Python
- Pandas
- NumPy
- SciPy
- Scikit-learn
- Implicit ALS
- FastAPI
- Uvicorn
- Joblib
- Pytest
- HTML
- CSS
- JavaScript
- Git
- GitHub
- GitHub Actions
- Vercel

## Current Limitations

The original recommendation models are trained on the MovieLens 100K dataset.

Because the dataset contains older movies, newly released movies are not currently part of the trained recommendation catalogue.

The deployed Vercel API currently focuses on popularity and content-based inference because the `implicit` collaborative filtering package depends on native OpenMP libraries that are not available in the standard Vercel Python runtime.

## Planned Improvements

Future improvements include:

- Search movies by title instead of MovieLens ID
- Search autocomplete
- Movie posters and backdrop images
- TMDB metadata integration
- Current and newly released movie catalogue
- Movie descriptions and ratings
- Genre filters
- Improved recommendation cards
- Updated recommendation dataset
- Collaborative filtering deployment on a compatible container-based platform
- User profiles and personalized recommendation history

## Author

**AnalyticPalmer**

GitHub:

https://github.com/AnalyticPalmer

## Project Status

Active development.

The recommendation engine, API, frontend, deployment, automated tests, and CI pipeline are operational.