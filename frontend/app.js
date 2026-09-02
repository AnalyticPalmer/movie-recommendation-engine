const API_BASE =
    "https://movie-recommendation-engine-nu.vercel.app";

const similarForm =
    document.getElementById("similarForm");

const movieIdInput =
    document.getElementById("movieId");

const topKSelect =
    document.getElementById("topK");

const similarButton =
    document.getElementById("similarButton");

const popularButton =
    document.getElementById("popularButton");

const statusElement =
    document.getElementById("status");

const resultsElement =
    document.getElementById("results");

const resultsTitle =
    document.getElementById("resultsTitle");

const resultCount =
    document.getElementById("resultCount");

const exampleButtons =
    document.querySelectorAll(".example-chip");


function setStatus(
    message = "",
    isError = false
) {
    statusElement.textContent = message;

    statusElement.className =
        isError
            ? "status-message error"
            : "status-message";
}


function showLoading(count = 6) {
    resultsElement.innerHTML = "";

    for (
        let index = 0;
        index < count;
        index += 1
    ) {
        const loader =
            document.createElement("div");

        loader.className = "loading-card";

        resultsElement.appendChild(loader);
    }
}


function formatGenres(genres) {
    if (!genres) {
        return [];
    }

    return genres
        .split(" ")
        .filter(Boolean)
        .slice(0, 4);
}


function renderMovies(movies) {
    resultsElement.innerHTML = "";

    if (!movies || movies.length === 0) {
        resultsElement.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">
                    ✦
                </div>

                <h3>
                    No recommendations found
                </h3>

                <p>
                    Try another movie ID.
                </p>
            </div>
        `;

        resultCount.textContent = "";

        return;
    }

    movies.forEach(
        (movie, index) => {
            const card =
                document.createElement("article");

            card.className = "movie-card";

            const rank =
                document.createElement("div");

            rank.className = "movie-rank";

            rank.textContent =
                `RECOMMENDATION ${String(index + 1)
                    .padStart(2, "0")}`;

            const title =
                document.createElement("h3");

            title.textContent =
                movie.title || "Unknown title";

            const meta =
                document.createElement("div");

            meta.className = "movie-meta";

            const idBadge =
                document.createElement("span");

            idBadge.className = "badge";

            idBadge.textContent =
                `Movie ID ${movie.movie_id}`;

            meta.appendChild(idBadge);

            formatGenres(movie.genres)
                .forEach((genre) => {
                    const badge =
                        document.createElement("span");

                    badge.className = "badge";

                    badge.textContent = genre;

                    meta.appendChild(badge);
                });

            const scoreValue =
                movie.similarity_score ??
                movie.weighted_score ??
                movie.average_rating;

            card.appendChild(rank);
            card.appendChild(title);
            card.appendChild(meta);

            if (
                scoreValue !== undefined &&
                scoreValue !== null
            ) {
                const score =
                    document.createElement("div");

                score.className = "score";

                score.textContent =
                    `AI Score · ${Number(scoreValue)
                        .toFixed(3)}`;

                card.appendChild(score);
            }

            resultsElement.appendChild(card);
        }
    );

    resultCount.textContent =
        `${movies.length} results`;
}


async function requestJson(url) {
    const response = await fetch(url);

    if (!response.ok) {
        let message =
            `Request failed (${response.status})`;

        try {
            const payload =
                await response.json();

            if (payload.detail) {
                message = payload.detail;
            }
        } catch {
            if (response.statusText) {
                message = response.statusText;
            }
        }

        throw new Error(message);
    }

    return response.json();
}


async function getSimilarMovies(movieId, topK) {
    resultsTitle.textContent =
        "Similar movies";

    resultCount.textContent = "";

    setStatus(
        "Analyzing movie similarities..."
    );

    showLoading(topK);

    similarButton.disabled = true;
    popularButton.disabled = true;

    try {
        const url =
            `${API_BASE}` +
            `/api/recommendations/similar/` +
            `${movieId}?top_k=${topK}`;

        const data =
            await requestJson(url);

        renderMovies(
            data.recommendations
        );

        setStatus(
            `Recommendation engine completed successfully.`
        );
    } catch (error) {
        resultsElement.innerHTML = "";

        setStatus(
            error.message ||
            "Unable to load recommendations.",
            true
        );
    } finally {
        similarButton.disabled = false;
        popularButton.disabled = false;
    }
}


async function getPopularMovies() {
    const topK =
        Number(topKSelect.value);

    resultsTitle.textContent =
        "Popular movies";

    resultCount.textContent = "";

    setStatus(
        "Loading audience favorites..."
    );

    showLoading(topK);

    similarButton.disabled = true;
    popularButton.disabled = true;

    try {
        const url =
            `${API_BASE}` +
            `/api/recommendations/popular` +
            `?top_k=${topK}`;

        const data =
            await requestJson(url);

        renderMovies(
            data.recommendations
        );

        setStatus(
            "Popular recommendations loaded."
        );
    } catch (error) {
        resultsElement.innerHTML = "";

        setStatus(
            error.message ||
            "Unable to load popular movies.",
            true
        );
    } finally {
        similarButton.disabled = false;
        popularButton.disabled = false;
    }
}


similarForm.addEventListener(
    "submit",
    async (event) => {
        event.preventDefault();

        const movieId =
            Number(movieIdInput.value);

        const topK =
            Number(topKSelect.value);

        if (
            !Number.isInteger(movieId) ||
            movieId <= 0
        ) {
            setStatus(
                "Enter a valid movie ID.",
                true
            );

            movieIdInput.focus();

            return;
        }

        await getSimilarMovies(
            movieId,
            topK
        );
    }
);


popularButton.addEventListener(
    "click",
    async () => {
        await getPopularMovies();
    }
);


exampleButtons.forEach(
    (button) => {
        button.addEventListener(
            "click",
            async () => {
                const movieId =
                    Number(
                        button.dataset.movieId
                    );

                movieIdInput.value =
                    movieId;

                await getSimilarMovies(
                    movieId,
                    Number(topKSelect.value)
                );
            }
        );
    }
);