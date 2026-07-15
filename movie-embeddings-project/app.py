from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st


DATA_PATH = Path("data/processed/movies_cleaned.csv")
EMBEDDINGS_PATH = Path("artifacts/embeddings/gte_modernbert_embeddings.npy")
GENRES_PATH = Path("artifacts/metadata/movie_genres.csv")
TOP_K = 5

st.set_page_config(
    page_title="Semantic Movie Recommender",
    page_icon="🎬",
    layout="centered",
)


def parse_genres(value: object) -> str:
    if pd.isna(value):
        return "Unknown"

    try:
        genres = json.loads(str(value))
    except (json.JSONDecodeError, TypeError):
        return "Unknown"

    return ", ".join(genres) if genres else "Unknown"


@st.cache_resource
def load_project_data() -> tuple[pd.DataFrame, np.ndarray]:
    required_paths = [DATA_PATH, EMBEDDINGS_PATH, GENRES_PATH]
    missing_paths = [str(path) for path in required_paths if not path.exists()]

    if missing_paths:
        raise FileNotFoundError(
            "Missing project files: " + ", ".join(missing_paths)
        )

    movies = pd.read_csv(DATA_PATH).reset_index(drop=True)
    genres = pd.read_csv(GENRES_PATH)

    required_columns = {"movie_id", "title", "overview"}
    missing_columns = required_columns - set(movies.columns)
    if missing_columns:
        raise ValueError(f"Missing movie columns: {sorted(missing_columns)}")

    movies["movie_id"] = movies["movie_id"].astype(int)
    genres["movie_id"] = genres["movie_id"].astype(int)
    genres["genre_names"] = genres["genres"].apply(parse_genres)

    movies = movies.merge(
        genres[["movie_id", "genre_names"]].drop_duplicates("movie_id"),
        on="movie_id",
        how="left",
        validate="one_to_one",
    )
    movies["genre_names"] = movies["genre_names"].fillna("Unknown")

    if "release_date" in movies.columns:
        years = pd.to_datetime(
            movies["release_date"],
            errors="coerce",
        ).dt.year.astype("Int64").astype(str)
        years = years.replace("<NA>", "Unknown year")
    else:
        years = pd.Series("Unknown year", index=movies.index)

    movies["year"] = years
    movies["movie_label"] = (
        movies["title"].astype(str)
        + " ("
        + movies["year"]
        + ") · ID "
        + movies["movie_id"].astype(str)
    )

    embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
    if embeddings.ndim != 2 or len(embeddings) != len(movies):
        raise ValueError(
            f"Embedding shape {embeddings.shape} does not match "
            f"{len(movies)} movies."
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinite values.")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("At least one embedding has zero norm.")
    embeddings /= norms

    return movies, embeddings


def get_top_similar_positions(
    embeddings: np.ndarray,
    query_position: int,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    similarities = embeddings @ embeddings[query_position]
    similarities[query_position] = -np.inf

    top_positions = np.argpartition(
        similarities,
        -top_k,
    )[-top_k:]
    top_positions = top_positions[
        np.argsort(similarities[top_positions])[::-1]
    ]

    return top_positions, similarities[top_positions]


st.title("🎬 Semantic Movie Recommender")
st.write(
    "Choose a movie and get five semantically similar recommendations "
    "based on GTE ModernBERT overview embeddings."
)

try:
    movies_df, gte_embeddings = load_project_data()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

selected_label = st.selectbox(
    "Movie title",
    options=movies_df["movie_label"].tolist(),
    index=None,
    placeholder="Start typing a movie title...",
)

search_clicked = st.button(
    "Find similar movies",
    type="primary",
    disabled=selected_label is None,
)

if search_clicked and selected_label is not None:
    query_position = int(
        movies_df.index[movies_df["movie_label"] == selected_label][0]
    )
    query_movie = movies_df.iloc[query_position]
    top_positions, top_scores = get_top_similar_positions(
        gte_embeddings,
        query_position,
        TOP_K,
    )

    st.subheader(f"Movies similar to {query_movie['title']}")
    st.caption(
        f"Selected movie: {query_movie['year']} · "
        f"{query_movie['genre_names']}"
    )

    for rank, (position, score) in enumerate(
        zip(top_positions, top_scores),
        start=1,
    ):
        movie = movies_df.iloc[position]

        with st.container(border=True):
            st.subheader(f"{rank}. {movie['title']}")
            similarity_column, year_column = st.columns(2)
            similarity_column.metric(
                "Cosine similarity",
                f"{float(score):.3f}",
            )
            year_column.metric("Release year", movie["year"])
            st.caption(f"Genres: {movie['genre_names']}")
            st.write(movie["overview"] if pd.notna(movie["overview"]) else "No overview available.")
