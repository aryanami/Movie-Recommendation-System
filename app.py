# app.py
import ast
import pickle
import warnings
import numpy as np
import pandas as pd
import requests
import bs4 as bs

from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

SENTIMENT_MODEL_PATH = "sentiment_model.pkl"
with open(SENTIMENT_MODEL_PATH, "rb") as f:
    sentiment_pipe = pickle.load(f)

class DataLoader:
    _instance = None
    data = None
    title_norm = None
    vectorizer = None
    nn_index = None
    count_matrix = None

    @classmethod
    def init(cls):
        if cls._instance is None:
            cls._instance = cls()
            # Load movie metadata 
            cls.data = pd.read_csv("main_data.csv")
            if "movie_title" not in cls.data.columns or "col_merge" not in cls.data.columns:
                raise ValueError("main_data.csv must contain columns: movie_title, col_merge")

            # Normalize titles once for lookup
            cls.data["title_norm"] = cls.data["movie_title"].astype(str).str.strip().str.lower()
            cls.title_norm = cls.data["title_norm"].values

            # Vectorize 
            cls.vectorizer = TfidfVectorizer()
            cls.count_matrix = cls.vectorizer.fit_transform(cls.data["col_merge"].astype(str))

            # Build a cosine nearest-neighbor index
            cls.nn_index = NearestNeighbors(metric="cosine", algorithm="brute")
            cls.nn_index.fit(cls.count_matrix)
        return cls.data, cls.nn_index, cls.count_matrix, cls.title_norm

def find_similar_titles(title: str, k: int = 10):
    data, nn, X, title_norm = DataLoader.init()
    q = (title or "").strip().lower()
    if q not in title_norm:
        return None  
    idx = int(np.where(title_norm == q)[0][0])
    vec = X[idx]
    # kneighbors returns distances; cosine distance = 1 - cosine similarity
    distances, indices = nn.kneighbors(vec, n_neighbors=min(k+1, X.shape[0]))  # +1 to skip itself
    indices = indices.flatten().tolist()
    distances = distances.flatten().tolist()

    out = []
    for j, d in zip(indices, distances):
        if j == idx:
            continue
        out.append((j, 1.0 - float(d)))  
        if len(out) == k:
            break
    # Return titles, preserving original casing
    return [data["movie_title"].iloc[j] for j, _ in out]

def parse_literal_list(s: str):
    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, list) else [v]
    except Exception:
        return []

def get_suggestions():
    data, *_ = DataLoader.init()
    return data["movie_title"].astype(str).tolist()

def fetch_imdb_reviews(imdb_id: str, limit: int = 6):
    reviews, status = [], []
    if not imdb_id:
        return ["No reviews available"], ["N/A"]
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        }
        url = f"https://www.imdb.com/title/{imdb_id}/reviews?ref_=tt_ov_rt"
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = bs.BeautifulSoup(resp.text, "lxml")
        # IMDb markup changes; this selector is current as of writing:
        nodes = soup.find_all("div", {"class": "text show-more__control"}, limit=limit)
        if not nodes:
            nodes = soup.find_all("div", {"data-testid": "review-content"}, limit=limit)

        for node in nodes:
            text = node.get_text(strip=True)
            if not text:
                continue
            reviews.append(text)
            # Sentiment prediction (0/1)
            label = int(sentiment_pipe.predict([text])[0])
            status.append("Good" if label == 1 else "Bad")

        if not reviews:
            reviews, status = ["No reviews available"], ["N/A"]

    except Exception:
        reviews, status = ["No reviews available"], ["N/A"]

    return reviews, status

@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template("home.html", suggestions=suggestions)

@app.route("/similarity", methods=["POST"])
def similarity_api():
    movie = request.form.get("name", "")
    similar = find_similar_titles(movie, k=10)
    if similar is None or len(similar) == 0:
        return "Sorry! The movie you requested is not in our database. Please check the spelling or try some other movie name"

    return "---".join(similar)

@app.route("/recommend", methods=["POST"])
def recommend():
    title = request.form.get("title", "")
    cast_ids = request.form.get("cast_ids", "[]")
    cast_names = request.form.get("cast_names", "[]")
    cast_chars = request.form.get("cast_chars", "[]")
    cast_bdays = request.form.get("cast_bdays", "[]")
    cast_bios = request.form.get("cast_bios", "[]")
    cast_places = request.form.get("cast_places", "[]")
    cast_profiles = request.form.get("cast_profiles", "[]")
    imdb_id = request.form.get("imdb_id", "")
    poster = request.form.get("poster", "")
    genres = request.form.get("genres", "")
    overview = request.form.get("overview", "")
    vote_average = request.form.get("rating", "")
    vote_count = request.form.get("vote_count", "")
    release_date = request.form.get("release_date", "")
    runtime = request.form.get("runtime", "")
    status = request.form.get("status", "")
    rec_movies = request.form.get("rec_movies", "[]")
    rec_posters = request.form.get("rec_posters", "[]")

    # Suggestions for autocomplete
    suggestions = get_suggestions()

    # Robust parsing for list-like fields
    rec_movies = parse_literal_list(rec_movies)
    rec_posters = parse_literal_list(rec_posters)
    cast_names = parse_literal_list(cast_names)
    cast_chars = parse_literal_list(cast_chars)
    cast_profiles = parse_literal_list(cast_profiles)
    cast_bdays = parse_literal_list(cast_bdays)
    cast_bios = parse_literal_list(cast_bios)
    cast_places = parse_literal_list(cast_places)
    cast_ids = parse_literal_list(cast_ids)

    cast_bios = [b.replace(r"\n", "\n").replace(r"\"", "\"") for b in cast_bios]

    movie_cards = {p: m for p, m in zip(rec_posters, rec_movies)}
    casts = {name: [cast_ids[i] if i < len(cast_ids) else "",
                    cast_chars[i] if i < len(cast_chars) else "",
                    cast_profiles[i] if i < len(cast_profiles) else ""]
             for i, name in enumerate(cast_names)}
    cast_details = {name: [cast_ids[i] if i < len(cast_ids) else "",
                           cast_profiles[i] if i < len(cast_profiles) else "",
                           cast_bdays[i] if i < len(cast_bdays) else "",
                           cast_places[i] if i < len(cast_places) else "",
                           cast_bios[i] if i < len(cast_bios) else ""]
                    for i, name in enumerate(cast_names)}

    reviews_list, reviews_status = fetch_imdb_reviews(imdb_id, limit=6)
    movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}

    return render_template(
        "recommend.html",
        title=title,
        poster=poster,
        overview=overview,
        vote_average=vote_average,
        vote_count=vote_count,
        release_date=release_date,
        runtime=runtime,
        status=status,
        genres=genres,
        movie_cards=movie_cards,
        reviews=movie_reviews,
        casts=casts,
        cast_details=cast_details,
        suggestions=suggestions
    )

if __name__ == "__main__":
    app.run(debug=True)
