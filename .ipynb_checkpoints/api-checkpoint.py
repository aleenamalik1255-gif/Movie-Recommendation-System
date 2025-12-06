# api.py
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from flask import Flask, render_template, request, jsonify
app = Flask(__name__)
# -----------------------------
# Initialize Flask app
# -----------------------------
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # still okay for local testing; can be left

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("merged_ratings_movies.csv")
df['movie_mean'] = df['movie_mean'].fillna(df['movie_mean'].mean())
df['movie_std'] = df['movie_std'].fillna(df['movie_std'].mean())
df['movie_year'] = df['movie_year'].fillna(df['movie_year'].median())
df_unique = df.drop_duplicates(subset=['title_clean']).reset_index(drop=True)

features = ['movie_mean', 'movie_std', 'movie_year']
X_scaled = StandardScaler().fit_transform(df_unique[features])
title_to_index = {title.strip(): idx for idx, title in enumerate(df_unique['title_clean'].values)}

# -----------------------------
# Serve frontend
# -----------------------------
@app.route("/")
def home():
    # renders templates/index.html
    return render_template('index.html')

# optional: if you want to serve other files from static manually:
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

# -----------------------------
# Recommendation route
# -----------------------------
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    movie_name = data.get("movie_name", "").strip()
    if movie_name not in title_to_index:
        return jsonify({"error": f"Movie '{movie_name}' not found."}), 200

    idx = title_to_index[movie_name]
    movie_vector = X_scaled[idx].reshape(1, -1)
    sim_scores = cosine_similarity(movie_vector, X_scaled).flatten()
    sim_scores[idx] = -1
    top_indices = sim_scores.argsort()[::-1][:5]
    recommendations = df_unique['title_clean'].iloc[top_indices].tolist()
    return jsonify({"recommendations": recommendations})

# -----------------------------
# Run the app
# -----------------------------

