import pandas as pd
import numpy as np

def load_and_preprocess_data(movies_path, ratings_path):
    """
    Loads and preprocesses the MovieLens dataset.
    
    Args:
        movies_path (str): Path to movies.csv
        ratings_path (str): Path to ratings.csv
        
    Returns:
        tuple: (movies_df, ratings_df, train_df, test_df)
    """
    # Load Data
    movies  = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

    movies["movieId"]  = movies["movieId"].astype(int)
    ratings["userId"]  = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"]  = ratings["rating"].astype(float)

    # Clean Movies
    movies = movies.drop_duplicates(subset="movieId").copy()
    movies["genres"] = movies["genres"].replace("(no genres listed)", np.nan)
    movies = movies.dropna(subset=["genres"]).reset_index(drop=True)

    movies["year"]  = movies["title"].str.extract(r"\((\d{4})\)$").astype(float)
    movies["title"] = movies["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True).str.strip()

    # Fix article at the end of titles (e.g., "Matrix, The" -> "The Matrix")
    movies["title"] = movies["title"].str.replace(r'(?i)^(.*),\s*(The|A|An)$', r'\2 \1', regex=True)

    # Clean Ratings
    ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")
    ratings = (ratings
            .sort_values("timestamp")
            .drop_duplicates(subset=["userId", "movieId"], keep="last")
            .copy())

    valid_ids = set(movies["movieId"])
    ratings   = ratings[ratings["movieId"].isin(valid_ids)].copy()
    ratings = ratings[(ratings["rating"] >= 0.5) & (ratings["rating"] <= 5.0)].copy()
    ratings = ratings.drop(columns=["timestamp"]).reset_index(drop=True)

    # Feature Engineering — Content-Based
    movies["genre_list"] = movies["genres"].str.split("|")
    movies["genres_str"] = movies["genre_list"].apply(lambda g: " ".join(g))

    # Filter Cold-Start Users & Movies
    MIN_RATINGS = 5
    prev_len, iteration = -1, 0
    while len(ratings) != prev_len:
        prev_len   = len(ratings)
        iteration += 1
        user_counts  = ratings["userId"].value_counts()
        movie_counts = ratings["movieId"].value_counts()
        valid_users  = user_counts[user_counts   >= MIN_RATINGS].index
        valid_movies = movie_counts[movie_counts >= MIN_RATINGS].index
        ratings = ratings[
            ratings["userId"].isin(valid_users) &
            ratings["movieId"].isin(valid_movies)
        ].copy()

    # Per-User Rating Normalisation
    user_mean = ratings.groupby("userId")["rating"].transform("mean")
    user_std  = ratings.groupby("userId")["rating"].transform("std").replace(0, 1)
    ratings["rating_normalised"] = ((ratings["rating"] - user_mean) / user_std).round(4)

    # Train / Test Split (80 / 20)
    ratings["rank"]  = ratings.groupby("userId").cumcount()
    ratings["total"] = ratings.groupby("userId")["rank"].transform("max") + 1
    ratings["pct"]   = ratings["rank"] / ratings["total"]

    test  = ratings[ratings["pct"] >= 0.8].drop(columns=["rank","total","pct"]).reset_index(drop=True)
    train = ratings[ratings["pct"] <  0.8].drop(columns=["rank","total","pct"]).reset_index(drop=True)

    return movies, ratings, train, test
