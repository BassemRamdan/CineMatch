# Hybrid Movie Recommendation System

This project implements a hybrid recommendation system using the MovieLens dataset. It combines Content-Based filtering (TF-IDF & Cosine Similarity) with Collaborative filtering (SVD using the `surprise` library).

## Project Structure
- `data/raw/`: Contains `movies.csv` and `ratings.csv` (MovieLens dataset).
- `src/preprocessing.py`: Data ingestion, cleaning, feature engineering, and train/test split.
- `src/content_engine.py`: Content-based recommender using movie genres.
- `src/collaborative_engine.py`: Collaborative recommender using Surprise SVD.
- `src/hybrid_engine.py`: Hybrid recommender combining both models using weighted average.
- `src/evaluation.py`: Model evaluation metrics (RMSE, MAE, Precision, Recall, F1-Score).
- `app/main.py`: Streamlit dashboard for the user interface.

## How to Run

1. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit Dashboard:**
   ```bash
   streamlit run app/main.py
   ```
