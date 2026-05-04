import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

class CollaborativeRecommender:
    def __init__(self, factors=50, random_state=42):
        """Initialize the Collaborative Filtering Recommender using Scikit-Learn TruncatedSVD."""
        self.factors = min(factors, 100) # Keep it reasonable
        self.model = TruncatedSVD(n_components=self.factors, random_state=random_state)
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        
        self.user_factors = None
        self.item_factors = None
        self.user_rated = {}
        
    def fit(self, train_df):
        """Fit the SVD model on the training data.
        
        Args:
            train_df (pd.DataFrame): Dataframe containing 'userId', 'movieId', and 'rating' columns.
        """
        # Save users' already rated movies for quick filtering
        self.user_rated = train_df.groupby('userId')['movieId'].apply(set).to_dict()
        
        # Create mappings from raw IDs to integer indices
        unique_users = train_df['userId'].unique()
        unique_items = train_df['movieId'].unique()
        
        self.user_mapping = {u: i for i, u in enumerate(unique_users)}
        self.item_mapping = {i: idx for idx, i in enumerate(unique_items)}
        self.reverse_user_mapping = {i: u for u, i in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: i for i, idx in self.item_mapping.items()}
        
        n_users = len(unique_users)
        n_items = len(unique_items)
        
        # We can construct a dense matrix for the SVD factorization
        matrix = np.zeros((n_users, n_items))
        
        row = train_df['userId'].map(self.user_mapping).values
        col = train_df['movieId'].map(self.item_mapping).values
        data = train_df['rating'].values
        
        matrix[row, col] = data
        
        # Factorize the matrix
        self.user_factors = self.model.fit_transform(matrix) # (n_users, factors)
        self.item_factors = self.model.components_.T         # (n_items, factors)
        
    def predict(self, user_id, movie_id):
        """Predict score for a single user and movie."""
        if self.user_factors is None:
            raise ValueError("Model must be fitted before predicting.")
            
        if user_id not in self.user_mapping or movie_id not in self.item_mapping:
            return 2.5 # default fallback
            
        u_idx = self.user_mapping[user_id]
        i_idx = self.item_mapping[movie_id]
        
        score = np.dot(self.user_factors[u_idx], self.item_factors[i_idx])
        return float(score)
        
    def recommend(self, user_id, movies_df, top_n=10):
        """Recommend movies for a given user."""
        if self.user_factors is None:
            raise ValueError("Model must be fitted before recommending.")
            
        if user_id not in self.user_mapping:
            return pd.DataFrame()
            
        u_idx = self.user_mapping[user_id]
        scores = np.dot(self.user_factors[u_idx], self.item_factors.T)
        
        top_indices = np.argsort(scores)[::-1]
        already_rated = self.user_rated.get(user_id, set())
        
        recommendations = []
        for i_idx in top_indices:
            raw_id = self.reverse_item_mapping[i_idx]
            if raw_id not in already_rated:
                recommendations.append((raw_id, scores[i_idx]))
            if len(recommendations) >= top_n:
                break
                
        if not recommendations:
            return pd.DataFrame()
            
        raw_movie_ids = [r[0] for r in recommendations]
        est_ratings = [r[1] for r in recommendations]
        
        recs_df = movies_df[movies_df['movieId'].isin(raw_movie_ids)].copy()
        recs_df = recs_df.set_index('movieId')
        recs_df = recs_df.loc[raw_movie_ids].reset_index()
        recs_df['est_rating'] = est_ratings
        
        return recs_df[['movieId', 'title', 'genres', 'est_rating']]