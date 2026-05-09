import pandas as pd
from surprise import SVD, Dataset, Reader

class CollaborativeRecommender:
    def __init__(self, factors=50, random_state=42):
        """Initialize the Collaborative Filtering Recommender using Surprise SVD."""
        self.factors = min(factors, 100) # Keep it reasonable
        self.model = SVD(n_factors=self.factors, random_state=random_state)
        self.trainset = None
        self.user_rated = {}
        self.all_movie_ids = []
        
    def fit(self, train_df):
        # Save users' already rated movies for quick filtering
        self.user_rated = train_df.groupby('userId')['movieId'].apply(set).to_dict()
        self.all_movie_ids = train_df['movieId'].unique().tolist()
        
        # Surprise requires a Reader object to parse the rating scale
        # Assuming typical 0.5 to 5.0 scale
        reader = Reader(rating_scale=(0.5, 5.0))
        
        # Dataset expects df columns: user, item, rating
        # We must only pass these exact three columns to Surprise
        data = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)
        
        # Build the full trainset for the model
        self.trainset = data.build_full_trainset()
        
        # Fit the SVD model
        self.model.fit(self.trainset)
        
    def predict(self, user_id, movie_id):
        """Predict score for a single user and movie."""
        if self.trainset is None:
            raise ValueError("Model must be fitted before predicting.")
            
        # The .est attribute of the prediction object holds the estimated rating
        pred = self.model.predict(user_id, movie_id)
        return float(pred.est)
        
    def recommend(self, user_id, movies_df, top_n=10):
        if self.trainset is None:
            raise ValueError("Model must be fitted before recommending.")
            
        # Check if the user is in the trainset
        try:
            self.trainset.to_inner_uid(user_id)
        except ValueError:
            # User not seen during training, return empty
            return pd.DataFrame()
            
        already_rated = self.user_rated.get(user_id, set())
        
        # Generate predictions for all movies the user hasn't rated yet
        predictions = []
        for movie_id in self.all_movie_ids:
            if movie_id not in already_rated:
                est = self.model.predict(user_id, movie_id).est
                predictions.append((movie_id, float(est)))
                
        # Sort by estimated rating descending
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-N
        top_preds = predictions[:top_n]
        
        if not top_preds:
            return pd.DataFrame()
            
        raw_movie_ids = [r[0] for r in top_preds]
        est_ratings = [r[1] for r in top_preds]
        
        recs_df = movies_df[movies_df['movieId'].isin(raw_movie_ids)].copy()
        recs_df = recs_df.set_index('movieId')
        recs_df = recs_df.loc[raw_movie_ids].reset_index()
        recs_df['est_rating'] = est_ratings
        
        return recs_df[['movieId', 'title', 'genres', 'est_rating']]