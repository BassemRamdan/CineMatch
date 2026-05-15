import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class HybridRecommender:
    def __init__(self, content_model, collab_model, content_weight=0.5):
        self.content_model = content_model
        self.collab_model = collab_model
        self.content_weight = content_weight
        self.collab_weight = 1.0 - content_weight
        
    def recommend(self, user_id, title, top_n=10):

        if self.content_model.movies_df is None or self.content_model.cosine_sim is None:
            raise ValueError("Content model must be fitted.")
            
        matches = self.content_model.movies_df[self.content_model.movies_df["title"].str.lower() == title.lower()]
        
        if len(matches) == 0:
            return None # Movie not found
            
        idx = matches.index[0]
        target_title = matches.iloc[0]["title"].lower()
        
        # 1. Get all similarity scores
        sim_scores = list(enumerate(self.content_model.cosine_sim[idx]))
        
        # Build dataframe for content scores
        content_df = pd.DataFrame(sim_scores, columns=['idx', 'similarity'])
        content_df['movieId'] = self.content_model.movies_df.iloc[content_df['idx']]['movieId'].values
        content_df['title'] = self.content_model.movies_df.iloc[content_df['idx']]['title'].values
        
        # Filter out identical titles from content_df immediately to prevent them from showing up
        content_df = content_df[content_df['title'].str.lower() != target_title]
        
        # Normalize similarity scores (0 to 1)
        scaler = MinMaxScaler()
        content_df['norm_sim'] = scaler.fit_transform(content_df[['similarity']])
        
        # 2. Get Collaborative Predictions for all these movies for the given user
        user_rated_raw_ids = self.collab_model.user_rated.get(user_id, set())

        # Predict ratings for all movies except the ones user already rated
        predictions = []
        for _, row in content_df.iterrows():
            m_id = row['movieId']
            if m_id not in user_rated_raw_ids:
                est_rating = self.collab_model.predict(user_id, m_id)
                predictions.append({'movieId': m_id, 'est_rating': est_rating})
                
        collab_df = pd.DataFrame(predictions)
        
        if collab_df.empty:
            return pd.DataFrame()
            
        # Normalize predicted ratings (0 to 1)
        collab_df['norm_est'] = scaler.fit_transform(collab_df[['est_rating']])
        
        # 3. Merge and compute hybrid score
        hybrid_df = pd.merge(content_df, collab_df, on='movieId')
        hybrid_df['hybrid_score'] = (hybrid_df['norm_sim'] * self.content_weight) + (hybrid_df['norm_est'] * self.collab_weight)
        
        # 4. Sort and return top_n
        hybrid_df = hybrid_df.sort_values(by='hybrid_score', ascending=False).head(top_n)
        
        # Join with movie metadata
        results = pd.merge(hybrid_df, self.content_model.movies_df, on='movieId')
        return results[['movieId', 'title', 'genres', 'similarity', 'est_rating', 'hybrid_score']]
