import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self):
        self.tfidf = TfidfVectorizer()
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.movies_df = None
        self.indices = None

    def fit(self, movies_df):
        self.movies_df = movies_df.copy()
        
        # Ensure genres_str exists for TF-IDF
        if "genres_str" not in self.movies_df.columns:
            if "genre_list" in self.movies_df.columns:
                self.movies_df["genres_str"] = self.movies_df["genre_list"].apply(lambda x: " ".join(x) if isinstance(x, list) else str(x))
            else:
                self.movies_df["genres_str"] = self.movies_df["genres"].fillna("").str.replace("|", " ")
                
        # Vectorize genres
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies_df["genres_str"])
        
        # Compute cosine similarity
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        # Map movie titles to index positions
        self.indices = pd.Series(self.movies_df.index, index=self.movies_df["title"]).drop_duplicates()
        
    def recommend(self, title, top_n=10):
        """Recommend movies based on cosine similarity of genres."""
        if self.movies_df is None or self.cosine_sim is None:
            raise ValueError("Model must be fitted before recommending.")
            
        matches = self.movies_df[self.movies_df["title"].str.lower() == title.lower()]
        
        if len(matches) == 0:
            return None
        
        idx = matches.index[0]
        
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Skip the first one since it's the movie itself
        sim_scores = sim_scores[1:top_n+1]
        
        movie_indices = [i[0] for i in sim_scores]
        
        # Return dataframe of recommendations with similarity score
        recs = self.movies_df.iloc[movie_indices].copy()
        recs['similarity'] = [i[1] for i in sim_scores]
        
        return recs[["movieId", "title", "genres", "similarity"]]

    def search_movie(self, keyword):
        """Helper to search for a movie by keyword."""
        if self.movies_df is None:
            raise ValueError("Model must be fitted first.")
        return self.movies_df[self.movies_df["title"].str.contains(keyword, case=False)][["title"]].head(20)
