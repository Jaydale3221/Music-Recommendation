"""
Music recommendation engine.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from src.models.similarity import SimilarityComputer
from src.models.index_builder import IndexBuilder

class MusicRecommender:
    """Content-based music recommendation system."""
    
    def __init__(self, model_dir: str = 'models', 
                 data_path: str = 'data/processed/tracks_processed.csv'):
        """
        Initialize recommender.
        
        Args:
            model_dir: Directory containing saved model files
            data_path: Path to processed tracks CSV
        """
        self.model_dir = model_dir
        self.data_path = data_path
        
        # Load index
        self.index_builder = IndexBuilder(model_dir)
        self.feature_matrix, self.track_index, self.config = self.index_builder.load_index()
        
        # Load full dataset for metadata
        self.df = pd.read_csv(data_path)
        
        # Initialize similarity computer
        self.similarity_computer = SimilarityComputer()
        
        # Apply weights to feature matrix
        feature_names = self.config['feature_columns']
        self.weighted_feature_matrix = self.similarity_computer.apply_weights(
            self.feature_matrix, feature_names
        )
        
        print(f"Recommender initialized with {len(self.track_index['id_to_idx'])} tracks")
    
    def find_track_by_name(self, track_name: str, artist_name: Optional[str] = None) -> List[Dict]:
        """
        Find tracks by name (and optionally artist).
        
        Args:
            track_name: Track name to search for
            artist_name: Optional artist name to filter by
            
        Returns:
            List of matching tracks with metadata
        """
        # Case-insensitive search
        mask = self.df['name'].str.contains(track_name, case=False, na=False)
        
        if artist_name:
            mask &= self.df['artists'].str.contains(artist_name, case=False, na=False)
        
        matches = self.df[mask][['id', 'name', 'artists', 'popularity', 'release_year']].head(10)
        
        return matches.to_dict('records')
    
    def get_recommendations(self, track_id: str, 
                          n_recommendations: int = 50,
                          diversity_filter: bool = True,
                          min_popularity: Optional[int] = None,
                          year_range: Optional[Tuple[int, int]] = None) -> List[Dict]:
        """
        Get recommendations for a given track.
        
        Args:
            track_id: Spotify track ID
            n_recommendations: Number of recommendations to return
            diversity_filter: Whether to filter out multiple tracks from same artist
            min_popularity: Minimum popularity score (0-100)
            year_range: Tuple of (min_year, max_year) to filter by
            
        Returns:
            List of recommended tracks with metadata and similarity scores
        """
        # Get track index
        if track_id not in self.track_index['id_to_idx']:
            raise ValueError(f"Track ID {track_id} not found in index")
        
        track_idx = self.track_index['id_to_idx'][track_id]
        
        # Get query vector
        query_vector = self.weighted_feature_matrix[track_idx]
        
        # Compute similarities (request more than needed for filtering)
        k = n_recommendations * 5 if diversity_filter else n_recommendations
        top_k_indices, top_k_scores = self.similarity_computer.get_top_k_similar(
            query_vector, 
            self.weighted_feature_matrix,
            k=min(k, len(self.feature_matrix)),
            exclude_indices=[track_idx]
        )
        
        # Build recommendations list
        recommendations = []
        seen_artists = set()
        
        for idx, score in zip(top_k_indices, top_k_scores):
            track_data = self.df.iloc[idx]
            
            # Apply filters
            if min_popularity and track_data.get('popularity', 0) < min_popularity:
                continue
            
            if year_range:
                year = track_data.get('release_year')
                if pd.isna(year) or year < year_range[0] or year > year_range[1]:
                    continue
            
            # Diversity filter: limit tracks per artist
            artist = track_data.get('artists', 'Unknown')
            if diversity_filter:
                if artist in seen_artists:
                    continue
                seen_artists.add(artist)
            
            recommendations.append({
                'id': track_data['id'],
                'name': track_data.get('name', 'Unknown'),
                'artists': artist,
                'popularity': int(track_data.get('popularity', 0)),
                'release_year': int(track_data.get('release_year', 0)) if pd.notna(track_data.get('release_year')) else None,
                'similarity_score': float(score),
                'danceability': float(track_data.get('danceability', 0)),
                'energy': float(track_data.get('energy', 0)),
                'valence': float(track_data.get('valence', 0)),
                'tempo': float(track_data.get('tempo', 0))
            })
            
            if len(recommendations) >= n_recommendations:
                break
        
        return recommendations
    
    def get_recommendations_by_features(self, 
                                       danceability: float,
                                       energy: float,
                                       valence: float,
                                       tempo: float,
                                       n_recommendations: int = 50) -> List[Dict]:
        """
        Get recommendations based on desired audio features.
        
        Args:
            danceability: Desired danceability (0-1)
            energy: Desired energy (0-1)
            valence: Desired valence (0-1)
            tempo: Desired tempo (BPM)
            n_recommendations: Number of recommendations
            
        Returns:
            List of recommended tracks
        """
        # Create a synthetic feature vector
        # This is simplified - in practice you'd need to match the full feature space
        feature_names = self.config['feature_columns']
        
        # Create query vector with average values for unspecified features
        query_vector = np.zeros(len(feature_names))
        
        for i, feature in enumerate(feature_names):
            if feature == 'danceability':
                query_vector[i] = danceability
            elif feature == 'energy':
                query_vector[i] = energy
            elif feature == 'valence':
                query_vector[i] = valence
            elif feature == 'tempo':
                query_vector[i] = tempo
            elif feature == 'tempo_normalized':
                query_vector[i] = tempo / 250.0  # Normalize
            else:
                # Use median value from dataset
                query_vector[i] = np.median(self.feature_matrix[:, i])
        
        # Normalize using the scaler
        query_vector = (query_vector - self.index_builder.scaler.mean_) / self.index_builder.scaler.scale_
        
        # Apply weights
        query_vector = self.similarity_computer.apply_weights(
            query_vector.reshape(1, -1), feature_names
        )[0]
        
        # Get similar tracks
        top_k_indices, top_k_scores = self.similarity_computer.get_top_k_similar(
            query_vector,
            self.weighted_feature_matrix,
            k=n_recommendations
        )
        
        # Build recommendations
        recommendations = []
        for idx, score in zip(top_k_indices, top_k_scores):
            track_data = self.df.iloc[idx]
            recommendations.append({
                'id': track_data['id'],
                'name': track_data.get('name', 'Unknown'),
                'artists': track_data.get('artists', 'Unknown'),
                'similarity_score': float(score),
                'danceability': float(track_data.get('danceability', 0)),
                'energy': float(track_data.get('energy', 0)),
                'valence': float(track_data.get('valence', 0)),
                'tempo': float(track_data.get('tempo', 0))
            })
        
        return recommendations


if __name__ == "__main__":
    # Test the recommender
    print("Testing MusicRecommender...\n")
    
    recommender = MusicRecommender()
    
    # Search for a track
    print("Searching for 'Bohemian Rhapsody'...")
    matches = recommender.find_track_by_name("Bohemian Rhapsody", "Queen")
    
    if matches:
        print(f"Found {len(matches)} matches:")
        for match in matches:
            print(f"  - {match['name']} by {match['artists']} ({match['release_year']})")
        
        # Get recommendations for first match
        track_id = matches[0]['id']
        print(f"\nGetting recommendations for: {matches[0]['name']}...")
        
        recommendations = recommender.get_recommendations(track_id, n_recommendations=10)
        
        print(f"\nTop 10 recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['name']} by {rec['artists']}")
            print(f"   Similarity: {rec['similarity_score']:.4f} | Popularity: {rec['popularity']}")
    else:
        print("No matches found")
