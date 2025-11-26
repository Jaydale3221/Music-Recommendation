"""
Similarity computation for music recommendation.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Optional, Tuple

class SimilarityComputer:
    """Compute similarity between tracks based on audio features."""
    
    def __init__(self, feature_weights: Optional[Dict[str, float]] = None):
        """
        Initialize similarity computer.
        
        Args:
            feature_weights: Dictionary mapping feature names to weights.
                           If None, uses default weights.
        """
        self.feature_weights = feature_weights or self._get_default_weights()
    
    @staticmethod
    def _get_default_weights() -> Dict[str, float]:
        """Get default feature weights based on importance for similarity."""
        return {
            # High importance - core musical characteristics
            'danceability': 2.0,
            'energy': 2.0,
            'valence': 2.0,
            'tempo_normalized': 2.0,
            
            # Medium importance - texture and style
            'acousticness': 1.5,
            'instrumentalness': 1.5,
            'loudness_normalized': 1.0,
            
            # Lower importance - specific characteristics
            'speechiness': 0.5,
            'liveness': 0.5,
            'mode': 0.3,
            'key': 0.3,
            
            # Derived features
            'mood_score': 2.5,
            'energy_danceability': 2.0,
            'vocal_presence': 1.0,
            'intensity': 1.5,
            'chill_factor': 1.5,
            
            # Temporal features (lower weight)
            'track_age_normalized': 0.5,
        }
    
    def apply_weights(self, feature_matrix: np.ndarray, 
                     feature_names: List[str]) -> np.ndarray:
        """
        Apply feature weights to the feature matrix.
        
        Args:
            feature_matrix: Matrix of shape (n_samples, n_features)
            feature_names: List of feature names corresponding to columns
            
        Returns:
            Weighted feature matrix
        """
        weights = np.array([
            self.feature_weights.get(name, 1.0) 
            for name in feature_names
        ])
        
        return feature_matrix * weights
    
    def compute_similarity(self, query_vector: np.ndarray, 
                          feature_matrix: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and all tracks.
        
        Args:
            query_vector: Feature vector of shape (n_features,)
            feature_matrix: Matrix of shape (n_samples, n_features)
            
        Returns:
            Similarity scores of shape (n_samples,)
        """
        # Reshape query vector to 2D
        query_2d = query_vector.reshape(1, -1)
        
        # Compute cosine similarity
        similarities = cosine_similarity(query_2d, feature_matrix)[0]
        
        return similarities
    
    def compute_pairwise_similarity(self, 
                                   feature_matrix: np.ndarray) -> np.ndarray:
        """
        Compute pairwise similarity matrix for all tracks.
        
        Args:
            feature_matrix: Matrix of shape (n_samples, n_features)
            
        Returns:
            Similarity matrix of shape (n_samples, n_samples)
        """
        return cosine_similarity(feature_matrix)
    
    def get_top_k_similar(self, query_vector: np.ndarray,
                         feature_matrix: np.ndarray,
                         k: int = 50,
                         exclude_indices: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get top-K most similar tracks.
        
        Args:
            query_vector: Feature vector of query track
            feature_matrix: Matrix of all track features
            k: Number of similar tracks to return
            exclude_indices: Indices to exclude from results (e.g., the query track itself)
            
        Returns:
            Tuple of (indices, similarity_scores) for top-K similar tracks
        """
        # Compute similarities
        similarities = self.compute_similarity(query_vector, feature_matrix)
        
        # Exclude specified indices
        if exclude_indices:
            similarities[exclude_indices] = -1
        
        # Get top-K indices
        top_k_indices = np.argsort(similarities)[::-1][:k]
        top_k_scores = similarities[top_k_indices]
        
        return top_k_indices, top_k_scores
    
    def compute_diversity_score(self, similarities: np.ndarray) -> float:
        """
        Compute diversity score for a set of recommendations.
        Higher score means more diverse recommendations.
        
        Args:
            similarities: Array of similarity scores
            
        Returns:
            Diversity score (0-1, higher is more diverse)
        """
        if len(similarities) < 2:
            return 1.0
        
        # Diversity is inversely related to variance in similarity scores
        # If all scores are very similar, diversity is low
        std = np.std(similarities)
        mean = np.mean(similarities)
        
        # Coefficient of variation as diversity metric
        if mean > 0:
            diversity = min(std / mean, 1.0)
        else:
            diversity = 0.0
        
        return diversity


if __name__ == "__main__":
    # Test similarity computation
    print("Testing SimilarityComputer...")
    
    # Create dummy feature matrix
    np.random.seed(42)
    n_tracks = 100
    n_features = 10
    feature_matrix = np.random.rand(n_tracks, n_features)
    
    # Create similarity computer
    computer = SimilarityComputer()
    
    # Test with first track as query
    query_vector = feature_matrix[0]
    top_k_indices, top_k_scores = computer.get_top_k_similar(
        query_vector, feature_matrix, k=10, exclude_indices=[0]
    )
    
    print(f"Top 10 similar tracks to track 0:")
    for idx, score in zip(top_k_indices, top_k_scores):
        print(f"  Track {idx}: similarity = {score:.4f}")
    
    print(f"\nDiversity score: {computer.compute_diversity_score(top_k_scores):.4f}")
