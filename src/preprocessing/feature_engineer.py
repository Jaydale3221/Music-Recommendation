"""
Feature engineering module for Spotify dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, List
import json

class FeatureEngineer:
    """Engineer features for music recommendation."""
    
    def __init__(self):
        self.scalers = {}
        self.feature_metadata = {}
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps.
        
        Args:
            df: Cleaned tracks dataframe
            
        Returns:
            Dataframe with engineered features
        """
        print("Starting feature engineering...")
        
        df_engineered = df.copy()
        
        # 1. Normalize audio features
        df_engineered = self._normalize_features(df_engineered)
        
        # 2. Create derived features
        df_engineered = self._create_derived_features(df_engineered)
        
        # 3. Engineer temporal features
        df_engineered = self._create_temporal_features(df_engineered)
        
        # 4. Parse genre information
        df_engineered = self._parse_genres(df_engineered)
        
        print("Feature engineering complete.")
        
        return df_engineered
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize audio features to [0, 1] range."""
        print("  - Normalizing audio features...")
        
        # Features already in [0, 1] range
        normalized_features = ['danceability', 'energy', 'speechiness', 
                              'acousticness', 'instrumentalness', 'liveness', 'valence']
        
        # Features that need normalization
        features_to_scale = {
            'loudness': (-60, 0),  # Typical range
            'tempo': (0, 250),      # Typical range
        }
        
        for feature, (min_val, max_val) in features_to_scale.items():
            if feature in df.columns:
                # Clip to expected range and normalize
                df[f'{feature}_normalized'] = df[feature].clip(min_val, max_val)
                df[f'{feature}_normalized'] = (df[f'{feature}_normalized'] - min_val) / (max_val - min_val)
        
        return df
    
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from existing ones."""
        print("  - Creating derived features...")
        
        # Energy-Danceability (party factor)
        if 'energy' in df.columns and 'danceability' in df.columns:
            df['energy_danceability'] = df['energy'] * df['danceability']
        
        # Mood score (combination of valence and energy)
        if 'valence' in df.columns and 'energy' in df.columns:
            df['mood_score'] = (df['valence'] + df['energy']) / 2
        
        # Acoustic ratio
        if 'acousticness' in df.columns and 'instrumentalness' in df.columns:
            df['acoustic_ratio'] = df['acousticness'] / (1 - df['instrumentalness'] + 0.001)
            df['acoustic_ratio'] = df['acoustic_ratio'].clip(0, 10)  # Clip extreme values
        
        # Vocal presence (inverse of instrumentalness)
        if 'instrumentalness' in df.columns:
            df['vocal_presence'] = 1 - df['instrumentalness']
        
        # Intensity score (energy + loudness)
        if 'energy' in df.columns and 'loudness_normalized' in df.columns:
            df['intensity'] = (df['energy'] + df['loudness_normalized']) / 2
        
        # Chill factor (low energy, high acousticness)
        if 'energy' in df.columns and 'acousticness' in df.columns:
            df['chill_factor'] = (1 - df['energy']) * df['acousticness']
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from release date."""
        print("  - Creating temporal features...")
        
        if 'release_date' not in df.columns:
            return df
        
        # Parse release date (handle different formats)
        df['release_date_parsed'] = pd.to_datetime(df['release_date'], errors='coerce')
        
        # Extract year
        df['release_year'] = df['release_date_parsed'].dt.year
        
        # Handle missing years (use median year)
        median_year = df['release_year'].median()
        df['release_year'] = df['release_year'].fillna(median_year).astype(int)
        
        # Create decade
        df['release_decade'] = (df['release_year'] // 10) * 10
        
        # Create era categories
        def categorize_era(year):
            if year < 1960:
                return 'vintage'
            elif year < 1980:
                return '60s-70s'
            elif year < 2000:
                return '80s-90s'
            elif year < 2010:
                return '2000s'
            elif year < 2020:
                return '2010s'
            else:
                return '2020s'
        
        df['release_era'] = df['release_year'].apply(categorize_era)
        
        # Track age (years since release)
        current_year = 2024
        df['track_age'] = current_year - df['release_year']
        
        # Normalize track age
        df['track_age_normalized'] = df['track_age'] / df['track_age'].max()
        
        return df
    
    def _parse_genres(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse and process genre information."""
        print("  - Processing genre information...")
        
        # For now, just count the number of artists per track
        if 'artists' in df.columns:
            df['artist_count'] = df['artists'].str.count(',') + 1
        
        # We'll handle genre embeddings later when we have artist data
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Get categorized feature columns."""
        
        original_audio_features = [
            'danceability', 'energy', 'key', 'loudness', 'mode', 
            'speechiness', 'acousticness', 'instrumentalness', 
            'liveness', 'valence', 'tempo'
        ]
        
        normalized_features = [
            'loudness_normalized', 'tempo_normalized'
        ]
        
        derived_features = [
            'energy_danceability', 'mood_score', 'acoustic_ratio',
            'vocal_presence', 'intensity', 'chill_factor'
        ]
        
        temporal_features = [
            'release_year', 'release_decade', 'track_age', 'track_age_normalized'
        ]
        
        # Filter to only include columns that exist in the dataframe
        return {
            'original_audio': [f for f in original_audio_features if f in df.columns],
            'normalized': [f for f in normalized_features if f in df.columns],
            'derived': [f for f in derived_features if f in df.columns],
            'temporal': [f for f in temporal_features if f in df.columns]
        }
    
    def save_feature_metadata(self, df: pd.DataFrame, filepath: str):
        """Save feature metadata to JSON."""
        feature_cols = self.get_feature_columns(df)
        
        metadata = {
            'total_features': sum(len(v) for v in feature_cols.values()),
            'feature_categories': feature_cols,
            'feature_statistics': {}
        }
        
        # Add statistics for numerical features
        all_features = []
        for category_features in feature_cols.values():
            all_features.extend(category_features)
        
        for feature in all_features:
            if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
                metadata['feature_statistics'][feature] = {
                    'mean': float(df[feature].mean()),
                    'std': float(df[feature].std()),
                    'min': float(df[feature].min()),
                    'max': float(df[feature].max())
                }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Feature metadata saved to {filepath}")


if __name__ == "__main__":
    # Test the feature engineer
    df = pd.read_csv('data/raw/tracks.csv')
    
    # Clean first
    from data_cleaner import DataCleaner
    cleaner = DataCleaner()
    df_clean = cleaner.clean(df)
    
    # Engineer features
    engineer = FeatureEngineer()
    df_engineered = engineer.engineer_features(df_clean)
    
    print(f"\nEngineered dataset shape: {df_engineered.shape}")
    print(f"Feature columns: {engineer.get_feature_columns(df_engineered)}")
