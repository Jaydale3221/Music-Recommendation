"""
Index builder for efficient similarity search.
"""

import pandas as pd
import numpy as np
import json
import os
from typing import List, Dict, Tuple
from sklearn.preprocessing import StandardScaler

class IndexBuilder:
    """Build and save feature index for fast similarity search."""
    
    def __init__(self, output_dir: str = 'models'):
        self.output_dir = output_dir
        self.scaler = StandardScaler()
        os.makedirs(output_dir, exist_ok=True)
    
    def build_index(self, df: pd.DataFrame, 
                   feature_columns: List[str]) -> Tuple[np.ndarray, Dict]:
        """
        Build feature index from processed dataset.
        
        Args:
            df: Processed tracks dataframe
            feature_columns: List of feature column names to use
            
        Returns:
            Tuple of (feature_matrix, track_index_mapping)
        """
        print("Building feature index...")
        
        # Extract features
        print(f"  - Extracting {len(feature_columns)} features...")
        feature_matrix = df[feature_columns].values
        
        # Handle any remaining NaN values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
        
        # Normalize features using StandardScaler
        print("  - Normalizing features...")
        feature_matrix = self.scaler.fit_transform(feature_matrix)
        
        # Create track index mapping
        print("  - Creating track index mapping...")
        track_index = {
            'id_to_idx': {track_id: idx for idx, track_id in enumerate(df['id'])},
            'idx_to_id': {idx: track_id for idx, track_id in enumerate(df['id'])},
            'track_metadata': {}
        }
        
        # Store essential metadata for each track
        for idx, row in df.iterrows():
            track_index['track_metadata'][str(idx)] = {
                'id': row['id'],
                'name': row.get('name', 'Unknown'),
                'artists': row.get('artists', 'Unknown'),
                'popularity': int(row.get('popularity', 0)),
                'release_year': int(row.get('release_year', 0)) if pd.notna(row.get('release_year')) else None
            }
        
        print(f"  - Index built: {feature_matrix.shape[0]} tracks, {feature_matrix.shape[1]} features")
        
        return feature_matrix, track_index
    
    def save_index(self, feature_matrix: np.ndarray, 
                  track_index: Dict,
                  feature_columns: List[str],
                  config: Dict = None):
        """
        Save feature index to disk.
        
        Args:
            feature_matrix: Normalized feature matrix
            track_index: Track ID to index mapping
            feature_columns: List of feature column names
            config: Additional configuration to save
        """
        print("Saving index to disk...")
        
        # Save feature matrix
        matrix_path = os.path.join(self.output_dir, 'feature_matrix.npy')
        np.save(matrix_path, feature_matrix)
        print(f"  - Feature matrix saved to {matrix_path}")
        
        # Save track index
        index_path = os.path.join(self.output_dir, 'track_index.json')
        with open(index_path, 'w') as f:
            json.dump(track_index, f, indent=2)
        print(f"  - Track index saved to {index_path}")
        
        # Save configuration
        config_data = {
            'feature_columns': feature_columns,
            'n_tracks': feature_matrix.shape[0],
            'n_features': feature_matrix.shape[1],
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_scale': self.scaler.scale_.tolist()
        }
        
        if config:
            config_data.update(config)
        
        config_path = os.path.join(self.output_dir, 'model_config.json')
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        print(f"  - Configuration saved to {config_path}")
    
    def load_index(self) -> Tuple[np.ndarray, Dict, Dict]:
        """
        Load feature index from disk.
        
        Returns:
            Tuple of (feature_matrix, track_index, config)
        """
        print("Loading index from disk...")
        
        # Load feature matrix
        matrix_path = os.path.join(self.output_dir, 'feature_matrix.npy')
        feature_matrix = np.load(matrix_path)
        print(f"  - Loaded feature matrix: {feature_matrix.shape}")
        
        # Load track index
        index_path = os.path.join(self.output_dir, 'track_index.json')
        with open(index_path, 'r') as f:
            track_index = json.load(f)
        print(f"  - Loaded track index: {len(track_index['id_to_idx'])} tracks")
        
        # Load configuration
        config_path = os.path.join(self.output_dir, 'model_config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"  - Loaded configuration")
        
        # Restore scaler
        self.scaler.mean_ = np.array(config['scaler_mean'])
        self.scaler.scale_ = np.array(config['scaler_scale'])
        
        return feature_matrix, track_index, config
    
    @staticmethod
    def get_recommended_features() -> List[str]:
        """Get recommended feature columns for similarity computation."""
        return [
            # Original audio features
            'danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence',
            'tempo', 'mode', 'key',
            
            # Normalized features
            'loudness_normalized', 'tempo_normalized',
            
            # Derived features
            'energy_danceability', 'mood_score', 'acoustic_ratio',
            'vocal_presence', 'intensity', 'chill_factor',
            
            # Temporal features
            'track_age_normalized'
        ]


if __name__ == "__main__":
    # Test index building
    print("Testing IndexBuilder...")
    
    # Load processed data
    df = pd.read_csv('data/processed/tracks_processed.csv')
    
    # Get recommended features
    feature_columns = IndexBuilder.get_recommended_features()
    
    # Filter to only existing columns
    feature_columns = [f for f in feature_columns if f in df.columns]
    print(f"Using {len(feature_columns)} features: {feature_columns}")
    
    # Build index
    builder = IndexBuilder()
    feature_matrix, track_index = builder.build_index(df, feature_columns)
    
    # Save index
    builder.save_index(feature_matrix, track_index, feature_columns)
    
    print("\nIndex building complete!")
