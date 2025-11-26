"""
Data cleaning module for Spotify dataset.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class DataCleaner:
    """Clean and validate Spotify track data."""
    
    def __init__(self):
        self.cleaning_stats = {}
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all cleaning steps to the dataframe.
        
        Args:
            df: Raw tracks dataframe
            
        Returns:
            Cleaned dataframe
        """
        print("Starting data cleaning...")
        original_count = len(df)
        
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # 1. Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # 2. Remove duplicates
        df_clean = self._remove_duplicates(df_clean)
        
        # 3. Handle outliers
        df_clean = self._handle_outliers(df_clean)
        
        # 4. Validate data types
        df_clean = self._validate_types(df_clean)
        
        # Store cleaning stats
        self.cleaning_stats = {
            'original_count': original_count,
            'final_count': len(df_clean),
            'removed_count': original_count - len(df_clean),
            'removal_percentage': ((original_count - len(df_clean)) / original_count) * 100
        }
        
        print(f"Cleaning complete. Removed {self.cleaning_stats['removed_count']} tracks "
              f"({self.cleaning_stats['removal_percentage']:.2f}%)")
        
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        print("  - Handling missing values...")
        
        # For track names, fill with "Unknown Track"
        if 'name' in df.columns:
            missing_names = df['name'].isnull().sum()
            if missing_names > 0:
                df['name'] = df['name'].fillna('Unknown Track')
                print(f"    Filled {missing_names} missing track names")
        
        # For audio features, drop rows with missing values (should be none based on exploration)
        audio_features = ['danceability', 'energy', 'loudness', 'speechiness', 
                         'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
        
        before_count = len(df)
        df = df.dropna(subset=audio_features)
        dropped = before_count - len(df)
        if dropped > 0:
            print(f"    Dropped {dropped} rows with missing audio features")
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate tracks."""
        print("  - Removing duplicates...")
        
        before_count = len(df)
        
        # Remove exact duplicates based on track ID
        df = df.drop_duplicates(subset=['id'], keep='first')
        
        duplicates_removed = before_count - len(df)
        if duplicates_removed > 0:
            print(f"    Removed {duplicates_removed} duplicate tracks")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in audio features."""
        print("  - Handling outliers...")
        
        # For tempo, remove tracks with tempo = 0 (likely errors)
        if 'tempo' in df.columns:
            before_count = len(df)
            df = df[df['tempo'] > 0]
            removed = before_count - len(df)
            if removed > 0:
                print(f"    Removed {removed} tracks with invalid tempo (0)")
        
        # For duration, remove extremely short tracks (< 10 seconds) or very long (> 30 minutes)
        if 'duration_ms' in df.columns:
            before_count = len(df)
            df = df[(df['duration_ms'] >= 10000) & (df['duration_ms'] <= 1800000)]
            removed = before_count - len(df)
            if removed > 0:
                print(f"    Removed {removed} tracks with invalid duration")
        
        return df
    
    def _validate_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and convert data types."""
        print("  - Validating data types...")
        
        # Ensure audio features are float
        audio_features = ['danceability', 'energy', 'loudness', 'speechiness', 
                         'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
        
        for feature in audio_features:
            if feature in df.columns:
                df[feature] = pd.to_numeric(df[feature], errors='coerce')
        
        # Ensure popularity is int
        if 'popularity' in df.columns:
            df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(0).astype(int)
        
        return df
    
    def get_cleaning_report(self) -> Dict:
        """Get a report of cleaning operations performed."""
        return self.cleaning_stats


if __name__ == "__main__":
    # Test the cleaner
    df = pd.read_csv('data/raw/tracks.csv')
    cleaner = DataCleaner()
    df_clean = cleaner.clean(df)
    print(f"\nCleaning report: {cleaner.get_cleaning_report()}")
    print(f"Clean dataset shape: {df_clean.shape}")
