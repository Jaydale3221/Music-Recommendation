#!/usr/bin/env python3
"""
Explore the downloaded Spotify dataset.
"""

import pandas as pd
import json

def explore_dataset():
    """Load and explore the Spotify dataset."""
    
    print("=" * 80)
    print("SPOTIFY DATASET EXPLORATION")
    print("=" * 80)
    
    # Load tracks data
    print("\n1. TRACKS DATA")
    print("-" * 80)
    tracks_df = pd.read_csv('data/raw/tracks.csv')
    print(f"Total tracks: {len(tracks_df):,}")
    print(f"Columns: {list(tracks_df.columns)}")
    print(f"\nFirst few rows:")
    print(tracks_df.head())
    print(f"\nData types:")
    print(tracks_df.dtypes)
    print(f"\nMissing values:")
    print(tracks_df.isnull().sum())
    
    # Check for audio features
    audio_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                     'speechiness', 'acousticness', 'instrumentalness', 
                     'liveness', 'valence', 'tempo']
    available_features = [f for f in audio_features if f in tracks_df.columns]
    print(f"\nAvailable audio features: {available_features}")
    
    # Load artists data
    print("\n\n2. ARTISTS DATA")
    print("-" * 80)
    artists_df = pd.read_csv('data/raw/artists.csv')
    print(f"Total artists: {len(artists_df):,}")
    print(f"Columns: {list(artists_df.columns)}")
    print(f"\nFirst few rows:")
    print(artists_df.head())
    
    # Sample statistics
    print("\n\n3. SAMPLE STATISTICS")
    print("-" * 80)
    if available_features:
        print(tracks_df[available_features].describe())
    
    # Save a summary
    summary = {
        'total_tracks': len(tracks_df),
        'total_artists': len(artists_df),
        'columns': list(tracks_df.columns),
        'audio_features': available_features,
        'date_range': {
            'min': str(tracks_df['release_date'].min()) if 'release_date' in tracks_df.columns else 'N/A',
            'max': str(tracks_df['release_date'].max()) if 'release_date' in tracks_df.columns else 'N/A'
        }
    }
    
    with open('data/raw/dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n\nSummary saved to data/raw/dataset_summary.json")
    print("=" * 80)

if __name__ == "__main__":
    explore_dataset()
