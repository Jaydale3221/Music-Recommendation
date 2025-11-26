# Music Recommendation System

A content-based music recommendation system that finds similar songs based on audio features using cosine similarity.

## Features

- 🎵 **585,967 tracks** from 1921-2020
- 🎯 **Content-based filtering** using audio features
- 🔍 **Smart search** by track name and artist
- 🎨 **Diversity filtering** to avoid repetitive recommendations
- ⚡ **Fast similarity search** with pre-built index
- 🎛️ **Customizable filters** (popularity, year range, etc.)
- 📊 **Feature-based search** for mood/style discovery

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Spotify API (Optional)

Create a `.env` file:

```bash
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
```

### 3. Run the Demo

```bash
python demo.py
```

## Usage

### Initialize the Recommender

```python
from src.models.recommender import MusicRecommender

recommender = MusicRecommender()
```

### Find Similar Songs

```python
# Search for a track
matches = recommender.find_track_by_name("Shape of You", "Ed Sheeran")
track_id = matches[0]['id']

# Get recommendations
recommendations = recommender.get_recommendations(
    track_id,
    n_recommendations=50,
    diversity_filter=True
)

for rec in recommendations:
    print(f"{rec['name']} by {rec['artists']} (similarity: {rec['similarity_score']:.4f})")
```

### Search by Audio Features

```python
# Find high-energy, danceable tracks
recommendations = recommender.get_recommendations_by_features(
    danceability=0.9,
    energy=0.9,
    valence=0.8,
    tempo=128,
    n_recommendations=50
)
```

### Apply Filters

```python
# Filter by popularity and year
recommendations = recommender.get_recommendations(
    track_id,
    n_recommendations=50,
    diversity_filter=True,
    min_popularity=40,
    year_range=(2015, 2024)
)
```

## Project Structure

```
music_rec_system/
├── data/
│   ├── raw/                    # Original dataset (586K tracks)
│   └── processed/              # Cleaned & engineered features
├── models/                     # Saved model files
│   ├── feature_matrix.npy      # Normalized feature matrix
│   ├── track_index.json        # Track ID mappings
│   └── model_config.json       # Model configuration
├── src/
│   ├── api/
│   │   └── spotify_client.py   # Spotify API client
│   ├── preprocessing/
│   │   ├── data_cleaner.py     # Data cleaning
│   │   ├── feature_engineer.py # Feature engineering
│   │   └── pipeline.py         # Preprocessing pipeline
│   ├── models/
│   │   ├── similarity.py       # Similarity computation
│   │   ├── index_builder.py    # Feature index builder
│   │   └── recommender.py      # Main recommendation engine
│   └── data/
│       └── collector.py        # Data collection utilities
├── scripts/
│   ├── download_dataset.py     # Download Spotify dataset
│   └── explore_dataset.py      # Data exploration
├── demo.py                     # Demo script
└── README.md                   # This file
```

## How It Works

### 1. Data Collection
- Downloaded 586K+ tracks from Kaggle's Spotify dataset
- Includes 11 audio features: danceability, energy, valence, tempo, etc.

### 2. Feature Engineering
Created 15 additional features:
- **Derived**: mood_score, energy_danceability, vocal_presence, intensity, chill_factor
- **Normalized**: loudness_normalized, tempo_normalized
- **Temporal**: release_year, track_age, release_era

### 3. Similarity Computation
- Uses **cosine similarity** with weighted features
- High weights for: danceability, energy, valence, tempo, mood_score
- Lower weights for: key, mode, liveness, speechiness

### 4. Recommendation Engine
- Pre-built feature index for fast search
- Diversity filtering to avoid duplicate artists
- Optional filters for popularity and year range

## Audio Features

| Feature | Description | Range |
|---------|-------------|-------|
| **danceability** | How suitable for dancing | 0.0 - 1.0 |
| **energy** | Intensity and activity | 0.0 - 1.0 |
| **valence** | Musical positiveness | 0.0 - 1.0 |
| **tempo** | Beats per minute | 0 - 250 |
| **acousticness** | Acoustic vs electronic | 0.0 - 1.0 |
| **instrumentalness** | Vocal vs instrumental | 0.0 - 1.0 |
| **speechiness** | Presence of spoken words | 0.0 - 1.0 |
| **liveness** | Live performance likelihood | 0.0 - 1.0 |
| **loudness** | Overall loudness (dB) | -60 - 0 |

## Performance

- **Dataset**: 585,967 tracks
- **Features**: 20 weighted features
- **Search time**: < 1 second for 50 recommendations
- **Similarity scores**: > 0.98 for top recommendations

## Rebuilding the Index

If you modify the dataset or features:

```bash
# Reprocess data
python -m src.preprocessing.pipeline

# Rebuild index
python -m src.models.index_builder
```

## Requirements

- Python 3.9+
- pandas
- numpy
- scikit-learn
- spotipy (for Spotify API)
- python-dotenv

## License

This project uses the [Spotify Dataset 1921-2020, 600k+ Tracks](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks) from Kaggle, licensed under Community Data License Agreement - Sharing - Version 1.0.

## Next Steps

Potential enhancements:
- REST API with FastAPI
- Web interface for user interaction
- Playlist generation
- Hybrid recommendations (collaborative + content-based)
- Cloud deployment

## Credits

Built with data from Spotify's Web API and Kaggle's Spotify dataset.
