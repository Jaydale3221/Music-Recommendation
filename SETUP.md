# Setup Instructions

## Prerequisites

- Python 3.9 or higher
- pip
- Kaggle account (for dataset download)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/music-recommendation-system.git
cd music-recommendation-system
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Kaggle API (for dataset download)

1. Go to https://www.kaggle.com/settings
2. Click "Create New API Token"
3. Save the downloaded `kaggle.json` to `~/.kaggle/`
4. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### 5. Download Dataset

```bash
python scripts/download_dataset.py
```

This will download ~193MB of data (586K+ tracks) from Kaggle.

### 6. Process Data

```bash
python -m src.preprocessing.pipeline
```

This will clean the data and engineer features (~2-3 minutes).

### 7. Build Model Index

```bash
python -m src.models.index_builder
```

This creates the feature index for fast similarity search (~1-2 minutes).

## Usage

### Interactive Mode (Recommended)

```bash
python interactive.py
```

Type song names and get instant recommendations!

### Demo Mode

```bash
python demo.py
```

See pre-configured examples of the recommendation system.

### Programmatic Usage

```python
from src.models.recommender import MusicRecommender

# Initialize
recommender = MusicRecommender()

# Search for a track
matches = recommender.find_track_by_name("Blinding Lights", "The Weeknd")
track_id = matches[0]['id']

# Get recommendations
recommendations = recommender.get_recommendations(
    track_id,
    n_recommendations=50,
    diversity_filter=True,
    min_popularity=40
)
```

## Optional: Spotify API Setup

If you want to use the Spotify API client (not required for recommendations):

1. Go to https://developer.spotify.com/dashboard
2. Create an app and get your credentials
3. Create a `.env` file:

```
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
```

## Troubleshooting

### Dataset Download Issues

If Kaggle download fails:
1. Ensure you've accepted the dataset terms on Kaggle website
2. Check your `kaggle.json` is in the correct location
3. Verify your Kaggle API token is valid

### Memory Issues

If you encounter memory errors during processing:
- The dataset is large (586K tracks)
- Ensure you have at least 4GB of available RAM
- Close other applications during processing

### Import Errors

If you get `ModuleNotFoundError`:
- Make sure you're in the project root directory
- Activate the virtual environment
- Run scripts as modules: `python -m src.module.name`

## File Sizes

After setup, expect these file sizes:
- `data/raw/tracks.csv`: ~150MB
- `data/processed/tracks_processed.csv`: ~180MB
- `models/feature_matrix.npy`: ~90MB
- `models/track_index.json`: ~200MB

**Note**: These large files are excluded from the repository via `.gitignore`. You must download and process them locally.
