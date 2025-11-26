import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
from typing import Dict, List, Optional

load_dotenv()

class SpotifyClient:
    def __init__(self):
        client_id = os.getenv("SPOTIFY_CLIENT_ID")
        client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        
        if not client_id or not client_secret:
            raise ValueError("Spotify credentials not found in environment variables.")

        self.sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret
        ))

    def search_track(self, query: str, limit: int = 1) -> List[Dict]:
        """
        Search for a track by name/artist.
        """
        results = self.sp.search(q=query, limit=limit, type='track')
        tracks = results['tracks']['items']
        return tracks

    def get_track_features(self, track_id: str) -> Optional[Dict]:
        """
        Get audio features for a specific track ID.
        """
        features = self.sp.audio_features([track_id])
        return features[0] if features else None

    def get_track_metadata(self, track_id: str) -> Dict:
        """
        Get detailed metadata for a track.
        """
        return self.sp.track(track_id)

    def get_recommendations(self, seed_tracks: List[str], limit: int = 10, **kwargs) -> List[Dict]:
        """
        Get recommendations based on seed tracks (using Spotify's native algo as a baseline).
        """
        return self.sp.recommendations(seed_tracks=seed_tracks, limit=limit, **kwargs)['tracks']

if __name__ == "__main__":
    # Simple test
    try:
        client = SpotifyClient()
        print("Spotify Client initialized successfully.")
    except Exception as e:
        print(f"Error initializing client: {e}")
