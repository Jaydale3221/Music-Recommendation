import pandas as pd
import time
from typing import List, Dict, Optional
from src.api.spotify_client import SpotifyClient

class DataCollector:
    def __init__(self):
        self.client = SpotifyClient()
        self.tracks_data = []

    def collect_from_genres(self, genres: List[str], limit_per_genre: int = 50):
        """
        Collect tracks for specific genres using search.
        """
        print(f"Starting collection for genres: {genres}")
        
        for genre in genres:
            print(f"Fetching tracks for genre: {genre}")
            try:
                # Search for tracks with this genre
                query = f"genre:{genre}"
                # Spotify search limit is max 50
                search_limit = min(limit_per_genre, 50)
                
                # We might need pagination if limit_per_genre > 50, 
                # but for now let's stick to simple search for the MVP
                results = self.client.sp.search(q=query, limit=search_limit, type='track')
                tracks = results['tracks']['items']
                
                self._process_tracks(tracks, genre)
                
                # Be nice to the API
                time.sleep(1)
                
            except Exception as e:
                print(f"Error collecting genre {genre}: {e}")

    def _process_tracks(self, tracks: List[Dict], genre: str = None):
        """
        Process a list of tracks: fetch features and append to internal list.
        """
        track_ids = [t['id'] for t in tracks if t and t.get('id')]
        
        # Fetch features in batches (Spotify allows max 100 ids per call)
        # Since we search with limit 50, we can do it in one go usually
        if not track_ids:
            return

        try:
            features_list = self.client.sp.audio_features(track_ids)
            
            for track, features in zip(tracks, features_list):
                if not features:
                    continue
                    
                track_info = {
                    'id': track['id'],
                    'name': track['name'],
                    'artist': track['artists'][0]['name'],
                    'artist_id': track['artists'][0]['id'],
                    'album': track['album']['name'],
                    'popularity': track['popularity'],
                    'genre': genre,  # Note: Tracks don't strictly have genres, artists do. This is a proxy.
                    # Audio features
                    'danceability': features['danceability'],
                    'energy': features['energy'],
                    'key': features['key'],
                    'loudness': features['loudness'],
                    'mode': features['mode'],
                    'speechiness': features['speechiness'],
                    'acousticness': features['acousticness'],
                    'instrumentalness': features['instrumentalness'],
                    'liveness': features['liveness'],
                    'valence': features['valence'],
                    'tempo': features['tempo'],
                    'duration_ms': features['duration_ms'],
                    'time_signature': features['time_signature']
                }
                self.tracks_data.append(track_info)
                
        except Exception as e:
            print(f"Error processing tracks: {e}")

    def save_to_csv(self, filepath: str):
        """
        Save collected data to CSV.
        """
        if not self.tracks_data:
            print("No data to save.")
            return
            
        df = pd.DataFrame(self.tracks_data)
        # Remove duplicates just in case
        df = df.drop_duplicates(subset=['id'])
        
        print(f"Saving {len(df)} tracks to {filepath}")
        df.to_csv(filepath, index=False)

if __name__ == "__main__":
    # Simple test run
    collector = DataCollector()
    genres = ['pop', 'rock', 'hip-hop', 'jazz', 'classical']
    collector.collect_from_genres(genres, limit_per_genre=10)
    collector.save_to_csv("data/raw/test_dataset.csv")
