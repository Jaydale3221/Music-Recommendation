import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.spotify_client import SpotifyClient

def test_spotify_client():
    print("Testing Spotify Client...")
    
    try:
        client = SpotifyClient()
    except ValueError as e:
        print(f"Skipping test: {e}")
        print("Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env")
        return

    # 1. Search for a song
    query = "Bohemian Rhapsody"
    print(f"\nSearching for '{query}'...")
    results = client.search_track(query)
    
    if not results:
        print("No results found.")
        return

    track = results[0]
    track_id = track['id']
    print(f"Found: {track['name']} by {track['artists'][0]['name']} (ID: {track_id})")

    # 2. Get Audio Features
    print(f"\nFetching audio features for {track_id}...")
    features = client.get_track_features(track_id)
    if features:
        print("Audio Features:")
        for key in ['danceability', 'energy', 'tempo', 'valence', 'key']:
            print(f"  - {key}: {features.get(key)}")
    else:
        print("No audio features found.")

if __name__ == "__main__":
    test_spotify_client()
