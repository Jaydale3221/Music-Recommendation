import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

load_dotenv()

def test_single_track_features():
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        print("Error: Credentials not found.")
        return

    try:
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret
        ))
        
        print(f"Searching for track...")
        results = sp.search(q='track:Bohemian Rhapsody artist:Queen', type='track', limit=1)
        if results['tracks']['items']:
            track = results['tracks']['items'][0]
            print(f"Track Name: {track['name']}")
            print(f"Track ID: {track['id']}")
            print(f"Preview URL: {track['preview_url']}")
        else:
            print("No track found.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_single_track_features()
