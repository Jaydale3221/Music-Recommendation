#!/usr/bin/env python3
"""
Interactive music recommendation CLI.
Type a song name and get instant recommendations!
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.recommender import MusicRecommender

def print_header():
    """Print welcome header."""
    print("\n" + "=" * 80)
    print("🎵  MUSIC RECOMMENDATION SYSTEM  🎵")
    print("=" * 80)
    print("Type a song name to get recommendations!")
    print("Commands: 'quit' or 'exit' to stop, 'help' for more options")
    print("=" * 80 + "\n")

def print_matches(matches):
    """Print search results."""
    print(f"\n📀 Found {len(matches)} matches:")
    print("-" * 80)
    for i, match in enumerate(matches, 1):
        year = match.get('release_year', 'Unknown')
        pop = match.get('popularity', 0)
        print(f"{i}. {match['name']}")
        print(f"   Artist: {match['artists']} | Year: {year} | Popularity: {pop}")
    print("-" * 80)

def print_recommendations(recommendations, query_track):
    """Print recommendations."""
    print(f"\n🎯 Top recommendations for: {query_track['name']}")
    print(f"   by {query_track['artists']}")
    print("=" * 80)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i:2d}. {rec['name']}")
        print(f"    Artist: {rec['artists']}")
        print(f"    Similarity: {rec['similarity_score']:.4f} | "
              f"Popularity: {rec['popularity']:3d} | "
              f"Year: {rec.get('release_year', 'N/A')}")
        print(f"    Features: Dance={rec['danceability']:.2f}, "
              f"Energy={rec['energy']:.2f}, "
              f"Valence={rec['valence']:.2f}, "
              f"Tempo={rec['tempo']:.0f}")
    
    print("\n" + "=" * 80)

def get_user_choice(max_choice):
    """Get user's choice from search results."""
    while True:
        try:
            choice = input(f"\nSelect a track (1-{max_choice}) or 'back' to search again: ").strip()
            
            if choice.lower() == 'back':
                return None
            
            choice_num = int(choice)
            if 1 <= choice_num <= max_choice:
                return choice_num - 1
            else:
                print(f"Please enter a number between 1 and {max_choice}")
        except ValueError:
            print("Please enter a valid number or 'back'")

def get_recommendation_settings():
    """Get recommendation settings from user."""
    print("\n⚙️  Recommendation Settings (press Enter for defaults)")
    print("-" * 80)
    
    # Number of recommendations
    while True:
        n_str = input("Number of recommendations (default: 10): ").strip()
        if not n_str:
            n_recommendations = 10
            break
        try:
            n_recommendations = int(n_str)
            if n_recommendations > 0:
                break
            print("Please enter a positive number")
        except ValueError:
            print("Please enter a valid number")
    
    # Diversity filter
    diversity_str = input("Enable diversity filter? (Y/n, default: Y): ").strip().lower()
    diversity_filter = diversity_str != 'n'
    
    # Popularity filter
    pop_str = input("Minimum popularity (0-100, default: none): ").strip()
    min_popularity = int(pop_str) if pop_str else None
    
    # Year range
    year_str = input("Year range (e.g., '2010-2020', default: all): ").strip()
    year_range = None
    if year_str and '-' in year_str:
        try:
            start, end = year_str.split('-')
            year_range = (int(start), int(end))
        except:
            print("Invalid year range, using all years")
    
    return n_recommendations, diversity_filter, min_popularity, year_range

def main():
    """Main interactive loop."""
    print_header()
    
    # Initialize recommender
    print("🔄 Initializing recommender (this may take a moment)...")
    try:
        recommender = MusicRecommender()
        print("✅ Ready!\n")
    except Exception as e:
        print(f"❌ Error initializing recommender: {e}")
        return
    
    while True:
        # Get search query
        query = input("🔍 Enter song name (or artist name): ").strip()
        
        if not query:
            continue
        
        # Check for commands
        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thanks for using the Music Recommendation System!")
            break
        
        if query.lower() == 'help':
            print("\n📖 Help:")
            print("  - Type a song name to search")
            print("  - You can include artist name for better results")
            print("  - Select a track from results to get recommendations")
            print("  - Type 'quit' or 'exit' to stop")
            print()
            continue
        
        # Search for tracks
        print(f"\n🔎 Searching for '{query}'...")
        
        # Try to split query into song and artist
        if ' by ' in query.lower():
            parts = query.split(' by ', 1)
            song_name = parts[0].strip()
            artist_name = parts[1].strip()
        else:
            song_name = query
            artist_name = None
        
        matches = recommender.find_track_by_name(song_name, artist_name)
        
        if not matches:
            print(f"❌ No tracks found for '{query}'")
            print("💡 Tip: Try different spelling or include artist name")
            continue
        
        # Show matches
        print_matches(matches)
        
        # Get user selection
        choice_idx = get_user_choice(len(matches))
        
        if choice_idx is None:
            continue
        
        selected_track = matches[choice_idx]
        
        # Get recommendation settings
        n_recs, diversity, min_pop, year_range = get_recommendation_settings()
        
        # Get recommendations
        print(f"\n🎲 Finding similar tracks...")
        try:
            recommendations = recommender.get_recommendations(
                selected_track['id'],
                n_recommendations=n_recs,
                diversity_filter=diversity,
                min_popularity=min_pop,
                year_range=year_range
            )
            
            if recommendations:
                print_recommendations(recommendations, selected_track)
            else:
                print("❌ No recommendations found with these filters")
                print("💡 Try relaxing the filters")
        
        except Exception as e:
            print(f"❌ Error getting recommendations: {e}")
        
        # Ask if user wants to continue
        print()
        continue_str = input("Search for another song? (Y/n): ").strip().lower()
        if continue_str == 'n':
            print("\n👋 Thanks for using the Music Recommendation System!")
            break
        print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
