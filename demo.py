#!/usr/bin/env python3
"""
Demo script for the music recommendation system.
Run this directly: python demo.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.recommender import MusicRecommender

def main():
    print("=" * 80)
    print("MUSIC RECOMMENDATION SYSTEM DEMO")
    print("=" * 80)
    
    # Initialize recommender
    print("\nInitializing recommender...")
    recommender = MusicRecommender()
    
    # Example 1: Search for a track
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Search for 'Shape of You' by Ed Sheeran")
    print("=" * 80)
    
    matches = recommender.find_track_by_name("Shape of You", "Ed Sheeran")
    
    if matches:
        print(f"\nFound {len(matches)} matches:")
        for i, match in enumerate(matches[:3], 1):
            print(f"{i}. {match['name']} by {match['artists']} ({match['release_year']})")
        
        # Get recommendations for first match
        track_id = matches[0]['id']
        print(f"\n{'='*80}")
        print(f"Getting recommendations for: {matches[0]['name']}")
        print("=" * 80)
        
        recommendations = recommender.get_recommendations(
            track_id, 
            n_recommendations=10,
            diversity_filter=True
        )
        
        print(f"\nTop 10 Similar Songs:")
        print("-" * 80)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i:2d}. {rec['name'][:50]:<50} by {rec['artists'][:30]:<30}")
            print(f"    Similarity: {rec['similarity_score']:.4f} | "
                  f"Popularity: {rec['popularity']:3d} | "
                  f"Year: {rec['release_year']}")
    
    # Example 2: Feature-based search
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Find high-energy, danceable tracks")
    print("=" * 80)
    
    recommendations = recommender.get_recommendations_by_features(
        danceability=0.9,
        energy=0.9,
        valence=0.8,
        tempo=128,
        n_recommendations=10
    )
    
    print(f"\nTop 10 High-Energy Dance Tracks:")
    print("-" * 80)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i:2d}. {rec['name'][:50]:<50} by {rec['artists'][:30]:<30}")
        print(f"    Dance: {rec['danceability']:.2f} | "
              f"Energy: {rec['energy']:.2f} | "
              f"Valence: {rec['valence']:.2f} | "
              f"Tempo: {rec['tempo']:.0f}")
    
    # Example 3: Search with filters
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Search for 'Blinding Lights' with filters")
    print("=" * 80)
    
    matches = recommender.find_track_by_name("Blinding Lights", "The Weeknd")
    
    if matches:
        track_id = matches[0]['id']
        print(f"\nFinding similar tracks to: {matches[0]['name']}")
        print("Filters: min_popularity=40, year_range=(2015, 2024)")
        
        recommendations = recommender.get_recommendations(
            track_id,
            n_recommendations=10,
            diversity_filter=True,
            min_popularity=40,
            year_range=(2015, 2024)
        )
        
        print(f"\nTop 10 Similar Modern Popular Tracks:")
        print("-" * 80)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i:2d}. {rec['name'][:50]:<50}")
            print(f"    Artist: {rec['artists'][:40]:<40} | "
                  f"Pop: {rec['popularity']:3d} | "
                  f"Year: {rec['release_year']}")
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
