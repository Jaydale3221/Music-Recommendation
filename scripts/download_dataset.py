#!/usr/bin/env python3
"""
Download Spotify dataset from Kaggle.

Prerequisites:
1. Install kaggle CLI: pip install kaggle
2. Set up Kaggle API credentials:
   - Go to https://www.kaggle.com/settings
   - Click "Create New API Token"
   - Place the downloaded kaggle.json in ~/.kaggle/
   - chmod 600 ~/.kaggle/kaggle.json
"""

import os
import subprocess
import sys

def download_dataset():
    """Download the Spotify dataset from Kaggle."""
    
    # Check if kaggle is installed
    try:
        subprocess.run(['kaggle', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Kaggle CLI not found. Installing...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'kaggle'], check=True)
    
    # Create data/raw directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    
    # Download the dataset
    dataset_name = 'yamaerenay/spotify-dataset-19212020-600k-tracks'
    print(f"Downloading dataset: {dataset_name}")
    
    try:
        subprocess.run([
            'kaggle', 'datasets', 'download',
            '-d', dataset_name,
            '-p', 'data/raw',
            '--unzip'
        ], check=True)
        
        print("Dataset downloaded successfully to data/raw/")
        
        # List the downloaded files
        print("\nDownloaded files:")
        for file in os.listdir('data/raw'):
            print(f"  - {file}")
            
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e}")
        print("\nPlease ensure you have:")
        print("1. A Kaggle account")
        print("2. Kaggle API token set up at ~/.kaggle/kaggle.json")
        print("3. Accepted the dataset's terms on Kaggle website")
        sys.exit(1)

if __name__ == "__main__":
    download_dataset()
