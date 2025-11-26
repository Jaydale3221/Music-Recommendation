"""
Preprocessing pipeline for Spotify dataset.
"""

import pandas as pd
import os
from src.preprocessing.data_cleaner import DataCleaner
from src.preprocessing.feature_engineer import FeatureEngineer

class PreprocessingPipeline:
    """Orchestrate data cleaning and feature engineering."""
    
    def __init__(self, raw_data_path: str = 'data/raw/tracks.csv',
                 output_dir: str = 'data/processed'):
        self.raw_data_path = raw_data_path
        self.output_dir = output_dir
        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def run(self, save_intermediate: bool = False) -> pd.DataFrame:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Processed dataframe
        """
        print("=" * 80)
        print("PREPROCESSING PIPELINE")
        print("=" * 80)
        
        # 1. Load raw data
        print("\n1. Loading raw data...")
        df_raw = pd.read_csv(self.raw_data_path)
        print(f"   Loaded {len(df_raw):,} tracks")
        
        # 2. Clean data
        print("\n2. Cleaning data...")
        df_clean = self.cleaner.clean(df_raw)
        
        if save_intermediate:
            clean_path = os.path.join(self.output_dir, 'tracks_cleaned.csv')
            df_clean.to_csv(clean_path, index=False)
            print(f"   Saved cleaned data to {clean_path}")
        
        # 3. Engineer features
        print("\n3. Engineering features...")
        df_processed = self.engineer.engineer_features(df_clean)
        
        # 4. Save processed data
        print("\n4. Saving processed data...")
        output_path = os.path.join(self.output_dir, 'tracks_processed.csv')
        df_processed.to_csv(output_path, index=False)
        print(f"   Saved to {output_path}")
        
        # 5. Save feature metadata
        metadata_path = os.path.join(self.output_dir, 'feature_metadata.json')
        self.engineer.save_feature_metadata(df_processed, metadata_path)
        
        # 6. Generate summary report
        self._generate_report(df_raw, df_clean, df_processed)
        
        print("\n" + "=" * 80)
        print("PREPROCESSING COMPLETE")
        print("=" * 80)
        
        return df_processed
    
    def _generate_report(self, df_raw: pd.DataFrame, 
                        df_clean: pd.DataFrame, 
                        df_processed: pd.DataFrame):
        """Generate a summary report of the preprocessing."""
        
        report = {
            'raw_data': {
                'total_tracks': len(df_raw),
                'total_columns': len(df_raw.columns)
            },
            'cleaned_data': {
                'total_tracks': len(df_clean),
                'total_columns': len(df_clean.columns),
                'tracks_removed': len(df_raw) - len(df_clean),
                'removal_percentage': ((len(df_raw) - len(df_clean)) / len(df_raw)) * 100
            },
            'processed_data': {
                'total_tracks': len(df_processed),
                'total_columns': len(df_processed.columns),
                'new_features_added': len(df_processed.columns) - len(df_clean.columns)
            },
            'cleaning_stats': self.cleaner.get_cleaning_report(),
            'feature_categories': self.engineer.get_feature_columns(df_processed)
        }
        
        # Save report
        import json
        report_path = os.path.join(self.output_dir, 'preprocessing_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n5. Preprocessing report saved to {report_path}")
        
        # Print summary
        print("\nSummary:")
        print(f"  - Original tracks: {report['raw_data']['total_tracks']:,}")
        print(f"  - Final tracks: {report['processed_data']['total_tracks']:,}")
        print(f"  - Tracks removed: {report['cleaned_data']['tracks_removed']:,} "
              f"({report['cleaned_data']['removal_percentage']:.2f}%)")
        print(f"  - Original features: {report['raw_data']['total_columns']}")
        print(f"  - Final features: {report['processed_data']['total_columns']}")
        print(f"  - New features added: {report['processed_data']['new_features_added']}")


if __name__ == "__main__":
    # Run the pipeline
    pipeline = PreprocessingPipeline()
    df_processed = pipeline.run(save_intermediate=True)
    
    print(f"\nProcessed dataset shape: {df_processed.shape}")
    print(f"Sample of processed data:")
    print(df_processed.head())
