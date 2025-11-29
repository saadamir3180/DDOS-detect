"""
Data loading utilities for CICDDoS2019 dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and basic validation of the CICDDoS2019 dataset."""
    
    def __init__(self, data_path: str):
        """
        Initialize DataLoader.
        
        Args:
            data_path: Path to the CSV file
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    def load_data(self, sample_size: Optional[int] = None, random_state: int = 42) -> pd.DataFrame:
        """
        Load the dataset from CSV.
        
        Args:
            sample_size: If specified, randomly sample this many rows
            random_state: Random seed for reproducibility
            
        Returns:
            DataFrame containing the dataset
        """
        logger.info(f"Loading data from {self.data_path}")
        
        try:
            # Load the full dataset
            df = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            
            # Sample if requested
            if sample_size and sample_size < len(df):
                df = df.sample(n=sample_size, random_state=random_state)
                logger.info(f"Sampled {sample_size} rows")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def load_in_chunks(self, chunk_size: int = 10000):
        """
        Load data in chunks for memory efficiency.
        
        Args:
            chunk_size: Number of rows per chunk
            
        Yields:
            DataFrame chunks
        """
        logger.info(f"Loading data in chunks of {chunk_size}")
        
        for chunk in pd.read_csv(self.data_path, chunksize=chunk_size):
            yield chunk
    
    def get_data_info(self) -> dict:
        """
        Get basic information about the dataset without loading it all.
        
        Returns:
            Dictionary with dataset metadata
        """
        # Read just the first few rows to get column info
        df_sample = pd.read_csv(self.data_path, nrows=5)
        
        return {
            'columns': df_sample.columns.tolist(),
            'num_columns': len(df_sample.columns),
            'dtypes': df_sample.dtypes.to_dict(),
            'file_size_mb': self.data_path.stat().st_size / (1024 * 1024)
        }


def load_dataset(data_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Convenience function to load the dataset.
    
    Args:
        data_path: Path to the CSV file
        sample_size: Optional number of rows to sample
        
    Returns:
        DataFrame containing the dataset
    """
    loader = DataLoader(data_path)
    return loader.load_data(sample_size=sample_size)


if __name__ == "__main__":
    # Example usage
    data_path = "data/raw/cicddos2019_dataset.csv"
    
    # Get info without loading full dataset
    loader = DataLoader(data_path)
    info = loader.get_data_info()
    print(f"Dataset info: {info}")
    
    # Load a sample
    df = load_dataset(data_path, sample_size=1000)
    print(f"Loaded sample: {df.shape}")
    print(df.head())
