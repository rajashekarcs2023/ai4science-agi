"""
Data loader for Battery Materials Property Database
Loads, cleans, and preprocesses experimental battery data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re


class BatteryDataLoader:
    """Load and preprocess battery materials data"""
    
    def __init__(self, data_path: str, sample_size: int = 10000, random_state: int = 42):
        """
        Initialize data loader
        
        Args:
            data_path: Path to battery_merged.csv
            sample_size: Number of samples to load (for speed)
            random_state: Random seed for reproducibility
        """
        self.data_path = Path(data_path)
        self.sample_size = sample_size
        self.random_state = random_state
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load and perform initial cleaning of battery data"""
        print(f"Loading data from {self.data_path}...")
        
        # Load full dataset
        df = pd.read_csv(self.data_path, low_memory=False)
        print(f"Loaded {len(df)} records")
        
        # Focus on key properties with voltage as primary target
        df = self._clean_columns(df)
        
        # Extract chemical formulas
        df = self._extract_formulas(df)
        
        # Filter for voltage data (our primary target)
        df = df[df['Property'].str.lower() == 'voltage'].copy()
        print(f"Filtered to {len(df)} voltage records")
        
        # Convert values to numeric
        df = self._convert_to_numeric(df)
        
        # Remove invalid entries
        df = df[df['voltage'] > 0].copy()
        df = df[df['voltage'] < 10].copy()  # Reasonable voltage range
        print(f"After filtering: {len(df)} valid voltage records")
        
        # Sample for manageable size
        if len(df) > self.sample_size:
            df = df.sample(n=self.sample_size, random_state=self.random_state)
            print(f"Sampled to {len(df)} records")
        
        # Reset index
        df = df.reset_index(drop=True)
        
        self.df = df
        return df
    
    def _clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize column names"""
        # Rename for consistency
        if 'Name' in df.columns:
            df = df.rename(columns={'Name': 'material_name'})
        if 'Value' in df.columns:
            df = df.rename(columns={'Value': 'value'})
        
        return df
    
    def _extract_formulas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract clean chemical formulas from material names"""
        def clean_formula(name):
            if pd.isna(name):
                return None
            
            # Remove common descriptors
            name = str(name)
            name = re.sub(r'\s*(nano|micro|meso|porous|@|/)\s*.*$', '', name, flags=re.IGNORECASE)
            name = name.strip()
            
            # Extract first chemical formula (simple heuristic)
            # Look for pattern: Capital letter followed by optional lowercase and numbers
            match = re.search(r'([A-Z][a-z]?\d*)+', name)
            if match:
                return match.group(0)
            return name
        
        df['formula'] = df['material_name'].apply(clean_formula)
        
        # Remove entries without valid formulas
        df = df[df['formula'].notna()].copy()
        df = df[df['formula'].str.len() > 0].copy()
        
        return df
    
    def _convert_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert property values to numeric"""
        # Create property-specific columns
        df['voltage'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Drop rows with invalid numeric values
        df = df[df['voltage'].notna()].copy()
        
        return df
    
    def get_unique_materials(self) -> pd.DataFrame:
        """Get unique materials with averaged properties"""
        if self.df is None:
            self.load_data()
        
        # Group by formula and average values
        unique_df = self.df.groupby('formula').agg({
            'voltage': 'mean',
            'material_name': 'first',
            'DOI': 'first',
            'Title': 'first'
        }).reset_index()
        
        print(f"Unique materials: {len(unique_df)}")
        return unique_df
    
    def get_processed_data(self) -> pd.DataFrame:
        """Get fully processed data ready for feature engineering"""
        if self.df is None:
            self.load_data()
        
        # Get unique materials
        df = self.get_unique_materials()
        
        # Add metadata
        df['source'] = 'experimental'
        
        return df


def load_battery_data(data_path: str = None, sample_size: int = 10000) -> pd.DataFrame:
    """
    Convenience function to load battery data
    
    Args:
        data_path: Path to battery_merged.csv (default: auto-detect)
        sample_size: Number of samples to load
        
    Returns:
        Processed DataFrame with unique materials
    """
    if data_path is None:
        # Auto-detect data path
        possible_paths = [
            'data/Battery Data/battery_merged.csv',
            '../data/Battery Data/battery_merged.csv',
            'data/battery_merged.csv'
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                data_path = path
                break
        
        if data_path is None:
            raise FileNotFoundError(
                "Could not find battery_merged.csv. Please provide data_path."
            )
    
    loader = BatteryDataLoader(data_path, sample_size=sample_size)
    return loader.get_processed_data()


if __name__ == "__main__":
    # Test data loading
    df = load_battery_data(
        data_path="data/Battery Data/battery_merged.csv",
        sample_size=5000
    )
    
    print("\n=== Data Summary ===")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nVoltage stats:\n{df['voltage'].describe()}")
    print(f"\nSample materials:\n{df[['formula', 'voltage', 'material_name']].head(10)}")
