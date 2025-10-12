"""
Feature engineering for battery materials using matminer and pymatgen
Converts chemical formulas into numerical descriptors
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition
    MATMINER_AVAILABLE = True
except ImportError:
    MATMINER_AVAILABLE = False
    print("Warning: matminer not available. Using fallback features.")


class MaterialFeaturizer:
    """Extract features from chemical formulas"""
    
    def __init__(self, use_matminer: bool = True):
        """
        Initialize featurizer
        
        Args:
            use_matminer: Whether to use matminer (if available)
        """
        self.use_matminer = use_matminer and MATMINER_AVAILABLE
        self.featurizer = None
        self.feature_names = []
        
        if self.use_matminer:
            try:
                # Use Magpie elemental features
                self.featurizer = ElementProperty.from_preset("magpie")
                print("Using matminer Magpie features")
            except Exception as e:
                print(f"Failed to initialize matminer: {e}")
                self.use_matminer = False
        
        if not self.use_matminer:
            print("Using fallback composition features")
    
    def featurize_formula(self, formula: str) -> Optional[dict]:
        """
        Convert a chemical formula to feature dict
        
        Args:
            formula: Chemical formula (e.g., 'LiCoO2')
            
        Returns:
            Dictionary of features or None if failed
        """
        try:
            comp = Composition(formula)
            
            if self.use_matminer:
                features = self.featurizer.featurize(comp)
                feature_names = self.featurizer.feature_labels()
                return dict(zip(feature_names, features))
            else:
                # Fallback: simple composition features
                return self._simple_composition_features(comp)
                
        except Exception as e:
            return None
    
    def _simple_composition_features(self, comp: Composition) -> dict:
        """Simple fallback features when matminer unavailable"""
        elements = comp.elements
        fractions = [comp.get_atomic_fraction(el) for el in elements]
        
        features = {
            'n_elements': len(elements),
            'mean_atomic_number': np.mean([el.Z for el in elements]),
            'std_atomic_number': np.std([el.Z for el in elements]),
            'mean_atomic_mass': comp.weight / comp.num_atoms,
            'density_estimate': comp.weight,  # Rough proxy
        }
        
        # Element-specific features
        element_names = ['Li', 'Co', 'Ni', 'Mn', 'O', 'C', 'Fe', 'Ti', 'V', 'S']
        for el_name in element_names:
            features[f'contains_{el_name}'] = float(el_name in comp)
            features[f'fraction_{el_name}'] = comp.get_atomic_fraction(el_name) if el_name in comp else 0.0
        
        return features
    
    def featurize_dataframe(self, df: pd.DataFrame, formula_col: str = 'formula') -> pd.DataFrame:
        """
        Add features to a dataframe
        
        Args:
            df: DataFrame with chemical formulas
            formula_col: Name of column containing formulas
            
        Returns:
            DataFrame with added feature columns
        """
        print(f"Featurizing {len(df)} materials...")
        
        features_list = []
        valid_indices = []
        
        for idx, formula in enumerate(df[formula_col]):
            if idx % 500 == 0:
                print(f"  Progress: {idx}/{len(df)}")
            
            features = self.featurize_formula(formula)
            if features is not None:
                features_list.append(features)
                valid_indices.append(idx)
        
        print(f"Successfully featurized {len(features_list)}/{len(df)} materials")
        
        # Create features dataframe
        features_df = pd.DataFrame(features_list)
        
        # Combine with original data
        result_df = df.iloc[valid_indices].reset_index(drop=True)
        result_df = pd.concat([result_df, features_df], axis=1)
        
        # Store feature names
        self.feature_names = features_df.columns.tolist()
        
        return result_df
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature column names"""
        return self.feature_names
    
    def clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean feature columns: handle missing values, remove low-variance
        
        Args:
            df: DataFrame with features
            
        Returns:
            Cleaned DataFrame
        """
        print("\nCleaning features...")
        
        # Get feature columns
        feature_cols = self.feature_names
        
        if not feature_cols:
            print("No features to clean")
            return df
        
        # Calculate missing percentage
        missing_pct = df[feature_cols].isnull().sum() / len(df) * 100
        
        # Drop columns with >40% missing
        high_missing = missing_pct[missing_pct > 40].index.tolist()
        if high_missing:
            print(f"Dropping {len(high_missing)} columns with >40% missing values")
            feature_cols = [col for col in feature_cols if col not in high_missing]
            df = df.drop(columns=high_missing)
        
        # Impute remaining missing with median
        for col in feature_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        # Remove low-variance features
        variances = df[feature_cols].var()
        low_var = variances[variances < 1e-6].index.tolist()
        if low_var:
            print(f"Dropping {len(low_var)} low-variance features")
            feature_cols = [col for col in feature_cols if col not in low_var]
            df = df.drop(columns=low_var)
        
        # Update feature names
        self.feature_names = feature_cols
        
        # Replace inf with large values
        df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
        
        print(f"Final feature count: {len(self.feature_names)}")
        
        return df


def prepare_features(df: pd.DataFrame, target_col: str = 'voltage') -> tuple:
    """
    Prepare features and target for modeling
    
    Args:
        df: DataFrame with materials and properties
        target_col: Name of target column
        
    Returns:
        (X_features, y_target, feature_names, df_processed)
    """
    # Featurize
    featurizer = MaterialFeaturizer()
    df = featurizer.featurize_dataframe(df)
    
    # Clean features
    df = featurizer.clean_features(df)
    
    # Get feature columns and target
    feature_cols = featurizer.get_feature_names()
    X = df[feature_cols].values
    y = df[target_col].values
    
    print(f"\n=== Feature Preparation Complete ===")
    print(f"Samples: {len(df)}")
    print(f"Features: {len(feature_cols)}")
    print(f"Target: {target_col} (mean={y.mean():.3f}, std={y.std():.3f})")
    
    return X, y, feature_cols, df


if __name__ == "__main__":
    # Test feature engineering
    from data_loader import load_battery_data
    
    print("Loading data...")
    df = load_battery_data(
        data_path="data/Battery Data/battery_merged.csv",
        sample_size=1000
    )
    
    print("\nPreparing features...")
    X, y, feature_names, df_processed = prepare_features(df)
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nFirst 10 feature names: {feature_names[:10]}")
