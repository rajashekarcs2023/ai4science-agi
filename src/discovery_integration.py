"""
Integration of novel materials discovery with trained models
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

from src.discovery import NovelMaterialsDiscovery
from src.feature_engineering import featurize_compositions
from pymatgen.core import Composition


def predict_novel_materials(
    model,
    feature_cols: List[str],
    novel_formulas: List[str],
    feature_method: str = 'magpie'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict properties for novel material compositions
    
    Args:
        model: Trained surrogate model
        feature_cols: List of feature column names from training
        novel_formulas: List of novel chemical formulas
        feature_method: Featurization method ('magpie' or 'composition')
        
    Returns:
        (predicted_values, uncertainties)
    """
    print(f"🔮 Featurizing {len(novel_formulas)} novel compositions...")
    
    # Create temporary dataframe
    temp_df = pd.DataFrame({'formula': novel_formulas})
    
    # Featurize using same method as training
    try:
        X_novel, _, novel_feature_cols, novel_df = featurize_compositions(
            temp_df,
            formula_col='formula',
            target_col=None,  # No target for novel materials
            method=feature_method
        )
        
        print(f"   Featurized {len(novel_df)} compositions successfully")
        
        # Align features with training features
        # Add missing features as zeros, remove extra features
        for col in feature_cols:
            if col not in novel_feature_cols:
                X_novel[col] = 0
        
        # Reorder to match training
        X_novel = X_novel[feature_cols]
        
        # Predict
        print(f"🔮 Predicting properties...")
        predictions, uncertainties = model.predict(X_novel.values, return_std=True)
        
        print(f"   ✓ Generated {len(predictions)} predictions")
        
        return predictions, uncertainties, novel_df['formula'].values
        
    except Exception as e:
        print(f"   ⚠ Featurization failed: {e}")
        # Return zeros for failed featurization
        return np.zeros(len(novel_formulas)), np.ones(len(novel_formulas)), novel_formulas


def run_discovery_pipeline(
    model,
    feature_cols: List[str],
    known_formulas: List[str],
    mp_api_key: str = None,
    n_generate: int = 500,
    n_validate: int = 30,
    n_top: int = 20,
    feature_method: str = 'magpie',
    min_voltage: float = 3.5
) -> Tuple[pd.DataFrame, Dict]:
    """
    Complete novel materials discovery pipeline
    
    Args:
        model: Trained surrogate model  
        feature_cols: Feature columns from training
        known_formulas: Formulas in training dataset
        mp_api_key: Materials Project API key
        n_generate: Number of compositions to generate
        n_validate: Number to validate via MP
        n_top: Number of top discoveries to return
        feature_method: Featurization method
        min_voltage: Minimum predicted voltage threshold
        
    Returns:
        (discoveries_df, stats_dict)
    """
    print("\n" + "="*70)
    print("🔬 AUTONOMOUS NOVEL MATERIALS DISCOVERY")
    print("="*70 + "\n")
    
    # Initialize discovery engine
    discovery = NovelMaterialsDiscovery(api_key=mp_api_key)
    
    # Step 1: Generate candidates
    print(f"📝 Step 1: Generating {n_generate} candidate compositions...")
    candidates = discovery.generate_candidate_compositions(n_candidates=n_generate)
    print(f"   ✓ Generated {len(candidates)} unique compositions\n")
    
    # Step 2: Filter known materials
    print(f"🔍 Step 2: Filtering out {len(known_formulas)} known materials...")
    novel_candidates = discovery.filter_known_compositions(candidates, known_formulas)
    print(f"   ✓ Found {len(novel_candidates)} novel compositions\n")
    
    if len(novel_candidates) == 0:
        print("⚠ No novel compositions found!")
        return pd.DataFrame(), {}
    
    # Step 3: Predict properties
    print(f"🤖 Step 3: Predicting properties using trained model...")
    predictions, uncertainties, valid_formulas = predict_novel_materials(
        model, feature_cols, novel_candidates, feature_method
    )
    print()
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'formula': valid_formulas,
        'predicted_voltage': predictions,
        'uncertainty': uncertainties
    })
    
    # Filter by minimum voltage
    high_voltage = results_df[results_df['predicted_voltage'] >= min_voltage].copy()
    print(f"⚡ Step 4: Found {len(high_voltage)} candidates with voltage ≥ {min_voltage}V\n")
    
    if len(high_voltage) == 0:
        print("⚠ No high-voltage candidates found!")
        return results_df.nlargest(n_top, 'predicted_voltage'), {
            'n_generated': len(candidates),
            'n_novel': len(novel_candidates),
            'n_high_voltage': 0,
            'n_validated': 0
        }
    
    # Step 5: Validate top candidates via Materials Project
    top_candidates = high_voltage.nlargest(min(n_validate, len(high_voltage)), 'predicted_voltage')
    
    print(f"🔬 Step 5: Validating top {len(top_candidates)} via Materials Project API...")
    validation_results = discovery.validate_via_mp(
        top_candidates['formula'].tolist(),
        max_queries=n_validate
    )
    print()
    
    # Merge validation results
    if len(validation_results) > 0:
        discoveries = top_candidates.merge(
            validation_results,
            on='formula',
            how='left'
        )
    else:
        discoveries = top_candidates
    
    # Calculate novelty scores
    if 'mp_exists' in discoveries.columns:
        discoveries['is_novel'] = ~discoveries['mp_exists'].fillna(False)
        discoveries['novelty_score'] = discoveries['is_novel'].astype(float)
    else:
        discoveries['is_novel'] = True
        discoveries['novelty_score'] = 1.0
    
    # Calculate discovery score
    discoveries['discovery_score'] = (
        discoveries['predicted_voltage'].rank(pct=True) * 0.5 +
        (1 - discoveries['uncertainty'].rank(pct=True)) * 0.3 +
        discoveries['novelty_score'] * 0.2
    )
    
    # Sort by discovery score
    discoveries = discoveries.sort_values('discovery_score', ascending=False)
    
    # Statistics
    stats = {
        'n_generated': len(candidates),
        'n_novel': len(novel_candidates),
        'n_high_voltage': len(high_voltage),
        'n_validated': len(validation_results) if len(validation_results) > 0 else 0,
        'n_truly_novel': int(discoveries.get('is_novel', pd.Series([False])).sum()),
        'max_predicted_voltage': float(discoveries['predicted_voltage'].max()),
        'mean_predicted_voltage': float(discoveries['predicted_voltage'].mean()),
    }
    
    # Print summary
    print("="*70)
    print("✅ DISCOVERY COMPLETE!")
    print("="*70)
    print(f"📊 Generated: {stats['n_generated']} compositions")
    print(f"🆕 Novel: {stats['n_novel']} unseen materials")
    print(f"⚡ High-voltage (≥{min_voltage}V): {stats['n_high_voltage']} candidates")
    print(f"🔬 Validated via MP: {stats['n_validated']} materials")
    if stats['n_validated'] > 0:
        print(f"🎯 Truly novel (not in MP): {stats['n_truly_novel']} discoveries")
    print(f"🏆 Best prediction: {stats['max_predicted_voltage']:.3f}V")
    print("="*70 + "\n")
    
    return discoveries.head(n_top), stats


if __name__ == "__main__":
    # Test discovery pipeline structure
    from src.discovery import NovelMaterialsDiscovery
    
    discovery = NovelMaterialsDiscovery()
    candidates = discovery.generate_candidate_compositions(n_candidates=50)
    
    print(f"Generated {len(candidates)} test candidates:")
    for i, comp in enumerate(candidates[:5]):
        print(f"  {i+1}. {comp}")
