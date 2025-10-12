"""
Novel Materials Discovery Module
Generates and validates new battery material compositions
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from itertools import product
from mp_api.client import MPRester
from pymatgen.core import Composition
import warnings
warnings.filterwarnings('ignore')


class NovelMaterialsDiscovery:
    """Generate and validate novel battery material compositions"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize discovery engine
        
        Args:
            api_key: Materials Project API key
        """
        self.api_key = api_key
        self.mp_client = None
        if api_key:
            try:
                self.mp_client = MPRester(api_key)
                print("✓ Materials Project API connected")
            except Exception as e:
                print(f"⚠ MP API connection failed: {e}")
    
    def generate_candidate_compositions(self,
                                       n_candidates: int = 1000,
                                       seed: int = 42) -> List[str]:
        """
        Generate novel battery cathode compositions
        
        Focus on Li-containing transition metal oxides (common cathode chemistry)
        Pattern: Li_x M_y O_z where M is transition metal
        
        Args:
            n_candidates: Number of compositions to generate
            seed: Random seed
            
        Returns:
            List of chemical formulas
        """
        np.random.seed(seed)
        
        # Transition metals known for good cathodes
        transition_metals = ['Mn', 'Co', 'Ni', 'Fe', 'V', 'Ti', 'Cr', 'Nb', 'Mo']
        
        # Also consider mixed transition metals
        multi_metal_combos = [
            ['Mn', 'Co'], ['Mn', 'Ni'], ['Co', 'Ni'], 
            ['Fe', 'Mn'], ['V', 'Mn'], ['Ti', 'V']
        ]
        
        compositions = []
        
        # Strategy 1: Single transition metal (Li_x M_y O_z)
        for metal in transition_metals:
            # Common stoichiometries for battery cathodes
            stoichiometries = [
                (1, 1, 2),   # LiMO2 (layered)
                (2, 1, 4),   # Li2MO4 (spinel-like)
                (3, 1, 4),   # Li3MO4
                (1, 2, 4),   # LiM2O4 (spinel)
                (4, 1, 5),   # Li4MO5
                (2, 2, 5),   # Li2M2O5
                (3, 2, 6),   # Li3M2O6
            ]
            
            for li, m, o in stoichiometries:
                formula = f"Li{li}{metal}{m}O{o}"
                compositions.append(formula)
        
        # Strategy 2: Mixed transition metals (Li_x M1_y M2_z O_w)
        for metals in multi_metal_combos:
            stoichiometries = [
                (1, metals[0], 0.5, metals[1], 0.5, 2),   # LiM0.5M'0.5O2
                (2, metals[0], 1, metals[1], 1, 4),       # Li2MM'O4
                (1, metals[0], 1, metals[1], 1, 4),       # LiMM'O4
            ]
            
            for li, m1_name, m1_amt, m2_name, m2_amt, o in stoichiometries:
                if m1_amt == int(m1_amt) and m2_amt == int(m2_amt):
                    formula = f"Li{li}{m1_name}{int(m1_amt)}{m2_name}{int(m2_amt)}O{o}"
                else:
                    # For fractional, use decimal notation
                    formula = f"Li{li}{m1_name}{m1_amt}{m2_name}{m2_amt}O{o}"
                compositions.append(formula)
        
        # Strategy 3: Add some dopants/substitutions (small amounts)
        dopants = ['Al', 'Mg', 'Ti', 'Zr']
        for metal in ['Mn', 'Co', 'Ni']:
            for dopant in dopants:
                formulas = [
                    f"Li{metal}0.9{dopant}0.1O2",
                    f"Li{metal}0.95{dopant}0.05O2",
                ]
                compositions.extend(formulas)
        
        # Remove duplicates and return
        compositions = list(set(compositions))
        
        # Limit to requested number
        if len(compositions) > n_candidates:
            np.random.shuffle(compositions)
            compositions = compositions[:n_candidates]
        
        return compositions
    
    def filter_known_compositions(self,
                                 candidates: List[str],
                                 known_formulas: List[str]) -> List[str]:
        """
        Filter out compositions that already exist in our dataset
        
        Args:
            candidates: Generated candidate formulas
            known_formulas: Formulas already in our dataset
            
        Returns:
            List of novel (unseen) formulas
        """
        # Normalize formulas for comparison
        known_set = set(f.strip().replace(' ', '') for f in known_formulas)
        
        novel = []
        for candidate in candidates:
            normalized = candidate.strip().replace(' ', '')
            if normalized not in known_set:
                novel.append(candidate)
        
        return novel
    
    def validate_via_mp(self,
                       formulas: List[str],
                       max_queries: int = 50) -> pd.DataFrame:
        """
        Validate compositions against Materials Project database
        
        Args:
            formulas: List of chemical formulas to validate
            max_queries: Maximum number of MP API queries (rate limiting)
            
        Returns:
            DataFrame with validation results
        """
        if not self.mp_client:
            print("⚠ Materials Project API not available, skipping validation")
            return pd.DataFrame({
                'formula': formulas[:max_queries],
                'mp_exists': [False] * min(len(formulas), max_queries),
                'mp_stable': [None] * min(len(formulas), max_queries),
                'mp_formation_energy': [None] * min(len(formulas), max_queries),
            })
        
        results = []
        
        print(f"🔍 Validating {min(len(formulas), max_queries)} compositions via Materials Project...")
        
        for i, formula in enumerate(formulas[:max_queries]):
            if i % 10 == 0:
                print(f"  Progress: {i}/{min(len(formulas), max_queries)}")
            
            try:
                # Parse composition
                comp = Composition(formula)
                reduced_formula = comp.reduced_formula
                
                # Query Materials Project
                docs = self.mp_client.materials.summary.search(
                    formula=reduced_formula,
                    fields=["material_id", "formula_pretty", "formation_energy_per_atom", 
                           "energy_above_hull", "is_stable"]
                )
                
                if docs:
                    # Material exists in MP database
                    doc = docs[0]  # Take first match
                    results.append({
                        'formula': formula,
                        'mp_exists': True,
                        'mp_id': doc.material_id,
                        'mp_formula': doc.formula_pretty,
                        'mp_formation_energy': doc.formation_energy_per_atom,
                        'mp_energy_above_hull': doc.energy_above_hull,
                        'mp_stable': doc.is_stable,
                    })
                else:
                    # Novel composition not in MP
                    results.append({
                        'formula': formula,
                        'mp_exists': False,
                        'mp_id': None,
                        'mp_formula': None,
                        'mp_formation_energy': None,
                        'mp_energy_above_hull': None,
                        'mp_stable': None,
                    })
                    
            except Exception as e:
                # Invalid composition or API error
                results.append({
                    'formula': formula,
                    'mp_exists': False,
                    'mp_id': None,
                    'mp_formula': None,
                    'mp_formation_energy': None,
                    'mp_energy_above_hull': None,
                    'mp_stable': None,
                    'error': str(e)
                })
        
        return pd.DataFrame(results)
    
    def rank_candidates(self,
                       candidates_df: pd.DataFrame,
                       predicted_voltage: np.ndarray,
                       predicted_uncertainty: np.ndarray) -> pd.DataFrame:
        """
        Rank candidate materials by predicted performance
        
        Args:
            candidates_df: DataFrame with candidate formulas
            predicted_voltage: Model predictions for voltage
            predicted_uncertainty: Model uncertainty estimates
            
        Returns:
            DataFrame with ranked candidates
        """
        candidates_df = candidates_df.copy()
        candidates_df['predicted_voltage'] = predicted_voltage
        candidates_df['uncertainty'] = predicted_uncertainty
        
        # Calculate discovery score: high voltage + low uncertainty + novel
        candidates_df['novelty_score'] = (~candidates_df.get('mp_exists', False)).astype(float)
        candidates_df['discovery_score'] = (
            candidates_df['predicted_voltage'] * 0.6 +
            (1 - candidates_df['uncertainty']) * 0.2 +
            candidates_df['novelty_score'] * 0.2
        )
        
        # Sort by discovery score
        candidates_df = candidates_df.sort_values('discovery_score', ascending=False)
        
        return candidates_df


def discover_novel_materials(
    trained_model,
    known_formulas: List[str],
    mp_api_key: str = None,
    n_generate: int = 500,
    n_validate: int = 50,
    n_top: int = 20
) -> Tuple[pd.DataFrame, Dict]:
    """
    End-to-end novel materials discovery pipeline
    
    Args:
        trained_model: Trained surrogate model
        known_formulas: List of formulas in training data
        mp_api_key: Materials Project API key
        n_generate: Number of compositions to generate
        n_validate: Number to validate via MP API
        n_top: Number of top discoveries to return
        
    Returns:
        (discoveries_df, stats_dict)
    """
    print("\n" + "="*60)
    print("🔬 NOVEL MATERIALS DISCOVERY PIPELINE")
    print("="*60 + "\n")
    
    # Initialize discovery engine
    discovery = NovelMaterialsDiscovery(api_key=mp_api_key)
    
    # Step 1: Generate candidates
    print(f"📝 Generating {n_generate} candidate compositions...")
    candidates = discovery.generate_candidate_compositions(n_candidates=n_generate)
    print(f"   Generated {len(candidates)} unique compositions")
    
    # Step 2: Filter known materials
    print(f"\n🔍 Filtering out {len(known_formulas)} known materials...")
    novel_candidates = discovery.filter_known_compositions(candidates, known_formulas)
    print(f"   Found {len(novel_candidates)} novel compositions")
    
    # Step 3: Predict properties using trained model
    print(f"\n🤖 Predicting properties for novel compositions...")
    # This will be implemented in the main integration
    # For now, return structure for integration
    
    stats = {
        'n_generated': len(candidates),
        'n_novel': len(novel_candidates),
        'n_validated': 0,
        'n_high_voltage': 0,
    }
    
    discoveries_df = pd.DataFrame({
        'formula': novel_candidates[:n_top]
    })
    
    return discoveries_df, stats


if __name__ == "__main__":
    # Test composition generation
    discovery = NovelMaterialsDiscovery()
    
    candidates = discovery.generate_candidate_compositions(n_candidates=100)
    print(f"Generated {len(candidates)} candidates")
    print("\nSample compositions:")
    for i, comp in enumerate(candidates[:10]):
        print(f"  {i+1}. {comp}")
