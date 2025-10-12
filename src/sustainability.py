"""
Sustainability scoring for battery materials
Based on element toxicity, supply risk, and cost
"""

import pandas as pd
import numpy as np
from typing import Dict

try:
    from pymatgen.core import Composition, Element
    PYMATGEN_AVAILABLE = True
except ImportError:
    PYMATGEN_AVAILABLE = False


# Element sustainability data (simplified proxies)
# Scale: 0 (best) to 1 (worst)
ELEMENT_SUSTAINABILITY = {
    # Low sustainability (rare, toxic, or expensive)
    'Co': {'supply_risk': 0.9, 'toxicity': 0.7, 'cost': 0.8},
    'Li': {'supply_risk': 0.6, 'toxicity': 0.3, 'cost': 0.5},
    'Ni': {'supply_risk': 0.5, 'toxicity': 0.6, 'cost': 0.4},
    'Mn': {'supply_risk': 0.3, 'toxicity': 0.4, 'cost': 0.2},
    'V': {'supply_risk': 0.5, 'toxicity': 0.5, 'cost': 0.4},
    'Cr': {'supply_risk': 0.4, 'toxicity': 0.8, 'cost': 0.3},
    'Pb': {'supply_risk': 0.3, 'toxicity': 1.0, 'cost': 0.2},
    'Cd': {'supply_risk': 0.6, 'toxicity': 0.9, 'cost': 0.5},
    
    # Medium sustainability
    'Fe': {'supply_risk': 0.1, 'toxicity': 0.2, 'cost': 0.1},
    'Ti': {'supply_risk': 0.2, 'toxicity': 0.2, 'cost': 0.3},
    'Al': {'supply_risk': 0.1, 'toxicity': 0.2, 'cost': 0.1},
    'Cu': {'supply_risk': 0.3, 'toxicity': 0.4, 'cost': 0.3},
    'Zn': {'supply_risk': 0.3, 'toxicity': 0.3, 'cost': 0.2},
    'Sn': {'supply_risk': 0.4, 'toxicity': 0.3, 'cost': 0.3},
    'S': {'supply_risk': 0.1, 'toxicity': 0.2, 'cost': 0.1},
    
    # High sustainability (abundant, safe, cheap)
    'C': {'supply_risk': 0.0, 'toxicity': 0.0, 'cost': 0.0},
    'O': {'supply_risk': 0.0, 'toxicity': 0.0, 'cost': 0.0},
    'H': {'supply_risk': 0.0, 'toxicity': 0.0, 'cost': 0.0},
    'N': {'supply_risk': 0.0, 'toxicity': 0.0, 'cost': 0.0},
    'Na': {'supply_risk': 0.1, 'toxicity': 0.2, 'cost': 0.1},
    'K': {'supply_risk': 0.1, 'toxicity': 0.2, 'cost': 0.1},
    'Mg': {'supply_risk': 0.1, 'toxicity': 0.1, 'cost': 0.1},
    'Ca': {'supply_risk': 0.1, 'toxicity': 0.1, 'cost': 0.1},
    'Si': {'supply_risk': 0.0, 'toxicity': 0.1, 'cost': 0.0},
    'P': {'supply_risk': 0.2, 'toxicity': 0.2, 'cost': 0.2},
}

# Default values for unknown elements
DEFAULT_SUSTAINABILITY = {'supply_risk': 0.5, 'toxicity': 0.5, 'cost': 0.5}


class SustainabilityScorer:
    """Calculate sustainability scores for materials"""
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize scorer
        
        Args:
            weights: Dict with keys 'supply_risk', 'toxicity', 'cost'
                    Default: equal weighting
        """
        if weights is None:
            weights = {'supply_risk': 0.33, 'toxicity': 0.34, 'cost': 0.33}
        
        self.weights = weights
    
    def score_composition(self, formula: str) -> Dict[str, float]:
        """
        Calculate sustainability score for a chemical formula
        
        Args:
            formula: Chemical formula (e.g., 'LiCoO2')
            
        Returns:
            Dict with scores: supply_risk, toxicity, cost, overall
            Range: 0 (best) to 1 (worst)
        """
        try:
            comp = Composition(formula)
            
            # Weight by atomic fraction
            total_supply_risk = 0.0
            total_toxicity = 0.0
            total_cost = 0.0
            
            for element, fraction in comp.get_el_amt_dict().items():
                el_str = str(element)
                atomic_fraction = fraction / comp.num_atoms
                
                # Get element sustainability data
                el_data = ELEMENT_SUSTAINABILITY.get(el_str, DEFAULT_SUSTAINABILITY)
                
                total_supply_risk += el_data['supply_risk'] * atomic_fraction
                total_toxicity += el_data['toxicity'] * atomic_fraction
                total_cost += el_data['cost'] * atomic_fraction
            
            # Calculate weighted overall score
            overall = (
                self.weights['supply_risk'] * total_supply_risk +
                self.weights['toxicity'] * total_toxicity +
                self.weights['cost'] * total_cost
            )
            
            return {
                'supply_risk': total_supply_risk,
                'toxicity': total_toxicity,
                'cost': total_cost,
                'sustainability_score': overall  # Lower is better
            }
            
        except Exception as e:
            # Return neutral score if parsing fails
            return {
                'supply_risk': 0.5,
                'toxicity': 0.5,
                'cost': 0.5,
                'sustainability_score': 0.5
            }
    
    def add_sustainability_scores(self, df: pd.DataFrame, formula_col: str = 'formula') -> pd.DataFrame:
        """
        Add sustainability scores to dataframe
        
        Args:
            df: DataFrame with chemical formulas
            formula_col: Name of column containing formulas
            
        Returns:
            DataFrame with added sustainability columns
        """
        print(f"Calculating sustainability scores for {len(df)} materials...")
        
        scores_list = []
        for formula in df[formula_col]:
            scores = self.score_composition(formula)
            scores_list.append(scores)
        
        # Add as new columns
        scores_df = pd.DataFrame(scores_list)
        result_df = pd.concat([df, scores_df], axis=1)
        
        print(f"Mean sustainability score: {result_df['sustainability_score'].mean():.3f} (0=best, 1=worst)")
        
        return result_df
    
    def get_pareto_front(self, df: pd.DataFrame, 
                         objective1: str = 'voltage',
                         objective2: str = 'sustainability_score',
                         maximize1: bool = True,
                         maximize2: bool = False) -> pd.DataFrame:
        """
        Find Pareto front for two objectives
        
        Args:
            df: DataFrame with objectives
            objective1: First objective column (e.g., voltage)
            objective2: Second objective column (e.g., sustainability)
            maximize1: Whether to maximize objective1
            maximize2: Whether to maximize objective2
            
        Returns:
            DataFrame with Pareto-optimal solutions
        """
        df = df.copy()
        
        # Convert to maximization problem
        if not maximize1:
            df['_obj1'] = -df[objective1]
        else:
            df['_obj1'] = df[objective1]
        
        if not maximize2:
            df['_obj2'] = -df[objective2]
        else:
            df['_obj2'] = df[objective2]
        
        # Find Pareto front
        is_pareto = np.ones(len(df), dtype=bool)
        
        for i in range(len(df)):
            if is_pareto[i]:
                # Check if any other point dominates this one
                is_pareto[i] = not np.any(
                    (df['_obj1'].values >= df['_obj1'].values[i]) &
                    (df['_obj2'].values >= df['_obj2'].values[i]) &
                    ((df['_obj1'].values > df['_obj1'].values[i]) |
                     (df['_obj2'].values > df['_obj2'].values[i]))
                )
        
        pareto_df = df[is_pareto].copy()
        pareto_df = pareto_df.drop(columns=['_obj1', '_obj2'])
        
        print(f"Pareto front: {len(pareto_df)} materials")
        
        return pareto_df


def add_sustainability(df: pd.DataFrame, formula_col: str = 'formula') -> pd.DataFrame:
    """
    Convenience function to add sustainability scores
    
    Args:
        df: DataFrame with formulas
        formula_col: Column containing chemical formulas
        
    Returns:
        DataFrame with sustainability scores
    """
    scorer = SustainabilityScorer()
    return scorer.add_sustainability_scores(df, formula_col)


if __name__ == "__main__":
    # Test sustainability scoring
    test_materials = [
        'LiCoO2',      # High Co content - low sustainability
        'LiFePO4',     # Iron-based - high sustainability
        'NaFeO2',      # Sodium + iron - high sustainability
        'LiNi0.8Co0.15Al0.05O2',  # Mixed
    ]
    
    scorer = SustainabilityScorer()
    
    print("=== Sustainability Scores ===")
    print("(0 = best, 1 = worst)\n")
    
    for formula in test_materials:
        scores = scorer.score_composition(formula)
        print(f"{formula:20s} | Overall: {scores['sustainability_score']:.3f} | "
              f"Supply: {scores['supply_risk']:.3f} | "
              f"Toxicity: {scores['toxicity']:.3f} | "
              f"Cost: {scores['cost']:.3f}")
