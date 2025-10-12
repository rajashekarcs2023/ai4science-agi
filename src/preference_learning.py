"""
Human preference learning for materials discovery
Implements "Tinder for Materials" - swipe interface for preference capture
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from dataclasses import dataclass, field


@dataclass
class Preference:
    """Single pairwise preference"""
    winner_idx: int
    loser_idx: int
    winner_features: np.ndarray
    loser_features: np.ndarray


@dataclass
class PreferenceHistory:
    """Collection of preferences"""
    preferences: List[Preference] = field(default_factory=list)
    
    def add(self, winner_idx: int, loser_idx: int, 
            winner_features: np.ndarray, loser_features: np.ndarray):
        """Add a preference"""
        pref = Preference(winner_idx, loser_idx, winner_features, loser_features)
        self.preferences.append(pref)
    
    def size(self) -> int:
        return len(self.preferences)


class PreferenceModel:
    """
    Learn human preferences from pairwise comparisons
    Uses Bradley-Terry model approximated with logistic regression
    """
    
    def __init__(self, feature_dim: int):
        """
        Initialize preference model
        
        Args:
            feature_dim: Dimension of feature vectors
        """
        self.feature_dim = feature_dim
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.is_fitted = False
        self.weights = None
    
    def fit(self, preferences: PreferenceHistory):
        """
        Fit preference model from pairwise comparisons
        
        Args:
            preferences: Collection of preferences
        """
        if preferences.size() < 3:
            print("Warning: Need at least 3 preferences for reliable training")
            return
        
        # Create training data: X = winner - loser, y = 1
        X_train = []
        y_train = []
        
        for pref in preferences.preferences:
            # Preference direction: winner > loser
            feature_diff = pref.winner_features - pref.loser_features
            X_train.append(feature_diff)
            y_train.append(1)  # Winner preferred
            
            # Add reversed pair for balance
            X_train.append(-feature_diff)
            y_train.append(0)  # Loser not preferred
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Train logistic regression
        self.model.fit(X_train, y_train)
        self.weights = self.model.coef_[0]
        self.is_fitted = True
        
        print(f"Trained preference model on {preferences.size()} preferences")
    
    def predict_preference_score(self, features: np.ndarray) -> np.ndarray:
        """
        Predict preference score for materials
        
        Args:
            features: Feature matrix (n_samples, n_features)
            
        Returns:
            Preference scores (higher = more preferred)
        """
        if not self.is_fitted:
            # Return neutral scores if not fitted
            return np.zeros(len(features))
        
        # Score = w^T * x
        scores = features @ self.weights
        return scores
    
    def get_feature_importance(self, feature_names: List[str], top_k: int = 10) -> pd.DataFrame:
        """
        Get most important features for preference
        
        Args:
            feature_names: Names of features
            top_k: Number of top features to return
            
        Returns:
            DataFrame with feature importances
        """
        if not self.is_fitted:
            return None
        
        importance = pd.DataFrame({
            'feature': feature_names,
            'weight': self.weights,
            'abs_weight': np.abs(self.weights)
        })
        
        importance = importance.nlargest(top_k, 'abs_weight')
        return importance


class MaterialComparator:
    """
    Interactive material comparison system
    Generates pairs for human evaluation
    """
    
    def __init__(self, df: pd.DataFrame, X: np.ndarray, 
                 predicted_values: np.ndarray,
                 uncertainties: np.ndarray):
        """
        Initialize comparator
        
        Args:
            df: DataFrame with materials
            X: Feature matrix
            predicted_values: Model predictions
            uncertainties: Model uncertainties
        """
        self.df = df
        self.X = X
        self.predicted_values = predicted_values
        self.uncertainties = uncertainties
        self.history = PreferenceHistory()
        
        self.compared_pairs = set()  # Track compared pairs
    
    def generate_pair(self, strategy: str = 'informative') -> Tuple[int, int]:
        """
        Generate a pair of materials for comparison
        
        Args:
            strategy: 'informative', 'pareto', or 'random'
            
        Returns:
            (index_A, index_B)
        """
        n = len(self.df)
        
        if strategy == 'informative':
            # Select materials with high uncertainty and good predicted value
            scores = self.predicted_values + 0.5 * self.uncertainties
            candidates = np.argsort(scores)[-50:]  # Top 50
            
            # Pick two from candidates
            np.random.shuffle(candidates)
            idx_a, idx_b = candidates[0], candidates[1]
        
        elif strategy == 'pareto':
            # Select from Pareto front (if sustainability available)
            if 'sustainability_score' in self.df.columns:
                # Simple Pareto: high voltage, low sustainability
                voltage_rank = self.predicted_values.argsort().argsort()
                sust_rank = (-self.df['sustainability_score'].values).argsort().argsort()
                pareto_score = voltage_rank + sust_rank
                
                candidates = np.argsort(pareto_score)[-50:]
                np.random.shuffle(candidates)
                idx_a, idx_b = candidates[0], candidates[1]
            else:
                # Fall back to informative
                return self.generate_pair('informative')
        
        else:  # random
            idx_a, idx_b = np.random.choice(n, size=2, replace=False)
        
        # Avoid duplicate pairs
        pair_key = tuple(sorted([idx_a, idx_b]))
        if pair_key in self.compared_pairs:
            # Try again (recursive with limit)
            if len(self.compared_pairs) < n * (n-1) / 2:
                return self.generate_pair(strategy)
        
        self.compared_pairs.add(pair_key)
        
        return int(idx_a), int(idx_b)
    
    def record_preference(self, winner_idx: int, loser_idx: int):
        """Record that winner is preferred over loser"""
        self.history.add(
            winner_idx, loser_idx,
            self.X[winner_idx], self.X[loser_idx]
        )
    
    def get_material_card(self, idx: int) -> dict:
        """
        Get material information for display
        
        Args:
            idx: Material index
            
        Returns:
            Dict with material info
        """
        row = self.df.iloc[idx]
        
        card = {
            'formula': row['formula'],
            'predicted_voltage': self.predicted_values[idx],
            'uncertainty': self.uncertainties[idx],
            'true_voltage': row.get('voltage', None)
        }
        
        # Add sustainability if available
        if 'sustainability_score' in row:
            card['sustainability'] = row['sustainability_score']
            card['toxicity'] = row.get('toxicity', None)
            card['supply_risk'] = row.get('supply_risk', None)
        
        return card
    
    def train_preference_model(self) -> PreferenceModel:
        """Train preference model from collected history"""
        model = PreferenceModel(self.X.shape[1])
        model.fit(self.history)
        return model


def create_preference_augmented_acquisition(base_scores: np.ndarray,
                                            preference_scores: np.ndarray,
                                            lambda_pref: float = 0.3) -> np.ndarray:
    """
    Combine acquisition scores with human preferences
    
    Args:
        base_scores: Base acquisition scores (e.g., UCB)
        preference_scores: Human preference scores
        lambda_pref: Weight for preferences (0-1)
        
    Returns:
        Combined scores
    """
    # Normalize both to same scale
    base_norm = (base_scores - base_scores.mean()) / (base_scores.std() + 1e-8)
    pref_norm = (preference_scores - preference_scores.mean()) / (preference_scores.std() + 1e-8)
    
    # Weighted combination
    combined = (1 - lambda_pref) * base_norm + lambda_pref * pref_norm
    
    return combined


if __name__ == "__main__":
    # Test preference learning
    np.random.seed(42)
    
    # Synthetic data
    n_materials = 100
    n_features = 20
    
    X = np.random.randn(n_materials, n_features)
    y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n_materials) * 0.1  # True utility
    
    df = pd.DataFrame({
        'formula': [f'Mat_{i}' for i in range(n_materials)],
        'voltage': y
    })
    
    predicted_values = y + np.random.randn(n_materials) * 0.2
    uncertainties = np.random.rand(n_materials) * 0.5
    
    # Create comparator
    comparator = MaterialComparator(df, X, predicted_values, uncertainties)
    
    # Simulate 10 comparisons
    print("=== Simulating Human Preferences ===\n")
    
    for i in range(10):
        idx_a, idx_b = comparator.generate_pair('informative')
        
        # Simulate human choosing based on true utility
        if y[idx_a] > y[idx_b]:
            winner, loser = idx_a, idx_b
        else:
            winner, loser = idx_b, idx_a
        
        comparator.record_preference(winner, loser)
        
        print(f"Comparison {i+1}: {df.iloc[winner]['formula']} > {df.iloc[loser]['formula']}")
    
    # Train preference model
    print("\n=== Training Preference Model ===\n")
    pref_model = comparator.train_preference_model()
    
    # Predict preferences
    pref_scores = pref_model.predict_preference_score(X)
    
    # Compare with true utilities
    from scipy.stats import spearmanr
    correlation = spearmanr(y, pref_scores)[0]
    
    print(f"Correlation between learned preferences and true utility: {correlation:.3f}")
    print("\nTop 5 materials by learned preference:")
    top_5_idx = np.argsort(pref_scores)[-5:][::-1]
    for idx in top_5_idx:
        print(f"  {df.iloc[idx]['formula']}: pref={pref_scores[idx]:.3f}, true={y[idx]:.3f}")
