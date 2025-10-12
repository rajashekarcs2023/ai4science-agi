"""
Autonomous Discovery Loop - The core active learning agent
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time

from .surrogate_model import SurrogateModel
from .acquisition import create_acquisition_function


@dataclass
class DiscoveryConfig:
    """Configuration for discovery loop"""
    n_init: int = 50  # Initial training size
    n_rounds: int = 10  # Number of discovery rounds
    batch_size: int = 5  # Materials to select per round
    acquisition_strategy: str = 'ucb'  # Acquisition strategy
    beta: float = 0.8  # Exploration parameter
    use_sustainability: bool = False  # Use sustainability constraints
    sustainability_weight: float = 0.2
    random_state: int = 42


@dataclass
class DiscoveryResults:
    """Results from discovery loop"""
    round_history: List[Dict] = field(default_factory=list)
    best_materials: pd.DataFrame = None
    baseline_history: List[Dict] = field(default_factory=list)
    final_model: SurrogateModel = None
    feature_cols: List[str] = field(default_factory=list)
    agent: 'AutonomousDiscoveryAgent' = None
    config: DiscoveryConfig = None
    
    def to_dict(self) -> Dict:
        """Convert results to dictionary"""
        return {
            'round_history': self.round_history,
            'baseline_history': self.baseline_history,
            'config': self.config.__dict__ if self.config else None
        }


class AutonomousDiscoveryAgent:
    """
    Autonomous agent for materials discovery using active learning
    
    The agent:
    1. Starts with small training set
    2. Trains surrogate model
    3. Uses acquisition function to select promising materials
    4. Evaluates selected materials (reveals true values)
    5. Retrains and repeats
    """
    
    def __init__(self, config: DiscoveryConfig = None):
        """
        Initialize discovery agent
        
        Args:
            config: Discovery configuration
        """
        self.config = config or DiscoveryConfig()
        self.model = None
        self.acquisition = None
        self.results = None
        
        np.random.seed(self.config.random_state)
    
    def run(self, 
            X: np.ndarray, 
            y: np.ndarray, 
            df: pd.DataFrame,
            feature_cols: List[str] = None,
            sustainability_scores: Optional[np.ndarray] = None,
            verbose: bool = True) -> DiscoveryResults:
        """
        Run full autonomous discovery loop
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,) - oracle values
            df: DataFrame with material metadata
            sustainability_scores: Sustainability scores (optional)
            verbose: Print progress
            
        Returns:
            DiscoveryResults object
        """
        if verbose:
            print("\n" + "="*60)
            print("🚀 AUTONOMOUS DISCOVERY AGENT STARTING")
            print("="*60)
            print(f"Total materials pool: {len(X)}")
            print(f"Initial training size: {self.config.n_init}")
            print(f"Discovery rounds: {self.config.n_rounds}")
            print(f"Batch size: {self.config.batch_size}")
            print(f"Acquisition: {self.config.acquisition_strategy.upper()} (β={self.config.beta})")
            print("="*60 + "\n")
        
        # Initialize results
        results = DiscoveryResults(config=self.config)
        
        # Phase 1: Random initialization
        all_indices = np.arange(len(X))
        np.random.shuffle(all_indices)
        
        train_indices = all_indices[:self.config.n_init]
        pool_indices = all_indices[self.config.n_init:]
        
        if verbose:
            print(f"📊 Round 0 (Initialization): {len(train_indices)} materials")
        
        # Track best value
        best_value = np.max(y[train_indices])
        best_idx = train_indices[np.argmax(y[train_indices])]
        
        if verbose:
            best_formula = df.iloc[best_idx]['formula']
            print(f"   Best so far: {best_formula} = {best_value:.3f}V\n")
        
        # Initialize acquisition function
        self.acquisition = create_acquisition_function(
            strategy=self.config.acquisition_strategy,
            beta=self.config.beta,
            use_sustainability=self.config.use_sustainability,
            sustainability_weight=self.config.sustainability_weight
        )
        
        # Phase 2: Discovery rounds
        for round_idx in range(1, self.config.n_rounds + 1):
            round_start = time.time()
            
            if verbose:
                print(f"🔬 Round {round_idx}/{self.config.n_rounds}")
            
            # Step 1: Train surrogate model
            X_train = X[train_indices]
            y_train = y[train_indices]
            
            self.model = SurrogateModel(model_type='lightgbm', random_state=self.config.random_state)
            self.model.fit(X_train, y_train, verbose=False)
            
            # Step 2: Predict on pool
            X_pool = X[pool_indices]
            mu_pool, sigma_pool = self.model.predict(X_pool, return_uncertainty=True)
            
            # Step 3: Select next batch using acquisition function
            if self.config.use_sustainability and sustainability_scores is not None:
                sust_pool = sustainability_scores[pool_indices]
                selected_pool_idx = self.acquisition.select_batch(
                    mu_pool, sigma_pool, self.config.batch_size,
                    best_value=best_value,
                    sustainability_scores=sust_pool
                )
            else:
                selected_pool_idx = self.acquisition.select_batch(
                    mu_pool, sigma_pool, self.config.batch_size,
                    best_value=best_value
                )
            
            # Convert pool indices to global indices
            selected_indices = pool_indices[selected_pool_idx]
            
            # Step 4: Evaluate selected materials (reveal true values)
            selected_values = y[selected_indices]
            
            # Step 5: Update best value
            round_best_value = np.max(selected_values)
            round_best_idx_local = np.argmax(selected_values)
            round_best_idx = selected_indices[round_best_idx_local]
            
            if round_best_value > best_value:
                best_value = round_best_value
                best_idx = round_best_idx
                improvement = "✨ NEW BEST!"
            else:
                improvement = ""
            
            # Step 6: Add to training set
            train_indices = np.concatenate([train_indices, selected_indices])
            pool_indices = np.array([idx for idx in pool_indices if idx not in selected_indices])
            
            # Calculate metrics
            mean_uncertainty = np.mean(sigma_pool)
            
            round_time = time.time() - round_start
            
            # Store history
            round_history = {
                'round': round_idx,
                'train_size': len(train_indices),
                'best_value': best_value,
                'best_formula': df.iloc[best_idx]['formula'],
                'best_material_idx': best_idx,
                'round_best_value': round_best_value,
                'round_best_formula': df.iloc[round_best_idx]['formula'],
                'mean_uncertainty': mean_uncertainty,
                'selected_indices': selected_indices.tolist(),
                'selected_values': selected_values.tolist(),
                'time': round_time
            }
            results.round_history.append(round_history)
            
            if verbose:
                print(f"   Trained on {len(train_indices)} materials")
                print(f"   Mean uncertainty: {mean_uncertainty:.3f}")
                print(f"   Selected: {df.iloc[round_best_idx]['formula']} = {round_best_value:.3f}V {improvement}")
                print(f"   Overall best: {df.iloc[best_idx]['formula']} = {best_value:.3f}V")
                print(f"   Time: {round_time:.2f}s\n")
        
        # Phase 3: Baseline comparison (random selection)
        if verbose:
            print("🎲 Running random baseline for comparison...\n")
        
        baseline_results = self._run_baseline(X, y, df, verbose=False)
        results.baseline_history = baseline_results
        
        # Final evaluation
        results.final_model = self.model
        results.feature_cols = feature_cols or []
        results.agent = self
        
        # Extract top materials
        all_mu, all_sigma = self.model.predict(X, return_uncertainty=True)
        df_results = df.copy()
        df_results['predicted_voltage'] = all_mu
        df_results['uncertainty'] = all_sigma
        df_results['true_voltage'] = y
        df_results['was_tested'] = False
        df_results.loc[train_indices, 'was_tested'] = True
        
        # Top materials
        results.best_materials = df_results.nlargest(20, 'predicted_voltage')
        
        if verbose:
            print("\n" + "="*60)
            print("✅ DISCOVERY COMPLETE")
            print("="*60)
            print(f"Final best: {df.iloc[best_idx]['formula']} = {best_value:.3f}V")
            print(f"Total materials evaluated: {len(train_indices)}/{len(X)}")
            print(f"Discovery efficiency: {len(train_indices)/len(X)*100:.1f}% of dataset")
            
            # Compare with baseline
            baseline_best = max([r['best_value'] for r in baseline_results])
            improvement_pct = (best_value - baseline_best) / baseline_best * 100
            print(f"\nAgent vs Random Baseline:")
            print(f"  Agent: {best_value:.3f}V")
            print(f"  Random: {baseline_best:.3f}V")
            print(f"  Improvement: {improvement_pct:+.1f}%")
            print("="*60 + "\n")
        
        self.results = results
        return results
    
    def _run_baseline(self, X: np.ndarray, y: np.ndarray, 
                     df: pd.DataFrame, verbose: bool = False) -> List[Dict]:
        """Run random selection baseline"""
        all_indices = np.arange(len(X))
        np.random.shuffle(all_indices)
        
        train_indices = all_indices[:self.config.n_init]
        pool_indices = all_indices[self.config.n_init:]
        
        baseline_history = []
        best_value = np.max(y[train_indices])
        
        for round_idx in range(1, self.config.n_rounds + 1):
            # Random selection
            selected_indices = pool_indices[:self.config.batch_size]
            pool_indices = pool_indices[self.config.batch_size:]
            
            train_indices = np.concatenate([train_indices, selected_indices])
            
            # Update best
            current_best = np.max(y[train_indices])
            if current_best > best_value:
                best_value = current_best
            
            baseline_history.append({
                'round': round_idx,
                'train_size': len(train_indices),
                'best_value': best_value
            })
        
        return baseline_history
    
    def get_top_materials(self, n: int = 10) -> pd.DataFrame:
        """Get top n predicted materials"""
        if self.results is None:
            raise ValueError("Must run discovery first")
        
        return self.results.best_materials.head(n)
    
    def run_discovery(self, X: np.ndarray, y: np.ndarray, df: pd.DataFrame,
                     feature_cols: List[str] = None,
                     sustainability_scores: Optional[np.ndarray] = None,
                     verbose: bool = True) -> DiscoveryResults:
        """Alias for run() method for backwards compatibility"""
        return self.run(X, y, df, feature_cols, sustainability_scores, verbose)


def run_autonomous_discovery(X: np.ndarray, y: np.ndarray, df: pd.DataFrame,
                            config: DiscoveryConfig = None,
                            sustainability_scores: Optional[np.ndarray] = None,
                            verbose: bool = True) -> DiscoveryResults:
    """
    Convenience function to run discovery
    
    Args:
        X: Features
        y: Targets
        df: Material metadata
        config: Discovery config
        sustainability_scores: Optional sustainability scores
        verbose: Print progress
        
    Returns:
        DiscoveryResults
    """
    agent = AutonomousDiscoveryAgent(config=config)
    return agent.run_discovery(X, y, df, sustainability_scores, verbose)


if __name__ == "__main__":
    # Test discovery loop
    from sklearn.datasets import make_regression
    
    print("Creating synthetic data...")
    X, y = make_regression(n_samples=500, n_features=20, noise=0.1, random_state=42)
    y = (y - y.min()) / (y.max() - y.min()) * 3 + 2  # Scale to 2-5V range
    
    df = pd.DataFrame({
        'formula': [f'Material_{i}' for i in range(len(X))],
        'voltage': y
    })
    
    config = DiscoveryConfig(
        n_init=30,
        n_rounds=5,
        batch_size=10,
        acquisition_strategy='ucb',
        beta=1.0
    )
    
    results = run_autonomous_discovery(X, y, df, config=config, verbose=True)
    
    print("\n=== Top 5 Discovered Materials ===")
    top5 = results.best_materials.head(5)
    print(top5[['formula', 'predicted_voltage', 'uncertainty', 'true_voltage']])
