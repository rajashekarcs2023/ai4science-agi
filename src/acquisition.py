"""
Acquisition functions for active learning
Implements UCB (Upper Confidence Bound) and other strategies
"""

import numpy as np
from typing import Optional, Callable


class AcquisitionFunction:
    """Acquisition functions for selecting next materials to evaluate"""
    
    def __init__(self, strategy: str = 'ucb', beta: float = 0.8, lambda_pref: float = 0.0):
        """
        Initialize acquisition function
        
        Args:
            strategy: 'ucb', 'ei', 'poi', or 'greedy'
            beta: Exploration parameter for UCB (higher = more exploration)
            lambda_pref: Weight for human preference (if available)
        """
        self.strategy = strategy
        self.beta = beta
        self.lambda_pref = lambda_pref
        
        self.strategies = {
            'ucb': self._upper_confidence_bound,
            'ei': self._expected_improvement,
            'greedy': self._greedy,
            'random': self._random,
            'uncertainty': self._uncertainty
        }
    
    def compute(self, mu: np.ndarray, sigma: np.ndarray, 
                best_value: Optional[float] = None,
                preference_scores: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute acquisition scores
        
        Args:
            mu: Mean predictions (n_samples,)
            sigma: Uncertainty estimates (n_samples,)
            best_value: Current best observed value (for EI)
            preference_scores: Human preference scores (optional)
            
        Returns:
            Acquisition scores (higher = better)
        """
        if self.strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Compute base acquisition
        scores = self.strategies[self.strategy](mu, sigma, best_value)
        
        # Add human preference if available
        if preference_scores is not None and self.lambda_pref > 0:
            # Normalize preference scores
            pref_norm = (preference_scores - preference_scores.mean()) / (preference_scores.std() + 1e-8)
            scores = scores + self.lambda_pref * pref_norm
        
        return scores
    
    def _upper_confidence_bound(self, mu: np.ndarray, sigma: np.ndarray, 
                               best_value: Optional[float] = None) -> np.ndarray:
        """
        UCB acquisition: μ + β * σ
        Balances exploitation (high μ) and exploration (high σ)
        """
        return mu + self.beta * sigma
    
    def _expected_improvement(self, mu: np.ndarray, sigma: np.ndarray, 
                             best_value: float) -> np.ndarray:
        """
        Expected Improvement over current best
        """
        from scipy.stats import norm
        
        if best_value is None:
            best_value = np.max(mu)
        
        z = (mu - best_value) / (sigma + 1e-8)
        ei = (mu - best_value) * norm.cdf(z) + sigma * norm.pdf(z)
        
        return ei
    
    def _greedy(self, mu: np.ndarray, sigma: np.ndarray, 
               best_value: Optional[float] = None) -> np.ndarray:
        """Pure exploitation: just pick highest predicted value"""
        return mu
    
    def _uncertainty(self, mu: np.ndarray, sigma: np.ndarray, 
                    best_value: Optional[float] = None) -> np.ndarray:
        """Pure exploration: pick highest uncertainty"""
        return sigma
    
    def _random(self, mu: np.ndarray, sigma: np.ndarray, 
               best_value: Optional[float] = None) -> np.ndarray:
        """Random selection (baseline)"""
        return np.random.rand(len(mu))
    
    def select_batch(self, mu: np.ndarray, sigma: np.ndarray, 
                     batch_size: int,
                     best_value: Optional[float] = None,
                     preference_scores: Optional[np.ndarray] = None,
                     exclude_indices: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Select a batch of candidates
        
        Args:
            mu: Mean predictions
            sigma: Uncertainty estimates
            batch_size: Number of candidates to select
            best_value: Current best value
            preference_scores: Human preferences
            exclude_indices: Indices to exclude from selection
            
        Returns:
            Indices of selected candidates
        """
        # Compute acquisition scores
        scores = self.compute(mu, sigma, best_value, preference_scores)
        
        # Exclude already-selected indices
        if exclude_indices is not None and len(exclude_indices) > 0:
            scores[exclude_indices] = -np.inf
        
        # Select top-k
        selected_indices = np.argsort(scores)[-batch_size:][::-1]
        
        return selected_indices


class SustainabilityAwareAcquisition(AcquisitionFunction):
    """Acquisition with sustainability constraints"""
    
    def __init__(self, strategy: str = 'ucb', beta: float = 0.8,
                 sustainability_weight: float = 0.2,
                 sustainability_threshold: float = 0.7):
        """
        Initialize sustainability-aware acquisition
        
        Args:
            strategy: Base acquisition strategy
            beta: Exploration parameter
            sustainability_weight: Weight for sustainability in scoring
            sustainability_threshold: Max acceptable sustainability score (0-1)
        """
        super().__init__(strategy, beta)
        self.sustainability_weight = sustainability_weight
        self.sustainability_threshold = sustainability_threshold
    
    def compute(self, mu: np.ndarray, sigma: np.ndarray,
                best_value: Optional[float] = None,
                preference_scores: Optional[np.ndarray] = None,
                sustainability_scores: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute acquisition with sustainability
        
        Args:
            mu: Mean predictions
            sigma: Uncertainty
            best_value: Current best
            preference_scores: Human preferences
            sustainability_scores: Sustainability scores (0=best, 1=worst)
            
        Returns:
            Acquisition scores
        """
        # Base acquisition
        scores = super().compute(mu, sigma, best_value, preference_scores)
        
        # Incorporate sustainability
        if sustainability_scores is not None:
            # Penalize unsustainable materials
            # Convert sustainability (0=best, 1=worst) to bonus (1=best, 0=worst)
            sustainability_bonus = 1.0 - sustainability_scores
            
            # Apply as multiplicative factor
            scores = scores * (1.0 - self.sustainability_weight + 
                             self.sustainability_weight * sustainability_bonus)
            
            # Hard constraint: exclude materials above threshold
            scores[sustainability_scores > self.sustainability_threshold] = -np.inf
        
        return scores
    
    def select_batch(self, mu: np.ndarray, sigma: np.ndarray,
                     batch_size: int,
                     best_value: Optional[float] = None,
                     preference_scores: Optional[np.ndarray] = None,
                     sustainability_scores: Optional[np.ndarray] = None,
                     exclude_indices: Optional[np.ndarray] = None) -> np.ndarray:
        """Select batch with sustainability awareness"""
        scores = self.compute(mu, sigma, best_value, preference_scores, sustainability_scores)
        
        if exclude_indices is not None and len(exclude_indices) > 0:
            scores[exclude_indices] = -np.inf
        
        selected_indices = np.argsort(scores)[-batch_size:][::-1]
        
        return selected_indices


def create_acquisition_function(strategy: str = 'ucb',
                                beta: float = 0.8,
                                use_sustainability: bool = False,
                                **kwargs) -> AcquisitionFunction:
    """
    Factory function to create acquisition function
    
    Args:
        strategy: Acquisition strategy
        beta: Exploration parameter
        use_sustainability: Whether to use sustainability-aware acquisition
        **kwargs: Additional parameters
        
    Returns:
        AcquisitionFunction instance
    """
    if use_sustainability:
        return SustainabilityAwareAcquisition(
            strategy=strategy,
            beta=beta,
            sustainability_weight=kwargs.get('sustainability_weight', 0.2),
            sustainability_threshold=kwargs.get('sustainability_threshold', 0.7)
        )
    else:
        return AcquisitionFunction(
            strategy=strategy,
            beta=beta,
            lambda_pref=kwargs.get('lambda_pref', 0.0)
        )


if __name__ == "__main__":
    # Test acquisition functions
    np.random.seed(42)
    
    n_candidates = 100
    mu = np.random.randn(n_candidates) * 2 + 3  # Mean predictions
    sigma = np.random.rand(n_candidates) * 0.5 + 0.1  # Uncertainties
    
    print("=== Testing Acquisition Functions ===\n")
    
    strategies = ['ucb', 'greedy', 'uncertainty', 'random']
    batch_size = 5
    
    for strategy in strategies:
        acq = AcquisitionFunction(strategy=strategy, beta=0.8)
        selected = acq.select_batch(mu, sigma, batch_size)
        
        print(f"{strategy.upper():12s} | Selected indices: {selected}")
        print(f"             | Means: {mu[selected].round(2)}")
        print(f"             | Uncertainties: {sigma[selected].round(2)}\n")
    
    # Test sustainability-aware acquisition
    print("\n=== Sustainability-Aware Acquisition ===\n")
    
    sustainability_scores = np.random.rand(n_candidates)
    
    acq_sust = SustainabilityAwareAcquisition(
        strategy='ucb',
        beta=0.8,
        sustainability_weight=0.3,
        sustainability_threshold=0.8
    )
    
    selected = acq_sust.select_batch(
        mu, sigma, batch_size,
        sustainability_scores=sustainability_scores
    )
    
    print(f"Selected indices: {selected}")
    print(f"Means: {mu[selected].round(2)}")
    print(f"Sustainability: {sustainability_scores[selected].round(2)}")
