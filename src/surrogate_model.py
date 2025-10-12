"""
Surrogate models for battery property prediction with uncertainty quantification
Supports LightGBM with quantile regression for confidence intervals
"""

import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    from sklearn.ensemble import RandomForestRegressor


class SurrogateModel:
    """Surrogate model with uncertainty quantification"""
    
    def __init__(self, model_type: str = 'lightgbm', random_state: int = 42):
        """
        Initialize surrogate model
        
        Args:
            model_type: 'lightgbm' or 'random_forest'
            random_state: Random seed
        """
        self.model_type = model_type
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        # Models for mean and uncertainty
        self.model_mean = None
        self.model_lower = None
        self.model_upper = None
        
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """
        Fit surrogate model with uncertainty quantification
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            verbose: Print training info
        """
        if verbose:
            print(f"\nTraining {self.model_type} surrogate model...")
            print(f"  Training samples: {len(X)}")
            print(f"  Features: {X.shape[1]}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        if self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            self._fit_lightgbm(X_scaled, y, verbose)
        else:
            self._fit_random_forest(X_scaled, y, verbose)
        
        self.is_fitted = True
        
        if verbose:
            print("  ✓ Model trained successfully")
    
    def _fit_lightgbm(self, X: np.ndarray, y: np.ndarray, verbose: bool):
        """Fit LightGBM with quantile regression"""
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': self.random_state,
            'n_estimators': 100,
        }
        
        # Train mean predictor
        self.model_mean = lgb.LGBMRegressor(**params)
        self.model_mean.fit(X, y)
        
        # Train quantile regressors for uncertainty
        params_lower = params.copy()
        params_lower.update({'objective': 'quantile', 'alpha': 0.16})  # ~1 std below
        self.model_lower = lgb.LGBMRegressor(**params_lower)
        self.model_lower.fit(X, y)
        
        params_upper = params.copy()
        params_upper.update({'objective': 'quantile', 'alpha': 0.84})  # ~1 std above
        self.model_upper = lgb.LGBMRegressor(**params_upper)
        self.model_upper.fit(X, y)
    
    def _fit_random_forest(self, X: np.ndarray, y: np.ndarray, verbose: bool):
        """Fallback: Random Forest with bootstrap for uncertainty"""
        from sklearn.ensemble import RandomForestRegressor
        
        self.model_mean = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model_mean.fit(X, y)
        
        # For RF, we'll use tree predictions for uncertainty
        self.model_lower = self.model_mean
        self.model_upper = self.model_mean
    
    def predict(self, X: np.ndarray, return_uncertainty: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict with uncertainty quantification
        
        Args:
            X: Feature matrix
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            (mean_predictions, std_predictions) or just mean_predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        
        # Mean prediction
        mu = self.model_mean.predict(X_scaled)
        
        if not return_uncertainty:
            return mu, None
        
        # Uncertainty estimation
        if self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            # Use quantile predictions
            lower = self.model_lower.predict(X_scaled)
            upper = self.model_upper.predict(X_scaled)
            
            # Convert to standard deviation (approximate)
            sigma = (upper - lower) / 2.0
        else:
            # Random Forest: use tree variance
            tree_preds = np.array([tree.predict(X_scaled) for tree in self.model_mean.estimators_])
            sigma = np.std(tree_preds, axis=0)
        
        # Ensure positive uncertainty
        sigma = np.maximum(sigma, 1e-6)
        
        return mu, sigma
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate model performance
        
        Args:
            X: Feature matrix
            y: True target values
            
        Returns:
            Dict with metrics: rmse, mae, r2
        """
        mu, sigma = self.predict(X, return_uncertainty=True)
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        rmse = np.sqrt(mean_squared_error(y, mu))
        mae = mean_absolute_error(y, mu)
        r2 = r2_score(y, mu)
        
        mean_uncertainty = np.mean(sigma)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mean_uncertainty': mean_uncertainty
        }


class EnsembleSurrogate:
    """Ensemble of surrogate models for improved uncertainty"""
    
    def __init__(self, n_models: int = 3, random_state: int = 42):
        """
        Initialize ensemble
        
        Args:
            n_models: Number of models in ensemble
            random_state: Random seed
        """
        self.n_models = n_models
        self.random_state = random_state
        self.models = []
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """Fit ensemble of models"""
        if verbose:
            print(f"\nTraining ensemble of {self.n_models} models...")
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.models = []
        for i in range(self.n_models):
            model = SurrogateModel(
                model_type='lightgbm',
                random_state=self.random_state + i
            )
            
            # Bootstrap sample
            indices = np.random.RandomState(self.random_state + i).choice(
                len(X), size=len(X), replace=True
            )
            X_boot = X_scaled[indices]
            y_boot = y[indices]
            
            model.scaler = self.scaler  # Share scaler
            model.is_fitted = False
            model.fit(X_boot, y_boot, verbose=False)
            
            self.models.append(model)
            
            if verbose and (i + 1) % max(1, self.n_models // 3) == 0:
                print(f"  Trained {i+1}/{self.n_models} models")
        
        self.is_fitted = True
        
        if verbose:
            print("  ✓ Ensemble trained successfully")
    
    def predict(self, X: np.ndarray, return_uncertainty: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict using ensemble"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            mu, _ = model.predict(X, return_uncertainty=False)
            predictions.append(mu)
        
        predictions = np.array(predictions)
        
        # Mean and std across models
        mu = np.mean(predictions, axis=0)
        
        if return_uncertainty:
            sigma = np.std(predictions, axis=0)
            sigma = np.maximum(sigma, 1e-6)
            return mu, sigma
        else:
            return mu, None
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate ensemble"""
        mu, sigma = self.predict(X, return_uncertainty=True)
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        rmse = np.sqrt(mean_squared_error(y, mu))
        mae = mean_absolute_error(y, mu)
        r2 = r2_score(y, mu)
        mean_uncertainty = np.mean(sigma)
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mean_uncertainty': mean_uncertainty
        }


if __name__ == "__main__":
    # Test surrogate model
    from sklearn.datasets import make_regression
    
    X, y = make_regression(n_samples=500, n_features=20, noise=0.1, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = SurrogateModel(model_type='lightgbm')
    model.fit(X_train, y_train)
    
    # Predict with uncertainty
    mu, sigma = model.predict(X_test, return_uncertainty=True)
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    
    print("\n=== Model Evaluation ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    print(f"\nPrediction examples:")
    for i in range(5):
        print(f"  True: {y_test[i]:.2f}, Pred: {mu[i]:.2f} ± {sigma[i]:.2f}")
