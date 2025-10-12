"""
Quick test script to verify the autonomous discovery pipeline works end-to-end
Run this before the demo to ensure everything is functioning
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_imports():
    """Test that all required modules can be imported"""
    print("=" * 60)
    print("TEST 1: Checking imports...")
    print("=" * 60)
    
    try:
        from src.data_loader import load_battery_data
        print("✓ data_loader")
        
        from src.feature_engineering import prepare_features
        print("✓ feature_engineering")
        
        from src.sustainability import add_sustainability
        print("✓ sustainability")
        
        from src.surrogate_model import SurrogateModel
        print("✓ surrogate_model")
        
        from src.acquisition import AcquisitionFunction
        print("✓ acquisition")
        
        from src.discovery_loop import AutonomousDiscoveryAgent, DiscoveryConfig
        print("✓ discovery_loop")
        
        from src.visualizations import plot_discovery_curve
        print("✓ visualizations")
        
        from src.preference_learning import PreferenceModel
        print("✓ preference_learning")
        
        print("\n✅ All imports successful!\n")
        return True
    except Exception as e:
        print(f"\n❌ Import failed: {e}\n")
        return False


def test_data_loading():
    """Test data loading"""
    print("=" * 60)
    print("TEST 2: Loading battery data...")
    print("=" * 60)
    
    try:
        from src.data_loader import load_battery_data
        
        df = load_battery_data(
            data_path="data/Battery Data/battery_merged.csv",
            sample_size=500  # Small sample for testing
        )
        
        print(f"\n✓ Loaded {len(df)} materials")
        print(f"✓ Columns: {df.columns.tolist()}")
        print(f"✓ Voltage range: {df['voltage'].min():.2f}V - {df['voltage'].max():.2f}V")
        print(f"\n✅ Data loading successful!\n")
        
        return df
    except Exception as e:
        print(f"\n❌ Data loading failed: {e}\n")
        import traceback
        traceback.print_exc()
        return None


def test_feature_engineering(df):
    """Test feature engineering"""
    print("=" * 60)
    print("TEST 3: Feature engineering...")
    print("=" * 60)
    
    try:
        from src.feature_engineering import prepare_features
        
        X, y, feature_names, df_processed = prepare_features(df, target_col='voltage')
        
        print(f"\n✓ Feature matrix shape: {X.shape}")
        print(f"✓ Target shape: {y.shape}")
        print(f"✓ Number of features: {len(feature_names)}")
        print(f"✓ Sample features: {feature_names[:5]}")
        print(f"\n✅ Feature engineering successful!\n")
        
        return X, y, feature_names, df_processed
    except Exception as e:
        print(f"\n❌ Feature engineering failed: {e}\n")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def test_surrogate_model(X, y):
    """Test surrogate model training"""
    print("=" * 60)
    print("TEST 4: Training surrogate model...")
    print("=" * 60)
    
    try:
        from src.surrogate_model import SurrogateModel
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = SurrogateModel(model_type='lightgbm')
        model.fit(X_train, y_train, verbose=True)
        
        mu, sigma = model.predict(X_test, return_uncertainty=True)
        
        metrics = model.evaluate(X_test, y_test)
        
        print(f"\n✓ Model trained successfully")
        print(f"✓ Test RMSE: {metrics['rmse']:.3f}")
        print(f"✓ Test R²: {metrics['r2']:.3f}")
        print(f"✓ Mean uncertainty: {metrics['mean_uncertainty']:.3f}")
        print(f"\n✅ Surrogate model working!\n")
        
        return True
    except Exception as e:
        print(f"\n❌ Surrogate model failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_discovery_loop(X, y, df):
    """Test autonomous discovery loop"""
    print("=" * 60)
    print("TEST 5: Running discovery loop (3 rounds)...")
    print("=" * 60)
    
    try:
        from src.discovery_loop import AutonomousDiscoveryAgent, DiscoveryConfig
        
        config = DiscoveryConfig(
            n_init=20,
            n_rounds=3,  # Short test
            batch_size=5,
            acquisition_strategy='ucb',
            beta=0.8
        )
        
        agent = AutonomousDiscoveryAgent(config=config)
        results = agent.run_discovery(X, y, df, verbose=True)
        
        final_best = results.round_history[-1]['best_value']
        initial_best = results.round_history[0]['best_value']
        improvement = ((final_best - initial_best) / initial_best) * 100
        
        print(f"\n✓ Discovery completed successfully")
        print(f"✓ Initial best: {initial_best:.3f}V")
        print(f"✓ Final best: {final_best:.3f}V")
        print(f"✓ Improvement: {improvement:+.1f}%")
        print(f"\n✅ Discovery loop working!\n")
        
        return results
    except Exception as e:
        print(f"\n❌ Discovery loop failed: {e}\n")
        import traceback
        traceback.print_exc()
        return None


def test_visualizations(results):
    """Test visualization generation"""
    print("=" * 60)
    print("TEST 6: Generating visualizations...")
    print("=" * 60)
    
    try:
        from src.visualizations import (
            plot_discovery_curve, 
            plot_uncertainty_collapse,
            plot_discovery_efficiency
        )
        
        fig1 = plot_discovery_curve(results.round_history, results.baseline_history)
        print("✓ Discovery curve created")
        
        fig2 = plot_uncertainty_collapse(results.round_history)
        print("✓ Uncertainty plot created")
        
        fig3 = plot_discovery_efficiency(results.round_history, results.baseline_history, len(results.best_materials))
        print("✓ Efficiency plot created")
        
        print(f"\n✅ Visualizations working!\n")
        return True
    except Exception as e:
        print(f"\n❌ Visualization failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n")
    print("🧪 " + "=" * 58)
    print("🧪 AUTONOMOUS DISCOVERY AGENT - PIPELINE TEST")
    print("🧪 " + "=" * 58)
    print("\n")
    
    # Test 1: Imports
    if not test_imports():
        print("❌ FAILED: Fix import errors before proceeding")
        return False
    
    # Test 2: Data loading
    df = test_data_loading()
    if df is None:
        print("❌ FAILED: Fix data loading before proceeding")
        return False
    
    # Test 3: Feature engineering
    X, y, feature_names, df_processed = test_feature_engineering(df)
    if X is None:
        print("❌ FAILED: Fix feature engineering before proceeding")
        return False
    
    # Test 4: Surrogate model
    if not test_surrogate_model(X, y):
        print("❌ FAILED: Fix surrogate model before proceeding")
        return False
    
    # Test 5: Discovery loop
    results = test_discovery_loop(X, y, df_processed)
    if results is None:
        print("❌ FAILED: Fix discovery loop before proceeding")
        return False
    
    # Test 6: Visualizations
    if not test_visualizations(results):
        print("❌ FAILED: Fix visualizations before proceeding")
        return False
    
    # All tests passed
    print("\n")
    print("🎉 " + "=" * 58)
    print("🎉 ALL TESTS PASSED - SYSTEM READY FOR DEMO!")
    print("🎉 " + "=" * 58)
    print("\n")
    print("Next steps:")
    print("  1. Run: streamlit run app.py")
    print("  2. Configure settings in sidebar")
    print("  3. Click '🚀 Start Discovery'")
    print("  4. Present to judges!")
    print("\n")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
