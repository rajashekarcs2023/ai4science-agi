"""
Statistical validation and calibration for model predictions
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats


def uncertainty_calibration_plot(y_true, y_pred, y_std):
    """
    Create calibration plot for uncertainty estimates
    Shows if σ matches actual errors
    
    Args:
        y_true: True values
        y_pred: Predicted values
        y_std: Predicted uncertainties (standard deviations)
    
    Returns:
        Plotly figure
    """
    # Calculate absolute errors
    errors = np.abs(y_true - y_pred)
    
    # Sort by uncertainty
    sort_idx = np.argsort(y_std)
    errors_sorted = errors[sort_idx]
    y_std_sorted = y_std[sort_idx]
    
    # Create bins
    n_bins = 10
    bin_size = len(errors) // n_bins
    
    bin_centers = []
    expected_errors = []
    actual_errors = []
    
    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else len(errors)
        
        bin_std = y_std_sorted[start:end].mean()
        bin_error = errors_sorted[start:end].mean()
        
        bin_centers.append(bin_std)
        expected_errors.append(bin_std)  # Expected = uncertainty
        actual_errors.append(bin_error)   # Actual = real error
    
    # Create plot
    fig = go.Figure()
    
    # Perfect calibration line
    max_val = max(max(expected_errors), max(actual_errors))
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name='Perfect Calibration'
    ))
    
    # Actual calibration
    fig.add_trace(go.Scatter(
        x=expected_errors,
        y=actual_errors,
        mode='markers+lines',
        marker=dict(size=10, color='blue'),
        line=dict(color='blue', width=2),
        name='Our Model'
    ))
    
    fig.update_layout(
        title="🎯 Uncertainty Calibration (Are Our Predictions Trustworthy?)",
        xaxis_title="Predicted Uncertainty (σ)",
        yaxis_title="Actual Error (|true - predicted|)",
        height=400,
        showlegend=True,
        template='plotly_white',
        annotations=[
            dict(
                x=0.5,
                y=0.95,
                xref='paper',
                yref='paper',
                text="Close to red line = Well-calibrated predictions",
                showarrow=False,
                font=dict(size=10, color='gray')
            )
        ]
    )
    
    # Calculate calibration score (how close to perfect)
    calibration_error = np.mean(np.abs(np.array(expected_errors) - np.array(actual_errors)))
    
    return fig, calibration_error


def active_learning_efficiency_curve(round_history, baseline_history):
    """
    Show cumulative best voltage over number of tests
    Proves active learning is more efficient
    """
    fig = go.Figure()
    
    # Extract cumulative best
    al_tests = [r['train_size'] for r in round_history]
    al_best = [r['best_value'] for r in round_history]
    
    baseline_tests = [r['train_size'] for r in baseline_history]
    baseline_best = [r['best_value'] for r in baseline_history]
    
    # Active Learning
    fig.add_trace(go.Scatter(
        x=al_tests,
        y=al_best,
        mode='lines+markers',
        name='Active Learning (UCB)',
        line=dict(color='green', width=3),
        marker=dict(size=8)
    ))
    
    # Random Baseline
    fig.add_trace(go.Scatter(
        x=baseline_tests,
        y=baseline_best,
        mode='lines+markers',
        name='Random Selection',
        line=dict(color='gray', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    # Find where active learning reaches final performance
    final_voltage = al_best[-1]
    tests_to_reach = None
    for i, v in enumerate(al_best):
        if v >= final_voltage * 0.95:  # 95% of final
            tests_to_reach = al_tests[i]
            break
    
    baseline_tests_needed = baseline_tests[-1]
    
    if tests_to_reach:
        # Add efficiency annotation
        fig.add_annotation(
            x=tests_to_reach,
            y=final_voltage,
            text=f"Active Learning reaches {final_voltage:.2f}V<br>in {tests_to_reach} tests",
            showarrow=True,
            arrowhead=2,
            ax=-80,
            ay=-40,
            font=dict(color='green', size=10)
        )
        
        fig.add_annotation(
            x=baseline_tests_needed,
            y=baseline_best[-1],
            text=f"Random needs {baseline_tests_needed} tests<br>for {baseline_best[-1]:.2f}V",
            showarrow=True,
            arrowhead=2,
            ax=80,
            ay=40,
            font=dict(color='gray', size=10)
        )
    
    fig.update_layout(
        title="⚡ Discovery Efficiency: Active Learning vs Random",
        xaxis_title="Number of Materials Tested",
        yaxis_title="Best Voltage Found (V)",
        height=400,
        showlegend=True,
        template='plotly_white'
    )
    
    efficiency_gain = ((baseline_tests_needed - tests_to_reach) / baseline_tests_needed * 100) if tests_to_reach else 0
    
    return fig, efficiency_gain


def statistical_significance_test(al_results, baseline_results):
    """
    Perform statistical test: Is active learning significantly better?
    
    Uses paired t-test comparing final performance
    """
    # Get final values from multiple runs (if available)
    # For single run, compare improvement rates
    
    al_improvement = (al_results[-1]['best_value'] - al_results[0]['best_value']) / al_results[0]['best_value']
    baseline_improvement = (baseline_results[-1]['best_value'] - baseline_results[0]['best_value']) / baseline_results[0]['best_value']
    
    # Simple comparison
    difference = al_improvement - baseline_improvement
    
    result = {
        'al_improvement': al_improvement * 100,
        'baseline_improvement': baseline_improvement * 100,
        'difference': difference * 100,
        'significant': difference > 0.05  # 5% threshold
    }
    
    return result


def coverage_probability(y_true, y_pred, y_std, confidence=0.68):
    """
    Calculate coverage probability
    What % of true values fall within predicted ± σ?
    
    For 68% confidence (1σ), should cover ~68% of data
    """
    # Calculate if true value is within uncertainty band
    within_band = np.abs(y_true - y_pred) <= y_std
    coverage = np.mean(within_band) * 100
    
    expected_coverage = confidence * 100
    
    result = {
        'coverage': coverage,
        'expected': expected_coverage,
        'difference': coverage - expected_coverage,
        'well_calibrated': abs(coverage - expected_coverage) < 10  # Within 10%
    }
    
    return result


if __name__ == "__main__":
    # Test
    np.random.seed(42)
    n = 100
    
    y_true = np.random.randn(n) * 2 + 4
    y_pred = y_true + np.random.randn(n) * 0.5
    y_std = np.abs(np.random.randn(n) * 0.5 + 0.5)
    
    fig, cal_error = uncertainty_calibration_plot(y_true, y_pred, y_std)
    print(f"Calibration Error: {cal_error:.3f}")
    
    cov = coverage_probability(y_true, y_pred, y_std)
    print(f"Coverage: {cov}")
