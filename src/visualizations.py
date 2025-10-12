"""
Visualization functions for autonomous discovery results
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Optional


def plot_discovery_curve(agent_history: List[Dict], 
                         baseline_history: List[Dict],
                         title: str = "Discovery Progress: Agent vs Random Baseline") -> go.Figure:
    """
    Plot discovery curve showing best value found over rounds
    
    Args:
        agent_history: Agent round history
        baseline_history: Baseline round history
        title: Plot title
        
    Returns:
        Plotly figure
    """
    # Extract data
    agent_rounds = [0] + [r['round'] for r in agent_history]
    agent_best = [agent_history[0]['best_value']] + [r['best_value'] for r in agent_history]
    
    baseline_rounds = [0] + [r['round'] for r in baseline_history]
    baseline_best = [baseline_history[0]['best_value']] + [r['best_value'] for r in baseline_history]
    
    # Create figure
    fig = go.Figure()
    
    # Agent line
    fig.add_trace(go.Scatter(
        x=agent_rounds,
        y=agent_best,
        mode='lines+markers',
        name='Autonomous Agent',
        line=dict(color='#00CC96', width=3),
        marker=dict(size=8),
        hovertemplate='Round: %{x}<br>Best Voltage: %{y:.3f}V<extra></extra>'
    ))
    
    # Baseline line
    fig.add_trace(go.Scatter(
        x=baseline_rounds,
        y=baseline_best,
        mode='lines+markers',
        name='Random Baseline',
        line=dict(color='#EF553B', width=2, dash='dash'),
        marker=dict(size=6),
        hovertemplate='Round: %{x}<br>Best Voltage: %{y:.3f}V<extra></extra>'
    ))
    
    # Styling
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="Discovery Round",
        yaxis_title="Best Voltage Found (V)",
        hovermode='x unified',
        template='plotly_white',
        height=400,
        legend=dict(x=0.7, y=0.15, bgcolor='rgba(255,255,255,0.8)')
    )
    
    return fig


def plot_uncertainty_collapse(agent_history: List[Dict],
                              title: str = "Uncertainty Reduction Over Time") -> go.Figure:
    """
    Plot how model uncertainty decreases as more data is collected
    
    Args:
        agent_history: Agent round history
        title: Plot title
        
    Returns:
        Plotly figure
    """
    rounds = [r['round'] for r in agent_history]
    uncertainties = [r['mean_uncertainty'] for r in agent_history]
    train_sizes = [r['train_size'] for r in agent_history]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=rounds,
        y=uncertainties,
        mode='lines+markers',
        name='Mean Uncertainty',
        line=dict(color='#AB63FA', width=3),
        marker=dict(size=8),
        fill='tozeroy',
        fillcolor='rgba(171, 99, 250, 0.2)',
        hovertemplate='Round: %{x}<br>Uncertainty: %{y:.3f}<br>Training Size: %{text}<extra></extra>',
        text=train_sizes
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="Discovery Round",
        yaxis_title="Mean Prediction Uncertainty (σ)",
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_pareto_front(df: pd.DataFrame,
                     objective1: str = 'voltage',
                     objective2: str = 'sustainability_score',
                     formula_col: str = 'formula',
                     highlight_tested: bool = True) -> go.Figure:
    """
    Plot Pareto front for multi-objective optimization
    
    Args:
        df: DataFrame with objectives
        objective1: First objective (to maximize)
        objective2: Second objective (to minimize)
        formula_col: Column with material formulas
        highlight_tested: Highlight tested materials
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Separate tested and untested
    if highlight_tested and 'was_tested' in df.columns:
        df_tested = df[df['was_tested']]
        df_untested = df[~df['was_tested']]
        
        # Untested points
        fig.add_trace(go.Scatter(
            x=df_untested[objective2],
            y=df_untested[objective1],
            mode='markers',
            name='Untested',
            marker=dict(size=6, color='lightgray', opacity=0.4),
            hovertemplate='%{text}<br>Voltage: %{y:.3f}V<br>Sustainability: %{x:.3f}<extra></extra>',
            text=df_untested[formula_col]
        ))
        
        # Tested points
        fig.add_trace(go.Scatter(
            x=df_tested[objective2],
            y=df_tested[objective1],
            mode='markers',
            name='Tested by Agent',
            marker=dict(size=10, color='#00CC96', symbol='diamond', 
                       line=dict(width=1, color='white')),
            hovertemplate='%{text}<br>Voltage: %{y:.3f}V<br>Sustainability: %{x:.3f}<extra></extra>',
            text=df_tested[formula_col]
        ))
    else:
        # All points
        fig.add_trace(go.Scatter(
            x=df[objective2],
            y=df[objective1],
            mode='markers',
            marker=dict(size=8, color=df[objective1], colorscale='Viridis',
                       showscale=True, colorbar=dict(title='Voltage')),
            hovertemplate='%{text}<br>Voltage: %{y:.3f}V<br>Sustainability: %{x:.3f}<extra></extra>',
            text=df[formula_col]
        ))
    
    fig.update_layout(
        title="Voltage vs Sustainability Trade-off (Pareto Front)",
        xaxis_title="Sustainability Score (lower = better)",
        yaxis_title="Voltage (V)",
        template='plotly_white',
        height=500,
        hovermode='closest'
    )
    
    # Add ideal region annotation
    fig.add_annotation(
        x=0.2, y=df[objective1].max() * 0.95,
        text="Ideal Region:<br>High voltage +<br>Low sustainability cost",
        showarrow=True,
        arrowhead=2,
        ax=-60, ay=-40,
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='#00CC96',
        borderwidth=2
    )
    
    return fig


def plot_acquisition_landscape(mu: np.ndarray, sigma: np.ndarray,
                               acquisition_scores: np.ndarray,
                               selected_indices: np.ndarray,
                               sample_size: int = 500) -> go.Figure:
    """
    Visualize acquisition function landscape
    
    Args:
        mu: Mean predictions
        sigma: Uncertainties
        acquisition_scores: Acquisition scores
        selected_indices: Indices of selected materials
        sample_size: Number of points to plot (for performance)
        
    Returns:
        Plotly figure
    """
    # Sample for visualization
    if len(mu) > sample_size:
        sample_idx = np.random.choice(len(mu), sample_size, replace=False)
    else:
        sample_idx = np.arange(len(mu))
    
    # Create figure with subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Mean Prediction vs Uncertainty", "Acquisition Score Distribution"),
        horizontal_spacing=0.12
    )
    
    # Plot 1: Mean vs Uncertainty
    fig.add_trace(
        go.Scatter(
            x=sigma[sample_idx],
            y=mu[sample_idx],
            mode='markers',
            marker=dict(size=5, color='lightblue', opacity=0.6),
            name='Candidates',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Highlight selected
    if len(selected_indices) > 0:
        fig.add_trace(
            go.Scatter(
                x=sigma[selected_indices],
                y=mu[selected_indices],
                mode='markers',
                marker=dict(size=12, color='red', symbol='star', 
                           line=dict(width=1, color='white')),
                name='Selected',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Plot 2: Acquisition score histogram
    fig.add_trace(
        go.Histogram(
            x=acquisition_scores[sample_idx],
            nbinsx=30,
            marker=dict(color='#AB63FA'),
            name='Acquisition Scores',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Add vertical line for threshold
    if len(selected_indices) > 0:
        threshold = np.min(acquisition_scores[selected_indices])
        fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                     row=1, col=2, annotation_text="Selection Threshold")
    
    fig.update_xaxes(title_text="Uncertainty (σ)", row=1, col=1)
    fig.update_yaxes(title_text="Mean Prediction (μ)", row=1, col=1)
    fig.update_xaxes(title_text="Acquisition Score", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    
    fig.update_layout(
        title="Acquisition Function Analysis",
        template='plotly_white',
        height=400
    )
    
    return fig


def create_leaderboard_table(df: pd.DataFrame, n_top: int = 10) -> go.Figure:
    """
    Create interactive leaderboard table
    
    Args:
        df: DataFrame with materials
        n_top: Number of top materials to show
        
    Returns:
        Plotly figure
    """
    top_df = df.nlargest(n_top, 'predicted_voltage')
    
    # Prepare table data
    formulas = top_df['formula'].values
    voltages = top_df['predicted_voltage'].values
    uncertainties = top_df['uncertainty'].values if 'uncertainty' in top_df.columns else [0] * len(top_df)
    tested = top_df['was_tested'].values if 'was_tested' in top_df.columns else [False] * len(top_df)
    
    # Format voltage with uncertainty
    voltage_str = [f"{v:.3f} ± {u:.3f}" for v, u in zip(voltages, uncertainties)]
    tested_str = ["✓" if t else "○" for t in tested]
    
    # Create table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Rank', 'Formula', 'Predicted Voltage (V)', 'Tested'],
            fill_color='#00CC96',
            font=dict(color='white', size=12),
            align='left'
        ),
        cells=dict(
            values=[
                list(range(1, len(top_df) + 1)),
                formulas,
                voltage_str,
                tested_str
            ],
            fill_color=[['white' if i % 2 == 0 else '#f0f0f0' for i in range(len(top_df))]]*4,
            font=dict(size=11),
            align='left',
            height=30
        )
    )])
    
    fig.update_layout(
        title=f"Top {n_top} Materials by Predicted Voltage",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig


def plot_discovery_efficiency(agent_history: List[Dict],
                              baseline_history: List[Dict],
                              total_pool_size: int) -> go.Figure:
    """
    Plot cumulative materials tested vs performance
    
    Args:
        agent_history: Agent history
        baseline_history: Baseline history  
        total_pool_size: Total size of material pool
        
    Returns:
        Plotly figure
    """
    agent_tested = [r['train_size'] for r in agent_history]
    agent_best = [r['best_value'] for r in agent_history]
    agent_pct = [t/total_pool_size*100 for t in agent_tested]
    
    baseline_tested = [r['train_size'] for r in baseline_history]
    baseline_best = [r['best_value'] for r in baseline_history]
    baseline_pct = [t/total_pool_size*100 for t in baseline_tested]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=agent_pct,
        y=agent_best,
        mode='lines+markers',
        name='Autonomous Agent',
        line=dict(color='#00CC96', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=baseline_pct,
        y=baseline_best,
        mode='lines+markers',
        name='Random Baseline',
        line=dict(color='#EF553B', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="Discovery Efficiency: Best Value vs % Materials Tested",
        xaxis_title="Percentage of Materials Pool Tested (%)",
        yaxis_title="Best Voltage Found (V)",
        template='plotly_white',
        height=400,
        legend=dict(x=0.7, y=0.15)
    )
    
    return fig


if __name__ == "__main__":
    # Test visualizations with synthetic data
    import numpy as np
    
    np.random.seed(42)
    
    # Mock history data
    agent_history = [
        {'round': i, 'best_value': 3.0 + i*0.15 + np.random.rand()*0.1,
         'mean_uncertainty': 0.5 - i*0.04, 'train_size': 50 + i*5}
        for i in range(1, 11)
    ]
    agent_history.insert(0, agent_history[0].copy())
    agent_history[0]['round'] = 0
    agent_history[0]['best_value'] = 3.0
    
    baseline_history = [
        {'round': i, 'best_value': 3.0 + i*0.08 + np.random.rand()*0.1,
         'train_size': 50 + i*5}
        for i in range(1, 11)
    ]
    baseline_history.insert(0, baseline_history[0].copy())
    baseline_history[0]['round'] = 0
    baseline_history[0]['best_value'] = 3.0
    
    # Create plots
    fig1 = plot_discovery_curve(agent_history, baseline_history)
    fig2 = plot_uncertainty_collapse(agent_history)
    
    print("Visualizations created successfully!")
    print("Run in Streamlit to see interactive plots.")
