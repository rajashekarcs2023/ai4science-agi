"""
Interpretability tools for materials discovery
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple


def get_feature_importance(model, feature_names: List[str], top_n: int = 15) -> pd.DataFrame:
    """
    Extract feature importance from trained model
    
    Args:
        model: Trained surrogate model
        feature_names: List of feature names
        top_n: Number of top features to return
    
    Returns:
        DataFrame with feature importance scores
    """
    try:
        # Get importance from LightGBM model
        if hasattr(model, 'model_mean') and hasattr(model.model_mean, 'feature_importances_'):
            importances = model.model_mean.feature_importances_
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            return None
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Sort and get top N
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        # Normalize to percentage
        importance_df['importance_pct'] = (importance_df['importance'] / importance_df['importance'].sum()) * 100
        
        return importance_df
        
    except Exception as e:
        print(f"Could not extract feature importance: {e}")
        return None


def plot_feature_importance(importance_df: pd.DataFrame) -> go.Figure:
    """
    Plot feature importance as horizontal bar chart
    """
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=importance_df['feature'][::-1],
        x=importance_df['importance_pct'][::-1],
        orientation='h',
        marker=dict(
            color=importance_df['importance_pct'][::-1],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Importance %")
        ),
        text=importance_df['importance_pct'][::-1].round(1).astype(str) + '%',
        textposition='auto',
    ))
    
    fig.update_layout(
        title="🔍 What Features Drive High Voltage?",
        xaxis_title="Importance (%)",
        yaxis_title="Feature",
        height=500,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig


def explain_material(
    formula: str,
    predicted_voltage: float,
    uncertainty: float,
    feature_values: Dict[str, float],
    top_features: List[str]
) -> str:
    """
    Generate natural language explanation for a material prediction
    
    Args:
        formula: Chemical formula
        predicted_voltage: Predicted voltage
        uncertainty: Prediction uncertainty
        feature_values: Dictionary of feature name -> value
        top_features: List of most important features
    
    Returns:
        Natural language explanation
    """
    explanation = f"## 🔬 Scientific Analysis: {formula}\n\n"
    
    # Voltage assessment
    if predicted_voltage >= 4.0:
        voltage_assessment = "**Excellent** - Competitive with commercial cathodes (LiCoO2: 3.7V, LiFePO4: 3.45V)"
    elif predicted_voltage >= 3.5:
        voltage_assessment = "**Good** - Suitable for battery applications"
    elif predicted_voltage >= 2.5:
        voltage_assessment = "**Moderate** - May be useful for specific applications"
    else:
        voltage_assessment = "**Low** - Below typical battery cathode range"
    
    explanation += f"### ⚡ Predicted Voltage: {predicted_voltage:.2f}V ± {uncertainty:.2f}V\n"
    explanation += f"{voltage_assessment}\n\n"
    
    # Confidence assessment
    if uncertainty < 0.5:
        confidence = "**High confidence** - Model is certain about this prediction"
    elif uncertainty < 1.0:
        confidence = "**Moderate confidence** - Reasonable prediction reliability"
    else:
        confidence = "**Lower confidence** - More experimental validation needed"
    
    explanation += f"### 📊 Prediction Confidence:\n{confidence}\n\n"
    
    # Feature analysis
    explanation += "### 🧪 Key Chemical Properties:\n"
    
    # Look for key features in the top features
    key_feature_groups = {
        'electronegativity': ['MagpieData mean Electronegativity', 'MagpieData range Electronegativity'],
        'atomic_number': ['MagpieData mean Number', 'MagpieData maximum Number'],
        'ionic_radius': ['MagpieData mean MendeleevNumber', 'MagpieData range MendeleevNumber'],
        'valence': ['MagpieData mean NValence', 'MagpieData mean NsValence']
    }
    
    for group, features in key_feature_groups.items():
        found_features = [f for f in features if f in feature_values]
        if found_features:
            feature = found_features[0]
            value = feature_values[feature]
            
            if 'Electronegativity' in feature:
                explanation += f"- **Electronegativity**: {value:.3f} - "
                if value > 2.0:
                    explanation += "High electronegative elements may enhance ionic character\n"
                else:
                    explanation += "Lower electronegativity\n"
            
            elif 'Number' in feature and 'Valence' not in feature:
                explanation += f"- **Average Atomic Number**: {value:.1f} - "
                if value > 20:
                    explanation += "Heavier transition metals present\n"
                else:
                    explanation += "Lighter elements\n"
    
    # Chemistry interpretation based on formula
    explanation += "\n### 🔬 Chemical Composition:\n"
    
    if 'Ni' in formula:
        explanation += "- **Nickel (Ni)**: Known for high capacity in Li-ion cathodes\n"
    if 'Co' in formula:
        explanation += "- **Cobalt (Co)**: Used in high-performance LiCoO2 cathodes\n"
    if 'Mn' in formula:
        explanation += "- **Manganese (Mn)**: Provides stability, used in LiMn2O4 spinel\n"
    if 'Fe' in formula:
        explanation += "- **Iron (Fe)**: Low-cost, used in LiFePO4 (olivine structure)\n"
    if 'Ti' in formula:
        explanation += "- **Titanium (Ti)**: Dopant for structural stability\n"
    if 'Al' in formula:
        explanation += "- **Aluminum (Al)**: Common dopant for improving cycle life\n"
    
    # Synthesis recommendation
    explanation += "\n### 🧑‍🔬 Recommended Next Steps:\n"
    
    if predicted_voltage >= 3.5 and uncertainty < 1.0:
        explanation += "1. **Priority synthesis** - High voltage with good confidence\n"
        explanation += "2. Perform DFT calculations to validate structure\n"
        explanation += "3. Check phase diagram for stability\n"
        explanation += "4. Synthesize via solid-state reaction\n"
    elif predicted_voltage >= 2.5:
        explanation += "1. Run DFT calculations first to validate prediction\n"
        explanation += "2. Check Materials Project for similar structures\n"
        explanation += "3. Consider as dopant or secondary phase\n"
    else:
        explanation += "1. Likely not suitable for high-performance cathode\n"
        explanation += "2. May be useful for other applications (anode, electrolyte)\n"
        explanation += "3. DFT validation recommended before synthesis\n"
    
    return explanation


def compare_to_known_materials(predicted_voltage: float) -> str:
    """
    Compare predicted voltage to known battery materials
    """
    known_materials = {
        'LiCoO2': 3.7,
        'LiFePO4': 3.45,
        'LiMn2O4': 4.1,
        'LiNi0.8Co0.15Al0.05O2 (NCA)': 3.6,
        'LiNi0.33Mn0.33Co0.33O2 (NMC)': 3.7,
        'LiNi0.5Mn1.5O4': 4.7
    }
    
    comparison = "\n### 📚 Comparison to Known Cathodes:\n"
    
    for material, voltage in sorted(known_materials.items(), key=lambda x: abs(x[1] - predicted_voltage)):
        diff = predicted_voltage - voltage
        if abs(diff) < 0.5:
            comparison += f"- Similar to **{material}** ({voltage}V, Δ={diff:+.2f}V)\n"
            break
    
    return comparison


if __name__ == "__main__":
    print("Interpretability module loaded successfully")
