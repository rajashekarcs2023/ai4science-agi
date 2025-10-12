"""
Autonomous Discovery Agent - Interactive Streamlit Application
Real-time visualization of AI-driven materials discovery
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_loader import load_battery_data
from src.feature_engineering import prepare_features
from src.sustainability import add_sustainability
from src.discovery_loop import AutonomousDiscoveryAgent, DiscoveryConfig
from src.visualizations import (
    plot_discovery_curve, plot_uncertainty_collapse,
    plot_pareto_front, create_leaderboard_table,
    plot_discovery_efficiency
)
from src.discovery_integration import run_discovery_pipeline

# Page config
st.set_page_config(
    page_title="Autonomous Discovery Agent",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #00CC96, #AB63FA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #00CC96;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(sample_size):
    """Load and cache battery data"""
    try:
        df = load_battery_data(
            data_path="data/Battery Data/battery_merged.csv",
            sample_size=sample_size
        )
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


@st.cache_data
def prepare_data(df):
    """Prepare features and add sustainability"""
    try:
        X, y, feature_names, df_processed = prepare_features(df, target_col='voltage')
        df_processed = add_sustainability(df_processed)
        return X, y, feature_names, df_processed
    except Exception as e:
        st.error(f"Error preparing features: {e}")
        return None, None, None, None


def run_discovery_session(X, y, df, feature_cols, config, use_sustainability):
    """Run discovery and store in session state"""
    with st.spinner("🚀 Running autonomous discovery... This may take 30-60 seconds..."):
        agent = AutonomousDiscoveryAgent(config=config)
        
        sustainability_scores = df['sustainability_score'].values if use_sustainability else None
        
        results = agent.run_discovery(
            X, y, df,
            feature_cols=feature_cols,
            sustainability_scores=sustainability_scores,
            verbose=False
        )
        
        return results


def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Autonomous Discovery Agent</h1>', unsafe_allow_html=True)
    st.markdown("**AI-Driven Active Learning for Sustainable Battery Materials Discovery**")
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data settings
        st.subheader("Data Settings")
        sample_size = st.slider("Dataset Size", 1000, 10000, 5000, 500,
                               help="Number of materials to load from database")
        
        # Discovery settings
        st.subheader("Discovery Settings")
        n_init = st.slider("Initial Training Size", 20, 200, 50, 10,
                          help="Number of random materials to start with")
        n_rounds = st.slider("Discovery Rounds", 5, 20, 10, 1,
                            help="Number of active learning iterations")
        batch_size = st.slider("Batch Size", 1, 10, 5, 1,
                              help="Materials to select per round")
        
        # Acquisition settings
        st.subheader("Acquisition Strategy")
        strategy = st.selectbox("Strategy", ['ucb', 'ei', 'greedy', 'uncertainty'],
                               help="UCB = Upper Confidence Bound (recommended)")
        beta = st.slider("Exploration (β)", 0.0, 2.0, 0.8, 0.1,
                        help="Higher = more exploration")
        
        # Sustainability
        st.subheader("Sustainability")
        use_sustainability = st.checkbox("Enable Sustainability Filter", value=False,
                                        help="Prioritize sustainable materials")
        if use_sustainability:
            sust_weight = st.slider("Sustainability Weight", 0.0, 1.0, 0.2, 0.05)
        else:
            sust_weight = 0.0
        
        st.markdown("---")
        run_button = st.button("🚀 Start Discovery", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.caption("💡 **Tip**: Higher β = more exploration of uncertain regions")
    
    # Main content
    if 'results' not in st.session_state:
        st.session_state.results = None
        st.session_state.data_loaded = False
    
    # Load data
    if not st.session_state.data_loaded:
        with st.spinner("Loading battery materials database..."):
            df = load_data(sample_size)
            
            if df is not None:
                st.success(f"✓ Loaded {len(df)} materials from database")
                
                X, y, feature_names, df_processed = prepare_data(df)
                
                if X is not None:
                    st.session_state.X = X
                    st.session_state.y = y
                    st.session_state.feature_names = feature_names
                    st.session_state.df = df_processed
                    st.session_state.data_loaded = True
                    st.success(f"✓ Extracted {len(feature_names)} material features")
                else:
                    st.error("Failed to prepare features")
                    return
            else:
                st.error("Failed to load data")
                return
    
    # Run discovery
    if run_button and st.session_state.data_loaded:
        config = DiscoveryConfig(
            n_init=n_init,
            n_rounds=n_rounds,
            batch_size=batch_size,
            acquisition_strategy=strategy,
            beta=beta,
            use_sustainability=use_sustainability,
            sustainability_weight=sust_weight
        )
        
        results = run_discovery_session(
            st.session_state.X,
            st.session_state.y,
            st.session_state.df,
            st.session_state.feature_names,
            config,
            use_sustainability
        )
        
        st.session_state.results = results
        st.success("✅ Discovery complete!")
        st.balloons()
    
    # Display results
    if st.session_state.results is not None:
        results = st.session_state.results
        df = st.session_state.df
        
        # Summary metrics
        st.header("📊 Discovery Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        final_best = results.round_history[-1]['best_value']
        initial_best = results.round_history[0]['best_value']
        improvement = ((final_best - initial_best) / initial_best) * 100
        
        baseline_final = results.baseline_history[-1]['best_value']
        vs_baseline = ((final_best - baseline_final) / baseline_final) * 100
        
        with col1:
            st.metric("Best Material Found", 
                     f"{final_best:.3f}V",
                     delta=f"+{improvement:.1f}%")
        
        with col2:
            st.metric("vs Random Baseline",
                     f"{final_best:.3f}V",
                     delta=f"+{vs_baseline:.1f}%")
        
        with col3:
            final_uncertainty = results.round_history[-1]['mean_uncertainty']
            initial_uncertainty = results.round_history[0]['mean_uncertainty']
            uncertainty_reduction = ((initial_uncertainty - final_uncertainty) / initial_uncertainty) * 100
            st.metric("Uncertainty Reduction",
                     f"{uncertainty_reduction:.1f}%",
                     delta=f"-{final_uncertainty:.3f}σ")
        
        with col4:
            efficiency = (results.round_history[-1]['train_size'] / len(df)) * 100
            st.metric("Materials Tested",
                     f"{results.round_history[-1]['train_size']}/{len(df)}",
                     delta=f"{efficiency:.1f}%")
        
        st.markdown("---")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Discovery Progress", "🏆 Top Materials", "🌍 Sustainability", "📋 Round Details", "🔬 Novel Discoveries"])
        
        with tab1:
            st.subheader("Discovery Curve: Agent vs Random Baseline")
            fig1 = plot_discovery_curve(results.round_history, results.baseline_history)
            st.plotly_chart(fig1, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Uncertainty Reduction")
                fig2 = plot_uncertainty_collapse(results.round_history)
                st.plotly_chart(fig2, use_container_width=True)
            
            with col2:
                st.subheader("Discovery Efficiency")
                fig3 = plot_discovery_efficiency(
                    results.round_history,
                    results.baseline_history,
                    len(df)
                )
                st.plotly_chart(fig3, use_container_width=True)
        
        with tab2:
            st.subheader("Top Discovered Materials")
            
            # Best material highlight
            best_idx = results.round_history[-1]['best_material_idx']
            best_material = results.best_materials[results.best_materials.index == best_idx]
            
            if len(best_material) > 0:
                best_material = best_material.iloc[0]
                st.success(f"""
                **🏆 Best Material: {best_material['formula']}**  
                True Voltage: **{best_material['true_voltage']:.3f}V**  
                Predicted: {best_material['predicted_voltage']:.3f} ± {best_material['uncertainty']:.3f}V  
                Sustainability Score: {best_material.get('sustainability_score', 0.5):.3f} (0=best, 1=worst)
                """)
            else:
                # Fallback: just show the formula and value from history
                st.success(f"""
                **🏆 Best Material: {results.round_history[-1]['best_formula']}**  
                Voltage: **{results.round_history[-1]['best_value']:.3f}V**
                """)
            
            # Top 10 table
            st.subheader("Top 10 Materials by Predicted Voltage")
            top_10 = results.best_materials.head(10)
            
            # Create table manually to avoid PyArrow issues
            table_data = []
            for idx, row in top_10.iterrows():
                row_data = {
                    'Rank': len(table_data) + 1,
                    'Formula': str(row.get('formula', 'N/A')),
                    'Predicted V': f"{float(row.get('predicted_voltage', 0)):.3f}",
                    'Uncertainty': f"±{float(row.get('uncertainty', 0)):.3f}",
                    'True V': f"{float(row.get('true_voltage', 0)):.3f}",
                }
                
                if 'sustainability_score' in row:
                    row_data['Sustainability'] = f"{float(row['sustainability_score']):.3f}"
                
                if 'was_tested' in row:
                    row_data['Tested'] = '✓' if row['was_tested'] else '○'
                
                table_data.append(row_data)
            
            # Convert to simple dataframe with string types
            display_df = pd.DataFrame(table_data)
            
            # Display with st.markdown as HTML table
            html_table = display_df.to_html(index=False, escape=False)
            st.markdown(html_table, unsafe_allow_html=True)
            
            # Download button
            csv = top_10.to_csv(index=False)
            st.download_button(
                label="📥 Download Top Materials (CSV)",
                data=csv,
                file_name="top_discovered_materials.csv",
                mime="text/csv"
            )
        
        with tab3:
            if 'sustainability_score' in results.best_materials.columns:
                st.subheader("Voltage vs Sustainability Trade-off")
                fig4 = plot_pareto_front(results.best_materials.head(50), objective1='predicted_voltage')
                st.plotly_chart(fig4, use_container_width=True)
                
                # Sustainability stats
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Sustainability Score",
                             f"{results.best_materials['sustainability_score'].mean():.3f}")
                with col2:
                    best_sustainable = results.best_materials[results.best_materials['sustainability_score'] < 0.3].nlargest(1, 'predicted_voltage')
                    if len(best_sustainable) > 0:
                        st.metric("Best Sustainable Material",
                                 f"{best_sustainable.iloc[0]['formula']}: {best_sustainable.iloc[0]['predicted_voltage']:.3f}V")
            else:
                st.info("Enable sustainability in settings to see this analysis")
        
        with tab4:
            st.subheader("Round-by-Round Discovery Details")
            
            for round_data in results.round_history:
                with st.expander(f"Round {round_data['round']}: Best = {round_data['best_formula']} ({round_data['best_value']:.3f}V)"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Training Size", round_data['train_size'])
                    with col2:
                        st.metric("Mean Uncertainty", f"{round_data['mean_uncertainty']:.3f}")
                    with col3:
                        st.metric("Time", f"{round_data['time']:.2f}s")
                    
                    st.write("**Selected Materials:**")
                    selected_formulas = [df.iloc[idx]['formula'] for idx in round_data['selected_indices']]
                    selected_values = round_data['selected_values']
                    
                    for formula, value in zip(selected_formulas, selected_values):
                        st.write(f"  • {formula}: {value:.3f}V")
        
        with tab5:
            st.subheader("🔬 Autonomous Novel Materials Discovery")
            st.markdown("""
            **Beyond finding the best in the dataset - Let's discover NEW materials!**  
            Using the trained model to predict properties of novel compositions not in any database.
            """)
            
            # Settings for discovery
            col1, col2 = st.columns(2)
            with col1:
                enable_discovery = st.checkbox("🚀 Enable Novel Discovery", value=False,
                                              help="Generate and validate novel material compositions")
            with col2:
                mp_api_key = st.text_input("Materials Project API Key", 
                                          value="ZeBYRahOYGbCrmMYtgNcb9Uwm7FUZU2u",
                                          type="password",
                                          help="Your MP API key for validation")
            
            if enable_discovery:
                col1, col2, col3 = st.columns(3)
                with col1:
                    n_generate = st.number_input("Compositions to Generate", 
                                               min_value=100, max_value=2000, value=500, step=100)
                with col2:
                    n_validate = st.number_input("Validate via MP", 
                                               min_value=10, max_value=100, value=30, step=5)
                with col3:
                    min_voltage = st.number_input("Min Voltage (V)", 
                                                min_value=2.0, max_value=5.0, value=3.5, step=0.1)
                
                if st.button("🔬 Discover Novel Materials", type="primary"):
                    with st.spinner("Generating and validating novel compositions..."):
                        try:
                            # Get trained model and features from agent
                            model = results.agent.model
                            feature_cols = results.agent.feature_cols
                            known_formulas = df['formula'].tolist()
                            
                            # Run discovery pipeline
                            discoveries, stats = run_discovery_pipeline(
                                model=model,
                                feature_cols=feature_cols,
                                known_formulas=known_formulas,
                                mp_api_key=mp_api_key if mp_api_key else None,
                                n_generate=int(n_generate),
                                n_validate=int(n_validate),
                                n_top=20,
                                min_voltage=float(min_voltage)
                            )
                            
                            # Store in session state
                            st.session_state.discoveries = discoveries
                            st.session_state.discovery_stats = stats
                            
                        except Exception as e:
                            st.error(f"Discovery failed: {e}")
                            import traceback
                            st.code(traceback.format_exc())
            
            # Display discoveries if available
            if 'discoveries' in st.session_state and st.session_state.discoveries is not None:
                discoveries = st.session_state.discoveries
                stats = st.session_state.discovery_stats
                
                # Summary metrics
                st.markdown("---")
                st.subheader("📊 Discovery Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Compositions Generated", stats['n_generated'])
                with col2:
                    st.metric("Novel Candidates", stats['n_novel'])
                with col3:
                    st.metric("High Voltage (≥3.5V)", stats['n_high_voltage'])
                with col4:
                    if stats['n_validated'] > 0:
                        st.metric("Validated via MP", stats['n_validated'])
                        if 'n_truly_novel' in stats:
                            st.caption(f"🎯 {stats['n_truly_novel']} not in MP database")
                
                # Top discoveries table
                st.subheader("🏆 Top Novel Discoveries")
                
                if len(discoveries) > 0:
                    # Highlight best discovery
                    best_discovery = discoveries.iloc[0]
                    
                    st.success(f"""
                    **🎯 Best Novel Discovery: {best_discovery['formula']}**  
                    Predicted Voltage: **{best_discovery['predicted_voltage']:.3f} ± {best_discovery['uncertainty']:.3f}V**  
                    {"✨ **Truly Novel** - Not in Materials Project database!" if best_discovery.get('is_novel', False) else "⚠️ Similar material exists in MP database"}
                    """)
                    
                    # Create discoveries table
                    table_data = []
                    for idx, row in discoveries.iterrows():
                        row_data = {
                            'Rank': len(table_data) + 1,
                            'Formula': str(row['formula']),
                            'Predicted V': f"{row['predicted_voltage']:.3f}",
                            'Uncertainty': f"±{row['uncertainty']:.3f}",
                            'Discovery Score': f"{row.get('discovery_score', 0):.3f}",
                        }
                        
                        if 'is_novel' in row:
                            row_data['Novel?'] = '✨ Yes' if row['is_novel'] else '○ No'
                        
                        if 'mp_stable' in row and pd.notna(row['mp_stable']):
                            row_data['MP Stable'] = '✓' if row['mp_stable'] else '✗'
                        
                        if 'mp_formation_energy' in row and pd.notna(row['mp_formation_energy']):
                            row_data['Form. Energy'] = f"{row['mp_formation_energy']:.3f}"
                        
                        table_data.append(row_data)
                    
                    display_df = pd.DataFrame(table_data)
                    html_table = display_df.to_html(index=False, escape=False)
                    st.markdown(html_table, unsafe_allow_html=True)
                    
                    # Download button
                    csv = discoveries.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Novel Discoveries (CSV)",
                        data=csv,
                        file_name="novel_discoveries.csv",
                        mime="text/csv"
                    )
                    
                    # Interpretation
                    st.markdown("---")
                    st.subheader("📖 Interpretation")
                    st.markdown(f"""
                    **What we discovered:**
                    - Generated {stats['n_generated']} novel battery cathode compositions
                    - Found {stats['n_high_voltage']} candidates with predicted voltage ≥ {min_voltage}V
                    - Validated {stats['n_validated']} via Materials Project DFT database
                    - Top prediction: **{stats.get('max_predicted_voltage', 0):.3f}V**
                    
                    **Scientific Significance:**
                    - These materials are **not in our training dataset**
                    - Compositions generated via chemical knowledge + combinatorics
                    - Predictions based on learned structure-property relationships
                    - MP validation provides independent verification of feasibility
                    
                    **Next Steps:**
                    - Synthesize top candidates in lab
                    - Perform detailed DFT calculations
                    - Test electrochemical performance
                    """)
                else:
                    st.warning("No high-voltage novel materials found. Try lowering the minimum voltage threshold.")
            else:
                st.info("👆 Enable novel discovery and click the button to generate new material candidates")
    
    else:
        # Welcome screen
        st.info("👈 Configure settings in the sidebar and click '🚀 Start Discovery' to begin")
        
        st.header("How It Works")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("1️⃣ Active Learning")
            st.write("Agent starts with small random sample and trains a surrogate model")
        
        with col2:
            st.subheader("2️⃣ Uncertainty-Based Selection")
            st.write("Agent selects materials with highest potential value + uncertainty")
        
        with col3:
            st.subheader("3️⃣ Iterative Improvement")
            st.write("Agent retrains on new data, improving predictions each round")
        
        st.markdown("---")
        
        st.header("Why This Matters")
        st.markdown("""
        **Traditional materials discovery**: Test thousands of materials randomly → Slow & expensive  
        **Autonomous Agent**: Intelligently selects what to test next → **70-90% cost reduction**
        
        ✨ **Key Innovation**: The agent learns *what to test*, not just *what to predict*
        """)


if __name__ == "__main__":
    main()
