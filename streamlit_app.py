"""
Life Expectancy Prediction Dashboard
=====================================
A Streamlit application that predicts life expectancy based on vaccine coverage
and provides comprehensive data visualizations.

Features:
- Multiple user flows: Visualization Only, Prediction Only, or Both
- Loads pre-trained model using joblib
- Interactive charts and visualizations
- Data upload capability
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import tempfile
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="VaccineLife | Life Expectancy Predictor",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
<style>
    /* Main app styling */
    .main {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0f0f23 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e4a 0%, #0d0d2b 100%);
        border-right: 1px solid #3d3d8f;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00d4ff !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1a4e 0%, #2d2d6e 100%);
        border: 1px solid #4d4d9f;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.1);
    }
    
    [data-testid="stMetricValue"] {
        color: #00ff88 !important;
        font-size: 2rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #a0a0ff !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #0088ff 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 30px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.5);
    }
    
    /* Info boxes */
    .stAlert {
        background: linear-gradient(135deg, #1a3a4e 0%, #0d2a3b 100%);
        border: 1px solid #00d4ff;
        border-radius: 10px;
    }
    
    /* Prediction result styling */
    .prediction-result {
        background: linear-gradient(135deg, #004d40 0%, #00695c 100%);
        border: 2px solid #00ff88;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0, 255, 136, 0.2);
    }
    
    .prediction-value {
        font-size: 4rem;
        font-weight: bold;
        color: #00ff88;
        text-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a4e;
        border-radius: 8px 8px 0 0;
        border: 1px solid #4d4d9f;
        color: #a0a0ff;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00d4ff 0%, #0088ff 100%);
        color: white !important;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00d4ff, transparent);
        margin: 30px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# VACCINE INFORMATION
# ============================================================================
VACCINE_INFO = {
    'BCG': 'Bacillus Calmette-Guérin (Tuberculosis)',
    'DTP1': 'Diphtheria-Tetanus-Pertussis (1st dose)',
    'DTP3': 'Diphtheria-Tetanus-Pertussis (3rd dose)',
    'HEPB3': 'Hepatitis B (3rd dose)',
    'HEPBB': 'Hepatitis B (birth dose)',
    'HIB3': 'Haemophilus influenzae type b (3rd dose)',
    'IPV1': 'Inactivated Polio Vaccine (1st dose)',
    'IPV2': 'Inactivated Polio Vaccine (2nd dose)',
    'MCV1': 'Measles-containing Vaccine (1st dose)',
    'MCV2': 'Measles-containing Vaccine (2nd dose)',
    'MENGA': 'Meningococcal A conjugate vaccine',
    'PCV3': 'Pneumococcal Conjugate Vaccine (3rd dose)',
    'POL3': 'Polio (3rd dose)',
    'RCV1': 'Rubella-containing Vaccine (1st dose)',
    'ROTAC': 'Rotavirus (completed series)',
    'YFV': 'Yellow Fever Vaccine'
}

VACCINE_COLS = list(VACCINE_INFO.keys())

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'imputer' not in st.session_state:
    st.session_state.imputer = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None

# ============================================================================
# DATA LOADING AND PROCESSING FUNCTIONS
# ============================================================================
def load_data(uploaded_file):
    """Load data from uploaded file"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.txt'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV, TXT, or Excel file.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def preprocess_data(df):
    """Preprocess the dataset for visualization"""
    df_processed = df.copy()
    
    for col in VACCINE_COLS:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    if 'life_expectancy' in df_processed.columns:
        df_processed['life_expectancy'] = pd.to_numeric(df_processed['life_expectancy'], errors='coerce')
    
    return df_processed


def load_model_artifacts(model_file, scaler_file=None, imputer_file=None, feature_names_file=None):
    """Load pre-trained model and preprocessing artifacts using joblib"""
    try:
        model = joblib.load(model_file)
        st.session_state.model = model
        
        if scaler_file is not None:
            st.session_state.scaler = joblib.load(scaler_file)
        
        if imputer_file is not None:
            st.session_state.imputer = joblib.load(imputer_file)
        
        if feature_names_file is not None:
            st.session_state.feature_names = joblib.load(feature_names_file)
        
        st.session_state.model_loaded = True
        return True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return False


def predict_life_expectancy(input_values):
    """Make a prediction for given vaccine coverages"""
    try:
        input_df = pd.DataFrame([input_values])
        
        if st.session_state.feature_names is not None:
            for feature in st.session_state.feature_names:
                if feature not in input_df.columns:
                    input_df[feature] = np.nan
            input_df = input_df[st.session_state.feature_names]
        
        if st.session_state.imputer is not None:
            input_imputed = st.session_state.imputer.transform(input_df)
        else:
            input_imputed = input_df.fillna(input_df.median()).values
        
        if st.session_state.scaler is not None:
            input_scaled = st.session_state.scaler.transform(input_imputed)
        else:
            input_scaled = input_imputed
        
        prediction = st.session_state.model.predict(input_scaled)[0]
        return prediction
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def create_global_overview(df):
    """Create global overview visualizations"""
    st.subheader("🌍 Global Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df, x='life_expectancy',
            nbins=50,
            title='Distribution of Life Expectancy',
            labels={'life_expectancy': 'Life Expectancy (years)', 'count': 'Frequency'},
            color_discrete_sequence=['#00d4ff']
        )
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.1)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'year' in df.columns:
            yearly_avg = df.groupby('year')['life_expectancy'].mean().reset_index()
            fig = px.line(
                yearly_avg, x='year', y='life_expectancy',
                title='Average Life Expectancy Over Time',
                labels={'year': 'Year', 'life_expectancy': 'Life Expectancy (years)'},
                markers=True
            )
            fig.update_traces(line_color='#00ff88', marker_color='#00ff88')
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.1)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Year column not found for time series analysis.")


def create_vaccine_coverage_analysis(df):
    """Create vaccine coverage analysis visualizations"""
    st.subheader("💉 Vaccine Coverage Analysis")
    
    available_vaccines = [col for col in VACCINE_COLS if col in df.columns]
    
    if not available_vaccines:
        st.warning("No vaccine columns found in the dataset.")
        return
    
    avg_coverage = df[available_vaccines].mean().sort_values(ascending=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            x=avg_coverage.values,
            y=avg_coverage.index,
            orientation='h',
            title='Average Global Vaccine Coverage',
            labels={'x': 'Coverage (%)', 'y': 'Vaccine'},
            color=avg_coverage.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.1)',
            font=dict(color='white'),
            showlegend=False,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'year' in df.columns:
            yearly_coverage = df.groupby('year')[available_vaccines].mean()
            fig = go.Figure()
            
            colors = px.colors.qualitative.Set3
            for i, vaccine in enumerate(available_vaccines[:8]):
                fig.add_trace(go.Scatter(
                    x=yearly_coverage.index,
                    y=yearly_coverage[vaccine],
                    mode='lines',
                    name=vaccine,
                    line=dict(color=colors[i % len(colors)])
                ))
            
            fig.update_layout(
                title='Vaccine Coverage Trends Over Time',
                xaxis_title='Year',
                yaxis_title='Coverage (%)',
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.1)',
                font=dict(color='white'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.box(
                df[available_vaccines],
                title='Vaccine Coverage Distribution',
                labels={'variable': 'Vaccine', 'value': 'Coverage (%)'}
            )
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.1)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)


def create_correlation_analysis(df):
    """Create correlation analysis between vaccines and life expectancy"""
    st.subheader("📊 Correlation Analysis")
    
    available_vaccines = [col for col in VACCINE_COLS if col in df.columns]
    
    if not available_vaccines or 'life_expectancy' not in df.columns:
        st.warning("Required columns not found for correlation analysis.")
        return
    
    correlations = df[available_vaccines + ['life_expectancy']].corr()['life_expectancy'].drop('life_expectancy')
    correlations = correlations.sort_values(ascending=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        colors = ['#ff6b6b' if x < 0 else '#00ff88' for x in correlations.values]
        fig = px.bar(
            x=correlations.values,
            y=correlations.index,
            orientation='h',
            title='Correlation: Vaccines vs Life Expectancy',
            labels={'x': 'Correlation Coefficient', 'y': 'Vaccine'}
        )
        fig.update_traces(marker_color=colors)
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.1)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        vaccine_corr = df[available_vaccines].corr()
        fig = px.imshow(
            vaccine_corr,
            title='Vaccine Coverage Correlation Matrix',
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.1)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)


def create_country_analysis(df):
    """Create country-level analysis"""
    st.subheader("🗺️ Country Analysis")
    
    if 'country' not in df.columns:
        st.warning("Country column not found in the dataset.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        countries = sorted(df['country'].unique())
        selected_country = st.selectbox("Select a Country", countries, key="country_analysis_select")
        
        if 'year' in df.columns:
            country_data = df[df['country'] == selected_country].sort_values('year')
            
            fig = px.line(
                country_data, x='year', y='life_expectancy',
                title=f'Life Expectancy in {selected_country}',
                markers=True
            )
            fig.update_traces(line_color='#00d4ff', marker_color='#00ff88')
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.1)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            country_data = df[df['country'] == selected_country]
            st.metric("Average Life Expectancy", f"{country_data['life_expectancy'].mean():.1f} years")
    
    with col2:
        if 'life_expectancy' in df.columns:
            country_avg = df.groupby('country')['life_expectancy'].mean().sort_values(ascending=False)
            
            top_10 = country_avg.head(10)
            bottom_10 = country_avg.tail(10).sort_values(ascending=True)
            
            fig = make_subplots(rows=1, cols=2, subplot_titles=('Top 10 Countries', 'Bottom 10 Countries'))
            
            fig.add_trace(
                go.Bar(y=top_10.index, x=top_10.values, orientation='h', marker_color='#00ff88', name='Top 10'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(y=bottom_10.index, x=bottom_10.values, orientation='h', marker_color='#ff6b6b', name='Bottom 10'),
                row=1, col=2
            )
            
            fig.update_layout(
                title='Countries by Average Life Expectancy',
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0.1)',
                font=dict(color='white'),
                showlegend=False,
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)


def create_scatter_analysis(df):
    """Create scatter plot analysis"""
    st.subheader("🔍 Scatter Plot Analysis")
    
    available_vaccines = [col for col in VACCINE_COLS if col in df.columns]
    
    if not available_vaccines or 'life_expectancy' not in df.columns:
        st.warning("Required columns not found for scatter analysis.")
        return
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_vaccine = st.selectbox("Select Vaccine", available_vaccines, key="scatter_vaccine_select")
        if selected_vaccine in VACCINE_INFO:
            st.info(f"**{selected_vaccine}**: {VACCINE_INFO[selected_vaccine]}")
    
    with col2:
        fig = px.scatter(
            df,
            x=selected_vaccine,
            y='life_expectancy',
            color='country' if 'country' in df.columns and df['country'].nunique() <= 20 else None,
            hover_data=['country', 'year'] if 'country' in df.columns and 'year' in df.columns else None,
            title=f'{selected_vaccine} Coverage vs Life Expectancy',
            labels={selected_vaccine: f'{selected_vaccine} Coverage (%)', 'life_expectancy': 'Life Expectancy (years)'},
            trendline='ols'
        )
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0.1)',
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)


def create_data_summary(df):
    """Create data summary statistics"""
    st.subheader("📋 Data Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        if 'country' in df.columns:
            st.metric("Countries", f"{df['country'].nunique():,}")
        else:
            st.metric("Columns", f"{len(df.columns):,}")
    with col3:
        if 'year' in df.columns:
            st.metric("Year Range", f"{int(df['year'].min())} - {int(df['year'].max())}")
        else:
            st.metric("Features", f"{len(df.columns):,}")
    with col4:
        if 'life_expectancy' in df.columns:
            st.metric("Avg Life Expectancy", f"{df['life_expectancy'].mean():.1f} yrs")
    
    with st.expander("📊 View Data Preview"):
        st.dataframe(df.head(20), use_container_width=True)
    
    with st.expander("📈 View Statistics"):
        st.dataframe(df.describe(), use_container_width=True)


# ============================================================================
# PREDICTION INTERFACE
# ============================================================================
def create_prediction_interface():
    """Create the prediction interface"""
    st.subheader("🎯 Life Expectancy Prediction")
    
    if not st.session_state.model_loaded:
        st.warning("⚠️ Please load a trained model first using the sidebar.")
        return
    
    st.markdown("Enter vaccine coverage percentages (0-100) for each vaccine:")
    
    if st.session_state.feature_names is not None:
        features_to_show = st.session_state.feature_names
    else:
        features_to_show = VACCINE_COLS
    
    input_values = {}
    cols = st.columns(4)
    
    for i, vaccine in enumerate(features_to_show):
        with cols[i % 4]:
            tooltip = VACCINE_INFO.get(vaccine, vaccine)
            input_values[vaccine] = st.number_input(
                vaccine,
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=1.0,
                help=tooltip,
                key=f"input_{vaccine}"
            )
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📉 Set Low (30%)", use_container_width=True):
            for vaccine in features_to_show:
                st.session_state[f"input_{vaccine}"] = 30.0
            st.rerun()
    with col2:
        if st.button("📊 Set Medium (60%)", use_container_width=True):
            for vaccine in features_to_show:
                st.session_state[f"input_{vaccine}"] = 60.0
            st.rerun()
    with col3:
        if st.button("📈 Set High (90%)", use_container_width=True):
            for vaccine in features_to_show:
                st.session_state[f"input_{vaccine}"] = 90.0
            st.rerun()
    
    st.markdown("---")
    
    if st.button("🔮 Predict Life Expectancy", type="primary", use_container_width=True):
        with st.spinner("Calculating prediction..."):
            prediction = predict_life_expectancy(input_values)
            
            if prediction is not None:
                st.markdown(f"""
                <div class="prediction-result">
                    <h3 style="color: white; margin-bottom: 10px;">Predicted Life Expectancy</h3>
                    <div class="prediction-value">{prediction:.1f}</div>
                    <p style="color: #a0ffcc; font-size: 1.2rem;">years</p>
                </div>
                """, unsafe_allow_html=True)
                
                if prediction >= 75:
                    st.success("🌟 High life expectancy - associated with well-developed healthcare systems.")
                elif prediction >= 65:
                    st.info("📊 Moderate life expectancy - improvements in coverage could increase this.")
                else:
                    st.warning("⚠️ Lower life expectancy - significant improvements in coverage may help.")


# ============================================================================
# SIDEBAR
# ============================================================================
def create_sidebar():
    """Create the sidebar with navigation and file uploads"""
    
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="color: #00d4ff; font-size: 1.8rem;">💉 VaccineLife</h1>
        <p style="color: #a0a0ff;">Life Expectancy Predictor</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Flow selection
    st.sidebar.subheader("🎯 Select Your Flow")
    flow = st.sidebar.radio(
        "What would you like to do?",
        options=["📊 Visualization Only", "🔮 Prediction Only", "📊🔮 Both"],
        index=2,
        help="Choose whether you want to visualize data, make predictions, or both"
    )
    
    st.sidebar.markdown("---")
    
    # Data upload section
    if flow in ["📊 Visualization Only", "📊🔮 Both"]:
        st.sidebar.subheader("📁 Data Upload")
        uploaded_data = st.sidebar.file_uploader(
            "Upload Dataset (CSV/TXT/Excel)",
            type=['csv', 'txt', 'xlsx', 'xls'],
            help="Upload your vaccine coverage dataset"
        )
        
        if uploaded_data is not None:
            df = load_data(uploaded_data)
            if df is not None:
                st.session_state.df = preprocess_data(df)
                st.session_state.data_loaded = True
                st.sidebar.success(f"✅ Data loaded: {len(df)} records")
    
    # Model upload section
    if flow in ["🔮 Prediction Only", "📊🔮 Both"]:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🤖 Model Upload")
        
        model_file = st.sidebar.file_uploader(
            "Upload Trained Model (.joblib/.pkl)",
            type=['joblib', 'pkl'],
            help="Upload your pre-trained model"
        )
        
        scaler_file = st.sidebar.file_uploader(
            "Upload Scaler (optional)",
            type=['joblib', 'pkl'],
            help="Upload the scaler used during training"
        )
        
        imputer_file = st.sidebar.file_uploader(
            "Upload Imputer (optional)",
            type=['joblib', 'pkl'],
            help="Upload the imputer used during training"
        )
        
        feature_names_file = st.sidebar.file_uploader(
            "Upload Feature Names (optional)",
            type=['joblib', 'pkl'],
            help="Upload the list of feature names"
        )
        
        if model_file is not None:
            if st.sidebar.button("Load Model", type="primary"):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as tmp:
                    tmp.write(model_file.getvalue())
                    model_path = tmp.name
                
                scaler_path = None
                if scaler_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as tmp:
                        tmp.write(scaler_file.getvalue())
                        scaler_path = tmp.name
                
                imputer_path = None
                if imputer_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as tmp:
                        tmp.write(imputer_file.getvalue())
                        imputer_path = tmp.name
                
                feature_path = None
                if feature_names_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as tmp:
                        tmp.write(feature_names_file.getvalue())
                        feature_path = tmp.name
                
                if load_model_artifacts(model_path, scaler_path, imputer_path, feature_path):
                    st.sidebar.success("✅ Model loaded successfully!")
                
                os.unlink(model_path)
                if scaler_path:
                    os.unlink(scaler_path)
                if imputer_path:
                    os.unlink(imputer_path)
                if feature_path:
                    os.unlink(feature_path)
    
    # Status indicators
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Status")
    
    if st.session_state.data_loaded:
        st.sidebar.success("✅ Data: Loaded")
    else:
        st.sidebar.warning("⚠️ Data: Not loaded")
    
    if st.session_state.model_loaded:
        st.sidebar.success("✅ Model: Loaded")
    else:
        st.sidebar.warning("⚠️ Model: Not loaded")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="padding: 10px; font-size: 0.8rem; color: #a0a0ff;">
        <h4 style="color: #00d4ff;">About</h4>
        <p>Predicts life expectancy based on vaccine coverage using ML.</p>
    </div>
    """, unsafe_allow_html=True)
    
    return flow


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Main application logic"""
    
    flow = create_sidebar()
    
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1>💉 Life Expectancy Prediction Dashboard</h1>
        <p style="font-size: 1.2rem; color: #a0a0ff;">
            Explore the relationship between vaccine coverage and life expectancy
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if flow == "📊 Visualization Only":
        if st.session_state.data_loaded and st.session_state.df is not None:
            df = st.session_state.df
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📋 Summary", "🌍 Global Overview", "💉 Vaccine Analysis",
                "📊 Correlations", "🗺️ Country Analysis"
            ])
            
            with tab1:
                create_data_summary(df)
            with tab2:
                create_global_overview(df)
            with tab3:
                create_vaccine_coverage_analysis(df)
                st.markdown("---")
                create_scatter_analysis(df)
            with tab4:
                create_correlation_analysis(df)
            with tab5:
                create_country_analysis(df)
        else:
            st.info("👆 Please upload your dataset using the sidebar to view visualizations.")
    
    elif flow == "🔮 Prediction Only":
        create_prediction_interface()
    
    else:  # Both
        tab_pred, tab_summary, tab_global, tab_vaccine, tab_corr, tab_country = st.tabs([
            "🔮 Prediction", "📋 Summary", "🌍 Global Overview",
            "💉 Vaccine Analysis", "📊 Correlations", "🗺️ Country Analysis"
        ])
        
        with tab_pred:
            create_prediction_interface()
        
        if st.session_state.data_loaded and st.session_state.df is not None:
            df = st.session_state.df
            
            with tab_summary:
                create_data_summary(df)
            with tab_global:
                create_global_overview(df)
            with tab_vaccine:
                create_vaccine_coverage_analysis(df)
                st.markdown("---")
                create_scatter_analysis(df)
            with tab_corr:
                create_correlation_analysis(df)
            with tab_country:
                create_country_analysis(df)
        else:
            for tab in [tab_summary, tab_global, tab_vaccine, tab_corr, tab_country]:
                with tab:
                    st.info("👆 Please upload your dataset using the sidebar to view visualizations.")


if __name__ == "__main__":
    main()
