"""
Life Expectancy Prediction Dashboard
=====================================
A Streamlit application that predicts life expectancy based on vaccine coverage
and provides comprehensive data visualizations.

REVISED VERSION - Matches exact preprocessing pipeline from WQD7001_GRP11_Modelling.ipynb

Features:
- Loads pre-trained models from models/ directory
- Interactive charts and visualizations
- Data upload and management capabilities
- Built-in feature engineering matching notebook pipeline
- Integrated StandardScaler preprocessing
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="VaccineLife | Life Expectancy Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# PATHS CONFIGURATION
# ============================================================================
MODELS_DIR = "models"
DATA_DIR = "data"
DATASET_PATH = "joined_cty_vacc.csv"

# Model artifact paths
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")
DEFAULT_DATA_PATH = os.path.join(DATA_DIR, DATASET_PATH)
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
    
    /* Delete button styling */
    .stButton > button[kind="secondary"] {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        box-shadow: 0 4px 15px rgba(255, 68, 68, 0.3);
    }
    
    .stButton > button[kind="secondary"]:hover {
        box-shadow: 0 6px 20px rgba(255, 68, 68, 0.5);
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
    
    /* Data management section */
    .data-management {
        background: linear-gradient(135deg, #2a2a5e 0%, #1a1a4e 100%);
        border: 1px solid #4d4d9f;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# VACCINE INFORMATION (Based on Training Data)
# ============================================================================
# Only vaccines present in the training dataset (from joined_cty_vacc.txt)
VACCINE_INFO = {
    'BCG': 'Bacillus Calmette-Guérin (Tuberculosis)',
    'DTP1': 'Diphtheria-Tetanus-Pertussis (1st dose)',
    'DTP3': 'Diphtheria-Tetanus-Pertussis (3rd dose)',
    'HEPB3': 'Hepatitis B (3rd dose)',
    'HEPBB': 'Hepatitis B (birth dose)',
    'HIB3': 'Haemophilus influenzae type b (3rd dose)',
    'IPV1': 'Inactivated Polio Vaccine (1st dose)',
    'MCV1': 'Measles-containing Vaccine (1st dose)',
    'MCV2': 'Measles-containing Vaccine (2nd dose)',
    'PCV3': 'Pneumococcal Conjugate Vaccine (3rd dose)',
    'POL3': 'Polio (3rd dose)',
    'RCV1': 'Rubella-containing Vaccine (1st dose)',
    'ROTAC': 'Rotavirus (completed series)',
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
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = None

# ============================================================================
# FEATURE ENGINEERING FUNCTIONS (Matching Notebook Pipeline)
# ============================================================================
def engineer_features(df_input, year_value, country_name, label_encoder=None, is_single_prediction=False):
    """
    Apply the exact feature engineering pipeline from the notebook.
    
    Pipeline steps:
    1. Vaccination coverage index (mean of all vaccines)
    2. Temporal features (years_since_2000, decade)
    3. Vaccination improvement rate (requires historical data)
    4. Country encoding
    
    Args:
        df_input: DataFrame with vaccine coverage values
        year_value: Year for prediction
        country_name: Country name for prediction
        label_encoder: Fitted LabelEncoder for country names
        is_single_prediction: Whether this is a single prediction or batch
    
    Returns:
        DataFrame with engineered features
    """
    df = df_input.copy()
    
    # 1. Vaccination coverage index - mean of all vaccine columns
    df['vacc_coverage_index'] = df[VACCINE_COLS].mean(axis=1)
    
    # 2. Temporal features
    df['years_since_2000'] = year_value - 2000
    df['decade'] = (year_value // 10) * 10
    
    # 3. Vaccination improvement rate
    # For single prediction, we set this to 0 (as we don't have historical context)
    # In production, you might want to calculate this from historical data
    if is_single_prediction:
        df['vacc_improvement'] = 0
    else:
        # For batch predictions, calculate if possible
        df['vacc_improvement'] = 0
    
    # 4. Country encoding
    if label_encoder is not None:
        try:
            df['country_encoded'] = label_encoder.transform([country_name])[0]
        except ValueError:
            # If country not in training data, use a default encoding
            st.warning(f"Country '{country_name}' not in training data. Using default encoding.")
            df['country_encoded'] = 0
    else:
        df['country_encoded'] = 0
    
    return df


def prepare_prediction_features(vaccine_values, year, country, label_encoder, feature_cols):
    """
    Prepare features for prediction matching the exact notebook pipeline.
    
    Args:
        vaccine_values: Dictionary of vaccine name -> coverage value
        year: Year for prediction
        country: Country name
        label_encoder: Fitted LabelEncoder
        feature_cols: List of feature column names from training
    
    Returns:
        DataFrame ready for prediction
    """
    # Create DataFrame with vaccine values
    df_pred = pd.DataFrame([vaccine_values])
    
    # Apply feature engineering
    df_pred = engineer_features(
        df_pred, 
        year, 
        country, 
        label_encoder, 
        is_single_prediction=True
    )
    # print('\n#1. df_pred aft engineer features:\n', df_pred.head())
    # print('\n#2. feature_cols:\n', feature_cols)
    # Ensure all required features are present in the correct order
    for col in feature_cols:
        if col not in df_pred.columns:
            df_pred[col] = 0  # Default value for missing features
    
    # Select only the features used in training, in the correct order
    df_pred = df_pred[feature_cols]
    # print('\n#3. df_pred aft feature_cols:\n', df_pred.head())
    return df_pred


# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================
def load_model_artifacts():
    """Load model and preprocessing artifacts"""
    try:
        if os.path.exists(MODEL_PATH):
            st.session_state.model = joblib.load(MODEL_PATH)
            st.session_state.model_loaded = True
            
            # Load scaler
            if os.path.exists(SCALER_PATH):
                st.session_state.scaler = joblib.load(SCALER_PATH)
            
            # Load label encoder
            if os.path.exists(LABEL_ENCODER_PATH):
                st.session_state.label_encoder = joblib.load(LABEL_ENCODER_PATH)
            
            # Load feature names
            if os.path.exists(FEATURE_NAMES_PATH):
                st.session_state.feature_names = joblib.load(FEATURE_NAMES_PATH)
            
            return True
        else:
            st.warning(f"Model file not found at {MODEL_PATH}")
            return False
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return False


# ============================================================================
# DATA LOADING AND PROCESSING FUNCTIONS
# ============================================================================
def load_data_from_file(file_path):
    """Load data from a file path"""
    try:
        if file_path.endswith('.csv') or file_path.endswith('.txt'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            st.error("Unsupported file format")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


def load_data_from_upload(uploaded_file):
    """Load data from uploaded file"""
    try:
        if uploaded_file.name.endswith('.csv') or uploaded_file.name.endswith('.txt'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


def preprocess_data(df):
    """Preprocess the data for visualization"""
    # Convert year to numeric
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
    
    # Convert vaccine columns to numeric
    for col in VACCINE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert life expectancy to numeric
    if 'life_expectancy' in df.columns:
        df['life_expectancy'] = pd.to_numeric(df['life_expectancy'], errors='coerce')
    
    return df


def clear_uploaded_data():
    """Clear the uploaded dataset"""
    st.session_state.df = None
    st.session_state.data_loaded = False
    st.session_state.data_source = None


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def create_data_summary(df):
    """Create data summary statistics"""
    st.header("Data Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    
    with col2:
        if 'country' in df.columns:
            st.metric("Countries", f"{df['country'].nunique()}")
    
    with col3:
        if 'year' in df.columns:
            year_range = f"{int(df['year'].min())}-{int(df['year'].max())}"
            st.metric("Year Range", year_range)
    
    with col4:
        if 'life_expectancy' in df.columns:
            avg_life_exp = df['life_expectancy'].mean()
            st.metric("Avg Life Expectancy", f"{avg_life_exp:.1f} years")
    
    st.markdown("---")
    
    # Dataset preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)


def create_global_overview(df):
    """Create global overview visualizations"""
    st.header("Global Overview")
    
    if 'life_expectancy' not in df.columns or 'year' not in df.columns:
        st.warning("Required columns not found in dataset")
        return
    
    # Life expectancy trends over time
    yearly_avg = df.groupby('year')['life_expectancy'].mean().reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yearly_avg['year'],
        y=yearly_avg['life_expectancy'],
        mode='lines+markers',
        name='Average Life Expectancy',
        line=dict(color='#00d4ff', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='Global Average Life Expectancy Trend',
        xaxis_title='Year',
        yaxis_title='Life Expectancy (years)',
        template='plotly_dark',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_vaccine_coverage_analysis(df):
    """Create vaccine coverage analysis"""
    st.header("Vaccine Coverage Analysis")
    
    # Average coverage by vaccine
    vaccine_data = []
    for vaccine in VACCINE_COLS:
        if vaccine in df.columns:
            avg_coverage = df[vaccine].mean()
            vaccine_data.append({
                'Vaccine': vaccine,
                'Average Coverage (%)': avg_coverage
            })
    
    if vaccine_data:
        vaccine_df = pd.DataFrame(vaccine_data).sort_values('Average Coverage (%)', ascending=False)
        
        fig = px.bar(
            vaccine_df,
            x='Vaccine',
            y='Average Coverage (%)',
            color='Average Coverage (%)',
            color_continuous_scale='Viridis',
            title='Average Vaccine Coverage by Type'
        )
        
        fig.update_layout(
            template='plotly_dark',
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)


def create_scatter_analysis(df):
    """Create scatter plot analysis"""
    st.subheader("Vaccination vs Life Expectancy")
    
    # Select vaccine for analysis
    available_vaccines = [v for v in VACCINE_COLS if v in df.columns]
    selected_vaccine = st.selectbox(
        "Select Vaccine",
        available_vaccines,
        key="scatter_vaccine"
    )
    
    if selected_vaccine and 'life_expectancy' in df.columns:
        fig = px.scatter(
            df,
            x=selected_vaccine,
            y='life_expectancy',
            color='year' if 'year' in df.columns else None,
            hover_data=['country'] if 'country' in df.columns else None,
            title=f'{VACCINE_INFO[selected_vaccine]} Coverage vs Life Expectancy',
            labels={
                selected_vaccine: f'{selected_vaccine} Coverage (%)',
                'life_expectancy': 'Life Expectancy (years)'
            }
        )
        
        fig.update_layout(
            template='plotly_dark',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)


def create_correlation_analysis(df):
    """Create correlation analysis"""
    st.header("Correlation Analysis")
    
    # Calculate correlation with life expectancy
    vaccine_cols_present = [v for v in VACCINE_COLS if v in df.columns]
    
    if vaccine_cols_present and 'life_expectancy' in df.columns:
        correlations = []
        for vaccine in vaccine_cols_present:
            corr = df[vaccine].corr(df['life_expectancy'])
            correlations.append({
                'Vaccine': vaccine,
                'Correlation': corr
            })
        
        corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=True)
        
        fig = px.bar(
            corr_df,
            x='Correlation',
            y='Vaccine',
            orientation='h',
            color='Correlation',
            color_continuous_scale='RdYlGn',
            title='Correlation between Vaccine Coverage and Life Expectancy'
        )
        
        fig.update_layout(
            template='plotly_dark',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)


def create_country_analysis(df):
    """Create country-specific analysis"""
    st.header("Country Analysis")
    
    if 'country' not in df.columns:
        st.warning("Country column not found in dataset")
        return
    
    # Country selector
    countries = sorted(df['country'].unique())
    selected_country = st.selectbox("Select Country", countries, key="country_analysis")
    
    if selected_country:
        country_data = df[df['country'] == selected_country].copy()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Life expectancy trend
            if 'year' in country_data.columns and 'life_expectancy' in country_data.columns:
                country_data = country_data.sort_values('year')
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=country_data['year'],
                    y=country_data['life_expectancy'],
                    mode='lines+markers',
                    name='Life Expectancy',
                    line=dict(color='#00ff88', width=3),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    title=f'Life Expectancy Trend - {selected_country}',
                    xaxis_title='Year',
                    yaxis_title='Life Expectancy (years)',
                    template='plotly_dark',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Vaccine coverage over time
            vaccine_cols_present = [v for v in VACCINE_COLS if v in country_data.columns]
            
            if vaccine_cols_present and 'year' in country_data.columns:
                selected_vaccines = st.multiselect(
                    "Select Vaccines to Display",
                    vaccine_cols_present,
                    default=vaccine_cols_present[:3],
                    key="country_vaccines"
                )
                
                if selected_vaccines:
                    fig = go.Figure()
                    
                    for vaccine in selected_vaccines:
                        fig.add_trace(go.Scatter(
                            x=country_data['year'],
                            y=country_data[vaccine],
                            mode='lines+markers',
                            name=vaccine,
                            marker=dict(size=6)
                        ))
                    
                    fig.update_layout(
                        title=f'Vaccine Coverage Trend - {selected_country}',
                        xaxis_title='Year',
                        yaxis_title='Coverage (%)',
                        template='plotly_dark',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PREDICTION INTERFACE
# ============================================================================
def create_prediction_interface():
    """Create the prediction interface matching notebook pipeline with comparison feature"""
    st.header("Life Expectancy Prediction")
    
    if not st.session_state.model_loaded:
        st.warning("Model not loaded. Please load the model from the sidebar.")
        
        if st.button("Try Loading Model Now", type="primary"):
            if load_model_artifacts():
                st.success("Model loaded successfully!")
                st.rerun()
            else:
                st.error("Failed to load model. Please ensure model files are in the models/ directory.")
        return
    
    st.success("Model loaded and ready for predictions")
    
    st.markdown("---")
    
    # Create tabs for Single Prediction and Comparison
    pred_tab1, pred_tab2 = st.tabs(["Single Prediction", "Comparison"])
    
    # ============================================================================
    # TAB 1: SINGLE PREDICTION
    # ============================================================================
    with pred_tab1:
        st.subheader("Input Parameters")
        
        # Year and Country inputs
        col1, col2 = st.columns(2)
        
        with col1:
            year_input = st.number_input(
                "Year",
                min_value=2000,
                max_value=2050,
                value=2023,
                step=1,
                help="Year for prediction (affects temporal features)",
                key="single_year"
            )
        
        with col2:
            # Get list of countries from label encoder if available
            if st.session_state.label_encoder is not None:
                try:
                    countries_list = list(st.session_state.label_encoder.classes_)
                    default_country = countries_list[0] if countries_list else "Afghanistan"
                except:
                    countries_list = ["Afghanistan", "United States", "China", "India", "Brazil"]
                    default_country = "Afghanistan"
            else:
                countries_list = ["Afghanistan", "United States", "China", "India", "Brazil"]
                default_country = "Afghanistan"
            
            country_input = st.selectbox(
                "Country",
                options=countries_list,
                index=0,
                help="Select country (affects country encoding feature)",
                key="single_country"
            )
        
        st.markdown("---")
        st.subheader("Vaccine Coverage (%)")
        st.info("Enter coverage values as percentages (0-100). These vaccines match the training dataset.")
        
        # Create vaccine input grid
        vaccine_values = {}
        
        # Split vaccines into groups of 3 for better layout
        vaccines_per_row = 3
        vaccine_groups = [VACCINE_COLS[i:i + vaccines_per_row] for i in range(0, len(VACCINE_COLS), vaccines_per_row)]
        
        for vaccine_group in vaccine_groups:
            cols = st.columns(vaccines_per_row)
            for idx, vaccine in enumerate(vaccine_group):
                with cols[idx]:
                    value = st.number_input(
                        f"{vaccine}",
                        min_value=0.0,
                        max_value=100.0,
                        value=75.0,
                        step=1.0,
                        key=f"single_vaccine_{vaccine}",
                        help=VACCINE_INFO[vaccine]
                    )
                    vaccine_values[vaccine] = value
        
        st.markdown("---")
        
        # Prediction button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_button = st.button("Predict Life Expectancy", type="primary", use_container_width=True, key="single_predict")
        
        if predict_button:
            try:
                with st.spinner("Processing prediction..."):
                    # Prepare features using the same pipeline as notebook
                    df_pred = prepare_prediction_features(
                        vaccine_values,
                        year_input,
                        country_input,
                        st.session_state.label_encoder,
                        st.session_state.feature_names
                    )
                    prediction = st.session_state.model.predict(df_pred)[0]
                    
                    # Display prediction
                    st.markdown("---")
                    st.markdown("""
                    <div class="prediction-result">
                        <h2 style="color: #00d4ff; margin-bottom: 10px;">Predicted Life Expectancy</h2>
                        <div class="prediction-value">{:.2f}</div>
                        <p style="color: #a0ffcc; font-size: 1.2rem;">years</p>
                    </div>
                    """.format(prediction), unsafe_allow_html=True)
                    
                    # Interpretation
                    if prediction >= 75:
                        st.success("High life expectancy - associated with well-developed healthcare systems.")
                    elif prediction >= 65:
                        st.info("Moderate life expectancy - improvements in coverage could increase this.")
                    else:
                        st.warning("Lower life expectancy - significant improvements in coverage may help.")
                    
                    # Show feature summary
                    with st.expander("Feature Engineering Summary"):
                        st.write("**Engineered Features:**")
                        st.write(f"- Vaccination Coverage Index: {df_pred['vacc_coverage_index'].values[0]:.2f}%")
                        st.write(f"- Years Since 2000: {df_pred['years_since_2000'].values[0]}")
                        st.write(f"- Decade: {df_pred['decade'].values[0]}")
                        st.write(f"- Vaccination Improvement: {df_pred['vacc_improvement'].values[0]:.2f}")
                        st.write(f"- Country Encoded: {df_pred['country_encoded'].values[0]}")
                        
                        st.write("\n**Input Vaccine Coverages:**")
                        for vaccine, value in vaccine_values.items():
                            st.write(f"- {vaccine} ({VACCINE_INFO[vaccine]}): {value}%")
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.error("Please ensure all model files are properly loaded and compatible.")
                with st.expander("Error Details"):
                    import traceback
                    st.code(traceback.format_exc())
    
    # ============================================================================
    # TAB 2: COMPARISON
    # ============================================================================
    with pred_tab2:
        st.subheader("Compare Two Vaccine Combinations")
        st.info("Compare life expectancy predictions for two different vaccine coverage scenarios.")
        
        # Get countries list
        if st.session_state.label_encoder is not None:
            try:
                countries_list = list(st.session_state.label_encoder.classes_)
            except:
                countries_list = ["Afghanistan", "United States", "China", "India", "Brazil"]
        else:
            countries_list = ["Afghanistan", "United States", "China", "India", "Brazil"]
        
        # Common parameters
        st.markdown("### Common Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            comp_year = st.number_input(
                "Year",
                min_value=2000,
                max_value=2050,
                value=2023,
                step=1,
                help="Year for prediction (affects temporal features)",
                key="comp_year"
            )
        
        with col2:
            comp_country = st.selectbox(
                "Country",
                options=countries_list,
                index=0,
                help="Select country (affects country encoding feature)",
                key="comp_country"
            )
        
        st.markdown("---")
        
        # Create two columns for comparison
        comp_col1, comp_col2 = st.columns(2)
        
        # ========== COMBINATION 1 ==========
        with comp_col1:
            st.markdown("### Combination 1")
            st.markdown("**Vaccine Coverage (%)**")
            
            vaccine_values_1 = {}
            vaccines_per_row = 2
            vaccine_groups = [VACCINE_COLS[i:i + vaccines_per_row] for i in range(0, len(VACCINE_COLS), vaccines_per_row)]
            
            for vaccine_group in vaccine_groups:
                cols = st.columns(vaccines_per_row)
                for idx, vaccine in enumerate(vaccine_group):
                    with cols[idx]:
                        value = st.number_input(
                            f"{vaccine}",
                            min_value=0.0,
                            max_value=100.0,
                            value=60.0,
                            step=1.0,
                            key=f"comp1_vaccine_{vaccine}",
                            help=VACCINE_INFO[vaccine],
                            label_visibility="visible"
                        )
                        vaccine_values_1[vaccine] = value
        
        # ========== COMBINATION 2 ==========
        with comp_col2:
            st.markdown("### Combination 2")
            st.markdown("**Vaccine Coverage (%)**")
            
            vaccine_values_2 = {}
            
            for vaccine_group in vaccine_groups:
                cols = st.columns(vaccines_per_row)
                for idx, vaccine in enumerate(vaccine_group):
                    with cols[idx]:
                        value = st.number_input(
                            f"{vaccine}",
                            min_value=0.0,
                            max_value=100.0,
                            value=90.0,
                            step=1.0,
                            key=f"comp2_vaccine_{vaccine}",
                            help=VACCINE_INFO[vaccine],
                            label_visibility="visible"
                        )
                        vaccine_values_2[vaccine] = value
        
        st.markdown("---")
        
        # Compare button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            compare_button = st.button("Compare Predictions", type="primary", use_container_width=True, key="compare_predict")
        
        if compare_button:
            try:
                with st.spinner("Processing predictions..."):
                    # Prepare features for both scenarios
                    df_pred_1 = prepare_prediction_features(
                        vaccine_values_1,
                        comp_year,
                        comp_country,
                        st.session_state.label_encoder,
                        st.session_state.feature_names
                    )
                    
                    df_pred_2 = prepare_prediction_features(
                        vaccine_values_2,
                        comp_year,
                        comp_country,
                        st.session_state.label_encoder,
                        st.session_state.feature_names
                    )
                    
                    # Make predictions
                    prediction_1 = st.session_state.model.predict(df_pred_1)[0]
                    prediction_2 = st.session_state.model.predict(df_pred_2)[0]
                    
                    # Calculate difference
                    difference = prediction_2 - prediction_1
                    percent_change = (difference / prediction_1) * 100 if prediction_1 != 0 else 0
                    
                    # Display comparison results
                    st.markdown("---")
                    st.markdown("### 🎯 Comparison Results")
                    
                    # Create two columns for side-by-side results
                    result_col1, result_col2 = st.columns(2)
                    
                    with result_col1:
                        st.markdown("""
                        <div class="prediction-result" style="background: linear-gradient(135deg, #1a4d5e 0%, #1a6b7a 100%);">
                            <h3 style="color: #00d4ff; margin-bottom: 10px;">Scenario 1</h3>
                            <div class="prediction-value" style="font-size: 3rem;">{:.2f}</div>
                            <p style="color: #a0ffcc; font-size: 1rem;">years</p>
                        </div>
                        """.format(prediction_1), unsafe_allow_html=True)
                    
                    with result_col2:
                        st.markdown("""
                        <div class="prediction-result" style="background: linear-gradient(135deg, #1a5e4d 0%, #1a7a6b 100%);">
                            <h3 style="color: #00d4ff; margin-bottom: 10px;">Scenario 2</h3>
                            <div class="prediction-value" style="font-size: 3rem;">{:.2f}</div>
                            <p style="color: #a0ffcc; font-size: 1rem;">years</p>
                        </div>
                        """.format(prediction_2), unsafe_allow_html=True)
                    
                    # Show difference
                    st.markdown("---")
                    st.markdown("### 📈 Impact Analysis")
                    
                    diff_col1, diff_col2, diff_col3 = st.columns(3)
                    
                    with diff_col1:
                        st.metric(
                            label="Difference",
                            value=f"{difference:.2f} years",
                            delta=f"{difference:.2f}"
                        )
                    
                    with diff_col2:
                        st.metric(
                            label="Percent Change",
                            value=f"{abs(percent_change):.2f}%",
                            delta=f"{percent_change:.2f}%"
                        )
                    
                    with diff_col3:
                        if difference > 0:
                            st.success("✅ Scenario 2 shows higher life expectancy")
                        elif difference < 0:
                            st.warning("⚠️ Scenario 1 shows higher life expectancy")
                        else:
                            st.info("➡️ Both scenarios show equal life expectancy")
                    
                    # Visualization
                    st.markdown("---")
                    st.markdown("### 📊 Visual Comparison")
                    
                    # Create comparison chart
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=['Scenario 1', 'Scenario 2'],
                        y=[prediction_1, prediction_2],
                        marker=dict(
                            color=['#1a7a6b', '#00d4ff'],
                            line=dict(color='#00ff88', width=2)
                        ),
                        text=[f'{prediction_1:.2f}', f'{prediction_2:.2f}'],
                        textposition='outside',
                        textfont=dict(size=16, color='white')
                    ))
                    
                    fig.update_layout(
                        title='Life Expectancy Comparison',
                        yaxis_title='Life Expectancy (years)',
                        template='plotly_dark',
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed comparison
                    with st.expander("📋 Detailed Comparison"):
                        comp_col1, comp_col2 = st.columns(2)
                        
                        with comp_col1:
                            st.markdown("**Scenario 1 Features:**")
                            st.write(f"- Vaccination Coverage Index: {df_pred_1['vacc_coverage_index'].values[0]:.2f}%")
                            st.write(f"- Vaccination Improvement: {df_pred_1['vacc_improvement'].values[0]:.2f}")
                            
                            st.markdown("**Scenario 1 Vaccines:**")
                            for vaccine, value in vaccine_values_1.items():
                                st.write(f"- {vaccine}: {value}%")
                        
                        with comp_col2:
                            st.markdown("**Scenario 2 Features:**")
                            st.write(f"- Vaccination Coverage Index: {df_pred_2['vacc_coverage_index'].values[0]:.2f}%")
                            st.write(f"- Vaccination Improvement: {df_pred_2['vacc_improvement'].values[0]:.2f}")
                            
                            st.markdown("**Scenario 2 Vaccines:**")
                            for vaccine, value in vaccine_values_2.items():
                                st.write(f"- {vaccine}: {value}%")
                    
            except Exception as e:
                st.error(f"Error making comparison: {str(e)}")
                st.error("Please ensure all model files are properly loaded and compatible.")
                with st.expander("Error Details"):
                    import traceback
                    st.code(traceback.format_exc())


# ============================================================================
# SIDEBAR
# ============================================================================
def create_sidebar():
    """Create the sidebar with navigation and file uploads"""
    
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h1 style="color: #00d4ff; font-size: 1.8rem;">VaccineLife</h1>
        <p style="color: #a0a0ff;">Life Expectancy Predictor</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Flow selection
    st.sidebar.subheader("Select Your Flow")
    flow = st.sidebar.radio(
        "What would you like to do?",
        options=["Visualization Only", "Prediction Only", "Both"],
        index=2,
        help="Choose whether you want to visualize data, make predictions, or both"
    )
    
    st.sidebar.markdown("---")
    
    # Data management section
    if flow in ["Visualization Only", "Both"]:
        st.sidebar.subheader("Data Management")
        
        # Show current data status
        if st.session_state.data_loaded:
            st.sidebar.markdown(
                f"""
                <div class="data-management">
                    <p style="color: #00ff88; font-weight: bold;">Data Loaded</p>
                    <p style="color: #a0a0ff; font-size: 0.9rem;">Source: {st.session_state.data_source}</p>
                    <p style="color: #a0a0ff; font-size: 0.9rem;">Records: {len(st.session_state.df):,}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Delete button
            if st.sidebar.button("Delete Current Dataset", type="secondary", use_container_width=True):
                clear_uploaded_data()
                st.sidebar.success("Dataset cleared!")
                st.rerun()
        
        st.sidebar.markdown("---")
        
        # Data upload section
        st.sidebar.subheader("Upload New Dataset")
        uploaded_data = st.sidebar.file_uploader(
            "Upload Dataset (CSV/TXT/Excel)",
            type=['csv', 'txt', 'xlsx', 'xls'],
            help="Upload your vaccine coverage dataset",
            key="data_uploader"
        )
        
        if uploaded_data is not None:
            df = load_data_from_upload(uploaded_data)
            if df is not None:
                st.session_state.df = preprocess_data(df)
                st.session_state.data_loaded = True
                st.session_state.data_source = uploaded_data.name
                st.sidebar.success(f"Data loaded: {len(df)} records")
                st.rerun()
        
        # Load default dataset button
        if not st.session_state.data_loaded:
            if os.path.exists(DEFAULT_DATA_PATH):
                if st.sidebar.button("Load Default Dataset", use_container_width=True):
                    df = load_data_from_file(DEFAULT_DATA_PATH)
                    if df is not None:
                        st.session_state.df = preprocess_data(df)
                        st.session_state.data_loaded = True
                        st.session_state.data_source = "Default Dataset"
                        st.sidebar.success("Default dataset loaded!")
                        st.rerun()
    
    # Model status section
    if flow in ["Prediction Only", "Both"]:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Model Status")
        
        if st.session_state.model_loaded:
            st.sidebar.success("Model: Loaded")
            if st.session_state.feature_names:
                # print("Feature names loaded:", st.session_state.feature_names)
                st.sidebar.info(f"Features: {len(st.session_state.feature_names)}")
            else:
                st.sidebar.info(f"Vaccines: {len(VACCINE_COLS)}")
        else:
            st.sidebar.warning("Model: Not loaded")
            if st.sidebar.button("Load Model", use_container_width=True):
                load_model_artifacts()
                st.rerun()
    
    # Overall status
    st.sidebar.markdown("---")
    st.sidebar.subheader("Overall Status")
    
    status_col1, status_col2 = st.sidebar.columns(2)
    
    with status_col1:
        if st.session_state.data_loaded:
            st.markdown("**Data Loaded**")
        else:
            st.markdown("**No Data**")
    
    with status_col2:
        if st.session_state.model_loaded:
            st.markdown("**Model Loaded**")
        else:
            st.markdown("**No Model**")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="padding: 10px; font-size: 0.8rem; color: #a0a0ff;">
        <h4 style="color: #00d4ff;">About</h4>
        <p>This application predicts life expectancy based on vaccine coverage data using machine learning.</p>
        <p style="margin-top: 10px; font-size: 0.75rem; color: #7080a0;">
        <b>Tip:</b> Place model files in the <code>models/</code> directory before running.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    return flow


# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    """Main application logic"""
    
    # Try to load model on startup
    if not st.session_state.model_loaded:
        load_model_artifacts()
    
    flow = create_sidebar()
    
    #### PREV ####
    # st.markdown("""
    # <div style="text-align: center; padding: 20px 0;">
    #     <h1>Life Expectancy Prediction Dashboard</h1>
    #     <p style="font-size: 1.2rem; color: #a0a0ff;">
    #         Explore the relationship between vaccine coverage and life expectancy
    #     </p>
    # </div>
    # """, unsafe_allow_html=True)

    #### IMPROVEMENT_V1 ####
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1>Life Expectancy Prediction Dashboard</h1>
        <p style="font-size: 1.2rem; color: #a0a0ff;">
            Predict the Life Expectancy for the Year using Vaccine Coverage
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if flow == "Visualization Only":
        if st.session_state.data_loaded and st.session_state.df is not None:
            df = st.session_state.df
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Summary", "Global Overview", "Vaccine Analysis",
                "Correlations", "Country Analysis"
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
            st.info("Please upload your dataset or load the default dataset using the sidebar to view visualizations.")
    
    elif flow == "Prediction Only":
        create_prediction_interface()
    
    else:  # Both
        tab_pred, tab_summary, tab_global, tab_vaccine, tab_corr, tab_country = st.tabs([
            "Prediction", "Summary", "Global Overview",
            "Vaccine Analysis", "Correlations", "Country Analysis"
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
                    st.info("Please upload your dataset or load the default dataset using the sidebar to view visualizations.")


if __name__ == "__main__":
    main()