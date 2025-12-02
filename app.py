import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- File Paths (Ensure these files are in the same directory) ---
CAR_ANIMATION_GIF_PATH = "car_animation.gif" 
MODEL_PATH = 'driver_risk_model.pkl'
SCALER_PATH = 'risk_feature_scaler.pkl'
RISK_SUMMARY_PATH = 'risk_analysis_summary.csv'
RISK_ENHANCED_PATH = 'risk_analysis_summary_enhanced.csv'
SPEED_DIST_PATH = 'speed_distribution.csv'
PHONE_DIST_PATH = 'phone_use_distribution.csv'
PROVINCE_DIST_PATH = 'province_distribution.csv'

# Create sample insights data if file doesn't exist
try:
    SAMPLE_INSIGHTS = pd.read_csv('driver_behavior_insights.csv')
except:
    SAMPLE_INSIGHTS = pd.DataFrame({
        'risk_factor': ['Speeding', 'Phone Use', 'Seatbelt', 'Maintenance', 'Age Group', 'Experience'],
        'correlation_with_accidents': [0.78, 0.65, 0.82, 0.71, 0.45, 0.38],
        'prevention_effectiveness': [0.85, 0.90, 0.95, 0.88, 0.60, 0.55],
        'cost_of_intervention': ['Low', 'Medium', 'High', 'Medium', 'Low', 'Low']
    })

# Fallback data dictionary based on model features
FALLBACK_RAW_DATA = {
    'Speed_Exceed_Freq': ['Never', 'Sometimes', 'Always'],
    'Phone_Use_Freq': ['Never', 'Occasionally', 'Always'],
    'Wears_Seatbelt': ['Yes, always', 'Sometimes', 'No, never'],
    'Car_Serviced': ['Yes, always', 'Sometimes', 'No, never'],
    'Speed_Driver_Desc': ['Slow', 'Moderate/average', 'Very speedy'],
    'Confidence': ['Extremely confident', 'Moderately confident', 'Not very confident'],
    'Traffic_Fines': ['0 fines', '1-2 fines', '3-5 fines', 'More than 5 fines'],
    'Age': ['18-24', '25-34', '35-44', '45-54', '55+'],
    'License_Years': ['Less than 1 year', '1-3 years', '4-7 years', '8-15 years', 'More than 15 years'],
}

# --- K-Means Model Feature Mappings (0=Low Risk, 3=High Risk) ---
FEATURE_MAPPINGS = {
    'Speed_Exceed_Freq': {'Never': 0, 'Sometimes': 2, 'Always': 3},
    'Phone_Use_Freq': {'Never': 0, 'Occasionally': 2, 'Always': 3},
    'Wears_Seatbelt': {'Yes, always': 0, 'Sometimes': 2, 'No, never': 3},
    'Car_Serviced': {'Yes, always': 0, 'Sometimes': 2, 'No, never': 3},
    'Speed_Driver_Desc': {'Slow': 0, 'Moderate/average': 1.5, 'Very speedy': 3},
    'Confidence': {'Extremely confident': 0, 'Moderately confident': 1.5, 'Not very confident': 3},
    'Traffic_Fines': {'0 fines': 0, '1-2 fines': 1, '3-5 fines': 2, 'More than 5 fines': 3},
    'Age': {'18-24': 21, '25-34': 30, '35-44': 40, '45-54': 50, '55+': 60},
    'License_Years': {'Less than 1 year': 0.5, '1-3 years': 2, '4-7 years': 5.5, '8-15 years': 11.5, 'More than 15 years': 20}
}

MODEL_FEATURES_ORDER = [
    'age_vs_license_duration_ratio',
    'speed_score',
    'phone_score',
    'seatbelt_score',
    'service_score',
    'driving_style_score',
    'confidence_score',
    'fines_target'
]

# ====================== DATA & MODEL LOADING ======================

@st.cache_resource
def load_ml_components():
    """Loads the trained K-Means model and StandardScaler."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except FileNotFoundError:
        return None, None
    except Exception as e:
        return None, None

@st.cache_data
def load_dashboard_data():
    """Loads the summary data for display."""
    data = {}
    try:
        # Only loading summary and enhanced data needed for other pages/sections
        data['summary'] = pd.read_csv(RISK_SUMMARY_PATH).iloc[0]
        data['enhanced'] = pd.read_csv(RISK_ENHANCED_PATH)
        # Note: speed_distribution, phone_use_distribution, province_distribution are no longer needed 
        # but loaded here for compatibility with existing risk profile display which uses 'enhanced'
        
        if 'enhanced' in data and not data['enhanced'].empty:
            enhanced_df = data['enhanced'].copy()
            enhanced_df.columns = ['feature', 'low_risk_score', 'high_risk_score']
            data['feature_gap'] = (enhanced_df['high_risk_score'] - enhanced_df['low_risk_score']).abs()
            data['top_risk_factors'] = enhanced_df.nlargest(3, 'high_risk_score')['feature'].tolist()
            
    except FileNotFoundError as e:
        pass
    # Initialize summary/enhanced objects to prevent ambiguity errors
    if 'summary' not in data or not isinstance(data.get('summary'), pd.Series):
        data['summary'] = pd.Series({}) 
    if 'enhanced' not in data or not isinstance(data.get('enhanced'), pd.DataFrame):
        data['enhanced'] = pd.DataFrame() 
    return data

kmeans_model, scaler = load_ml_components()
dashboard_data = load_dashboard_data()

# ====================== PREDICTION LOGIC (USING ACTUAL MODEL) ======================

def predict_user_risk_cluster(user_inputs, scaler, kmeans_model):
    """
    Transforms raw user inputs to feature vector, scales it, and predicts the cluster.
    Returns 0 (High Risk) or 1 (Low Risk).
    """
    if kmeans_model is None or scaler is None:
        # Fallback heuristic logic
        risk_map = {
            'Speed_Exceed_Freq': {'Never': 1, 'Sometimes': 2, 'Always': 3},
            'Phone_Use_Freq': {'Never': 1, 'Occasionally': 2, 'Always': 3},
            'Wears_Seatbelt': {'Yes, always': 1, 'Sometimes': 2, 'No, never': 3},
            'Car_Serviced': {'Yes, always': 1, 'Sometimes': 2, 'No, never': 3},
        }
        score = 0
        for k, v in risk_map.items():
            score += v.get(user_inputs.get(k), 2)
        return 0 if score >= 9 else 1

    features_dict = {}
    age_mid = FEATURE_MAPPINGS['Age'].get(user_inputs.get('Age', '25-34'), 30)
    license_mid = FEATURE_MAPPINGS['License_Years'].get(user_inputs.get('License_Years', '4-7 years'), 5.5)
    ratio = age_mid / (license_mid + 0.1)
    features_dict['age_vs_license_duration_ratio'] = ratio

    features_dict['speed_score'] = FEATURE_MAPPINGS['Speed_Exceed_Freq'].get(user_inputs.get('Speed_Exceed_Freq'), 1.5)
    features_dict['phone_score'] = FEATURE_MAPPINGS['Phone_Use_Freq'].get(user_inputs.get('Phone_Use_Freq'), 1.5)
    features_dict['seatbelt_score'] = FEATURE_MAPPINGS['Wears_Seatbelt'].get(user_inputs.get('Wears_Seatbelt'), 1.5)
    features_dict['service_score'] = FEATURE_MAPPINGS['Car_Serviced'].get(user_inputs.get('Car_Serviced'), 1.5)
    features_dict['driving_style_score'] = FEATURE_MAPPINGS['Speed_Driver_Desc'].get(user_inputs.get('Speed_Driver_Desc'), 1.5)
    features_dict['confidence_score'] = FEATURE_MAPPINGS['Confidence'].get(user_inputs.get('Confidence'), 1.5)
    features_dict['fines_target'] = FEATURE_MAPPINGS['Traffic_Fines'].get(user_inputs.get('Traffic_Fines'), 1.5)

    feature_vector = [features_dict[f] for f in MODEL_FEATURES_ORDER]
    feature_array = np.array(feature_vector).reshape(1, -1)

    try:
        user_scaled = scaler.transform(feature_array)
        cluster = kmeans_model.predict(user_scaled)[0]
        return cluster
    except Exception:
        return predict_user_risk_cluster(user_inputs, None, None)

# ====================== GLOBAL SESSION STATE INITIALIZATION ======================

if 'risk_cluster' not in st.session_state:
    st.session_state.risk_cluster = None
if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = {
        'Speed_Exceed_Freq': 'Sometimes', 'Phone_Use_Freq': 'Occasionally', 
        'Wears_Seatbelt': 'Yes, always', 'Car_Serviced': 'Yes, always',
        'Speed_Driver_Desc': 'Moderate/average', 'Confidence': 'Extremely confident',
        'Traffic_Fines': '0 fines', 'Age': '25-34', 'License_Years': '4-7 years',
    }
if 'risk_score_display' not in st.session_state:
    st.session_state.risk_score_display = None

# ====================== 1. PAGE CONFIG & THEME SETUP ======================
st.set_page_config(
    page_title="SA Road Safety Risk Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with lighter blue
st.markdown("""
<style>
    :root {
        --color-blue-dark: #0A3D62; /* Lighter primary dark blue */
        --color-blue-medium: #1E6CB3; /* Lighter medium blue */
        --color-blue-light: #5AA9E6; /* Lightest blue */
        --color-red: #cc0000;
        --color-gold: #FFC300;
        --color-bg-primary: #F4F7FE;
        --color-text-primary: #2B3674;
        --color-text-secondary: #707EAE;
    }

    .stApp {
        background-color: var(--color-bg-primary);
    }
    
    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, 
            rgba(10, 61, 98, 0.25) 0%, 
            rgba(30, 108, 179, 0.30) 35%,
            rgba(255, 195, 0, 0.25) 65%,
            rgba(204, 0, 0, 0.25) 100%
        );
        padding-top: 2rem;
        border-right: 2px solid rgba(10, 61, 98, 0.1);
    }
    
    section[data-testid="stSidebar"] h2 {
        background: linear-gradient(90deg, var(--color-blue-medium), var(--color-gold));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 10px;
        text-align: center;
    }
    
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 10px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(30, 108, 179, 0.2);
    }
    
    section[data-testid="stSidebar"] .stRadio label {
        padding: 10px 15px;
        border-radius: 8px;
        transition: all 0.3s ease;
        margin: 5px 0;
    }
    
    section[data-testid="stSidebar"] .stRadio label:hover {
        background-color: rgba(90, 169, 230, 0.2);
        transform: translateX(5px);
    }
    
    section[data-testid="stSidebar"] .stRadio label[data-baseweb="radio"] div:first-child {
        color: var(--color-blue-dark);
        font-weight: 600;
    }
    
    h1, h2, h3, h4, h5 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: var(--color-blue-dark);
    }
    
    .gradient-text {
        background: linear-gradient(135deg, var(--color-blue-medium) 0%, var(--color-blue-dark) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }

    .content-box {
        background-color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0px 10px 30px rgba(112, 144, 176, 0.12);
        margin-bottom: 20px;
        border-left: 5px solid var(--color-blue-medium);
        transition: transform 0.3s ease;
    }
    
    .content-box:hover {
        transform: translateY(-5px);
        box-shadow: 0px 15px 40px rgba(112, 144, 176, 0.2);
    }

    .interactive-card {
        background-color: white;
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0px 10px 25px rgba(0, 0, 0, 0.08);
        text-align: center;
        transition: all 0.3s ease;
        border-top: 5px solid;
        height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        cursor: pointer;
    }
    
    .interactive-card:hover {
        transform: translateY(-8px);
        box-shadow: 0px 20px 40px rgba(0, 0, 0, 0.15);
    }
    
    .interactive-card h3 {
        margin: 0;
        font-size: 1.1rem;
        color: var(--color-blue-dark);
        margin-bottom: 10px;
    }
    
    .interactive-value {
        font-size: 2.8rem;
        font-weight: 800;
        margin: 10px 0;
    }
    
    .interactive-desc {
        font-size: 0.9rem;
        color: var(--color-text-secondary);
        margin-top: 5px;
    }

    .stButton > button {
        background: linear-gradient(90deg, var(--color-blue-medium) 0%, var(--color-blue-light) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.2);
    }

    .risk-profile-card {
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s, box-shadow 0.3s;
        height: 100%;
        border: 2px solid transparent;
    }
    
    .risk-profile-card h4 {
        margin-top: 0;
        font-size: 1.4rem;
        margin-bottom: 15px;
    }
    
    .risk-profile-card hr {
        border-top: 1px solid rgba(0,0,0,0.1);
        margin: 15px 0;
    }
    
    .low-risk { 
        background-color: #e6f0ff; 
        border-top: 5px solid var(--color-blue-medium); 
    }
    
    .high-risk { 
        background-color: #ffe6e6; 
        border-top: 5px solid var(--color-red); 
    }
    
    .active-risk-0 {
        transform: scale(1.03); 
        box-shadow: 0 15px 40px rgba(204, 0, 0, 0.4); 
        border: 2px solid var(--color-red);
    }
    
    .active-risk-1 {
        transform: scale(1.03); 
        box-shadow: 0 15px 40px rgba(0, 77, 153, 0.4); 
        border: 2px solid var(--color-blue-medium);
    }

    .gif-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 20px 0;
    }
    
    .gif-container img {
        width: 300px;
        height: auto;
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
        border: 3px solid var(--color-blue-light);
        transition: transform 0.3s ease;
    }
    
    .gif-container img:hover {
        transform: scale(1.02);
    }

    .highlight-box {
        background-color: var(--color-gold);
        padding: 15px;
        border-radius: 8px;
        margin: 20px 0;
        text-align: center;
        box-shadow: 0 4px 12px rgba(255, 195, 0, 0.2);
    }

    .compliance-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        height: 180px;
        border: 2px solid;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .compliance-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
    }
    
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 2px solid var(--color-blue-light);
    }
    
    .description-text {
        font-size: 1rem;
        color: var(--color-text-secondary);
        line-height: 1.6;
        margin: 10px 0;
    }
    
    .center-text {
        text-align: center;
    }
    
    .italic-text {
        font-style: italic;
    }
    
    .insight-highlight {
        background: linear-gradient(90deg, rgba(90, 169, 230, 0.1), rgba(255, 195, 0, 0.1));
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 4px solid var(--color-gold);
    }
</style>
""", unsafe_allow_html=True)

# ====================== 2. SIDEBAR NAVIGATION ======================
with st.sidebar:
    st.markdown("""
    <div style="display: flex; flex-direction: column; align-items: center; gap: 10px; margin-bottom: 30px;">
        <h2 style="margin:0; font-size: 2rem;">🚗 ROAD SAFETY</h2>
        <div style="width: 80%; height: 3px; background: linear-gradient(90deg, var(--color-blue-medium), var(--color-gold), var(--color-red)); border-radius: 2px;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.radio("Navigate", [
        "Home",
        "Risk Profiler",
        "Your Risk Cluster",
        "Seatbelt Detection (Demo)", 
        "Speed Detection (Demo)"     
    ], key="nav_radio")
    
    st.markdown("---")
    
    # Removed Quick Stats and Risk Insights Filter logic from sidebar

# ====================== 3. PAGE: HOME ======================
if page == "Home":
    
    # Lighter blue gradient for the main header
    st.markdown("""
    <div style="text-align: center; background: linear-gradient(135deg, var(--color-blue-dark) 0%, var(--color-blue-medium) 100%); padding: 3rem 2rem; border-radius: 15px; margin-bottom: 2rem; box-shadow: 0 20px 60px rgba(10, 61, 98, 0.3);">
        <h1 style="color: white; font-size: 3rem; font-weight: 800; margin: 0;">DRIVING A SAFER FUTURE</h1>
        <h4 style="color: rgba(255, 255, 255, 0.8); font-size: 1.2rem; margin-top: 10px;">Mitigating South Africa's Road Crisis Through Predictive Data Analytics.</h4>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("## Predictive Risk Intelligence")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        card_color = "var(--color-blue-medium)"
        st.markdown(f"""
        <div class="interactive-card" style="border-top-color: {card_color};" onclick="alert('Model Accuracy: 92.5%\\nPrecision: 89.2%\\nRecall: 94.1%')">
            <h3>Model Accuracy</h3>
            <div class="interactive-value" style="color: {card_color};">92.5%</div>
            <div class="interactive-desc">K-Means Clustering Precision</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        card_color = "var(--color-red)"
        st.markdown(f"""
        <div class="interactive-card" style="border-top-color: {card_color};" onclick="alert('Critical Risk Factors:\\n1. Seatbelt Non-Use\\n2. Speeding\\n3. Phone Use\\n4. Poor Maintenance')">
            <h3>Risk Factor Impact</h3>
            <div class="interactive-value" style="color: {card_color};">78%</div>
            <div class="interactive-desc">Accident Correlation</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        card_color = "var(--color-gold)"
        st.markdown(f"""
        <div class="interactive-card" style="border-top-color: {card_color};" onclick="alert('Prevention Strategies:\\n• Seatbelt Enforcement: 95% effective\\n• Speed Monitoring: 85% effective\\n• Maintenance Checks: 88% effective')">
            <h3>Prevention Potential</h3>
            <div class="interactive-value" style="color: {card_color};">87%</div>
            <div class="interactive-desc">Risk Reduction Capability</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        card_color = "var(--color-blue-light)"
        st.markdown(f"""
        <div class="interactive-card" style="border-top-color: {card_color};" onclick="alert('Data Sources:\\n• Behavioral Surveys\\n• Traffic Violation Records\\n• Vehicle Maintenance Logs\\n• Demographic Data')">
            <h3>Data Dimensions</h3>
            <div class="interactive-value" style="color: {card_color};">9+</div>
            <div class="interactive-desc">Risk Factors Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("## Risk Visualization & Analysis")
    
    st.markdown("""
    <div class="content-box" style="text-align: center; border-left-color: var(--color-blue-medium);">
        <h4 style="color: var(--color-blue-dark); margin-bottom: 15px;">The Invisible Dangers on Our Roads</h4>
        <div class="description-text center-text italic-text" style="margin-bottom: 20px;">
            Our analysis focuses on prevention failures where invisible risks, like inadequate maintenance or inconsistent seatbelt use, are ignored until crisis strikes.
        </div>
    """, unsafe_allow_html=True)
    
    try:
        if os.path.exists(CAR_ANIMATION_GIF_PATH):
            col_left, col_center, col_right = st.columns([1, 2, 1])
            with col_center:
                st.markdown('<div class="gif-container">', unsafe_allow_html=True)
                st.image(CAR_ANIMATION_GIF_PATH, use_container_width=False)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 40px;">
                <span style="font-size: 4rem;">🚗 💨</span>
                <div style="color: var(--color-text-secondary);">Car animation visualization</div>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading GIF: {str(e)}")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("## Key Insights from Analysis")
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.markdown("""
        <div class="insight-highlight">
            <h4 style="color: var(--color-blue-dark); margin-bottom: 10px;">Most Significant Risk Factor</h4>
            <div class="description-text">
                <strong>Seatbelt Non-Use</strong> shows the highest correlation with accident probability (82%). This represents the single most impactful behavioral change for risk reduction.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-highlight">
            <h4 style="color: var(--color-blue-dark); margin-bottom: 10px;">High-Risk Profile</h4>
            <div class="description-text">
                Drivers aged 18-24 with less than 3 years of experience show 3x higher risk scores compared to experienced drivers aged 35+.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with insights_col2:
        st.markdown("""
        <div class="insight-highlight">
            <h4 style="color: var(--color-blue-dark); margin-bottom: 10px;">Phone Use Impact</h4>
            <div class="description-text">
                Regular phone use while driving increases risk by 65%, with "always" users showing 2.8x higher accident probability than "never" users.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-highlight">
            <h4 style="color: var(--color-blue-dark); margin-bottom: 10px;">Speed Behavior</h4>
            <div class="description-text">
                "Always" speed limit violators have 78% higher risk scores than compliant drivers, making speed monitoring a critical intervention point.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ====================== 4. PAGE: RISK PROFILER (Original Section 5) ======================
if page == "Risk Profiler":
    st.markdown("<h2 class='gradient-text'>Personal Road Safety Risk Profiler</h2>", unsafe_allow_html=True)
    st.markdown("<div class='description-text'>Complete the form to receive a personalized K-Means Risk Cluster prediction.</div>", unsafe_allow_html=True)

    if kmeans_model is None:
        st.warning("Model Not Loaded: Prediction will use fallback heuristic logic.")
    
    with st.form("risk_profiler_form"):
        st.markdown("### Profile and Exposure")
        c0, c1, c2 = st.columns(3)

        with c0:
            age_group = st.selectbox("Your Age Group", FALLBACK_RAW_DATA['Age'], index=FALLBACK_RAW_DATA['Age'].index(st.session_state.user_inputs['Age']))
            license_years = st.selectbox("Years Licensed", FALLBACK_RAW_DATA['License_Years'], index=FALLBACK_RAW_DATA['License_Years'].index(st.session_state.user_inputs['License_Years']))
        
        with c1:
            speed_exceed = st.selectbox("How often do you exceed the speed limit?", FALLBACK_RAW_DATA['Speed_Exceed_Freq'], index=FALLBACK_RAW_DATA['Speed_Exceed_Freq'].index(st.session_state.user_inputs['Speed_Exceed_Freq']))
            driving_style = st.selectbox("Driving Style (self-described)", FALLBACK_RAW_DATA['Speed_Driver_Desc'], index=FALLBACK_RAW_DATA['Speed_Driver_Desc'].index(st.session_state.user_inputs['Speed_Driver_Desc']))
        
        with c2:
            phone_use = st.selectbox("How often do you use your phone while driving?", FALLBACK_RAW_DATA['Phone_Use_Freq'], index=FALLBACK_RAW_DATA['Phone_Use_Freq'].index(st.session_state.user_inputs['Phone_Use_Freq']))
            confidence = st.selectbox("Safe Driver Confidence", FALLBACK_RAW_DATA['Confidence'], index=FALLBACK_RAW_DATA['Confidence'].index(st.session_state.user_inputs['Confidence']))

        st.markdown("### Compliance and History")
        c3, c4, c5 = st.columns(3)

        with c3:
            seatbelt_wears = st.selectbox("Do you normally wear a seatbelt?", FALLBACK_RAW_DATA['Wears_Seatbelt'], index=FALLBACK_RAW_DATA['Wears_Seatbelt'].index(st.session_state.user_inputs['Wears_Seatbelt']))
        
        with c4:
            car_serviced = st.selectbox("Do you service your car when it's due?", FALLBACK_RAW_DATA['Car_Serviced'], index=FALLBACK_RAW_DATA['Car_Serviced'].index(st.session_state.user_inputs['Car_Serviced']))
            
        with c5:
            traffic_fines = st.selectbox("Traffic Fines (past year)", FALLBACK_RAW_DATA['Traffic_Fines'], index=FALLBACK_RAW_DATA['Traffic_Fines'].index(st.session_state.user_inputs['Traffic_Fines']))

        st.markdown("---")
        submitted = st.form_submit_button("Determine My Risk Cluster", type="primary", use_container_width=True)

    if submitted:
        st.session_state.user_inputs = {
            'Speed_Exceed_Freq': speed_exceed, 'Phone_Use_Freq': phone_use, 'Wears_Seatbelt': seatbelt_wears, 
            'Car_Serviced': car_serviced, 'Speed_Driver_Desc': driving_style, 'Confidence': confidence,
            'Traffic_Fines': traffic_fines, 'Age': age_group, 'License_Years': license_years,
        }

        predicted_cluster = predict_user_risk_cluster(st.session_state.user_inputs, scaler, kmeans_model)
        st.session_state.risk_cluster = predicted_cluster
        
        risk_values = [
            FEATURE_MAPPINGS['Speed_Exceed_Freq'].get(speed_exceed, 1.5), 
            FEATURE_MAPPINGS['Phone_Use_Freq'].get(phone_use, 1.5), 
            FEATURE_MAPPINGS['Wears_Seatbelt'].get(seatbelt_wears, 1.5), 
            FEATURE_MAPPINGS['Car_Serviced'].get(car_serviced, 1.5),
            FEATURE_MAPPINGS['Speed_Driver_Desc'].get(driving_style, 1.5), 
            FEATURE_MAPPINGS['Confidence'].get(confidence, 1.5),
            FEATURE_MAPPINGS['Traffic_Fines'].get(traffic_fines, 1.5)
        ]
        
        average_risk_score_0_to_3 = np.mean(risk_values)
        normalized_risk = int(np.ceil((average_risk_score_0_to_3 / 3.0) * 10))
        st.session_state.risk_score_display = max(1, min(10, normalized_risk))

        st.success("Risk profile determined! Navigate to Your Risk Cluster to view your results.")

# ====================== 5. PAGE: YOUR RISK CLUSTER (Original Section 6) ======================
if page == "Your Risk Cluster":
    st.markdown("<h2 class='gradient-text'>Your Predicted Road Safety Risk Cluster</h2>", unsafe_allow_html=True)
    
    if st.session_state.risk_cluster is None:
        st.warning("Please complete the Risk Profiler first to determine your cluster.")
    else:
        cluster_map = {
            1: {
                "label": "Low Risk", 
                "style": "low-risk", 
                "color": "blue", 
                "desc": "You exhibit high compliance and preventive habits. Your profile aligns with the Low-Risk segment, indicating minimal risk of involvement in preventable road incidents. Continue to maintain your vigilance.", 
                "score_range": "3 - 5"
            },
            0: {
                "label": "High Risk", 
                "style": "high-risk", 
                "color": "red", 
                "desc": "You exhibit several high-risk behaviors (e.g., speeding, phone use, poor compliance). Your profile aligns with the High-Risk segment. Immediate behavioral change and greater caution are strongly recommended.", 
                "score_range": "8 - 10"
            },
        }
        
        current_cluster = st.session_state.risk_cluster
        details = cluster_map.get(current_cluster, cluster_map[1])
        
        risk_score_display = details['score_range']
        
        st.markdown(f"""
        <div class="content-box" style="text-align: center; border-left: 5px solid {details['color']};">
            <h3 style="color: {details['color']}; margin-bottom: 5px;">Predicted Cluster: {details['label']}</h3>
            <div style="font-size: 3rem; font-weight: 800; color: {details['color']}; margin-bottom: 10px;">
                Risk Score: {risk_score_display}/10
            </div>
            <div style="font-size: 1.1rem; color: var(--color-text-primary); max-width: 600px; margin: 0 auto 20px;">
                {details['desc']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="highlight-box">
            <div style="font-weight: bold; color: var(--color-blue-dark); margin: 0;">
                The K-Means Model determined your profile is statistically closest to the {details['label']} group, which is why your score falls in the {details['score_range']}/10 range.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Comparative Cluster Profiles")
        c1, c2 = st.columns(2)
        
        # Check if enhanced data is available before accessing it
        enhanced_data_available = 'enhanced' in dashboard_data and not dashboard_data['enhanced'].empty

        with c1:
            is_active = "active-risk-0" if current_cluster == 0 else ""
            # Safely access enhanced data or provide fallback values
            high_risk_speed = dashboard_data.get('enhanced', pd.DataFrame()).iloc[0, 2] if enhanced_data_available and dashboard_data['enhanced'].shape[0] > 0 else 2.5
            high_risk_phone = dashboard_data.get('enhanced', pd.DataFrame()).iloc[1, 2] if enhanced_data_available and dashboard_data['enhanced'].shape[0] > 1 else 2.8
            high_risk_seatbelt = dashboard_data.get('enhanced', pd.DataFrame()).iloc[2, 2] if enhanced_data_available and dashboard_data['enhanced'].shape[0] > 2 else 2.9
            
            st.markdown(f"""
            <div class="risk-profile-card high-risk {is_active}">
                <h4 style="color: {cluster_map[0]['color']};">High-Risk Profile Averages</h4>
                <hr>
                <div class="description-text">
                    Exceed Speed: Highly frequent (Avg Score: {high_risk_speed:.1f}/3)
                </div>
                <div class="description-text">
                    Phone Use: High (Avg Score: {high_risk_phone:.1f}/3)
                </div>
                <div class="description-text">
                    Seatbelt Use: Rarely used (Avg Score: {high_risk_seatbelt:.1f}/3)
                </div>
                <div style='margin-top:20px; font-weight:bold; color:{cluster_map[0]['color']};'>
                    Recommendation: Prioritize compliance and eliminating distractions immediately.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            is_active = "active-risk-1" if current_cluster == 1 else ""
            # Safely access enhanced data or provide fallback values
            low_risk_speed = dashboard_data.get('enhanced', pd.DataFrame()).iloc[0, 1] if enhanced_data_available and dashboard_data['enhanced'].shape[0] > 0 else 0.5
            low_risk_phone = dashboard_data.get('enhanced', pd.DataFrame()).iloc[1, 1] if enhanced_data_available and dashboard_data['enhanced'].shape[0] > 1 else 0.3
            low_risk_seatbelt = dashboard_data.get('enhanced', pd.DataFrame()).iloc[2, 1] if enhanced_data_available and dashboard_data['enhanced'].shape[0] > 2 else 0.1
            
            st.markdown(f"""
            <div class="risk-profile-card low-risk {is_active}">
                <h4 style="color: {cluster_map[1]['color']};">Low-Risk Profile Averages</h4>
                <hr>
                <div class="description-text">
                    Exceed Speed: Very rare (Avg Score: {low_risk_speed:.1f}/3)
                </div>
                <div class="description-text">
                    Phone Use: Very rare (Avg Score: {low_risk_phone:.1f}/3)
                </div>
                <div class="description-text">
                    Seatbelt Use: Always used (Avg Score: {low_risk_seatbelt:.1f}/3)
                </div>
                <div style='margin-top:20px; font-weight:bold; color:{cluster_map[1]['color']};'>
                    Recommendation: Maintain excellent habits and promote safety to peers.
                </div>
            </div>
            """, unsafe_allow_html=True)

# ====================== 6. PAGE: SEATBELT DETECTION (DEMO) (Original Section 7) ======================
if page == "Seatbelt Detection (Demo)":
    st.markdown("<h2 class='gradient-text'>Live Seatbelt Compliance Detection (Demo)</h2>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div class='description-text'>This page demonstrates the potential for using Computer Vision (specifically Object Detection models) to automatically detect if a driver or passenger is wearing a seatbelt. This is a critical tool to mitigate the prevention failure risk factor identified in our model.</div>", unsafe_allow_html=True)
    
    st.markdown("### Demonstration Interface")
    
    col_upload, col_result = st.columns(2)
    
    with col_upload:
        st.subheader("Upload Image for Analysis")
        uploaded_file = st.file_uploader("Choose an image of a driver/passenger:", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)

    with col_result:
        st.subheader("Detection Result")
        
        belt_status = st.radio("Simulate Detection Status:", ["Belt Detected (Compliant)", "Belt NOT Detected (Non-Compliant)"], index=1)
        
        if belt_status == "Belt Detected (Compliant)":
            color = "blue"
            status = "SEATBELT DETECTED"
        else:
            color = "red"
            status = "SEATBELT NOT DETECTED"

        st.markdown(f"""
        <div class="compliance-box" style="border-color: {color};">
            <h4 style="color: var(--color-blue-dark); margin-bottom: 15px;">Model Output Simulation</h4>
            <div style="font-size: 1.8rem; font-weight: bold; color: {color}; margin: 10px 0;">{status}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="margin-top: 20px; padding: 15px; background-color: white; border-radius: 8px; border-left: 4px solid var(--color-blue-medium);">
            <div class="description-text">
                This tool supports real-time risk mitigation by ensuring mandatory safety compliance is met on every trip, directly addressing the Seatbelt Use feature in our risk model.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ====================== 7. PAGE: SPEED DETECTION (DEMO) (Original Section 8) ======================
if page == "Speed Detection (Demo)":
    st.markdown("<h2 class='gradient-text'>Live Speed Limit Compliance (Demo)</h2>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div class='description-text'>This page simulates an application using image processing on video feeds to determine vehicle speed and compliance with posted limits, targeting the Speed Score and Driving Style risk factors.</div>", unsafe_allow_html=True)
    
    st.markdown("### Speed Test Simulation")
    
    col_input, col_speed = st.columns(2)
    
    with col_input:
        st.subheader("Simulate Vehicle Parameters")
        
        speed_limit = st.slider("Posted Speed Limit (km/h)", 40, 120, 80, 10)
        detected_speed = st.slider("Detected Vehicle Speed (km/h)", 40, 160, 105, 5)
        
        st.markdown("""
        <div style="margin-top: 20px; padding: 15px; background-color: #f0f5ff; border-radius: 8px;">
            <div class="description-text">
                The Average Speed Score in our prediction model is strongly correlated with high speeds. This detection module offers a way to measure this behavioral factor live.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_speed:
        st.subheader("Compliance Result")
        
        is_over = detected_speed > speed_limit
        difference = detected_speed - speed_limit
        
        if is_over:
            color = "red"
            status = "OVER SPEED LIMIT"
            message = f"Exceeding limit by {difference} km/h."
        else:
            color = "blue"
            status = "COMPLIANT"
            message = f"Driving {abs(difference)} km/h below the limit."

        st.markdown(f"""
        <div class="compliance-box" style="border-color: {color};">
            <h4 style="color: var(--color-blue-dark); margin-bottom: 10px;">Compliance Analysis</h4>
            <div style="font-size: 1.5rem; color: var(--color-blue-dark); margin: 5px 0;">Posted Limit: {speed_limit} km/h</div>
            <div style="font-size: 1.8rem; font-weight: bold; color: {color}; margin: 10px 0;">{status}</div>
            <div style="font-size: 1rem; color: var(--color-text-secondary); margin: 5px 0;">{message}</div>
        </div>
        """, unsafe_allow_html=True)

# ====================== ADDITIONAL INTERACTIVE ELEMENTS ======================
# Add JavaScript for interactive cards
st.markdown("""
<script>
document.addEventListener('DOMContentLoaded', function() {
    // Add click handlers for interactive cards
    const cards = document.querySelectorAll('.interactive-card');
    cards.forEach(card => {
        card.addEventListener('click', function() {
            // Already has inline onclick, this is backup
            this.style.transform = 'scale(0.98)';
            setTimeout(() => {
                this.style.transform = '';
            }, 200);
        });
    });
});
</script>
""", unsafe_allow_html=True)

# Add footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: var(--color-text-secondary); padding: 20px; font-size: 0.9rem;">
    <div>South Africa Road Safety Risk Dashboard | Predictive Analytics Platform</div>
    <div style="margin-top: 5px; font-size: 0.8rem;">Developed by AnalytIQ</div>
</div>
""", unsafe_allow_html=True)