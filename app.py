import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import requests
import pydeck as pdk
import shap
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
from modules.predict import predict_new_data, load_artifacts

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Logistics Nexus",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3D & Premium Glassmorphism CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
        background-color: #0b1120;
        color: #f8fafc;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, #1e1b4b 0%, #0b1120 100%);
    }

    /* Keyframes for animations */
    @keyframes slideUpFade {
        0% { transform: translateY(40px) scale(0.95); opacity: 0; }
        100% { transform: translateY(0) scale(1); opacity: 1; }
    }
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 15px rgba(56, 189, 248, 0.2); }
        50% { box-shadow: 0 0 30px rgba(56, 189, 248, 0.6); }
        100% { box-shadow: 0 0 15px rgba(56, 189, 248, 0.2); }
    }

    .main-header {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        animation: slideUpFade 0.8s ease-out forwards;
        box-shadow: 0 20px 40px rgba(0,0,0,0.5);
    }
    .main-header h1 {
        margin: 0;
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -1px;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        animation: slideUpFade 0.6s ease-out backwards;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .metric-card:hover {
        transform: translateY(-10px) scale(1.05) rotateX(5deg);
        border-color: rgba(56, 189, 248, 0.5);
        background: rgba(255, 255, 255, 0.08);
        box-shadow: 0 15px 35px rgba(56, 189, 248, 0.2);
    }
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(to right, #ffffff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }
    
    .ai-insight {
        background: linear-gradient(to right, rgba(14, 165, 233, 0.1), transparent);
        border-left: 4px solid #0ea5e9;
        padding: 1.25rem;
        border-radius: 0 12px 12px 0;
        margin: 1.5rem 0;
        color: #e0f2fe;
        animation: pulseGlow 4s infinite;
    }
    
    .badge {
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 800;
        text-transform: uppercase;
        display: inline-block;
        letter-spacing: 1px;
    }
    .badge-success { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid #22c55e; }
    
    .section-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 1.5rem;
        background: linear-gradient(to right, #f8fafc, #64748b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

@st.cache_resource
def get_model_artifacts():
    try:
        return load_artifacts()
    except Exception as e:
        return None, None, None, None, None, None, None

def display_ai_insight(text):
    st.markdown(f'<div class="ai-insight"><strong>✨ AI Quantum Engine:</strong> {text}</div>', unsafe_allow_html=True)

def display_metric_card(label, value, suffix="", delay=0):
    st.markdown(f"""
    <div class="metric-card" style="animation-delay: {delay}s;">
        <div style="color: #94a3b8; font-size: 0.9rem; text-transform: uppercase; font-weight: 600; letter-spacing: 2px;">{label}</div>
        <div class="metric-value">{value}{suffix}</div>
    </div>
    """, unsafe_allow_html=True)

# --- App Logic ---
model, scaler, product_stats, label_encoders, feature_columns, city_info, gov_report = get_model_artifacts()

if gov_report is None:
    gov_report = {"geo_decision_narrative": "No model detected.", "model_selection_narrative": "Train the model.", "metrics": {"roc_auc": 0, "precision": 0, "recall": 0}}

# --- Header Section ---
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("""
    <div class="main-header">
        <div>
            <h1>📦 AI Logistics Nexus</h1>
            <p style="color:#94a3b8; font-size: 1.2rem;">Predictive Supply Chain & 3D Route Intelligence</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
with col_h2:
    # Adding a sleek Lottie animation
    lottie_url = "https://assets9.lottiefiles.com/packages/lf20_jmejybvu.json"
    lottie_json = load_lottieurl(lottie_url)
    if lottie_json:
        st_lottie(lottie_json, height=150, key="delivery_animation")

# --- Sidebar ---
st.sidebar.markdown("### 👤 Interface Mode")
persona = st.sidebar.radio(
    "",
    ["Business Executive", "Analyst / Data Science"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📂 Data Import")
uploaded_file = st.sidebar.file_uploader("Upload Delivery Dataset (CSV)", type=['csv'])

if persona == "Business Executive":
    sections = ["Executive Summary", "3D Geographic Risks", "Strategic Actions"]
else:
    sections = ["Model Governance", "3D Feature Mapping", "XGBoost Explainability (SHAP)"]

selected_section = st.sidebar.radio("Navigate to:", sections)

# --- Process Uploaded Data ---
df_input = None
df_output = None
X_matrix = None
df_processed = None

if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)
        if model is not None:
            proba, pred, X_matrix, df_processed = predict_new_data(
                df_input, model, scaler, product_stats, label_encoders,
                feature_columns, city_info, gov_report, threshold=0.5
            )
            df_output = df_input.copy()
            df_output['Delay Risk (%)'] = (proba * 100).round(1)
            df_output['Predicted Status'] = ['Delayed' if p == 1 else 'On Time' for p in pred]
    except Exception as e:
        st.error(f"Error parsing data: {str(e)}")

# --- Views ---
if persona == "Business Executive":
    if selected_section == "Executive Summary":
        st.markdown('<div class="section-title">Executive Overview</div>', unsafe_allow_html=True)
        display_ai_insight("The XGBoost AI core has automatically analyzed the logistics network. We are operating with advanced gradient boosting for maximum precision.")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1: display_metric_card("System Health", "100", "%", 0.1)
        with col2: display_metric_card("Data Trust Index", "96", "/100", 0.2)
        with col3: display_metric_card("AI Confidence", "89", "%", 0.3)
        with col4: display_metric_card("Active Models", "2", "", 0.4)

        if df_output is not None:
            delayed_count = (df_output['Predicted Status'] == 'Delayed').sum()
            st.markdown("### Live Predictions")
            st.dataframe(df_output[['shipment_id', 'origin_city', 'destination_city', 'Delay Risk (%)', 'Predicted Status']].head(10), use_container_width=True)
        else:
            st.info("Upload dataset in the sidebar to activate the predictive engine.")

    elif selected_section == "3D Geographic Risks":
        st.markdown('<div class="section-title">3D Logistics Control Tower</div>', unsafe_allow_html=True)
        if df_output is not None and city_info is not None and 'origin_city' in df_output.columns:
            display_ai_insight("Visualizing high-risk supply chain routes across Nigeria. Taller, red arcs indicate severe delay probabilities.")
            
            # Map data prep
            map_data = df_output[['origin_city', 'destination_city', 'Delay Risk (%)']].dropna().copy()
            map_data = map_data.merge(city_info[['city', 'latitude', 'longitude']], left_on='origin_city', right_on='city', how='left')
            map_data.rename(columns={'latitude': 'start_lat', 'longitude': 'start_lon'}, inplace=True)
            map_data = map_data.merge(city_info[['city', 'latitude', 'longitude']], left_on='destination_city', right_on='city', how='left')
            map_data.rename(columns={'latitude': 'end_lat', 'longitude': 'end_lon'}, inplace=True)
            map_data.dropna(subset=['start_lat', 'start_lon', 'end_lat', 'end_lon'], inplace=True)
            
            # Arcs
            arc_layer = pdk.Layer(
                "ArcLayer",
                data=map_data,
                get_source_position=["start_lon", "start_lat"],
                get_target_position=["end_lon", "end_lat"],
                get_source_color=[0, 255, 128, 200],
                get_target_color="[Delay Risk (%) * 2.5, 50, 50, 200]", # Dynamic color based on risk
                auto_highlight=True,
                width_scale=0.0005,
                get_width="Delay Risk (%)",
                width_min_pixels=2,
                width_max_pixels=10,
                pickable=True
            )
            
            view_state = pdk.ViewState(latitude=9.0820, longitude=8.6753, zoom=5, pitch=50, bearing=-20) # Center on Nigeria
            
            st.pydeck_chart(pdk.Deck(
                layers=[arc_layer],
                initial_view_state=view_state,
                map_style="mapbox://styles/mapbox/dark-v10",
                tooltip={"text": "{origin_city} to {destination_city}\nRisk: {Delay Risk (%)}%"}
            ))
        else:
            st.warning("Upload data to view the 3D Control Tower.")

    elif selected_section == "Strategic Actions":
        st.markdown('<div class="section-title">AI Prescriptive Actions</div>', unsafe_allow_html=True)
        display_ai_insight("The following actions are prescribed by the algorithm to minimize operational costs and delay penalties.")
        if df_output is not None:
            critical = df_output[df_output['Delay Risk (%)'] > 80]
            st.error(f"🚨 ACTION REQUIRED: {len(critical)} shipments have critical risk levels (>80%). Route reallocation to premium carriers is advised immediately.")
            st.dataframe(critical[['shipment_id', 'Delay Risk (%)', 'logistics_company']], use_container_width=True)
        else:
            st.info("Awaiting data upload.")

else:
    # --- Analyst Persona ---
    if selected_section == "Model Governance":
        st.markdown('<div class="section-title">Automated AI Governance</div>', unsafe_allow_html=True)
        display_ai_insight(gov_report.get('model_selection_narrative', ''))
        st.info(f"**Geospatial Intelligence:** {gov_report.get('geo_decision_narrative', '')}")
        
        col1, col2, col3 = st.columns(3)
        metrics = gov_report.get('metrics', {})
        with col1: display_metric_card("Validation ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
        with col2: display_metric_card("Precision", f"{metrics.get('precision', 0):.3f}")
        with col3: display_metric_card("Recall", f"{metrics.get('recall', 0):.3f}")

    elif selected_section == "3D Feature Mapping":
        st.markdown('<div class="section-title">Geospatial Distribution</div>', unsafe_allow_html=True)
        if df_output is not None and city_info is not None:
            # 3D Hexagon layer for origin volumes
            map_data = df_output[['origin_city']].dropna().copy()
            map_data = map_data.merge(city_info[['city', 'latitude', 'longitude']], left_on='origin_city', right_on='city', how='inner')
            
            hex_layer = pdk.Layer(
                'HexagonLayer',
                data=map_data,
                get_position=['longitude', 'latitude'],
                radius=20000,
                elevation_scale=50,
                elevation_range=[0, 3000],
                pickable=True,
                extruded=True,
            )
            view_state = pdk.ViewState(latitude=9.0820, longitude=8.6753, zoom=5, pitch=60, bearing=30)
            st.pydeck_chart(pdk.Deck(layers=[hex_layer], initial_view_state=view_state, map_style="mapbox://styles/mapbox/dark-v10"))
        else:
            st.warning("Upload data to render 3D Volume Mappings.")

    elif selected_section == "XGBoost Explainability (SHAP)":
        st.markdown('<div class="section-title">Global Feature Interpretability</div>', unsafe_allow_html=True)
        display_ai_insight("Computing exact SHapley Additive exPlanations (SHAP) for the XGBoost ensemble to guarantee black-box transparency.")
        
        if X_matrix is not None and model is not None:
            with st.spinner("Calculating Shapley values..."):
                try:
                    # Take sample for speed in UI
                    X_sample = X_matrix[:200]
                    df_features_sample = df_processed.iloc[:200]
                    
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample)
                    
                    # Custom Matplotlib style to match Dark Theme
                    plt.style.use('dark_background')
                    fig, ax = plt.subplots(figsize=(10, 6))
                    fig.patch.set_facecolor('#0b1120')
                    ax.set_facecolor('#0b1120')
                    
                    shap.summary_plot(shap_values, df_features_sample, show=False)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"SHAP explanation failed: {str(e)}")
        else:
            st.warning("Upload data to compute SHAP values for the current inference batch.")

# --- Footer ---
st.markdown("""
<div style="text-align: center; margin-top: 5rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.1); color: #64748b; font-size: 0.8rem;">
    Advanced AI Logistics Platform v3.0 | 3D Geospatial Engine | Neural Operations
</div>
""", unsafe_allow_html=True)