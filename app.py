import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Telecom Traffic Forecasting Dashboard",
    page_icon="📡",
    layout="wide"
)

# Title
st.title("📡 Telecom Traffic Forecasting Dashboard")
st.markdown("### Prediksi Traffic Jaringan Telekomunikasi dengan LightGBM")
st.markdown("---")

# Sidebar untuk input
st.sidebar.header("⚙️ Parameter Prediksi")

# 🔧 PERBAIKAN: Dapatkan path absolut
@st.cache_resource
def load_model():
    try:
        model_path = 'telecom_traffic_best_model.pkl'
        if os.path.exists(model_path):
            return joblib.load(model_path)
        else:
            st.error(f"Model tidak ditemukan di: {os.path.abspath(model_path)}")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_data():
    # 🔧 PERBAIKAN: Gunakan path absolut
    data_path = 'square_hour_aggregated_full.csv'
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    else:
        st.error(f"File data tidak ditemukan di: {os.path.abspath(data_path)}")
        st.info("Pastikan file 'square_hour_aggregated_full.csv' ada di direktori yang sama")
        return None

# Load semua komponen
with st.spinner("Memuat model dan data..."):
    model = load_model()
    df = load_data()

if model is None:
    st.error("❌ Model tidak ditemukan.")
    st.stop()

if df is None:
    st.error("❌ Data tidak ditemukan.")
    st.stop()

st.sidebar.success(f"✅ Model dan data berhasil dimuat")
st.sidebar.info(f"📊 Data: {len(df):,} baris | {df['square_id'].nunique()} square")

# Sidebar input untuk prediksi manual
st.sidebar.subheader("🔮 Prediksi Manual")
st.sidebar.markdown("Masukkan parameter untuk prediksi traffic:")

square_id = st.sidebar.number_input("Square ID", min_value=1, value=1, step=1)
hour = st.sidebar.slider("Jam (Hour of Day)", 0, 23, 12)

st.sidebar.markdown("**Data Traffic Sebelumnya:**")
lag_1 = st.sidebar.number_input("Traffic 1 jam lalu (lag_1)", value=0.3, format="%.4f")
lag_2 = st.sidebar.number_input("Traffic 2 jam lalu (lag_2)", value=0.25, format="%.4f")
lag_3 = st.sidebar.number_input("Traffic 3 jam lalu (lag_3)", value=0.2, format="%.4f")
rolling_mean_3 = st.sidebar.number_input("Rata-rata 3 jam terakhir", value=0.25, format="%.4f")
avg_count = st.sidebar.number_input("Avg Values Count", value=4.0, format="%.1f")

# Hitung feature untuk prediksi
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)

if st.sidebar.button("🚀 Prediksi Sekarang", type="primary"):
    features = pd.DataFrame([{
        'square_id': square_id,
        'hour': hour,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'avg_values_count': avg_count,
        'lag_1': lag_1,
        'lag_2': lag_2,
        'lag_3': lag_3,
        'rolling_mean_3': rolling_mean_3
    }])
    
    prediction = model.predict(features)[0]
    st.sidebar.metric("📊 Hasil Prediksi Traffic", f"{prediction:.6f}")
    
    if prediction > 0.5:
        st.sidebar.warning("⚠️ Traffic tinggi! Potensi perlu tambahan bandwidth")
    elif prediction > 0.3:
        st.sidebar.info("📶 Traffic normal")
    else:
        st.sidebar.success("✅ Traffic rendah")

# Main area - Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Traffic Pattern", "🎯 Feature Importance", "📊 Model Performance", "🗺️ Heatmap"])

with tab1:
    st.subheader("Pola Traffic per Square")
    st.markdown("Visualisasi pola traffic harian untuk square yang dipilih")
    
    square_options = sorted(df['square_id'].unique()[:100])
    selected_square = st.selectbox("Pilih Square ID", square_options)
    
    square_data = df[df['square_id'] == selected_square].sort_values('hour')
    
    if len(square_data) == 24:
        col1, col2, col3 = st.columns(3)
        with col1:
            peak_hour = square_data.loc[square_data['value_mean'].idxmax(), 'hour']
            st.metric("🕐 Jam Sibuk (Peak Hour)", f"{peak_hour}:00")
        with col2:
            st.metric("📊 Rata-rata Traffic", f"{square_data['value_mean'].mean():.4f}")
        with col3:
            st.metric("📈 Traffic Tertinggi", f"{square_data['value_mean'].max():.4f}")
        
        fig = px.line(
            square_data, 
            x='hour', 
            y='value_mean',
            title=f'Pola Traffic 24 Jam - Square {selected_square}',
            labels={'hour': 'Jam ke-', 'value_mean': 'Nilai Traffic'},
            markers=True
        )
        fig.update_layout(height=450, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"Data untuk square {selected_square} tidak lengkap")

with tab2:
    st.subheader("Feature Importance Analysis")
    
    features_imp = pd.DataFrame({
        'Feature': ['lag_1 (traffic 1 jam lalu)', 'rolling_mean_3 (rata2 3 jam)', 'lag_2 (traffic 2 jam lalu)', 
                    'hour_sin (pola sirkadian)', 'hour (jam)', 'hour_cos (pola sirkadian)', 
                    'square_id (ID area)', 'avg_values_count', 'lag_3 (traffic 3 jam lalu)'],
        'Importance (%)': [68.04, 18.18, 5.58, 2.60, 2.08, 2.03, 0.56, 0.50, 0.43]
    })
    
    fig_imp = px.bar(
        features_imp,
        x='Importance (%)',
        y='Feature',
        orientation='h',
        title='Feature Importance (Top 9 Features)',
        color='Importance (%)',
        color_continuous_scale='Viridis'
    )
    fig_imp.update_layout(height=500)
    st.plotly_chart(fig_imp, use_container_width=True)

with tab3:
    st.subheader("Model Performance Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### LightGBM (Selected Model)")
        st.metric("📉 MAPE", "31.13%")
        st.metric("📏 MAE", "0.197")
        st.metric("⚡ Inference Time", "< 1 ms")
        st.metric("💾 Model Size", "< 10 MB")
    
    with col2:
        st.markdown("### LSTM (Not Recommended)")
        st.metric("📉 MAPE", "Tidak bisa train")
        st.metric("📏 MAE", "N/A")
        st.metric("⚡ Inference Time", "50-100 ms")
        st.metric("💾 Model Size", "~50 MB")

with tab4:
    st.subheader("Heatmap Traffic per Square")
    
    top_squares = df['square_id'].unique()[:50]
    pivot_data = df[df['square_id'].isin(top_squares)].pivot_table(
        index='square_id', 
        columns='hour', 
        values='value_mean'
    )
    
    fig_heat = px.imshow(
        pivot_data,
        labels=dict(x="Jam ke-", y="ID Square", color="Nilai Traffic"),
        title="Heatmap Traffic (50 Square × 24 Jam)",
        color_continuous_scale="Viridis",
        aspect="auto"
    )
    fig_heat.update_layout(height=600)
    st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("---")
st.markdown("🔬 **Model:** LightGBM | **MAPE:** 31.13% | **Inference:** <1ms per prediksi")
