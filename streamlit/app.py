import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_echarts import st_echarts
from tensorflow.keras.models import load_model
from styles import HEADER_STYLE     
import folium
from streamlit_folium import st_folium
import streamlit.components.v1 as components
from parcel_coords import parcel_coords  
import os

##--------------------------------------------------------------------------------------- RUTAS
BASE_DIR = os.path.dirname(__file__)
LOGO_PATH = os.path.join(BASE_DIR, "images", "logo.png")
MODEL_STACK_PATH = os.path.join(BASE_DIR, "models", "modelo_stack.pkl")
MODEL_LSTM_PATH = os.path.join(BASE_DIR, "models", "modelo_lstm.h5")

##--------------------------------------------------------------------------------------- CONFIGURACIÓN
st.set_page_config(
    page_title="GRUPO AMUTIO Predictive Dashboard",
    page_icon=LOGO_PATH,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(HEADER_STYLE, unsafe_allow_html=True)

##--------------------------------------------------------------------------------------- CABECERA
col_logo, col_text = st.columns([1, 12])
with col_logo:
    st.image(LOGO_PATH, width=65)
with col_text:
    st.markdown("""
        <div class="header-container">
            <div>
                <div class="header-title">GRUPO AMUTIO Predictive IA</div>
                <div class="header-subtitle">Centro de Control — Dashboard 2025</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("""<hr style="margin-top:-10px; margin-bottom:20px;">""", unsafe_allow_html=True)

##--------------------------------------------------------------------------------------- SIDEBAR
with st.sidebar:
    st.header("Datos de entrada")
    uploaded_file = st.file_uploader("Archivo de seguimiento semanal (CSV)", type=["csv"])
    uploaded_forecast = st.file_uploader("Archivo de predicción meteorológica (CSV)", type=["csv"])
    st.markdown("---")
    st.caption("Versión MVP 2025")

    with st.expander("ℹ️ Información del Proyecto"):
        st.markdown("""
**Proyecto Inicial Prioritario: Sistema de Predicción de Rendimiento con IA**

Este proyecto busca modernizar y optimizar las predicciones de cosecha de Amutio, que hoy se basan en la experiencia de campo. Con IA, se espera estimar el rendimiento de forma mucho más precisa, integrando datos de manejo agrícola, clima, predicciones meteorológicas y potencialmente datos satelitales. Esto permitirá:

- Ajustar almacenamiento y ventas con antelación
- Reducir pérdidas por sobreproducción o falta de producto
- Mejorar la rentabilidad estimada en 5–10%
- Optimizar la logística de cosecha y contratos
- Posicionarse como referente en agricultura de precisión

**Plan de implementación tentativo:**
- **Mes 0–1**: recopilación y análisis de datos históricos
- **Mes 2–3**: desarrollo y entrenamiento de modelos (ensambles y redes LSTM)
- **Mes 4**: validación piloto con campaña real
- **Mes 5**: despliegue en la nube con dashboard
- **Mes 6**: evaluación y plan de escalado

A medio plazo, la solución será escalable a todas las zonas de cultivo y adaptable a otros cultivos o regiones, dando a Amutio una ventaja tecnológica sólida para el futuro.
        """)
    
##--------------------------------------------------------------------------------------- MODELOS
with st.spinner("Cargando modelos..."):
    stack_model = joblib.load(MODEL_STACK_PATH)
    lstm_model = load_model(MODEL_LSTM_PATH, compile=False)

##--------------------------------------------------------------------------------------- CONTENIDO
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["fecha"])
    parcelas = df["parcela"].unique()

    if uploaded_forecast:
        df_forecast = pd.read_csv(uploaded_forecast, parse_dates=["fecha"])
    else:
        st.sidebar.warning("No se ha cargado predicción meteorológica")

    st.selectbox("Modo de análisis:", ["Visión general", "Detalle por parcela"], key="modo")

    if st.session_state.modo == "Visión general":
        st.subheader("Visión general de predicciones")

        global_preds = []
        for parcela in parcelas:
            datos_p = df[df["parcela"] == parcela].sort_values("fecha")
            mes_cosecha = datos_p["fecha"].max().month
            variedad = datos_p["variedad"].iloc[-1]
            variedad_code = {"Agria": 0, "Monalisa": 1, "Spunta": 2}[variedad]
            tam_parcela_ha = datos_p["tam_parcela_ha"].iloc[-1]
            temp_media = datos_p["temp_media"].mean()
            temp_std = datos_p["temp_media"].std()
            lluvia_total = datos_p["lluvia"].sum()
            fertilizante_total = datos_p["fertilizante"].sum()
            fertilizante_medio = datos_p["fertilizante"].mean()
            riego_total = datos_p["riego"].sum()
            riego_medio = datos_p["riego"].mean()

            X_input = np.array([[variedad_code, tam_parcela_ha, temp_media, temp_std,
                                 lluvia_total, fertilizante_total, fertilizante_medio,
                                 riego_total, riego_medio, mes_cosecha]])
            pred = stack_model.predict(X_input)[0]
            global_preds.append(pred)

        col_a, col_b, col_c, col_d = st.columns(4)

        with col_a:
            st.metric("Rendimiento medio", f"{np.mean(global_preds):.2f} ton/ha")
        with col_b:
            st.metric("Mejor parcela", f"{np.max(global_preds):.2f} ton/ha")
        with col_c:
            st.metric("Peor parcela", f"{np.min(global_preds):.2f} ton/ha")
        with col_d:
            st.metric("Variabilidad", f"{np.std(global_preds):.2f} ton/ha")

        st.markdown("#### Ranking de parcelas")
        ranking_df = pd.DataFrame({
            "Parcela": parcelas,
            "Predicción rendimiento (ton/ha)": global_preds
        }).sort_values(by="Predicción rendimiento (ton/ha)", ascending=False)
        st.dataframe(ranking_df, use_container_width=True)

        st.markdown("#### Distribución de rendimiento por parcela")
        bar_chart_opt = {
            "xAxis": {"type": "category", "data": ranking_df["Parcela"].tolist()},
            "yAxis": {"type": "value", "name": "ton/ha", "nameLocation": "middle", "nameGap": 30},
            "series": [{
                "data": [round(x, 2) for x in ranking_df["Predicción rendimiento (ton/ha)"]],
                "type": "bar",
                "label": {"show": True, "position": "top"},
                "itemStyle": {"color": "#4CAF50"}
            }],
            "tooltip": {"trigger": "axis"},
            "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
        }
        st_echarts(options=bar_chart_opt, height="400px")

        st.markdown("#### Mapa de parcelas")

        def show_static_map(global_preds, ranking_df, parcel_coords):
            min_pred = np.min(global_preds)
            max_pred = np.max(global_preds)

            def rendimiento_color(value):
                ratio = (value - min_pred) / (max_pred - min_pred + 1e-8)
                r = int(255 * (1 - ratio))
                g = int(180 * ratio)
                b = 0
                return f"rgb({r},{g},{b})"

            all_coords = [c for coords in parcel_coords.values() for c in coords]
            mean_lat = np.mean([c[0] for c in all_coords])
            mean_lon = np.mean([c[1] for c in all_coords])

            m = folium.Map(location=[mean_lat, mean_lon], zoom_start=13, tiles="cartodbpositron")
            m.fit_bounds([[min(c[0] for c in all_coords), min(c[1] for c in all_coords)],
                          [max(c[0] for c in all_coords), max(c[1] for c in all_coords)]])

            for parcela in ranking_df["Parcela"].tolist():
                coords = parcel_coords.get(parcela)
                if coords:
                    rendimiento = ranking_df.loc[ranking_df["Parcela"] == parcela, "Predicción rendimiento (ton/ha)"].values[0]
                    color = rendimiento_color(rendimiento)
                    folium.Polygon(
                        locations=coords, color="black", fill=True, fill_opacity=0.7, fill_color=color,
                        weight=1, popup=f"<b>{parcela}</b><br>Rendimiento: {rendimiento:.2f} ton/ha"
                    ).add_to(m)
            return m._repr_html_()

        if "static_map" not in st.session_state:
            st.session_state["static_map"] = show_static_map(global_preds, ranking_df, parcel_coords)

        components.html(st.session_state["static_map"], height=600, scrolling=False)

    else:
        st.info("Funcionalidad 'Detalle por parcela' disponible tras elegir archivo.")
else:
    st.info("Por favor sube el archivo CSV de seguimiento semanal para comenzar.")

st.markdown("""
<div class="footer">
GRUPO AMUTIO Predictive IA — Monitorización en vivo | 2025
</div>
""", unsafe_allow_html=True)
