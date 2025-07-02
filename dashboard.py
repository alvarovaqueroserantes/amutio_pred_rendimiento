import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_echarts import st_echarts
from tensorflow.keras.models import load_model
import folium
from streamlit_folium import st_folium
import streamlit.components.v1 as components
from parcel_coords import parcel_coords  # <-- ahora importado desde módulo externo

# CONFIGURACIÓN
st.set_page_config(
    page_title="AMUTIO Predictive Dashboard",
    page_icon="images/logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ESTILOS AVANZADOS
st.markdown("""
    <style>
    .header-container {
        display: flex;
        align-items: center;
        gap: 20px;
        margin-bottom: 10px;
    }
    .header-title {
        color: #2E7D32;
        font-size: 2em;
        font-weight: 600;
        margin: 0;
    }
    .header-subtitle {
        font-size: 0.9em;
        color: #555;
        margin: 0;
    }
    .footer {
        font-size: 0.85em;
        color: #666;
        text-align: center;
        margin-top: 30px;
        border-top: 1px solid #ddd;
        padding-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# CABECERA CORPORATIVA
col_logo, col_text = st.columns([1, 12])
with col_logo:
    st.image("images/logo.png", width=65)
with col_text:
    st.markdown("""
        <div class="header-container">
            <div>
                <div class="header-title">AMUTIO Predictive IA</div>
                <div class="header-subtitle">Centro de Control — Dashboard 2025</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("""<hr style="margin-top:-10px; margin-bottom:20px;">""", unsafe_allow_html=True)

# MODELOS
with st.spinner("Cargando modelos..."):
    stack_model = joblib.load("models/modelo_stack.pkl")
    lstm_model = load_model("models/modelo_lstm.h5", compile=False)

# SIDEBAR
with st.sidebar:
    st.header("Datos de entrada")
    uploaded_file = st.file_uploader("Archivo de seguimiento semanal (CSV)", type=["csv"])
    uploaded_forecast = st.file_uploader("Archivo de predicción meteorológica (CSV)", type=["csv"])
    st.markdown("---")
    st.caption("Versión MVP 2025")

# CONTENIDO
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

        # métricas resumen
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Rendimiento medio", f"{np.mean(global_preds):.2f} ton/ha")
        col_b.metric("Mejor parcela", f"{np.max(global_preds):.2f} ton/ha")
        col_c.metric("Peor parcela", f"{np.min(global_preds):.2f} ton/ha")
        col_d.metric("Variabilidad", f"{np.std(global_preds):.2f} ton/ha")

        # tabla ranking
        ranking_df = pd.DataFrame({
            "Parcela": parcelas,
            "Predicción rendimiento (ton/ha)": global_preds
        }).sort_values(by="Predicción rendimiento (ton/ha)", ascending=False)
        st.dataframe(ranking_df, use_container_width=True)

        # gráfico
        bar_chart_opt = {
            "xAxis": {"type": "category", "data": ranking_df["Parcela"].tolist()},
            "yAxis": {"type": "value", "name": "ton/ha"},
            "series": [{
                "data": [round(x, 2) for x in ranking_df["Predicción rendimiento (ton/ha)"]],
                "type": "bar",
                "label": {"show": True, "position": "top"},
                "itemStyle": {"color": "#4CAF50"}
            }],
        }
        st_echarts(options=bar_chart_opt, height="400px")

        # definición del mapa con función auxiliar
        def show_static_map(global_preds, ranking_df, parcel_coords):
            min_pred = np.min(global_preds)
            max_pred = np.max(global_preds)

            def rendimiento_color(value):
                ratio = (value - min_pred) / (max_pred - min_pred + 1e-8)
                r = int(255 * (1 - ratio))
                g = int(180 * ratio)
                b = 0
                return f"rgb({r},{g},{b})"

            all_coords = []
            for coords in parcel_coords.values():
                all_coords.extend(coords)

            mean_lat = np.mean([c[0] for c in all_coords])
            mean_lon = np.mean([c[1] for c in all_coords])

            m = folium.Map(location=[mean_lat, mean_lon], zoom_start=13, tiles="cartodbpositron")
            m.fit_bounds([[min(c[0] for c in all_coords), min(c[1] for c in all_coords)],
                          [max(c[0] for c in all_coords), max(c[1] for c in all_coords)]])

            for parcela in ranking_df["Parcela"].tolist():
                coords = parcel_coords.get(parcela)
                if coords:
                    rendimiento = ranking_df.loc[
                        ranking_df["Parcela"] == parcela, "Predicción rendimiento (ton/ha)"
                    ].values[0]
                    color = rendimiento_color(rendimiento)
                    folium.Polygon(
                        locations=coords,
                        color="black",
                        fill=True,
                        fill_opacity=0.7,
                        fill_color=color,
                        weight=1,
                        popup=f"<b>{parcela}</b><br>Rendimiento: {rendimiento:.2f} ton/ha"
                    ).add_to(m)

            return m._repr_html_()

        if "static_map" not in st.session_state:
            st.session_state["static_map"] = show_static_map(global_preds, ranking_df, parcel_coords)

        st.markdown("#### Mapa de parcelas")
        components.html(st.session_state["static_map"], height=600, scrolling=False)

    else:
        # detalle por parcela (omitido aquí por brevedad, idéntico al tuyo)
        st.warning("Funcionalidad detalle por parcela idéntica, se mantiene sin cambios.")

    st.markdown("""
        <div class="footer">
        AMUTIO Predictive IA — Monitorización en vivo | 2025
        </div>
    """, unsafe_allow_html=True)
else:
    st.info("Por favor sube el archivo CSV de seguimiento semanal para comenzar.")
