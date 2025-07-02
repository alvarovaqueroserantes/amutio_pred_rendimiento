import streamlit as st
import pandas as pd
import numpy as np
from streamlit_echarts import st_echarts
from load_models import load_stack_model, load_lstm_model
from data_processing import get_features_stack, get_lstm_series
from visuals import render_metric_card
from styles import HEADER_STYLE     
import folium
import streamlit.components.v1 as components
from parcel_coords import parcel_coords  

# CONFIGURACIÓN
st.set_page_config(
    page_title="AMUTIO Predictive Dashboard",
    page_icon="images/logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(HEADER_STYLE, unsafe_allow_html=True)

# CABECERA
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
    stack_model = load_stack_model()
    lstm_model = load_lstm_model()

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

    # VISIÓN GENERAL
    if st.session_state.modo == "Visión general":
        st.subheader("Visión general de predicciones")

        global_preds = []
        for parcela in parcelas:
            datos_p = df[df["parcela"] == parcela].sort_values("fecha")
            X_input = get_features_stack(datos_p)
            pred = stack_model.predict(X_input)[0]
            global_preds.append(pred)

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.markdown(render_metric_card("Rendimiento medio", f"{np.mean(global_preds):.2f} ton/ha", "success"), unsafe_allow_html=True)
        with col_b:
            st.markdown(render_metric_card("Mejor parcela", f"{np.max(global_preds):.2f} ton/ha", "warning"), unsafe_allow_html=True)
        with col_c:
            st.markdown(render_metric_card("Peor parcela", f"{np.min(global_preds):.2f} ton/ha", "danger"), unsafe_allow_html=True)
        with col_d:
            st.markdown(render_metric_card("Variabilidad", f"{np.std(global_preds):.2f} ton/ha", "info"), unsafe_allow_html=True)

        st.markdown("#### Ranking de parcelas")
        ranking_df = pd.DataFrame({
            "Parcela": parcelas,
            "Predicción rendimiento (ton/ha)": global_preds
        }).sort_values(by="Predicción rendimiento (ton/ha)", ascending=False)
        st.dataframe(ranking_df, use_container_width=True)

        # gráfico de barras
        st.markdown("#### Distribución de rendimiento por parcela")
        bar_chart_opt = {
            "xAxis": {"type": "category", "data": ranking_df["Parcela"].tolist()},
            "yAxis": {"type": "value", "name": "ton/ha", "nameLocation": "middle", "nameGap": 30},
            "series": [{
                "data": [round(x, 2) for x in ranking_df["Predicción rendimiento (ton/ha)"]],
                "type": "bar",
                "label": {"show": True, "position": "top"},
                "itemStyle": {"color": "#4CAF50"},
            }],
            "tooltip": {"trigger": "axis"},
        }
        st_echarts(options=bar_chart_opt, height="400px")

        # mapa
        def show_static_map(global_preds, ranking_df, parcel_coords):
            min_pred = np.min(global_preds)
            max_pred = np.max(global_preds)
            def rendimiento_color(value):
                ratio = (value - min_pred) / (max_pred - min_pred + 1e-8)
                r = int(255 * (1 - ratio))
                g = int(180 * ratio)
                return f"rgb({r},{g},0)"
            m = folium.Map(location=[np.mean([c[0] for coords in parcel_coords.values() for c in coords]),
                                     np.mean([c[1] for coords in parcel_coords.values() for c in coords])],
                           zoom_start=13, tiles="cartodbpositron")
            m.fit_bounds([
                [min(c[0] for coords in parcel_coords.values() for c in coords), min(c[1] for coords in parcel_coords.values() for c in coords)],
                [max(c[0] for coords in parcel_coords.values() for c in coords), max(c[1] for coords in parcel_coords.values() for c in coords)]
            ])
            for parcela in ranking_df["Parcela"]:
                coords = parcel_coords.get(parcela)
                if coords:
                    rendimiento = ranking_df.loc[ranking_df["Parcela"] == parcela, "Predicción rendimiento (ton/ha)"].values[0]
                    folium.Polygon(
                        locations=coords,
                        color="black",
                        fill=True,
                        fill_opacity=0.7,
                        fill_color=rendimiento_color(rendimiento),
                        weight=1,
                        popup=f"<b>{parcela}</b><br>Rendimiento: {rendimiento:.2f} ton/ha"
                    ).add_to(m)
            return m._repr_html_()
        if "static_map" not in st.session_state:
            st.session_state["static_map"] = show_static_map(global_preds, ranking_df, parcel_coords)
        st.markdown("#### Mapa de parcelas")
        components.html(st.session_state["static_map"], height=600, scrolling=False)

    # DETALLE PARCELA
    else:
        parcela_sel = st.selectbox("Selecciona parcela", parcelas)
        datos_p = df[df["parcela"] == parcela_sel].sort_values("fecha")
        st.subheader(f"Detalle parcela: {parcela_sel}")

        X_input = get_features_stack(datos_p)
        pred_stack = stack_model.predict(X_input)[0]

        week_series_norm = get_lstm_series(datos_p)
        pred_lstm = lstm_model.predict(week_series_norm)[0].flatten()
        pred_lstm_py = [round(float(v),2) for v in pred_lstm]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(render_metric_card("Actual (XGBoost)", f"{pred_stack:.2f} ton/ha", "success"), unsafe_allow_html=True)
        with col2:
            st.markdown(render_metric_card("Última semana (LSTM)", f"{pred_lstm_py[-1]:.2f} ton/ha", "warning"), unsafe_allow_html=True)
        with col3:
            st.markdown(render_metric_card("Mejor semana", f"Semana {np.argmax(pred_lstm_py)+1}", "info"), unsafe_allow_html=True)

        if uploaded_forecast is not None:
            df_forecast_p = df_forecast[df_forecast["parcela"]==parcela_sel]
            st.markdown("#### Recomendaciones próximas semanas")

            reco_data = []
            for _, row in df_forecast_p.iterrows():
                reco_data.append({
                    "Fecha": row['fecha'].strftime("%d/%m/%Y"),
                    "Temperatura alta": "Sí" if row["temp_media"]>28 else "No",
                    "Lluvia baja": "Sí" if row["lluvia"]<2 else "No",
                    "Riego sugerido": f"{row['riego_sugerido_mm']:.1f} mm/día",
                    "Fertilizante sugerido": f"{row['fertilizante_sugerido_kg']:.2f} kg/día"
                })
            st.dataframe(pd.DataFrame(reco_data), use_container_width=True)

        st.markdown("#### Últimos registros")
        st.dataframe(datos_p.tail(5), use_container_width=True)

    # FOOTER
    st.markdown("""
        <div class="footer">
        AMUTIO Predictive IA — Monitorización en vivo | 2025
        </div>
    """, unsafe_allow_html=True)

else:
    st.info("Por favor sube el archivo CSV de seguimiento semanal para comenzar.")
