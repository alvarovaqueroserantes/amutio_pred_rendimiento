import streamlit as st
import pandas as pd
import numpy as np

# módulos de tu estructura real
from app import (
    config,
    styles,
    models,
    metrics,
    visuals,
    map_utils,
    parcels,
    footer
)

# configuración de página y estilos
config.set_page_config()
styles.load_custom_styles()

# cabecera
visuals.render_header()

# carga de modelos
with st.spinner("Cargando modelos..."):
    stack_model, lstm_model = models.load_models()

# sidebar
uploaded_file = visuals.sidebar_upload()
uploaded_forecast = visuals.sidebar_forecast()

# contenido
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["fecha"])
    parcelas_list = df["parcela"].unique()

    if uploaded_forecast:
        df_forecast = pd.read_csv(uploaded_forecast, parse_dates=["fecha"])
    else:
        st.sidebar.warning("No se ha cargado predicción meteorológica")

    modo = st.selectbox("Modo de análisis:", ["Visión general", "Detalle por parcela"])

    if modo == "Visión general":
        global_preds, ranking_df = metrics.calculate_global_metrics(df, parcelas_list, stack_model)

        metrics.display_global_metrics(global_preds)
        metrics.display_environmental_metrics(df)

        st.markdown("#### Ranking de parcelas")
        st.dataframe(ranking_df, use_container_width=True)

        visuals.plot_bar_chart(ranking_df)

        st.markdown("#### Mapa de parcelas")
        static_map = map_utils.show_static_map(global_preds, ranking_df, parcels.parcel_coords)
        st.components.v1.html(static_map, height=600, scrolling=False)

    else:
        parcela_sel = st.selectbox("Selecciona parcela", parcelas_list)
        datos_p = df[df["parcela"] == parcela_sel].sort_values("fecha")
        st.subheader(f"Detalle parcela: {parcela_sel}")

        pred_stack, pred_lstm_py = metrics.calculate_parcela_metrics(datos_p, stack_model, lstm_model)

        col1, col2, col3 = st.columns(3)
        col1.metric("Actual (XGBoost)", f"{pred_stack:.2f} ton/ha")
        col2.metric("Última semana (LSTM)", f"{pred_lstm_py[-1]:.2f} ton/ha")
        col3.metric("Mejor semana", f"{np.argmax(pred_lstm_py)+1}")

        if uploaded_forecast:
            df_forecast_p = df_forecast[df_forecast["parcela"] == parcela_sel]
            metrics.display_recommendations(df_forecast_p, datos_p, pred_lstm_py, lstm_model)

        st.markdown("#### Últimos registros")
        st.dataframe(datos_p.tail(5), use_container_width=True)

# footer
footer.render_footer()

if not uploaded_file:
    st.info("Por favor sube el archivo CSV de seguimiento semanal para comenzar.")
