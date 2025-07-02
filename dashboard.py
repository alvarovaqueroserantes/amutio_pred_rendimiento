import streamlit as st
import pandas as pd
import numpy as np

# importar módulos reales según estructura que me diste
from app import (
    config,
    styles,
    data_load,
    models,
    metrics,
    visuals,
    map_utils,
    footer
)

# aplicar configuración de página y estilos
st.set_page_config(page_title=config.PAGE_TITLE, page_icon=config.PAGE_ICON)
st.markdown(styles.HEADER_STYLE, unsafe_allow_html=True)

# render cabecera corporativa
st.markdown(f"""
<div class="header-container">
  <img src="{config.PAGE_ICON}" alt="Logo" width="60">
  <div>
    <h1 class="header-title">{config.PAGE_TITLE}</h1>
    <p class="header-subtitle">Monitorización de parcelas agrícolas con IA</p>
  </div>
</div>
""", unsafe_allow_html=True)

# cargar modelos
with st.spinner("Cargando modelos..."):
    stack_model = models.load_stack_model(config.STACK_MODEL_PATH)
    lstm_model = models.load_lstm_model(config.LSTM_MODEL_PATH)

# sidebar
st.sidebar.header("Datos de entrada")
uploaded_file = st.sidebar.file_uploader("Archivo de seguimiento (CSV)", type="csv")
uploaded_forecast = st.sidebar.file_uploader("Predicción meteorológica (CSV opcional)", type="csv")

# contenido principal
if uploaded_file:
    df = data_load.load_tracking_data(uploaded_file)
    parcelas = df["parcela"].unique()

    if uploaded_forecast:
        df_forecast = data_load.load_forecast_data(uploaded_forecast)
    else:
        st.sidebar.warning("No se ha cargado predicción meteorológica")

    # seleccionar modo
    modo = st.selectbox("Modo de análisis:", ["Visión general", "Detalle por parcela"])

    if modo == "Visión general":
        # calcular predicciones globales
        global_preds = []
        ranking = []

        for parcela in parcelas:
            datos_p = df[df["parcela"] == parcela]
            feature_vector = data_load.prepare_features(df, parcela)
            pred_stack = stack_model.predict(feature_vector.reshape(1, -1))[0]
            global_preds.append(pred_stack)
            ranking.append({
                "Parcela": parcela,
                "Predicción rendimiento (ton/ha)": round(pred_stack, 2)
            })

        ranking_df = pd.DataFrame(ranking).sort_values(by="Predicción rendimiento (ton/ha)", ascending=False)

        # render métricas
        visuals.render_kpis(global_preds)
        visuals.render_global_conditions(df)

        # ranking tabla
        st.markdown("#### Ranking de parcelas")
        st.dataframe(ranking_df, use_container_width=True)

        # gráfico de barras
        visuals.render_bar_chart(ranking_df)

        # mapa
        st.markdown("#### Mapa de parcelas")
        static_map = map_utils.show_static_map(global_preds, ranking_df, parcel_coords=config.DEFAULT_LATLON)
        st.components.v1.html(static_map, height=600, scrolling=False)

    else:
        parcela_sel = st.selectbox("Selecciona parcela", parcelas)
        datos_p = df[df["parcela"] == parcela_sel].sort_values("fecha")

        st.subheader(f"Detalle de la parcela **{parcela_sel}**")

        # predicción con stack
        feature_vector = data_load.prepare_features(df, parcela_sel)
        pred_stack = stack_model.predict(feature_vector.reshape(1, -1))[0]

        # predicción con lstm
        week_sequence = data_load.prepare_features(df, parcela_sel)
        pred_lstm = lstm_model.predict(week_sequence)[0]

        # métricas
        col1, col2, col3 = st.columns(3)
        col1.metric("Pred. Stack", f"{pred_stack:.2f} ton/ha")
        col2.metric("Última semana (LSTM)", f"{pred_lstm[-1]:.2f} ton/ha")
        col3.metric("Mejor semana", f"{np.argmax(pred_lstm)+1}")

        # recomendaciones
        if uploaded_forecast is not None:
            df_forecast_p = df_forecast[df_forecast["parcela"] == parcela_sel]
            from app import forecast_utils
            recos = forecast_utils.generate_recommendations(df_forecast_p)
            st.markdown("#### Recomendaciones")
            st.dataframe(recos, use_container_width=True)

        # últimos registros
        st.markdown("#### Últimos registros")
        st.dataframe(datos_p.tail(5), use_container_width=True)

# pie de página
st.markdown(styles.FOOTER_STYLE, unsafe_allow_html=True)
footer.render_footer()

# mensaje si no hay CSV
if not uploaded_file:
    st.info("Por favor sube el archivo CSV de seguimiento semanal para comenzar.")
