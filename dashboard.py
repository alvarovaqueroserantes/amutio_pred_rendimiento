import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_echarts import st_echarts
from tensorflow.keras.models import load_model

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
    stack_model = joblib.load("modelo_stack.pkl")
    lstm_model = load_model("modelo_lstm.h5", compile=False)

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
            variedad_code = {"Agria":0, "Monalisa":1, "Spunta":2}[variedad]
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

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Rendimiento medio (XGBoost)", f"{np.mean(global_preds):.2f} ton/ha")
        col_b.metric("Mejor parcela", f"{np.max(global_preds):.2f} ton/ha")
        col_c.metric("Peor parcela", f"{np.min(global_preds):.2f} ton/ha")

        st.markdown("#### Ranking de parcelas")
        ranking_df = pd.DataFrame({
            "Parcela": parcelas,
            "Predicción rendimiento (ton/ha)": global_preds
        }).sort_values(by="Predicción rendimiento (ton/ha)", ascending=False)
        st.dataframe(ranking_df, use_container_width=True)

    else:
        parcela_sel = st.selectbox("Selecciona parcela", parcelas)
        datos_p = df[df["parcela"] == parcela_sel].sort_values("fecha")
        st.subheader(f"Detalle parcela: {parcela_sel}")

        mes_cosecha = datos_p["fecha"].max().month
        variedad = datos_p["variedad"].iloc[-1]
        variedad_code = {"Agria":0,"Monalisa":1,"Spunta":2}[variedad]
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
        pred_stack = stack_model.predict(X_input)[0]

        week_series = datos_p[["temp_media", "lluvia", "fertilizante", "riego"]].values
        ciclo_weeks = 16
        if len(week_series) < ciclo_weeks:
            pad = np.tile(week_series[-1], (ciclo_weeks - len(week_series), 1))
            week_series = np.vstack([week_series, pad])
        else:
            week_series = week_series[-ciclo_weeks:]

        week_series_norm = (week_series - week_series.min(axis=0)) / (
            week_series.max(axis=0) - week_series.min(axis=0) + 1e-8
        )
        week_series_norm = week_series_norm.reshape((1, ciclo_weeks, 4))
        pred_lstm = lstm_model.predict(week_series_norm)[0].flatten()
        pred_lstm_py = [round(float(v),2) for v in pred_lstm]

        col1, col2, col3 = st.columns(3)
        col1.metric("Actual (XGBoost)", f"{pred_stack:.2f} ton/ha")
        col2.metric("Última semana (LSTM)", f"{pred_lstm_py[-1]:.2f} ton/ha")
        col3.metric("Mejor semana", f"{np.argmax(pred_lstm_py)+1}")

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

            future_opt = [round(pred_lstm_py[-1],2)]
            seq_opt = week_series.tolist()
            fert_acum_opt = datos_p["fertilizante"].sum()
            riego_acum_opt = datos_p["riego"].sum()

            for i in range(4):
                row = df_forecast_p.iloc[min(i,len(df_forecast_p)-1)]
                fert_acum_opt += row["fertilizante_sugerido_kg"]*7
                riego_acum_opt += row["riego_sugerido_mm"]*7
                semana_o = [
                    row["temp_media"],
                    row["lluvia"],
                    fert_acum_opt/(len(seq_opt)+1),
                    riego_acum_opt/(len(seq_opt)+1)
                ]
                seq_opt.append(semana_o)
                seq_o_np = np.array(seq_opt[-ciclo_weeks:])
                seq_o_norm = (seq_o_np - seq_o_np.min(axis=0)) / (
                    seq_o_np.max(axis=0) - seq_o_np.min(axis=0) + 1e-8
                )
                pred_next = lstm_model.predict(seq_o_norm.reshape(1, ciclo_weeks,4))[0][-1]
                future_opt.append(round(float(pred_next),2))

            st.markdown("#### Proyección rendimiento")
            line_opt = {
                "xAxis": {"type": "category", "data": list(range(1,len(pred_lstm_py)+5))},
                "yAxis": {
                    "type": "value",
                    "name": "Toneladas por hectárea",
                    "nameLocation": "middle",
                    "nameGap": 35
                },
                "series": [
                    {
                        "data": pred_lstm_py + [None]*4,
                        "type":"line",
                        "smooth":True,
                        "name":"Histórico",
                        "label": {"show": True, "formatter": "{c:.2f}", "position":"top"},
                        "symbolSize": 6
                    },
                    {
                        "data": [None]*(len(pred_lstm_py)-1)+future_opt,
                        "type":"line",
                        "smooth":True,
                        "name":"Escenario",
                        "label": {"show": True, "formatter": """function(params){return params.value.toFixed(2)}""", "position":"top"},
                        "symbolSize": 6,
                        "lineStyle": {"type":"dashed"}
                    }
                ],
                "tooltip": {"trigger": "axis"}
            }
            st_echarts(options=line_opt, height="500px")

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
