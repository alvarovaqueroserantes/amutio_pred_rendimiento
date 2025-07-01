import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_echarts import st_echarts
from tensorflow.keras.models import load_model

# CONFIG
# CONFIG
st.set_page_config(
    page_title="AMUTIO Predictive Dashboard",
    page_icon="images/logo.png",   # usa tu propio logo
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 20px;">
        <img src="images/logo.png" alt="Amutio logo" style="width:60px; height:auto;">
        <div style="color:#4CAF50; font-size: 2.2em; font-weight: bold;">
            AMUTIO Predictive IA - Centro de Control
        </div>
    </div>
    """, unsafe_allow_html=True
)

# MODELS
stack_model = joblib.load("modelo_stack.pkl")
lstm_model = load_model("modelo_lstm.h5", compile=False)

# FILE UPLOADS
with st.sidebar:
    st.markdown("## 📁 Datos de entrada")
    uploaded_file = st.file_uploader("Archivo de seguimiento semanal (CSV)", type=["csv"])

    uploaded_forecast = st.file_uploader("Archivo de predicción meteorológica (CSV)", type=["csv"])

    st.divider()

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["fecha"])
    parcelas = df["parcela"].unique()

    if uploaded_forecast:
        df_forecast = pd.read_csv(uploaded_forecast, parse_dates=["fecha"])
    else:
        st.sidebar.warning("⚠️ No se ha cargado predicción meteorológica")

    st.divider()

    modo = st.selectbox(
        "Selecciona vista de análisis",
        ["Resumen global", "Detalle por parcela"]
    )

    if modo == "Resumen global":
        st.subheader("Resumen global de predicciones")

        global_preds = []
        for parcela in parcelas:
            datos_p = df[df["parcela"]==parcela].sort_values("fecha")
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
            pred = stack_model.predict(X_input)[0]
            global_preds.append(pred)

        st.metric("🔹 Rendimiento medio (XGBoost)", f"{np.mean(global_preds):.2f} ton/ha")
        st.metric("🔹 Mejor parcela", f"{np.max(global_preds):.2f} ton/ha")
        st.metric("🔹 Peor parcela", f"{np.min(global_preds):.2f} ton/ha")

        ranking_df = pd.DataFrame({
            "Parcela": parcelas,
            "Predicción rendimiento actual (ton/ha)": global_preds
        }).sort_values(by="Predicción rendimiento actual (ton/ha)",ascending=False)
        
        st.subheader("Ranking de parcelas")
        st.dataframe(ranking_df, use_container_width=True)

    else:
        parcela_sel = st.selectbox("*Selecciona la parcela a analizar*", parcelas)
        datos_p = df[df["parcela"]==parcela_sel].sort_values("fecha")
        st.subheader(f"🟢 Parcela: {parcela_sel}")

        # XGBoost
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
        
        # LSTM
        week_series = datos_p[["temp_media","lluvia","fertilizante","riego"]].values
        ciclo_weeks = 16
        if len(week_series) < ciclo_weeks:
            pad = np.tile(week_series[-1], (ciclo_weeks-len(week_series),1))
            week_series = np.vstack([week_series, pad])
        else:
            week_series = week_series[-ciclo_weeks:]
        
        week_series_norm = (week_series - week_series.min(axis=0)) / (
            week_series.max(axis=0) - week_series.min(axis=0) + 1e-8
        )
        week_series_norm = week_series_norm.reshape((1,ciclo_weeks,4))
        pred_lstm = lstm_model.predict(week_series_norm)[0].flatten()
        pred_lstm_py = [round(float(v),2) for v in pred_lstm]

        semanas_restantes = ciclo_weeks - datos_p["semana_cultivo"].max()
        st.metric("Rendimiento actual (XGBoost)", f"{pred_stack:.2f} ton/ha")
        st.metric("Proyección última semana (LSTM)", f"{pred_lstm_py[-1]:.2f} ton/ha")
        semana_optima = np.argmax(pred_lstm_py)+1
        st.metric(f"Mejor semana para cosechar", f"{semana_optima}")
        st.metric(f"Semanas hasta cosecha", f"{semanas_restantes}")

        if df_forecast is not None:
            df_forecast_p = df_forecast[df_forecast["parcela"]==parcela_sel]
            st.subheader("Resumen de recomendaciones próximas 4 semanas")

            reco_data = []
            for _, row in df_forecast_p.iterrows():
                fecha = row['fecha'].strftime('%d/%m/%Y')
                temp_alert = "Si" if row["temp_media"] > 28 else "No"
                lluvia_alert = "Si" if row["lluvia"] < 2 else "No"
                riego_sugerido = f"{row['riego_sugerido_mm']:.1f} mm/día"
                fert_sugerido = f"{row['fertilizante_sugerido_kg']:.2f} kg/día"
                
                reco_data.append({
                    "Fecha": fecha,
                    "Temp alta": temp_alert,
                    "Lluvia baja": lluvia_alert,
                    "Riego sugerido": riego_sugerido,
                    "Fertilizante sugerido": fert_sugerido
                })
            st.dataframe(pd.DataFrame(reco_data), use_container_width=True)

            # proyecciones
            future_opt = [round(pred_lstm_py[-1],2)]
            seq_opt = week_series.copy().tolist()
            fert_acum_opt = datos_p["fertilizante"].sum()
            riego_acum_opt = datos_p["riego"].sum()

            for i in range(4):
                row = df_forecast_p.iloc[min(i, len(df_forecast_p)-1)]
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
                pred_next = lstm_model.predict(seq_o_norm.reshape(1,ciclo_weeks,4))[0][-1]
                future_opt.append(round(float(pred_next),2))

            st.subheader("Escenarios proyectados")
            line_opt = {
                "xAxis": {"type": "category", "data": list(range(1, len(pred_lstm_py)+5))},
                "yAxis": {"type": "value"},
                "series": [
                    {
                        "data": pred_lstm_py + [None]*4,
                        "type": "line",
                        "smooth": True,
                        "name": "Histórico",
                        "lineStyle": {"color": "#91c7ae"},
                        "label": {"show": True, "formatter": "{c}", "position": "top"},
                        "symbol": "circle",
                        "symbolSize": 6
                    },
                    {
                        "data": [None]*(len(pred_lstm_py)-1) + future_opt,
                        "type": "line",
                        "smooth": True,
                        "name": "Recomendado",
                        "lineStyle": {"color": "green", "type": "dashed"},
                        "label": {"show": True, "formatter": "{c}", "position": "top"},
                        "symbol": "circle",
                        "symbolSize": 6
                    }
                ],
                "tooltip": {"trigger": "axis"}
            }
            st_echarts(options=line_opt, height="500px")

        else:
            st.info("No hay predicción meteorológica para generar escenarios futuros.")

        st.subheader("Datos recientes")
        st.dataframe(datos_p.tail(5), use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns([1, 10])

    with col1:
        st.image("images/logo.png", width=30, caption="Amutio")

    with col2:
        st.markdown(
            """
            <span style="font-size:0.9em; color: #555;">
                <em>AMUTIO Predictive IA | Dashboard 2025 | Monitorización en vivo</em>
            </span>
            """,
            unsafe_allow_html=True
        )


else:
    st.warning("⚠️ Por favor sube el CSV de seguimiento semanal para comenzar.")
