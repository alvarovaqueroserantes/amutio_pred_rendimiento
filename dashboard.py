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

        # métricas con colores profesionales
        col_a, col_b, col_c, col_d = st.columns(4)

        with col_a:
            st.markdown(f"""
            <div style="background:#E8F5E9; border-radius:10px; padding:15px; text-align:center; border:1px solid #C8E6C9;">
                <div style="font-size:0.9em; color:#2E7D32; font-weight:bold;">Rendimiento medio</div>
                <div style="font-size:1.6em; font-weight:bold; color:#1B5E20;">{np.mean(global_preds):.2f} ton/ha</div>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown(f"""
            <div style="background:#FFFDE7; border-radius:10px; padding:15px; text-align:center; border:1px solid #FFF9C4;">
                <div style="font-size:0.9em; color:#F57F17; font-weight:bold;">Mejor parcela</div>
                <div style="font-size:1.6em; font-weight:bold; color:#F57F17;">{np.max(global_preds):.2f} ton/ha</div>
            </div>
            """, unsafe_allow_html=True)

        with col_c:
            st.markdown(f"""
            <div style="background:#FFEBEE; border-radius:10px; padding:15px; text-align:center; border:1px solid #FFCDD2;">
                <div style="font-size:0.9em; color:#C62828; font-weight:bold;">Peor parcela</div>
                <div style="font-size:1.6em; font-weight:bold; color:#B71C1C;">{np.min(global_preds):.2f} ton/ha</div>
            </div>
            """, unsafe_allow_html=True)

        with col_d:
            st.markdown(f"""
            <div style="background:#E3F2FD; border-radius:10px; padding:15px; text-align:center; border:1px solid #BBDEFB;">
                <div style="font-size:0.9em; color:#1565C0; font-weight:bold;">Variabilidad</div>
                <div style="font-size:1.6em; font-weight:bold; color:#0D47A1;">{np.std(global_preds):.2f} ton/ha</div>
            </div>
            """, unsafe_allow_html=True)

        # indicadores ambientales
        media_temp = df["temp_media"].mean()
        media_lluvia = df["lluvia"].mean()
        media_fertilizante = df["fertilizante"].mean()
        media_riego = df["riego"].mean()

        st.markdown("#### Condiciones globales promedio")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div style="background:#F1F8E9; border-radius:10px; padding:10px; text-align:center;">
                <span style="font-size:0.9em; color:#558B2F; font-weight:bold;">Temperatura media</span><br>
                <span style="font-size:1.4em; font-weight:bold;">{media_temp:.1f} °C</span>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div style="background:#E8F5E9; border-radius:10px; padding:10px; text-align:center;">
                <span style="font-size:0.9em; color:#33691E; font-weight:bold;">Lluvia media</span><br>
                <span style="font-size:1.4em; font-weight:bold;">{media_lluvia:.1f} mm/semana</span>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div style="background:#FFF8E1; border-radius:10px; padding:10px; text-align:center;">
                <span style="font-size:0.9em; color:#FF8F00; font-weight:bold;">Fertilizante medio</span><br>
                <span style="font-size:1.4em; font-weight:bold;">{media_fertilizante:.1f} kg/semana</span>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div style="background:#E1F5FE; border-radius:10px; padding:10px; text-align:center;">
                <span style="font-size:0.9em; color:#0288D1; font-weight:bold;">Riego medio</span><br>
                <span style="font-size:1.4em; font-weight:bold;">{media_riego:.1f} mm/semana</span>
            </div>
            """, unsafe_allow_html=True)

        # ranking
        st.markdown("#### Ranking de parcelas")
        ranking_df = pd.DataFrame({
            "Parcela": parcelas,
            "Predicción rendimiento (ton/ha)": global_preds
        }).sort_values(by="Predicción rendimiento (ton/ha)", ascending=False)
        st.dataframe(ranking_df, use_container_width=True)

        # gráfico de barras
        st.markdown("#### Distribución de rendimiento por parcela")
        bar_chart_opt = {
            "xAxis": {
                "type": "category",
                "data": ranking_df["Parcela"].tolist(),
            },
            "yAxis": {
                "type": "value",
                "name": "ton/ha",
                "nameLocation": "middle",
                "nameGap": 30,
            },
            "series": [
                {
                    "data": [round(x, 2) for x in ranking_df["Predicción rendimiento (ton/ha)"]],
                    "type": "bar",
                    "label": {"show": True, "position": "top"},
                    "itemStyle": {"color": "#4CAF50"},
                }
            ],
            "tooltip": {"trigger": "axis"},
            "grid": {"left": "3%", "right": "4%", "bottom": "3%", "containLabel": True},
        }
        
        import folium
        from streamlit_folium import st_folium

        # coordenadas aproximadas de parcelas ficticias alrededor de Cartagena
        parcel_coords = {
            # AGRIA
            "Agria_p1": [(37.64, -0.98), (37.64, -0.975), (37.642, -0.975), (37.642, -0.98)],
            "Agria_p2": [(37.638, -0.972), (37.638, -0.968), (37.640, -0.968), (37.640, -0.972)],
            "Agria_p3": [(37.635, -0.98), (37.635, -0.975), (37.637, -0.974), (37.637, -0.978)],
            "Agria_p4": [(37.633, -0.97), (37.633, -0.966), (37.636, -0.966), (37.636, -0.97)],
            "Agria_p5": [(37.630, -0.985), (37.630, -0.980), (37.633, -0.980), (37.633, -0.985)],
            "Agria_p6": [(37.628, -0.972), (37.628, -0.968), (37.631, -0.968), (37.631, -0.972)],
            "Agria_p7": [(37.625, -0.978), (37.625, -0.975), (37.628, -0.975), (37.628, -0.978)],
            "Agria_p8": [(37.623, -0.985), (37.623, -0.980), (37.626, -0.980), (37.626, -0.985)],
            "Agria_p9": [(37.620, -0.975), (37.620, -0.972), (37.623, -0.972), (37.623, -0.975)],
            "Agria_p10": [(37.618, -0.985), (37.618, -0.980), (37.621, -0.979), (37.621, -0.984)],

            # MONALISA
            "Monalisa_p1": [(37.615, -0.97), (37.615, -0.966), (37.618, -0.966), (37.618, -0.970)],
            "Monalisa_p2": [(37.612, -0.978), (37.612, -0.974), (37.615, -0.974), (37.615, -0.978)],
            "Monalisa_p3": [(37.610, -0.985), (37.610, -0.980), (37.613, -0.980), (37.613, -0.985)],
            "Monalisa_p4": [(37.608, -0.975), (37.608, -0.970), (37.611, -0.970), (37.611, -0.975)],
            "Monalisa_p5": [(37.605, -0.985), (37.605, -0.980), (37.608, -0.980), (37.608, -0.985)],
            "Monalisa_p6": [(37.603, -0.978), (37.603, -0.974), (37.606, -0.974), (37.606, -0.978)],
            "Monalisa_p7": [(37.600, -0.97), (37.600, -0.966), (37.603, -0.966), (37.603, -0.970)],
            "Monalisa_p8": [(37.598, -0.978), (37.598, -0.974), (37.601, -0.974), (37.601, -0.978)],
            "Monalisa_p9": [(37.596, -0.985), (37.596, -0.980), (37.599, -0.980), (37.599, -0.985)],
            "Monalisa_p10": [(37.594, -0.975), (37.594, -0.970), (37.597, -0.970), (37.597, -0.975)],

            # SPUNTA
            "Spunta_p1": [(37.590, -0.985), (37.590, -0.980), (37.593, -0.980), (37.593, -0.985)],
            "Spunta_p2": [(37.588, -0.975), (37.588, -0.970), (37.591, -0.970), (37.591, -0.975)],
            "Spunta_p3": [(37.585, -0.985), (37.585, -0.980), (37.588, -0.980), (37.588, -0.985)],
            "Spunta_p4": [(37.583, -0.975), (37.583, -0.970), (37.586, -0.970), (37.586, -0.975)],
            "Spunta_p5": [(37.580, -0.985), (37.580, -0.980), (37.583, -0.980), (37.583, -0.985)],
            "Spunta_p6": [(37.578, -0.975), (37.578, -0.970), (37.581, -0.970), (37.581, -0.975)],
            "Spunta_p7": [(37.575, -0.985), (37.575, -0.980), (37.578, -0.980), (37.578, -0.985)],
            "Spunta_p8": [(37.573, -0.975), (37.573, -0.970), (37.576, -0.970), (37.576, -0.975)],
            "Spunta_p9": [(37.570, -0.985), (37.570, -0.980), (37.573, -0.980), (37.573, -0.985)],
            "Spunta_p10": [(37.568, -0.975), (37.568, -0.970), (37.571, -0.970), (37.571, -0.975)]
        }



        def show_map():
            min_pred = np.min(global_preds)
            max_pred = np.max(global_preds)

            def rendimiento_color(value):
                ratio = (value - min_pred) / (max_pred - min_pred + 1e-8)
                r = int(255 * (1 - ratio))
                g = int(180 * ratio)
                b = 0
                return f"rgb({r},{g},{b})"

            m = folium.Map(location=[37.625, -0.990], zoom_start=12, tiles="cartodbpositron")

            for parcela in ranking_df["Parcela"].tolist():
                coords = parcel_coords.get(parcela)
                if coords:
                    rendimiento = ranking_df.loc[ranking_df["Parcela"]==parcela, "Predicción rendimiento (ton/ha)"].values[0]
                    color = rendimiento_color(rendimiento)
                    folium.Polygon(
                        locations=coords,
                        color="black",
                        fill=True,
                        fill_opacity=0.7,
                        fill_color=color,
                        weight=1,
                        popup=f"""
                            <b>{parcela}</b><br>
                            Rendimiento: {rendimiento:.2f} ton/ha
                        """
                    ).add_to(m)

            return m

        # botón de control
        if st.button("Mostrar mapa de parcelas"):
            st.session_state["mapa"] = show_map()

        # si existe en la sesión, se dibuja SIN volver a recalcular
        if "mapa" in st.session_state:
            st.markdown("#### Mapa de parcelas")
            st_folium(st.session_state["mapa"], width=900, height=600)
        
        
        st_echarts(options=bar_chart_opt, height="400px")

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
            # valores históricos
            historico_labels = [round(v, 2) if v is not None else None for v in pred_lstm_py + [None]*4]
            # escenario futuro
            escenario_labels = [None]*(len(pred_lstm_py)-1) + [round(v, 2) for v in future_opt]

            line_opt = {
                "xAxis": {"type": "category", "data": list(range(1, len(pred_lstm_py) + 5))},
                "yAxis": {
                    "type": "value",
                    "name": "Toneladas por hectárea",
                    "nameLocation": "middle",
                    "nameGap": 35
                },
                "series": [
                    {
                        "data": historico_labels,
                        "type": "line",
                        "smooth": True,
                        "name": "Histórico",
                        "label": {
                            "show": True,
                            "position": "top"
                        },
                        "symbolSize": 6
                    },
                    {
                        "data": escenario_labels,
                        "type": "line",
                        "smooth": True,
                        "name": "Escenario",
                        "label": {
                            "show": True,
                            "position": "top"
                        },
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
