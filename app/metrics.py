# app/metrics.py

import streamlit as st
import numpy as np

def render_global_metrics(global_preds):
    """
    Renderiza los cuadros de métricas principales del rendimiento de todas las parcelas.
    """
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

def render_environmental_metrics(df):
    """
    Renderiza las métricas ambientales promedio
    """
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
