# app/footer.py

import streamlit as st

def render_footer():
    """
    Renderiza el pie de página corporativo
    """
    st.markdown("""
        <div class="footer" style="font-size:0.85em; color:#666; text-align:center; margin-top:30px; border-top:1px solid #ddd; padding-top:10px;">
            AMUTIO Predictive IA — Monitorización en vivo | 2025
        </div>
    """, unsafe_allow_html=True)
