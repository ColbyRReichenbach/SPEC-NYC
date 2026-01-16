"""
S.P.E.C. NYC - Streamlit Dashboard
Placeholder until Phase 1.8 implementation
"""
import streamlit as st

st.set_page_config(
    page_title="S.P.E.C. NYC",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏢 S.P.E.C. NYC Valuation Engine")
st.markdown("### Status: V1.0 In Development")

st.info("""
**This dashboard is under construction.**

Current progress:
- ✅ Project scaffold created
- ✅ Agent workflows configured
- ⏳ Data pipeline (Phase 1.2-1.6)
- ⏳ Model training (Phase 1.7)
- ⏳ Dashboard implementation (Phase 1.8)

See [Implementation Plan](docs/NYC_IMPLEMENTATION_PLAN.md) for details.
""")

# Placeholder sections
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Model Metrics")
    st.metric("PPE10", "—", help="Percentage within ±10%")
    st.metric("MdAPE", "—", help="Median Absolute Percentage Error")
    st.metric("R²", "—", help="Coefficient of Determination")

with col2:
    st.subheader("📈 Data Status")
    st.metric("Records", "0", help="Cleaned transaction records")
    st.metric("Boroughs", "0/5", help="Manhattan, Brooklyn first")
    st.metric("Features", "0", help="Engineered features")

st.markdown("---")
st.caption("S.P.E.C. NYC - Spatial · Predictive · Explainable · Conversational")
