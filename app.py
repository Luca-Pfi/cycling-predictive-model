"""Pro-cycling predictor: two Streamlit pages.

    streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="Pro Cycling Predictor", page_icon="🚴", layout="wide")
st.navigation([
    st.Page("views/stage_designer.py", title="Who suits this stage?", icon="🏔️", default=True),
    st.Page("views/career.py", title="Career trajectory", icon="📈"),
]).run()
