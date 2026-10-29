"""Pro-cycling predictor: two Streamlit pages, on real or synthetic data.

    streamlit run app.py
"""
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Pro Cycling Predictor", page_icon="🚴", layout="wide")

DATASETS = {  # label -> folder with metrics.json (stage models) and career/ (career models)
    "Real riders: Tour de France 1985–2017": "models/tdf",
    "Demo: synthetic riders": "models",
}
available = {k: v for k, v in DATASETS.items() if (Path(v) / "metrics.json").exists()}
choice = st.sidebar.radio("Data", list(available) or list(DATASETS),
                          help="Real data comes from open Tour de France datasets (see README). "
                               "The demo data has more variables (lab tests, U23 results) "
                               "but fictional riders.")
st.session_state["models_root"] = available.get(choice, "models")

st.navigation([
    st.Page("views/stage_designer.py", title="Who suits this stage?", icon="🏔️", default=True),
    st.Page("views/career.py", title="Career trajectory", icon="📈"),
]).run()
