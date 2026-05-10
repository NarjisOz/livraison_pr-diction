"""Explainable preprocessing page."""

import streamlit as st

from delivery_delay.streamlit_app import (
    apply_app_style,
    dataframe_download,
    dataset_uploader,
    get_dataset,
    initialize_session_state,
    is_advanced_mode,
    json_download,
    markdown_download,
    render_dataset_status,
    render_messages,
    render_mode_banner,
    render_mode_selector,
    render_score_cards,
    target_selector,
)
from delivery_delay.performance import cached_analyze_dataset, cached_preprocessing_explanations, lightweight_table
from delivery_delay.understanding import (
    columns_report_table,
    plan_summary_table,
    preprocessing_explanations_markdown,
    understanding_to_dict,
)


st.set_page_config(page_title="Preprocessing Intelligence", page_icon=":gear:", layout="wide")
initialize_session_state()
apply_app_style()
st.title("Explainable Preprocessing")

with st.sidebar:
    render_mode_selector()

df = get_dataset()
if df is None:
    df = dataset_uploader("Upload dataset once", key="preprocessing_upload")
if df is None:
    st.info("Upload a dataset first. The preprocessing page will reuse the same session dataset.")
    st.stop()

render_dataset_status(df)
render_mode_banner()
target, data = target_selector(df, key="preprocessing_target")
if target is None:
    st.warning("No target was found yet. Preprocessing can still be profiled, but training needs a target.")

problem_type = st.selectbox("Planning mode", options=["auto", "classification", "regression"], index=0)
understanding = cached_analyze_dataset(data, target_column=target, problem_type=problem_type)
st.session_state["dataset_understanding"] = understanding

st.subheader("Decision Scores")
render_score_cards(understanding)
render_messages(understanding.recommendations)

st.subheader("Why Each Preprocessing Decision Was Made")
explanations = cached_preprocessing_explanations(understanding)
if is_advanced_mode():
    st.dataframe(explanations, use_container_width=True, hide_index=True)
else:
    st.dataframe(lightweight_table(explanations, max_rows=8, max_columns=5), use_container_width=True, hide_index=True)
    st.info("Business Mode shows the highest-signal decisions. Switch to Advanced Mode for full diagnostics.")

download_left, download_mid, download_right = st.columns(3)
with download_left:
    dataframe_download(data, "Download cleaned dataset", "cleaned_preprocessed_input.csv")
with download_mid:
    markdown_download(
        preprocessing_explanations_markdown(understanding),
        "Download preprocessing report",
        "preprocessing_explanations.md",
    )
with download_right:
    json_download(understanding_to_dict(understanding), "Download preprocessing JSON", "preprocessing_plan.json")

if not is_advanced_mode():
    st.stop()

tab_plan, tab_columns, tab_governance = st.tabs(["Transformation Plan", "Column Diagnostics", "Governance"])

with tab_plan:
    st.write("This is the machine-readable plan used to construct the sklearn preprocessing pipeline.")
    st.dataframe(plan_summary_table(understanding.preprocessing_plan), use_container_width=True, hide_index=True)
    st.json(understanding.preprocessing_plan)

with tab_columns:
    st.dataframe(columns_report_table(understanding), use_container_width=True, hide_index=True)

with tab_governance:
    st.subheader("Leakage Warnings")
    st.json(understanding.leakage_warnings)
    st.subheader("Sensitive Data Warnings")
    st.json(understanding.governance_warnings)
    st.subheader("Date Relationships")
    st.json(understanding.date_relationships)
