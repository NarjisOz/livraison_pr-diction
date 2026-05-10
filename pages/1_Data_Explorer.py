"""Explainable EDA page."""

import streamlit as st

from delivery_delay.eda import (
    plot_categorical_counts,
    plot_correlation_heatmap,
    plot_missing_values,
    plot_numeric_distribution,
    plot_target_distribution,
)
from delivery_delay.insights import insights_to_frame, insights_to_markdown
from delivery_delay.performance import cached_build_eda_report, cached_build_eda_story, cached_detect_columns
from delivery_delay.streamlit_app import (
    apply_app_style,
    dataset_uploader,
    get_dataset,
    initialize_session_state,
    is_advanced_mode,
    markdown_download,
    render_dataset_status,
    render_insight_cards,
    render_mode_banner,
    render_mode_selector,
    target_selector,
)


st.set_page_config(page_title="EDA Storytelling", page_icon=":bar_chart:", layout="wide")
initialize_session_state()
apply_app_style()
st.title("Explainable Data Explorer")

with st.sidebar:
    render_mode_selector()

df = get_dataset()
if df is None:
    df = dataset_uploader("Upload dataset once", key="eda_upload")
if df is None:
    st.info("Upload a dataset from the welcome page or here. It will be reused across every page.")
    st.stop()

render_dataset_status(df)
render_mode_banner()
target, data = target_selector(df, key="eda_target")
schema = cached_detect_columns(data)
story = cached_build_eda_story(data, target_column=target)

st.subheader("Story First")
st.write(
    "The cards below prioritize what an operations user should notice immediately before looking at charts."
)
render_insight_cards(story, max_cards=8)
markdown_download(insights_to_markdown(story, title="Explainable EDA Story"), "Download EDA story", "eda_story.md")

if not is_advanced_mode():
    st.info("Business Mode keeps this page lightweight. Switch to Advanced/Data Science Mode for profiles and charts.")
    with st.expander("Insight Table", expanded=False):
        st.dataframe(insights_to_frame(story), use_container_width=True, hide_index=True)
    st.stop()

with st.spinner("Loading cached EDA profiles and visual diagnostics..."):
    report = cached_build_eda_report(data, target_column=target)

tab_story, tab_profiles, tab_visuals, tab_quality = st.tabs(
    ["Insight Table", "Profiles", "Visuals", "Quality and Anomalies"]
)

with tab_story:
    st.dataframe(insights_to_frame(story), use_container_width=True, hide_index=True)

with tab_profiles:
    st.dataframe(report["overview"], use_container_width=True, hide_index=True)
    st.subheader("Numerical profile")
    st.dataframe(report["numeric_profile"], use_container_width=True, hide_index=True)
    st.subheader("Categorical profile")
    st.dataframe(report["categorical_profile"], use_container_width=True, hide_index=True)
    st.subheader("Datetime profile")
    st.dataframe(report["datetime_profile"], use_container_width=True, hide_index=True)

with tab_visuals:
    st.plotly_chart(plot_target_distribution(data, target), use_container_width=True)
    st.plotly_chart(plot_missing_values(data), use_container_width=True)
    st.plotly_chart(plot_correlation_heatmap(data), use_container_width=True)

    if schema.numeric:
        numeric_choice = st.selectbox("Numerical column", options=schema.numeric, key="eda_numeric")
        st.plotly_chart(plot_numeric_distribution(data, numeric_choice), use_container_width=True)

    categorical_options = schema.categorical + schema.boolean
    if categorical_options:
        categorical_choice = st.selectbox("Categorical column", options=categorical_options, key="eda_categorical")
        st.plotly_chart(plot_categorical_counts(data, categorical_choice), use_container_width=True)

with tab_quality:
    st.subheader("Missing values")
    st.dataframe(report["missing_values"], use_container_width=True, hide_index=True)
    st.subheader("Outlier report")
    st.dataframe(report["outliers"], use_container_width=True, hide_index=True)
    st.subheader("Correlation with target")
    st.dataframe(report["correlation"], use_container_width=True)
