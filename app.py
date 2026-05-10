"""Streamlit entry point for the intelligent logistics analytics platform."""

import streamlit as st

from delivery_delay.dual_prediction import tasks_to_frame
from delivery_delay.external_intelligence import (
    external_result_markdown,
    external_result_to_dict,
)
from delivery_delay.insights import insights_to_frame, insights_to_markdown
from delivery_delay.performance import (
    cached_analyze_dataset,
    cached_build_eda_story,
    cached_discover_prediction_tasks,
    cached_external_weather_intelligence,
    lightweight_table,
)
from delivery_delay.reporting import dataset_summary
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
    render_decision_badge,
    render_executive_brief,
    render_feature_cards,
    render_insight_cards,
    render_messages,
    render_mode_banner,
    render_mode_selector,
    render_platform_header,
    render_score_cards,
    render_schema_table,
    show_dataframe_preview,
)
from delivery_delay.targeting import available_targets
from delivery_delay.understanding import understanding_to_dict


st.set_page_config(page_title="Logistics Intelligence Platform", page_icon=":truck:", layout="wide")
initialize_session_state()
apply_app_style()

render_platform_header(
    "Adaptive Logistics Intelligence",
    (
        "Upload once, then explore data quality, preprocess transparently, train classification and regression "
        "models, test Nigeria weather intelligence, explain decisions, and turn predictions into operational action."
    ),
    eyebrow="Intelligent analytics and decision support",
)

st.write("")

with st.sidebar:
    render_mode_selector()
    st.divider()
    st.subheader("Single Upload")
    df = dataset_uploader("Upload CSV, Excel, or Parquet", key="home_dataset_upload")
    st.caption(
        "The same uploaded dataset is reused by EDA, preprocessing, training, prediction, dashboards, and explainability."
    )

df = get_dataset()
render_mode_banner()

if df is None:
    st.subheader("What This Platform Does")
    render_feature_cards(
        [
            {
                "title": "Dual ML System",
                "body": "Trains delay-risk classification and delivery-duration regression when the uploaded data supports them.",
            },
            {
                "title": "Weather Intelligence",
                "body": "Tests Nigeria city weather, latitude/longitude, and distance features before deciding whether to use them.",
            },
            {
                "title": "Explainable Decisions",
                "body": "Narrates missing values, anomalies, preprocessing choices, model behavior, and operational risks.",
            },
        ]
    )

    st.subheader("Supported Data")
    render_feature_cards(
        [
            {
                "title": "File Types",
                "body": "CSV, Excel, and Parquet files with shipment, delivery, order, route, supplier, carrier, city, date, or status fields.",
            },
            {
                "title": "Best Targets",
                "body": "Delivery status, delayed flag, expected and actual delivery dates, delay days, lead time, or delivery duration.",
            },
            {
                "title": "Limitations",
                "body": "The system will not force prediction or external data integration when targets, joins, or validation quality are weak.",
            },
        ]
    )

    st.subheader("Workflow")
    render_feature_cards(
        [
            {
                "title": "1. Upload",
                "body": "Upload once from the sidebar; every page reads the same session dataset.",
            },
            {
                "title": "2. Understand",
                "body": "Review EDA storytelling, schema roles, missingness, outliers, and operational hotspots.",
            },
            {
                "title": "3. Decide",
                "body": "Inspect preprocessing rules and the weather usefulness decision before training.",
            },
            {
                "title": "4. Train",
                "body": "Compare compatible classification and regression models dynamically.",
            },
            {
                "title": "5. Act",
                "body": "Predict, prioritize high-risk records, download reports, and explain model drivers.",
            },
        ]
    )
    st.stop()

render_dataset_status(df)

targets = available_targets(df)
selected_target = targets[0] if targets else None
if targets:
    selected_target = st.selectbox("Primary target for understanding", options=targets, index=0)

understanding = cached_analyze_dataset(df, target_column=selected_target)
st.session_state["dataset_understanding"] = understanding
tasks = cached_discover_prediction_tasks(df)
compatible_tasks = [task for task in tasks if task.compatible]

st.subheader("Executive Readiness")
render_score_cards(understanding)
render_executive_brief(
    [
        "Dataset quality is {:.0f}/100 and logistics compatibility is {:.0f}/100.".format(
            understanding.quality_score,
            understanding.compatibility_score,
        ),
        "{} compatible prediction system{} detected: {}.".format(
            len(compatible_tasks),
            "" if len(compatible_tasks) == 1 else "s",
            ", ".join(task.label for task in compatible_tasks) or "none yet",
        ),
        "Governance score is {:.0f}/100; sensitive fields, leakage risks, and weak predictors are checked before modeling.".format(
            understanding.governance_score
        ),
    ],
    title="AI Readiness Brief",
)

st.subheader("Immediate Intelligence")
story = cached_build_eda_story(df, target_column=selected_target)
render_insight_cards(story, max_cards=6)

download_left, download_mid, download_right = st.columns(3)
with download_left:
    dataframe_download(df, "Download cleaned dataset", "cleaned_logistics_dataset.csv")
with download_mid:
    markdown_download(insights_to_markdown(story), "Download insight summary", "logistics_insights.md")
with download_right:
    json_download(
        understanding_to_dict(understanding), "Download understanding JSON", "dataset_understanding.json"
    )

st.subheader("Dataset Understanding")
render_messages(dataset_summary(df))

left, right = st.columns([1.1, 1])
with left:
    st.markdown("#### Column Roles")
    if is_advanced_mode():
        render_schema_table(df)
    else:
        st.info("Switch to Advanced/Data Science Mode to inspect full schema diagnostics.")
with right:
    st.markdown("#### Prediction Compatibility")
    st.dataframe(tasks_to_frame(tasks), use_container_width=True, hide_index=True)

st.subheader("External Weather Intelligence")
st.write(
    "The platform checks the bundled Nigeria city weather dataset, city joins, latitude/longitude patterns, "
    "distance features, and predictive value before recommending integration."
)
if st.button("Analyze Nigeria weather dataset", type="primary"):
    with st.spinner("Testing weather, geography, distance, and predictive usefulness..."):
        result = cached_external_weather_intelligence(df, target_column=selected_target)
        st.session_state["external_intelligence"] = result
        if result.enriched_data is not None:
            st.session_state["external_enriched_dataset"] = result.enriched_data

external_result = st.session_state.get("external_intelligence")
if external_result is not None:
    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Weather match rate", "{:.1%}".format(external_result.match_rate))
    metric_2.metric("Matched records", "{:,}".format(external_result.matched_rows))
    metric_3.metric("Added features", "{:,}".format(len(external_result.feature_columns)))
    render_decision_badge(
        "Weather integration recommended"
        if external_result.recommended
        else "Weather integration not forced",
        external_result.recommended,
    )
    render_messages(external_result.reasons)
    if external_result.predictive_test.get("weather_feature_importance"):
        st.markdown("#### External Feature Contribution")
        st.dataframe(
            external_result.predictive_test["weather_feature_importance"],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Weather importance share: {:.1%}".format(
                external_result.predictive_test.get("weather_importance_share", 0.0)
            )
        )
    markdown_download(
        external_result_markdown(external_result),
        "Download weather intelligence report",
        "weather_intelligence.md",
    )
    json_download(
        external_result_to_dict(external_result),
        "Download weather intelligence JSON",
        "weather_intelligence.json",
    )

st.subheader("Data Preview")
if is_advanced_mode():
    show_dataframe_preview(df)
else:
    st.dataframe(lightweight_table(df, max_rows=12, max_columns=12), use_container_width=True)

with st.expander("Insight Table", expanded=False):
    st.dataframe(insights_to_frame(story), use_container_width=True, hide_index=True)
