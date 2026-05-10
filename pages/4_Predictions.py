"""Prediction and decision dashboard page."""

import pandas as pd
import streamlit as st

from delivery_delay.config import DEFAULT_MODEL_PATH, MODEL_PORTFOLIO_PATH
from delivery_delay.dual_prediction import is_portfolio, portfolio_model_options, select_portfolio_model
from delivery_delay.eda import categorical_profile, missing_values_table, numeric_profile
from delivery_delay.performance import (
    cached_detect_columns,
    cached_external_weather_intelligence,
    cached_load_model_artifact,
    cached_predict_dataframe,
    lightweight_table,
    model_file_signature,
)
from delivery_delay.prediction_dashboard import (
    anomaly_alerts,
    executive_ai_summary,
    enrich_prediction_segments,
    group_risk_table,
    plot_group_risk,
    plot_prediction_value_distribution,
    plot_probability_distribution,
    plot_risk_gauge,
    plot_risk_segments,
    prediction_kpis,
    prediction_reliability_indicators,
    readable_priority_columns,
    recommended_group_columns,
    top_priority_rows,
)
from delivery_delay.recommendations import generate_operational_recommendations, recommendations_table
from delivery_delay.reporting import (
    build_prediction_dashboard_html,
    build_prediction_report_markdown,
    prediction_dashboard_summary,
)
from delivery_delay.schema import summarize_schema
from delivery_delay.streamlit_app import (
    apply_app_style,
    dataframe_download,
    dataset_uploader,
    get_dataset,
    html_download,
    initialize_session_state,
    is_advanced_mode,
    markdown_download,
    render_executive_brief,
    render_dataset_status,
    render_messages,
    render_mode_banner,
    render_mode_selector,
)


BUSINESS_OUTPUT_COLUMNS = (
    "predicted_delay_class",
    "delay_probability",
    "predicted_delay_value",
    "estimated_delivery_days",
    "estimated_delivery_date",
)


def _complete_dual_business_outputs(artifact, prediction_input, predictions, selected_key):
    """Add complementary classification/regression outputs when a portfolio has both."""

    if not is_portfolio(artifact):
        return predictions

    combined = predictions.copy()
    for key in artifact.get("models", {}):
        if key == selected_key:
            continue
        needs_classification = key == "classification" and "delay_probability" not in combined.columns
        needs_regression = key == "regression" and "estimated_delivery_days" not in combined.columns
        if not (needs_classification or needs_regression):
            continue
        try:
            auxiliary = cached_predict_dataframe(artifact, prediction_input, model_key=key)
        except Exception:
            continue
        for column in BUSINESS_OUTPUT_COLUMNS:
            if column in auxiliary.columns and column not in combined.columns:
                combined[column] = auxiliary[column]
    return combined


st.set_page_config(page_title="Prediction Dashboard", page_icon=":chart_with_upwards_trend:", layout="wide")
initialize_session_state()
apply_app_style()
st.title("Prediction Dashboard")

with st.sidebar:
    render_mode_selector()

df = get_dataset()
if df is None:
    df = dataset_uploader("Upload dataset once", key="prediction_upload")
if df is None:
    st.info("Upload a dataset once, then predictions will reuse it automatically.")
    st.stop()

render_dataset_status(df)
render_mode_banner()

artifact = st.session_state.get("model_artifact") or st.session_state.get("model_portfolio")
if artifact is None and MODEL_PORTFOLIO_PATH.exists():
    artifact = cached_load_model_artifact(
        str(MODEL_PORTFOLIO_PATH), model_file_signature(MODEL_PORTFOLIO_PATH)
    )
    st.session_state["model_artifact"] = artifact
if artifact is None and DEFAULT_MODEL_PATH.exists():
    artifact = cached_load_model_artifact(str(DEFAULT_MODEL_PATH), model_file_signature(DEFAULT_MODEL_PATH))
    st.session_state["model_artifact"] = artifact

if artifact is None:
    st.warning("Train the dual prediction system first, or place a valid artifact in the artifacts folder.")
    st.stop()

model_options = portfolio_model_options(artifact)
selected_key = model_options[0][0]
if is_portfolio(artifact) and len(model_options) > 1:
    option_labels = {label: key for key, label in model_options}
    selected_label = st.selectbox("Prediction system", options=list(option_labels.keys()), index=0)
    selected_key = option_labels[selected_label]
selected_artifact, resolved_key = select_portfolio_model(artifact, model_key=selected_key)

prediction_input = df
external_result = st.session_state.get("external_intelligence")
weather_used_by_model = bool(
    isinstance(artifact, dict) and artifact.get("metadata", {}).get("weather_features_used")
)
if weather_used_by_model and st.session_state.get("external_enriched_dataset") is None:
    with st.spinner("Rebuilding weather-enhanced features expected by the trained portfolio..."):
        external_result = cached_external_weather_intelligence(df, run_predictive_test=False)
        st.session_state["external_intelligence"] = external_result
        if external_result.enriched_data is not None:
            st.session_state["external_enriched_dataset"] = external_result.enriched_data

if (
    (external_result is not None and external_result.recommended) or weather_used_by_model
) and st.session_state.get("external_enriched_dataset") is not None:
    prediction_input = st.session_state["external_enriched_dataset"]
    st.info("Using weather-enhanced session data because the selected model was trained with those features.")

with st.spinner("Generating predictions and dashboard segments..."):
    try:
        predictions = cached_predict_dataframe(artifact, prediction_input, model_key=selected_key)
    except Exception as exc:
        st.error("Predictions could not be generated: {}".format(exc))
        st.stop()

predictions = _complete_dual_business_outputs(artifact, prediction_input, predictions, selected_key)
predictions = enrich_prediction_segments(predictions)
st.session_state["prediction_results"] = predictions
group_options = recommended_group_columns(predictions)
selected_group = group_options[0] if group_options else None
kpis = prediction_kpis(predictions)

if group_options:
    selected_group = st.selectbox("Business view", options=group_options, index=0)

recommendations = generate_operational_recommendations(predictions=predictions)
model_name = selected_artifact.get("best_model_name", "selected model")
target_name = selected_artifact.get("target_column", "target")

st.subheader("Executive AI Brief")
render_executive_brief(
    executive_ai_summary(
        predictions,
        artifact=selected_artifact,
        group_column=selected_group,
        recommendations=recommendations,
    )
)

model_col_1, model_col_2, model_col_3 = st.columns(3)
model_col_1.metric("Selected best model", model_name)
model_col_2.metric("Prediction system", resolved_key.title())
model_col_3.metric("Target", target_name)

st.subheader("Key Alerts")
if "delay_probability" in predictions.columns:
    metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)
    metric_1.metric("Records analyzed", "{:,}".format(kpis.get("rows", 0)))
    metric_2.metric("Average delay risk", "{:.1%}".format(kpis.get("average_risk", 0.0)))
    metric_3.metric("High-risk records", "{:,}".format(kpis.get("high_risk_count", 0)))
    if "average_estimated_days" in kpis:
        metric_4.metric("Avg estimated days", "{:.2f}".format(kpis.get("average_estimated_days", 0.0)))
    else:
        metric_4.metric("Median risk", "{:.1%}".format(kpis.get("median_risk", 0.0)))
    metric_5.metric(
        "Avg confidence",
        "{:.1%}".format(kpis.get("average_confidence") or 0.0),
        help="Classification confidence is the probability of the model's selected class.",
    )
    if kpis.get("high_risk_count", 0) > 0:
        st.error("{:,} high-risk records should be reviewed first.".format(kpis.get("high_risk_count", 0)))
else:
    metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)
    metric_1.metric("Records analyzed", "{:,}".format(kpis.get("rows", 0)))
    metric_2.metric("Avg estimated days", "{:.2f}".format(kpis.get("average_estimated_days", 0.0)))
    metric_3.metric("Median estimate", "{:.2f}".format(kpis.get("median_prediction", 0.0)))
    metric_4.metric(
        "Model confidence",
        "{:.1%}".format(kpis.get("average_confidence") or 0.0),
        help="Regression confidence is derived from validation strength and should be read as a planning signal.",
    )
    metric_5.metric(
        "Date coverage",
        "{:.1%}".format(kpis.get("estimated_date_coverage") or 0.0),
        help="Share of rows with enough date context to estimate a delivery date.",
    )

alerts = anomaly_alerts(predictions, group_column=selected_group)
if alerts:
    st.dataframe(pd.DataFrame(alerts), use_container_width=True, hide_index=True)

with st.expander("Reliability Indicators", expanded=not is_advanced_mode()):
    st.dataframe(
        prediction_reliability_indicators(predictions, artifact=selected_artifact),
        use_container_width=True,
        hide_index=True,
    )

st.subheader("Decision Narrative")
render_messages(prediction_dashboard_summary(predictions, group_column=selected_group))

tab_names = ["Prediction Story", "Risk by Group", "Priority Rows", "Recommendations"]
if is_advanced_mode():
    tab_names.append("Input Health")
tabs = st.tabs(tab_names)
tab_results, tab_groups, tab_priority, tab_recommendations = tabs[:4]
tab_data = tabs[4] if is_advanced_mode() else None

with tab_results:
    left, right = st.columns([0.9, 1.1])
    with left:
        if "delay_probability" in predictions.columns:
            st.plotly_chart(plot_risk_gauge(predictions), use_container_width=True)
        else:
            st.plotly_chart(plot_prediction_value_distribution(predictions), use_container_width=True)
    with right:
        st.plotly_chart(plot_risk_segments(predictions), use_container_width=True)

    if "delay_probability" in predictions.columns and is_advanced_mode():
        st.plotly_chart(plot_probability_distribution(predictions), use_container_width=True)

with tab_groups:
    if not selected_group:
        st.info("No categorical business column was found for grouping.")
    else:
        grouped = group_risk_table(predictions, selected_group)
        st.plotly_chart(plot_group_risk(grouped, selected_group), use_container_width=True)
        st.dataframe(grouped, use_container_width=True, hide_index=True)
        dataframe_download(grouped, "Download grouped dashboard data", "grouped_prediction_dashboard.csv")

with tab_priority:
    priority = top_priority_rows(predictions)
    priority_columns = readable_priority_columns(priority)
    if priority_columns:
        st.dataframe(priority[priority_columns], use_container_width=True, hide_index=True)
    else:
        st.dataframe(priority, use_container_width=True, hide_index=True)
    st.info(
        "These are the records to review first. Risk score ranks delay probability for classification "
        "and longest estimated durations for regression."
    )

with tab_recommendations:
    st.dataframe(recommendations_table(recommendations), use_container_width=True, hide_index=True)

if tab_data is not None:
    with tab_data:
        schema = cached_detect_columns(prediction_input)
        col_1, col_2, col_3 = st.columns(3)
        col_1.metric("Rows in active file", "{:,}".format(len(prediction_input)))
        col_2.metric("Detected numerical columns", "{:,}".format(len(schema.numeric)))
        col_3.metric("Detected categorical columns", "{:,}".format(len(schema.categorical)))
        st.subheader("Detected Column Roles")
        st.dataframe(summarize_schema(schema), use_container_width=True, hide_index=True)

        profile_tabs = st.tabs(["Missing Values", "Numerical Columns", "Categorical Columns"])
        with profile_tabs[0]:
            st.dataframe(missing_values_table(prediction_input), use_container_width=True, hide_index=True)
        with profile_tabs[1]:
            st.dataframe(numeric_profile(prediction_input), use_container_width=True, hide_index=True)
        with profile_tabs[2]:
            st.dataframe(categorical_profile(prediction_input), use_container_width=True, hide_index=True)

if not is_advanced_mode():
    st.caption("Input-health profiling is hidden in Business Mode to keep the dashboard fast.")

st.subheader("Download Center")
st.write("Export operational outputs for reporting, Power BI, audit trails, or thesis appendices.")
download_a, download_b, download_c = st.columns(3)
with download_a:
    dataframe_download(predictions, "Download predictions", "delivery_predictions.csv")
with download_b:
    markdown_download(
        build_prediction_report_markdown(predictions, recommendations),
        "Download prediction report",
        "prediction_report.md",
    )
with download_c:
    html_download(
        build_prediction_dashboard_html(predictions, recommendations),
        "Download dashboard HTML",
        "prediction_dashboard.html",
    )

st.subheader("Prediction Table")
st.dataframe(lightweight_table(predictions, max_rows=300, max_columns=60), use_container_width=True)
