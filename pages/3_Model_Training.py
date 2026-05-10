"""Dual model training and comparison page."""

import streamlit as st

from delivery_delay.config import DEFAULT_MODEL_PATH, MODEL_PORTFOLIO_PATH
from delivery_delay.dual_prediction import (
    tasks_to_frame,
    train_dual_prediction_system,
)
from delivery_delay.eda import plot_model_comparison
from delivery_delay.explainability import natural_language_feature_insights
from delivery_delay.external_intelligence import (
    external_result_markdown,
    external_result_to_dict,
)
from delivery_delay.performance import (
    cached_detect_columns,
    cached_discover_prediction_tasks,
    cached_external_weather_intelligence,
    cached_model_feature_importance,
)
from delivery_delay.recommendations import generate_operational_recommendations, recommendations_table
from delivery_delay.reporting import build_training_report_markdown, model_summary
from delivery_delay.streamlit_app import (
    apply_app_style,
    dataset_uploader,
    get_dataset,
    initialize_session_state,
    is_advanced_mode,
    markdown_download,
    render_dataset_status,
    render_decision_badge,
    render_messages,
    render_mode_banner,
    render_mode_selector,
    select_columns,
)


st.set_page_config(page_title="Dual Model Training", page_icon=":brain:", layout="wide")
initialize_session_state()
apply_app_style()
st.title("Dual Prediction Training")

with st.sidebar:
    render_mode_selector()

df = get_dataset()
if df is None:
    df = dataset_uploader("Upload dataset once", key="training_upload")
if df is None:
    st.info("Upload a dataset before training. The same upload is reused here.")
    st.stop()

render_dataset_status(df)
render_mode_banner()

st.subheader("Prediction Compatibility")
tasks = cached_discover_prediction_tasks(df)
st.dataframe(tasks_to_frame(tasks), use_container_width=True, hide_index=True)

st.subheader("External Data Decision")
external_result = st.session_state.get("external_intelligence")
if external_result is None:
    st.write(
        "The platform automatically compares models with and without Nigeria weather, route distance, and "
        "geolocation features before training. External features are used only when validation improves."
    )
    with st.spinner("Automatically checking city joins, geography, distance, and predictive usefulness..."):
        external_result = cached_external_weather_intelligence(df)
        st.session_state["external_intelligence"] = external_result
        if external_result.enriched_data is not None:
            st.session_state["external_enriched_dataset"] = external_result.enriched_data

if external_result is not None:
    cols = st.columns(3)
    cols[0].metric("Weather match rate", "{:.1%}".format(external_result.match_rate))
    cols[1].metric("Matched records", "{:,}".format(external_result.matched_rows))
    cols[2].metric("Weather features", "{:,}".format(len(external_result.feature_columns)))
    render_decision_badge(
        "Weather features will be used"
        if external_result.recommended
        else "Weather features will not be forced",
        external_result.recommended,
    )
    render_messages(external_result.reasons)
    if external_result.predictive_test.get("weather_feature_importance"):
        with st.expander("Weather/geolocation feature contribution", expanded=True):
            st.dataframe(
                external_result.predictive_test["weather_feature_importance"],
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "External feature importance share in the quick validation model: {:.1%}".format(
                    external_result.predictive_test.get("weather_importance_share", 0.0)
                )
            )
    markdown_download(
        external_result_markdown(external_result), "Download weather report", "weather_training_decision.md"
    )

training_data = df
if external_result is not None and external_result.recommended and external_result.enriched_data is not None:
    training_data = external_result.enriched_data
    st.success("Training will use the weather-enhanced dataset because it improved validation quality.")
else:
    st.info("Training will use the uploaded dataset without forced external weather features.")

schema = cached_detect_columns(training_data)
default_exclusions = schema.id_like

with st.sidebar:
    st.subheader("Training Controls")
    feature_percentile = st.slider("Feature selection percentile", 20, 100, 80, step=5)
    max_rows = st.number_input(
        "Maximum training rows", min_value=1000, max_value=200000, value=100000, step=1000
    )
    exclude_columns = select_columns(
        "Additional columns to exclude",
        options=list(training_data.columns),
        default=default_exclusions,
        key="training_exclusions",
    )

train_clicked = st.button("Train compatible classification and regression systems", type="primary")

if train_clicked:
    with st.spinner("Training model portfolio and comparing classification vs regression usefulness..."):
        portfolio_result = train_dual_prediction_system(
            training_data,
            tasks=cached_discover_prediction_tasks(training_data),
            exclude_columns=exclude_columns,
            feature_selection_percentile=feature_percentile,
            max_rows=int(max_rows),
            output_path=MODEL_PORTFOLIO_PATH,
            active_model_output_path=DEFAULT_MODEL_PATH,
            metadata={
                "external_intelligence": external_result_to_dict(external_result)
                if external_result is not None
                else None,
                "weather_features_used": bool(
                    external_result is not None
                    and external_result.recommended
                    and external_result.enriched_data is not None
                ),
            },
        )
        st.session_state["model_portfolio"] = portfolio_result["portfolio"]
        st.session_state["model_portfolio_path"] = portfolio_result["portfolio_path"]
        st.session_state["model_artifact"] = portfolio_result["portfolio"]
        st.session_state["model_artifact_path"] = portfolio_result["portfolio_path"]

portfolio = st.session_state.get("model_portfolio") or st.session_state.get("model_artifact")
if not portfolio:
    st.write("Ready to train the compatible systems detected above.")
    st.stop()

comparison = portfolio.get("comparison", {})
st.success(
    "Model portfolio trained. Recommended default: {}".format(
        comparison.get("recommended_model_key") or "none"
    )
)
render_messages(comparison.get("messages", []))
st.caption(
    "Portfolio artifact: {}".format(st.session_state.get("model_portfolio_path") or MODEL_PORTFOLIO_PATH)
)

for key, artifact in portfolio.get("models", {}).items():
    st.subheader("{} Model".format(key.title()))
    render_messages(model_summary(artifact))
    leaderboard = artifact.get("leaderboard")
    if leaderboard is not None and is_advanced_mode():
        st.plotly_chart(plot_model_comparison(leaderboard), use_container_width=True)
        st.dataframe(leaderboard, use_container_width=True, hide_index=True)
    elif leaderboard is not None:
        st.caption("Leaderboard and diagnostic charts are available in Advanced/Data Science Mode.")

    importance = cached_model_feature_importance(artifact["pipeline"])
    if not importance.empty and is_advanced_mode():
        with st.expander(
            "{} feature explanations".format(key.title()),
            expanded=key == comparison.get("recommended_model_key"),
        ):
            st.bar_chart(importance.set_index("feature"))
            render_messages(natural_language_feature_insights(importance))

    markdown_download(
        build_training_report_markdown(artifact),
        "Download {} model report".format(key),
        "{}_model_report.md".format(key),
    )

st.subheader("Operational Recommendations")
all_recommendations = []
for artifact in portfolio.get("models", {}).values():
    all_recommendations.extend(generate_operational_recommendations(artifact=artifact))
st.dataframe(recommendations_table(all_recommendations), use_container_width=True, hide_index=True)

if portfolio.get("failed_models"):
    with st.expander("Candidate failures and skipped tasks"):
        st.dataframe(portfolio["failed_models"], use_container_width=True, hide_index=True)
