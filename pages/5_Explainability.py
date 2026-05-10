"""Model and decision explainability page."""

import streamlit as st

from delivery_delay.config import DEFAULT_MODEL_PATH, MODEL_PORTFOLIO_PATH
from delivery_delay.dual_prediction import is_portfolio, portfolio_model_options, select_portfolio_model
from delivery_delay.explainability import (
    natural_language_feature_insights,
)
from delivery_delay.performance import (
    cached_analyze_dataset,
    cached_load_model_artifact,
    cached_model_feature_importance,
    cached_preprocessing_explanations,
    cached_shap_importance,
    lightweight_table,
    model_file_signature,
)
from delivery_delay.streamlit_app import (
    apply_app_style,
    dataset_uploader,
    get_dataset,
    initialize_session_state,
    is_advanced_mode,
    markdown_download,
    render_dataset_status,
    render_messages,
    render_mode_banner,
    render_mode_selector,
)
from delivery_delay.understanding import (
    preprocessing_explanations_markdown,
)


def _importance_markdown(importance, model_key):
    lines = ["# Feature Importance Explanation", "", "- Model: {}".format(model_key), ""]
    for _, row in importance.iterrows():
        lines.append("- {}: {:.5f}".format(row["feature"], float(row["importance"])))
    return "\n".join(lines)


st.set_page_config(page_title="Explainability", page_icon=":mag:", layout="wide")
initialize_session_state()
apply_app_style()
st.title("Explainability Center")

with st.sidebar:
    render_mode_selector()

df = get_dataset()
if df is None:
    df = dataset_uploader("Upload dataset once", key="explainability_upload")
if df is None:
    st.info("Upload a dataset first so explanations can connect model behavior to the active data.")
    st.stop()

render_dataset_status(df)
render_mode_banner()

artifact = st.session_state.get("model_artifact") or st.session_state.get("model_portfolio")
if artifact is None and MODEL_PORTFOLIO_PATH.exists():
    artifact = cached_load_model_artifact(str(MODEL_PORTFOLIO_PATH), model_file_signature(MODEL_PORTFOLIO_PATH))
if artifact is None and DEFAULT_MODEL_PATH.exists():
    artifact = cached_load_model_artifact(str(DEFAULT_MODEL_PATH), model_file_signature(DEFAULT_MODEL_PATH))

if artifact is None:
    st.warning("Train a model before opening model explanations. Preprocessing explanations are still available below.")
else:
    model_options = portfolio_model_options(artifact)
    selected_key = model_options[0][0]
    if is_portfolio(artifact) and len(model_options) > 1:
        labels = {label: key for key, label in model_options}
        selected_label = st.selectbox("Model to explain", options=list(labels.keys()), index=0)
        selected_key = labels[selected_label]
    model_artifact, resolved_key = select_portfolio_model(artifact, model_key=selected_key)

    st.subheader("Model Explanation")
    st.write(
        "Feature importance explains which transformed inputs were most influential. SHAP can be computed on demand "
        "for a deeper local/global explanation when the installed model supports it."
    )
    importance = cached_model_feature_importance(model_artifact["pipeline"])
    if importance.empty:
        st.info("This model does not expose native coefficients or feature importances.")
    else:
        st.bar_chart(importance.head(12).set_index("feature"))
        render_messages(natural_language_feature_insights(importance))
        markdown_download(
            _importance_markdown(importance, resolved_key),
            "Download feature-importance explanation",
            "{}_feature_importance.md".format(resolved_key),
        )

    if is_advanced_mode():
        with st.expander("Compute SHAP summary", expanded=False):
            st.caption("SHAP is intentionally lazy and cached because it is one of the heaviest explainability steps.")
            if st.button("Run SHAP explanation", key="compute_shap_explainability"):
                with st.spinner("Computing cached SHAP values on a bounded sample..."):
                    try:
                        shap_df = cached_shap_importance(model_artifact["pipeline"], df, max_rows=80)
                        st.dataframe(lightweight_table(shap_df), use_container_width=True, hide_index=True)
                        if not shap_df.empty:
                            st.bar_chart(shap_df.set_index("feature"))
                            render_messages(natural_language_feature_insights(shap_df))
                    except Exception as exc:
                        st.warning("SHAP could not be computed for this model: {}".format(exc))
    else:
        st.info("SHAP is available in Advanced/Data Science Mode and runs only when requested.")

st.subheader("Preprocessing Explanation")
understanding = st.session_state.get("dataset_understanding")
if understanding is None:
    target = artifact.get("target_column") if isinstance(artifact, dict) and "target_column" in artifact else None
    understanding = cached_analyze_dataset(df, target_column=target)

explanations = cached_preprocessing_explanations(understanding)
st.dataframe(
    explanations if is_advanced_mode() else lightweight_table(explanations, max_rows=8, max_columns=5),
    use_container_width=True,
    hide_index=True,
)
markdown_download(
    preprocessing_explanations_markdown(understanding),
    "Download preprocessing explanation",
    "explainability_preprocessing.md",
)

external_result = st.session_state.get("external_intelligence")
if external_result is not None:
    st.subheader("External Data Explainability")
    render_messages(external_result.reasons)
    st.json(external_result.predictive_test)
