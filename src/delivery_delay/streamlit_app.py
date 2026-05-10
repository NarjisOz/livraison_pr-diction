"""Streamlit helpers shared by all analytics pages."""

import hashlib
import html
import json

import pandas as pd
import streamlit as st

from .cleaning import clean_dataframe
from .data_io import read_tabular_file
from .performance import estimate_memory_mb
from .schema import detect_columns, summarize_schema
from .targeting import add_derived_delivery_targets, available_targets


DERIVED_SESSION_KEYS = (
    "dataset_understanding",
    "training_result",
    "model_artifact",
    "model_artifact_path",
    "model_portfolio",
    "model_portfolio_path",
    "prediction_results",
    "external_intelligence",
    "external_enriched_dataset",
)


def apply_app_style():
    """Apply a modern analytics visual system across Streamlit pages."""

    st.markdown(
        """
        <style>
        :root {
            --ink: #102033;
            --muted: #5B677A;
            --panel: #FFFFFF;
            --line: #DDE6EF;
            --surface: #F5F8FB;
            --teal: #0E7C7B;
            --blue: #2563EB;
            --amber: #B7791F;
            --rose: #C2413D;
            --green: #15803D;
        }
        .stApp {
            background:
                linear-gradient(135deg, rgba(14, 124, 123, 0.08), rgba(37, 99, 235, 0.06) 42%, rgba(245, 248, 251, 0.94)),
                linear-gradient(180deg, #F7FAFC 0%, #EEF4F8 100%);
            color: var(--ink);
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 3rem;
            max-width: 1320px;
        }
        h1, h2, h3 {
            letter-spacing: 0;
            color: var(--ink);
        }
        p, li, label, .stMarkdown {
            color: #223247;
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.82), rgba(255, 255, 255, 0.62));
            border: 1px solid rgba(255, 255, 255, 0.70);
            border-radius: 8px;
            padding: 15px 16px;
            box-shadow: 0 14px 35px rgba(16, 32, 51, 0.07);
            backdrop-filter: blur(12px);
            animation: metricRise 420ms ease both;
            transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            border-color: rgba(14, 124, 123, 0.35);
            box-shadow: 0 18px 42px rgba(16, 32, 51, 0.10);
        }
        div[data-testid="stMetricLabel"] {
            color: var(--muted);
        }
        div[data-testid="stTabs"] button {
            font-weight: 600;
        }
        .stAlert {
            border-radius: 8px;
        }
        .stButton > button,
        .stDownloadButton > button {
            border-radius: 8px;
            border: 1px solid rgba(16, 32, 51, 0.12);
            transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 22px rgba(16, 32, 51, 0.10);
            border-color: rgba(14, 124, 123, 0.35);
        }
        .codex-hero {
            border: 1px solid rgba(16, 32, 51, 0.10);
            border-radius: 8px;
            background:
                linear-gradient(135deg, rgba(16, 32, 51, 0.96), rgba(14, 124, 123, 0.92)),
                url("https://images.unsplash.com/photo-1586528116311-ad8dd3c8310d?auto=format&fit=crop&w=1600&q=80");
            background-size: cover;
            background-position: center;
            background-blend-mode: multiply;
            padding: clamp(24px, 4vw, 46px);
            min-height: 260px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-shadow: 0 24px 60px rgba(16, 32, 51, 0.20);
            overflow: hidden;
        }
        .codex-hero small,
        .codex-hero h1,
        .codex-hero p {
            color: #FFFFFF;
        }
        .codex-hero small {
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0;
            opacity: 0.82;
        }
        .codex-hero h1 {
            font-size: clamp(2.0rem, 4vw, 4.2rem);
            line-height: 1.02;
            margin: 0.45rem 0 0.75rem;
            max-width: 900px;
        }
        .codex-hero p {
            font-size: 1.05rem;
            max-width: 800px;
            opacity: 0.92;
        }
        .platform-card,
        .insight-card,
        .workflow-card {
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.82), rgba(255, 255, 255, 0.64));
            border: 1px solid rgba(255, 255, 255, 0.72);
            border-radius: 8px;
            padding: 16px 17px;
            box-shadow: 0 14px 34px rgba(16, 32, 51, 0.07);
            backdrop-filter: blur(14px);
            transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
            height: 100%;
        }
        .platform-card:hover,
        .insight-card:hover,
        .workflow-card:hover {
            transform: translateY(-2px);
            border-color: rgba(37, 99, 235, 0.24);
            box-shadow: 0 18px 40px rgba(16, 32, 51, 0.10);
        }
        .platform-card h3,
        .workflow-card h3,
        .insight-card h3 {
            margin: 0 0 0.35rem;
            font-size: 1rem;
        }
        .platform-card p,
        .workflow-card p,
        .insight-card p {
            margin: 0.35rem 0;
            color: var(--muted);
            font-size: 0.94rem;
        }
        .priority-critical {
            border-left: 5px solid var(--rose);
        }
        .priority-high {
            border-left: 5px solid #E11D48;
        }
        .priority-medium {
            border-left: 5px solid var(--amber);
        }
        .priority-low,
        .priority-info {
            border-left: 5px solid var(--teal);
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 4px 9px;
            font-size: 0.76rem;
            font-weight: 700;
            color: #102033;
            background: #E6F4F1;
            border: 1px solid rgba(14, 124, 123, 0.18);
        }
        .decision-yes {
            color: #FFFFFF;
            background: var(--green);
        }
        .decision-no {
            color: #FFFFFF;
            background: var(--rose);
        }
        .muted-caption {
            color: var(--muted);
            font-size: 0.9rem;
        }
        .mode-banner {
            background: linear-gradient(135deg, rgba(16, 32, 51, 0.88), rgba(14, 124, 123, 0.84));
            border: 1px solid rgba(255, 255, 255, 0.18);
            color: #FFFFFF;
            border-radius: 8px;
            padding: 12px 14px;
            box-shadow: 0 14px 32px rgba(16, 32, 51, 0.12);
        }
        .mode-banner strong,
        .mode-banner span {
            color: #FFFFFF;
        }
        .glass-panel {
            background: rgba(255, 255, 255, 0.68);
            border: 1px solid rgba(255, 255, 255, 0.70);
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: 0 16px 38px rgba(16, 32, 51, 0.08);
            backdrop-filter: blur(14px);
        }
        .executive-brief {
            background: linear-gradient(135deg, rgba(16, 32, 51, 0.94), rgba(14, 124, 123, 0.88));
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 8px;
            padding: 17px 18px;
            box-shadow: 0 18px 44px rgba(16, 32, 51, 0.16);
        }
        .executive-brief h3 {
            margin: 0 0 0.65rem;
            color: #FFFFFF;
            font-size: 1.05rem;
        }
        .executive-brief ul {
            margin: 0;
            padding-left: 1.1rem;
        }
        .executive-brief li {
            color: #FFFFFF;
            margin: 0.35rem 0;
        }
        @keyframes metricRise {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @media (max-width: 760px) {
            .codex-hero {
                min-height: 230px;
                padding: 22px;
            }
            .codex-hero h1 {
                font-size: 2rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_session_state():
    """Create stable state keys used across all pages."""

    defaults = {
        "dataset": None,
        "dataset_name": None,
        "dataset_fingerprint": None,
        "dataset_uploaded_at": None,
        "user_mode": "Business Mode",
        "advanced_charts_enabled": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def dataset_uploader(label="Dataset", key="dataset_uploader"):
    initialize_session_state()
    uploaded = st.file_uploader(label, type=["csv", "xlsx", "xls", "parquet"], key=key)
    if uploaded is None:
        return st.session_state.get("dataset")

    try:
        df = read_tabular_file(uploaded, filename=uploaded.name)
        cleaned = clean_dataframe(df)
    except Exception as exc:
        st.error("The dataset could not be loaded: {}".format(exc))
        st.info(
            "Use a CSV, Excel, or Parquet file with a header row and at least one delivery, route, date, status, "
            "duration, city, carrier, supplier, or cost column."
        )
        return st.session_state.get("dataset")

    fingerprint = dataframe_fingerprint(cleaned)
    if fingerprint != st.session_state.get("dataset_fingerprint"):
        clear_derived_state()
    store_dataset(cleaned, uploaded.name, fingerprint=fingerprint)
    memory_mb = estimate_memory_mb(cleaned)
    if memory_mb > 250:
        st.warning(
            "Large dataset loaded ({:.0f} MB). The app will keep previews and advanced visuals bounded for responsiveness.".format(
                memory_mb
            )
        )
    if cleaned.shape[0] > 200000:
        st.info(
            "This file has {:,} rows. Training controls can cap the training sample while predictions still run on the active file.".format(
                cleaned.shape[0]
            )
        )
    if cleaned.shape[1] > 180:
        st.info(
            "This file has {:,} columns. Identifier-like and weak columns will be filtered before modeling.".format(
                cleaned.shape[1]
            )
        )
    return cleaned


def get_dataset():
    initialize_session_state()
    return st.session_state.get("dataset")


def store_dataset(df, name="uploaded dataset", fingerprint=None):
    """Store the active dataset and metadata in Streamlit session state."""

    st.session_state["dataset"] = df
    st.session_state["dataset_name"] = name
    st.session_state["dataset_fingerprint"] = fingerprint or dataframe_fingerprint(df)
    st.session_state["dataset_uploaded_at"] = pd.Timestamp.utcnow().isoformat()


def clear_derived_state():
    """Clear stale models and reports when a new dataset arrives."""

    for key in DERIVED_SESSION_KEYS:
        st.session_state.pop(key, None)


def dataframe_fingerprint(df, sample_rows=2000):
    """Create a lightweight fingerprint for session invalidation."""

    if df is None:
        return None
    sample = df.head(sample_rows).copy()
    payload = "|".join(map(str, df.shape)) + "|" + "|".join(map(str, df.columns))
    try:
        row_hash = pd.util.hash_pandas_object(sample.astype(str), index=True).values.tobytes()
        return hashlib.sha256(payload.encode("utf-8") + row_hash).hexdigest()
    except Exception:
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def render_platform_header(title, subtitle, eyebrow="Logistics Intelligence"):
    """Render the first-viewport platform header."""

    st.markdown(
        """
        <div class="codex-hero">
            <small>{eyebrow}</small>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """.format(
            eyebrow=html.escape(str(eyebrow)),
            title=html.escape(str(title)),
            subtitle=html.escape(str(subtitle)),
        ),
        unsafe_allow_html=True,
    )


def render_dataset_status(df=None):
    """Show active dataset status in a compact row."""

    df = df if df is not None else get_dataset()
    if df is None:
        st.info("No dataset is active yet. Upload once and every page will reuse it automatically.")
        return

    st.caption("Active dataset: {}".format(st.session_state.get("dataset_name") or "uploaded dataset"))
    render_quality_metrics(df)
    render_dataset_warnings(df)


def render_mode_selector(location="sidebar"):
    """Render global Business/Advanced mode control and persist it."""

    initialize_session_state()
    options = ["Business Mode", "Advanced/Data Science Mode"]
    current = st.session_state.get("user_mode", options[0])
    index = options.index(current) if current in options else 0
    label = "User mode"
    help_text = (
        "Business Mode keeps pages fast and insight-first. Advanced/Data Science Mode unlocks EDA tables, "
        "diagnostics, SHAP, and heavier charts on demand."
    )
    container = st.sidebar if location == "sidebar" else st
    with container:
        try:
            selected = st.segmented_control(label, options=options, default=options[index], help=help_text)
        except Exception:
            selected = st.radio(label, options=options, index=index, help=help_text)
        st.session_state["user_mode"] = selected
        return selected


def is_advanced_mode():
    initialize_session_state()
    return st.session_state.get("user_mode") == "Advanced/Data Science Mode"


def render_mode_banner():
    """Show a compact explanation of the current mode."""

    mode = st.session_state.get("user_mode", "Business Mode")
    if mode == "Business Mode":
        body = (
            "Fast executive workflow: alerts, predictions, estimated dates, confidence, and recommendations."
        )
    else:
        body = "Detailed workflow: EDA, preprocessing logic, model diagnostics, feature importance, and SHAP on demand."
    st.markdown(
        """
        <div class="mode-banner">
            <strong>{mode}</strong><br>
            <span>{body}</span>
        </div>
        """.format(mode=html.escape(mode), body=html.escape(body)),
        unsafe_allow_html=True,
    )


def render_quality_metrics(df):
    missing_rate = df.isna().sum().sum() / max(df.shape[0] * df.shape[1], 1)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", "{:,}".format(df.shape[0]))
    col2.metric("Columns", "{:,}".format(df.shape[1]))
    col3.metric("Missing cells", "{:.2%}".format(missing_rate))
    col4.metric("Duplicate rows", "{:,}".format(int(df.duplicated().sum())))


def dataset_health_warnings(df):
    """Return practical guidance for data issues that affect reliability."""

    if df is None or df.empty:
        return []

    warnings = []
    rows, columns = df.shape
    missing_rate = df.isna().sum().sum() / max(rows * columns, 1)
    duplicate_rate = df.duplicated().mean() if rows else 0.0
    targets = available_targets(df)

    if missing_rate >= 0.25:
        warnings.append(
            "High missing-data rate ({:.1%}). Predictions can still run, but low-completeness rows should be reviewed carefully.".format(
                missing_rate
            )
        )
    elif missing_rate >= 0.10:
        warnings.append(
            "Moderate missing-data rate ({:.1%}). The adaptive preprocessor will impute values and explain the choices.".format(
                missing_rate
            )
        )

    if duplicate_rate >= 0.05:
        warnings.append(
            "Duplicate rows represent {:.1%} of the file. Training removes exact duplicates to reduce memorization risk.".format(
                duplicate_rate
            )
        )

    if not targets:
        warnings.append(
            "No clear delay target was detected yet. Add delivery status, expected/actual delivery dates, delay days, or lead time for training."
        )

    if columns < 3:
        warnings.append(
            "The dataset has very few columns. Richer route, supplier, carrier, date, cost, and city fields usually improve reliability."
        )

    return warnings


def render_dataset_warnings(df):
    """Show compact trust and usability warnings without blocking exploration."""

    for warning in dataset_health_warnings(df)[:3]:
        st.warning(warning)


def render_schema_table(df):
    schema = detect_columns(df)
    st.dataframe(summarize_schema(schema), use_container_width=True, hide_index=True)
    return schema


def target_selector(df, key="target_selector"):
    data, _ = add_derived_delivery_targets(df)
    targets = available_targets(data)
    if not targets:
        st.warning("No automatic target candidate was found.")
        return None, data
    target = st.selectbox("Target column", options=targets, index=0, key=key)
    return target, data


def dataframe_download(df, label, filename):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")


def json_download(data, label, filename):
    payload = json.dumps(data, indent=2, default=str).encode("utf-8")
    st.download_button(label=label, data=payload, file_name=filename, mime="application/json")


def markdown_download(markdown, label, filename):
    st.download_button(label=label, data=markdown.encode("utf-8"), file_name=filename, mime="text/markdown")


def html_download(markup, label, filename):
    st.download_button(label=label, data=markup.encode("utf-8"), file_name=filename, mime="text/html")


def render_score_cards(understanding):
    col_1, col_2, col_3, col_4 = st.columns(4)
    col_1.metric("Quality score", "{:.0f}/100".format(understanding.quality_score))
    col_2.metric("Compatibility", "{:.0f}/100".format(understanding.compatibility_score))
    col_3.metric("Governance", "{:.0f}/100".format(understanding.governance_score))
    col_4.metric("Semantic coverage", "{:.0f}%".format(understanding.semantic_coverage))


def render_messages(messages):
    for message in messages:
        st.markdown("- {}".format(message))


def render_executive_brief(messages, title="Executive AI Brief"):
    """Render high-signal business narrative in a premium panel."""

    if not messages:
        return
    items = "".join("<li>{}</li>".format(html.escape(str(message))) for message in messages)
    st.markdown(
        """
        <div class="executive-brief">
            <h3>{title}</h3>
            <ul>{items}</ul>
        </div>
        """.format(title=html.escape(str(title)), items=items),
        unsafe_allow_html=True,
    )


def render_insight_cards(insights, max_cards=6):
    """Render prioritized story cards."""

    if not insights:
        st.info("No major insight was detected yet.")
        return

    columns = st.columns(2)
    for index, insight in enumerate(insights[:max_cards]):
        priority = html.escape(str(getattr(insight, "priority", "info")).lower())
        with columns[index % 2]:
            st.markdown(
                """
                <div class="insight-card priority-{priority}">
                    <span class="status-pill">{priority_label}</span>
                    <h3>{title}</h3>
                    <p><strong>Observation:</strong> {observation}</p>
                    <p><strong>Why it matters:</strong> {impact}</p>
                    <p><strong>Next action:</strong> {action}</p>
                </div>
                """.format(
                    priority=priority,
                    priority_label=html.escape(str(getattr(insight, "priority", "info")).title()),
                    title=html.escape(str(getattr(insight, "title", ""))),
                    observation=html.escape(str(getattr(insight, "observation", ""))),
                    impact=html.escape(str(getattr(insight, "impact", ""))),
                    action=html.escape(str(getattr(insight, "recommended_action", ""))),
                ),
                unsafe_allow_html=True,
            )


def render_feature_cards(cards):
    """Render small feature/workflow cards from dictionaries."""

    if not cards:
        return
    columns = st.columns(min(len(cards), 3))
    for index, card in enumerate(cards):
        with columns[index % len(columns)]:
            st.markdown(
                """
                <div class="platform-card">
                    <h3>{title}</h3>
                    <p>{body}</p>
                </div>
                """.format(
                    title=html.escape(str(card.get("title", ""))),
                    body=html.escape(str(card.get("body", ""))),
                ),
                unsafe_allow_html=True,
            )


def render_decision_badge(label, positive):
    class_name = "decision-yes" if positive else "decision-no"
    st.markdown(
        '<span class="status-pill {class_name}">{label}</span>'.format(
            class_name=class_name,
            label=html.escape(str(label)),
        ),
        unsafe_allow_html=True,
    )


def select_columns(label, options, default=None, key=None):
    options = list(options)
    default = default or []
    return st.multiselect(label, options=options, default=[c for c in default if c in options], key=key)


def show_dataframe_preview(df, rows=20):
    preview = df.head(rows)
    st.dataframe(preview, use_container_width=True)


def get_training_dataframe_with_targets(df):
    data, _ = add_derived_delivery_targets(df)
    return pd.DataFrame(data)
