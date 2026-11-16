"""Streamlit UI for the Sales Intelligence Agent: run the agent and watch every step live.

    streamlit run app.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "data"))

from sales_agent.agent_llm import check_tool_support, run_llm_agent  # noqa: E402
from sales_agent.agent_offline import run_offline_agent  # noqa: E402
from sales_agent.config import MODEL_PRESETS, AgentConfig  # noqa: E402
from sales_agent.data import DEFAULT_DATA_PATH, DataError, load_sales  # noqa: E402
from sales_agent.periods import parse_period  # noqa: E402
from sales_agent.tools import SalesContext, get_product_performance  # noqa: E402
from sales_agent.trace import TraceEvent, Tracer  # noqa: E402

EXAMPLE_Q = "Analyse my sales performance this month and tell me which products are underperforming."

# Chart colors (reference palette): series-1 blue, muted reference gray, diverging blue <-> red
C_CURRENT = "#2a78d6"
C_PREVIOUS = "#c3c2b7"
C_POS, C_NEG = "#2a78d6", "#e34948"

st.set_page_config(page_title="Sales Intelligence Agent", page_icon="📊", layout="wide")


# ---- data -------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_path(path: str, mtime: float) -> pd.DataFrame:  # mtime busts the cache on regenerate
    return load_sales(path)


@st.cache_data(show_spinner=False)
def _load_bytes(data: bytes) -> pd.DataFrame:
    return load_sales(io.BytesIO(data))


def ensure_sample() -> None:
    if not DEFAULT_DATA_PATH.exists():
        from generate_sample import generate

        DEFAULT_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        generate().to_csv(DEFAULT_DATA_PATH, index=False)


def ollama_status(base: str) -> tuple[bool, list[str]]:
    try:
        r = requests.get(f"{base.rstrip('/')}/api/tags", timeout=1.5)
        r.raise_for_status()
        return True, [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return False, []


# ---- sidebar ----------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Agent settings")
    cfg = AgentConfig.from_env()
    # Default to offline when the configured local LLM isn't available.
    llm_ready = not cfg.is_ollama or ollama_status(cfg.effective_api_base or "")[0]
    mode = st.radio("Mode", ["LLM agent", "Offline pipeline"], index=0 if llm_ready else 1, horizontal=True,
                    help="LLM: the model chooses which tools to call. Offline: fixed tool order, no LLM.")
    if mode == "LLM agent":
        presets = list(dict.fromkeys([cfg.model, *MODEL_PRESETS]))
        choice = st.selectbox("Model", presets + ["Custom…"],
                              help="Any LiteLLM model string, e.g. ollama_chat/qwen2.5:7b, anthropic/…, openai/…")
        cfg.model = st.text_input("Custom model string", value=cfg.model) if choice == "Custom…" else choice
        cfg.api_base = st.text_input("API base URL (optional)", value=cfg.effective_api_base or "",
                                     help="Leave empty for hosted providers.") or None
        cfg.max_iterations = st.slider("Max agent iterations", 3, 20, cfg.max_iterations)

        if cfg.is_ollama:
            up, models = ollama_status(cfg.effective_api_base or "")
            tag = cfg.model.split("/", 1)[-1]
            if not up:
                st.error(f"Ollama not reachable at {cfg.effective_api_base}. Start it with `ollama serve`.", icon="🔴")
            elif not any(m == tag or m.split(":")[0] == tag for m in models):
                st.warning(f"Ollama is running, but `{tag}` isn't pulled.\n\n`ollama pull {tag}`", icon="🟡")
                if models:
                    st.caption("Installed: " + ", ".join(models))
            else:
                st.success(f"Ollama running · `{tag}` available", icon="🟢")
        support = check_tool_support(cfg.model)
        if support is False:
            st.warning("LiteLLM reports this model may not support tool calling.", icon="⚠️")

    st.divider()
    st.header("📁 Data")
    upload = st.file_uploader("Upload your own CSV", type="csv",
                              help="Required columns: date, product_id, product_name, units, revenue. "
                                   "Optional: order_id, category, region, unit_price, discount.")
    if st.button("Regenerate sample data", width="stretch"):
        from generate_sample import generate

        generate().to_csv(DEFAULT_DATA_PATH, index=False)
        st.toast("Sample data regenerated")

try:
    if upload is not None:
        df = _load_bytes(upload.getvalue())
        source = upload.name
    else:
        ensure_sample()
        df = _load_path(str(DEFAULT_DATA_PATH), DEFAULT_DATA_PATH.stat().st_mtime)
        source = "sample data"
except DataError as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    st.caption(f"Using **{source}**: {len(df):,} rows, {df['product_id'].nunique()} products, "
               f"{df['date'].min():%Y-%m-%d} → {df['date'].max():%Y-%m-%d}")


# ---- rendering helpers ------------------------------------------------------
def render_event(e: TraceEvent) -> None:
    with st.expander(f"{e.title}  ·  {e.output_summary[:90]}" if e.output_summary else e.title,
                     expanded=e.kind == "error"):
        cols = st.columns(4)
        cols[0].caption(f"⏱ {e.duration_ms:.0f} ms" if e.duration_ms is not None else "⏱ –")
        if e.tokens:
            cols[1].caption(f"🔤 {e.tokens.get('prompt', 0)} in / {e.tokens.get('completion', 0)} out")
        if e.model:
            cols[2].caption(f"🤖 {e.model}")
        cols[3].caption(f"🕒 {e.timestamp[11:]}")
        if e.kind == "error":
            st.error(e.output_summary)
        if e.input is not None:
            st.markdown("**Input**")
            st.json(e.input, expanded=True)
        if e.output is not None and e.kind != "final":
            st.markdown("**Output**")
            st.json(e.output, expanded=False)


def render_charts(perf: dict) -> None:
    prods = sorted(perf["products"], key=lambda p: p["current_revenue"])
    flagged = {u["product"] for u in perf["underperformers"]}
    labels = [("⚠ " if p["product"] in flagged else "") + p["product"] for p in prods]

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_bar(y=labels, x=[p["previous_revenue"] for p in prods], orientation="h", name="Previous period",
                    marker_color=C_PREVIOUS, hovertemplate="%{y}<br>Previous: $%{x:,.0f}<extra></extra>")
        fig.add_bar(y=labels, x=[p["current_revenue"] for p in prods], orientation="h", name="Current period",
                    marker_color=C_CURRENT, hovertemplate="%{y}<br>Current: $%{x:,.0f}<extra></extra>")
        fig.update_layout(title="Revenue by product (⚠ = underperforming)", barmode="group", bargap=0.25,
                          bargroupgap=0.08, height=max(420, 26 * len(prods)), margin=dict(l=10, r=10, t=50, b=10),
                          legend=dict(orientation="h", y=1.04, x=0), xaxis=dict(tickprefix="$", gridcolor="#e1e0d9"))
        st.plotly_chart(fig, width="stretch")
    with c2:
        g = sorted([p for p in prods if p["growth_pct"] is not None], key=lambda p: p["growth_pct"])
        fig = go.Figure(go.Bar(
            y=[("⚠ " if p["product"] in flagged else "") + p["product"] for p in g],
            x=[p["growth_pct"] for p in g], orientation="h",
            marker_color=[C_NEG if p["growth_pct"] < 0 else C_POS for p in g],
            text=[f"{p['growth_pct']:+.1f}%" for p in g], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br>Growth: %{x:+.1f}%<extra></extra>",
        ))
        thr = perf["thresholds"]["decline_pct"]
        fig.add_vline(x=thr, line_dash="dot", line_color="#898781",
                      annotation_text=f"flag threshold {thr:.0f}%", annotation_position="top")
        fig.update_layout(title="Revenue growth vs previous period", height=max(420, 26 * len(g)),
                          margin=dict(l=10, r=40, t=50, b=10), showlegend=False,
                          xaxis=dict(ticksuffix="%", zeroline=True, zerolinecolor="#c3c2b7", gridcolor="#e1e0d9"))
        st.plotly_chart(fig, width="stretch")


def render_kpis(ctx_results: dict) -> None:
    growth = ctx_results.get("calculate_growth", {}).get("overall")
    perf = ctx_results.get("get_product_performance")
    rev = ctx_results.get("calculate_revenue")
    cols = st.columns(4)
    if growth:
        gp = growth["growth_pct"]
        cols[0].metric("Revenue (current)", f"${growth['current_revenue']:,.0f}",
                       f"{gp:+.1f}% vs previous" if gp is not None else None)
        ug = growth["units_growth_pct"]
        cols[1].metric("Units", f"{growth['current_units']:,}", f"{ug:+.1f}%" if ug is not None else None)
    if rev:
        cols[2].metric("Avg order value", f"${rev['average_order_value']:,.2f}")
    if perf:
        cols[3].metric("Underperforming products", f"{len(perf['underperformers'])} / {perf['product_count']}")


def render_results(run: dict) -> None:
    if not run.get("report"):
        st.warning("The agent finished without producing a report. Check the trace above for errors.")
        return
    render_kpis(run["results"])
    tab_report, tab_charts, tab_trace = st.tabs(["📄 Report", "📈 Charts", "🧾 Trace JSON"])
    with tab_report:
        st.markdown(run["report"])
        st.download_button("Download report (.md)", run["report"], file_name="sales_report.md",
                           key=f"dl_md_{run['id']}")
    with tab_charts:
        if run.get("perf"):
            render_charts(run["perf"])
    with tab_trace:
        st.download_button("Download trace (.json)", run["trace_json"], file_name="agent_trace.json",
                           mime="application/json", key=f"dl_tr_{run['id']}")
        st.json(json.loads(run["trace_json"]), expanded=False)


def totals_line(t: dict) -> str:
    s = f"{t['tool_calls']} tool calls · {t['steps']} steps · {t['duration_ms'] / 1000:.1f}s"
    if t["prompt_tokens"] or t["completion_tokens"]:
        s += f" · {t['prompt_tokens']:,} prompt + {t['completion_tokens']:,} completion tokens"
    if t["errors"]:
        s += f" · {t['errors']} error(s)"
    return s


# ---- main -------------------------------------------------------------------
st.title("📊 Sales Intelligence Agent")
st.caption("Ask a question about your sales. The agent retrieves the data, calculates revenue and growth, ranks "
           "products, and writes a report. Every step is traced below.")

if "runs" not in st.session_state:
    st.session_state.runs = []

with st.form("ask"):
    question = st.text_area("Question", value=EXAMPLE_Q, height=80)
    submitted = st.form_submit_button("▶ Run analysis", type="primary")

if submitted and question.strip():
    ctx = SalesContext(df=df)
    model_name = cfg.model if mode == "LLM agent" else None
    tracer = Tracer("llm" if mode == "LLM agent" else "offline", question, model_name)
    agent = run_llm_agent(question, ctx, tracer, cfg) if mode == "LLM agent" else run_offline_agent(question, ctx, tracer)

    st.subheader("🔍 Execution trace")
    status = st.status(f"Running {mode.lower()}…", expanded=False)
    trace_box = st.container(border=True)
    for event in agent:
        with trace_box:
            render_event(event)
        label = {"llm_call": "🧠 Model deciding next step…",
                 "tool_call": f"🔧 Running {event.tool_name}…"}.get(event.kind)
        if label:
            status.update(label=label)
    totals = tracer.totals()
    status.update(label=f"{'Done' if ctx.report_markdown else 'Stopped'} · {totals_line(totals)}",
                  state="complete" if ctx.report_markdown else "error")

    try:
        tracer.save()
    except OSError:
        pass

    perf = ctx.results.get("get_product_performance")
    if perf is None and ctx.report_markdown:  # LLM skipped ranking: compute for charts
        p = parse_period(question, ctx.today)
        get_product_performance(ctx, **{k: p[k] for k in ("current_start", "current_end", "previous_start",
                                                          "previous_end")})
        perf = ctx.results["get_product_performance"]

    run = {
        "id": len(st.session_state.runs),
        "question": question,
        "mode": mode,
        "model": model_name,
        "events": tracer.events,
        "totals": totals,
        "report": ctx.report_markdown,
        "results": dict(ctx.results),
        "perf": perf,
        "trace_json": tracer.to_json(),
    }
    st.session_state.runs.append(run)
    render_results(run)

elif st.session_state.runs:
    run = st.session_state.runs[-1]
    st.subheader("🔍 Execution trace")
    st.status(f"Last run · {totals_line(run['totals'])}", state="complete" if run["report"] else "error",
              expanded=False)
    with st.container(border=True):
        for e in run["events"]:
            render_event(e)
    render_results(run)

if len(st.session_state.runs) > 1:
    st.divider()
    st.markdown(f"**🕘 Previous runs ({len(st.session_state.runs) - 1})**")
    for r in reversed(st.session_state.runs[:-1]):
        model_txt = f" · `{r['model']}`" if r["model"] else ""
        st.markdown(f"**#{r['id'] + 1}** · {r['mode']}{model_txt} · {totals_line(r['totals'])}  \n> {r['question']}")
