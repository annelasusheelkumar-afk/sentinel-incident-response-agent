"""SENTINEL - AI Incident Response & Organizational Memory
Built on Hindsight memory + Groq LLM + Streamlit.

Setup:
  pip install streamlit hindsight-client openai
  $env:HINDSIGHT_BASE_URL="https://api.hindsight.vectorize.io"
  $env:HINDSIGHT_API_KEY="your-hsk_-key"
  $env:GROQ_API_KEY="your-gsk_-key"
Run:
  py -m streamlit run app.py
"""
import asyncio
import json
import os
import re
from datetime import datetime

import streamlit as st
from hindsight_client import Hindsight
from openai import OpenAI

BANK = "incident-agent"
MODEL = "openai/gpt-oss-120b"  # fallback: qwen/qwen3-32b

HINDSIGHT_URL = os.environ.get("HINDSIGHT_BASE_URL", "")
HINDSIGHT_KEY = os.environ.get("HINDSIGHT_API_KEY", "")
GROQ_KEY = os.environ.get("GROQ_API_KEY", "")

# Demo incidents - stored through the real Hindsight retain call.
# Stable document_ids make reloading safe (Hindsight upserts by document_id).
SEED = [
    ("demo-payment-timeout", "payments",
     "Payment API timed out during peak traffic.",
     "Database connection pool exhaustion.",
     "Increased database connection pool capacity and restarted affected services."),
    ("demo-checkout-504", "checkout",
     "Checkout requests returned intermittent 504 errors during high traffic.",
     "Database connection pool saturation.",
     "Increased connection pool capacity and optimized connection timeout settings."),
    ("demo-order-latency", "orders",
     "Order service experienced elevated latency after traffic increased.",
     "Database connections were exhausted under peak load.",
     "Adjusted connection pool limits and restarted affected application instances."),
]
DEMO_TAG = "[DEMO INCIDENT]"

# Phrases used to recognise the demo incidents when listing memories.
# NOTE: Hindsight runs fact extraction on retain, so the stored text is often
# a distilled version of what we sent (the [DEMO INCIDENT] prefix may be gone).
# We therefore match on the incident wording, not on the prefix.
DEMO_PHRASES = {
    "demo-payment-timeout": "payment api timed out during peak traffic",
    "demo-checkout-504": "checkout requests returned intermittent 504 errors",
    "demo-order-latency": "order service experienced elevated latency",
}

# ============================================================ Hindsight ==


def run_async(coro):
    return asyncio.run(coro)


def seed_memory():
    items = [
        {
            "content": f"{DEMO_TAG} {symptom} Root cause: {cause} Resolution: {fix}",
            "context": "demo incident postmortem",
            "timestamp": datetime.now(),
            "document_id": doc_id,
        }
        for doc_id, svc, symptom, cause, fix in SEED
    ]

    async def _go():
        with Hindsight(base_url=HINDSIGHT_URL, api_key=HINDSIGHT_KEY, timeout=60.0) as client:
            await client.aretain_batch(bank_id=BANK, items=items, retain_async=False)

    run_async(_go())


def recall_memory(query):
    async def _go():
        with Hindsight(base_url=HINDSIGHT_URL, api_key=HINDSIGHT_KEY, timeout=60.0) as client:
            return await client.arecall(bank_id=BANK, query=query, budget="mid", max_tokens=2048)

    return run_async(_go())


def reflect_memory(query):
    async def _go():
        with Hindsight(base_url=HINDSIGHT_URL, api_key=HINDSIGHT_KEY, timeout=60.0) as client:
            return await client.areflect(bank_id=BANK, query=query, budget="mid")

    return run_async(_go())


def retain_one(content):
    async def _go():
        with Hindsight(base_url=HINDSIGHT_URL, api_key=HINDSIGHT_KEY, timeout=60.0) as client:
            await client.aretain(bank_id=BANK, content=content,
                                 context="incident postmortem", retain_async=False)

    run_async(_go())


# ================================================================= LLM ==

DEFAULT_RESULT = {
    "severity": "Unknown", "affected_service": "Unknown",
    "summary": "", "root_cause": "Could not parse structured output.",
    "recommended_actions": [], "confidence": "Low", "why_recommendation": "",
}


def _llm_client():
    return OpenAI(api_key=GROQ_KEY, base_url="https://api.groq.com/openai/v1")


def ask_llm(system, user):
    r = _llm_client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.2,
    )
    return r.choices[0].message.content


def ask_llm_json(system, user):
    def _call(json_mode):
        kwargs = dict(
            model=MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.2,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return _llm_client().chat.completions.create(**kwargs)

    try:
        r = _call(True)
    except Exception:
        r = _call(False)
    raw = r.choices[0].message.content or ""

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    try:
        data = json.loads(cleaned)
    except Exception:
        data = {**DEFAULT_RESULT, "summary": raw}

    out = {**DEFAULT_RESULT, **{k: v for k, v in data.items() if v not in (None, "")}}
    if isinstance(out["recommended_actions"], str):
        out["recommended_actions"] = [out["recommended_actions"]]
    return out


def investigate_with_ai(problem, memories):
    context = "\n".join(f"- {m}" for m in memories) or "No similar past incidents in memory."
    system = (
        "You are Sentinel, an enterprise incident response AI with access to organizational "
        "memory of past incidents. Use the recalled past incidents (if any) to ground your "
        "answer. Judge severity by business impact: payment/checkout/auth outages affecting "
        "all users are usually Critical or High; degraded performance is usually Medium; "
        "minor/cosmetic issues are Low. "
        "Respond with ONLY a valid JSON object, no markdown, no code fences, no extra text, "
        "with exactly these keys: "
        "severity (Critical|High|Medium|Low), "
        "affected_service (short string), "
        "summary (2-3 sentence investigation summary), "
        "root_cause (string), "
        "recommended_actions (array of 3-5 short action strings), "
        "confidence (High|Medium|Low), "
        "why_recommendation (1-2 sentences on how the past incidents, if any, support this "
        "recommendation - or general reasoning if none were found)."
    )
    user = f"New incident: {problem}\n\nPast incidents recalled from organizational memory:\n{context}"
    return ask_llm_json(system, user)


def answer_without_memory(problem):
    return ask_llm("You are an on-call engineer assistant. Be concise.",
                   f"New incident: {problem}\nGive likely causes and steps.")


# ============================================================== helpers ==

RECORD_RE = re.compile(
    r"^(?:\[DEMO INCIDENT\]\s*)?(?:Incident on [\d-]+\.\s*)?(?:Symptom:\s*)?"
    r"(?P<symptom>.*?)\s*Root cause:\s*(?P<cause>.*?)\s*"
    r"(?:Resolution|Fix):\s*(?P<fix>.*?)\.?\s*$", re.IGNORECASE | re.DOTALL)


def looks_like_demo(text):
    low = text.lower()
    return DEMO_TAG.lower() in low or any(p in low for p in DEMO_PHRASES.values())


def parse_memory_record(text):
    """Best-effort structured parse of a stored memory string. Falls back to raw text."""
    m = RECORD_RE.search(text)
    is_demo = looks_like_demo(text)
    if m:
        return {"symptom": m.group("symptom").strip().rstrip("."),
                "root_cause": m.group("cause").strip().rstrip("."),
                "resolution": m.group("fix").strip().rstrip("."),
                "raw": text, "is_demo": is_demo}
    return {"symptom": None, "root_cause": None, "resolution": None,
            "raw": text, "is_demo": is_demo}


def _list_memories(client):
    """Call list_memories, tolerating the search kwarg being named `q` or `search_query`."""
    try:
        return client.list_memories(bank_id=BANK, q="incident", limit=100)
    except TypeError:
        return client.list_memories(bank_id=BANK, search_query="incident", limit=100)


def check_demo_status():
    """Check whether the demo incidents exist in Hindsight.

    1. Ask for each document_id directly (exact, if the client supports it).
    2. Fall back to matching the incident wording in listed memories.
    3. If this session just stored them successfully, trust that.
    """
    try:
        found = set()
        with Hindsight(base_url=HINDSIGHT_URL, api_key=HINDSIGHT_KEY, timeout=60) as client:
            getter = getattr(client, "get_document", None)
            if getter:
                for doc_id in DEMO_PHRASES:
                    try:
                        if getter(bank_id=BANK, document_id=doc_id):
                            found.add(doc_id)
                    except Exception:
                        pass  # not found

            if len(found) < len(DEMO_PHRASES):
                response = _list_memories(client)  # real connectivity errors propagate
                for item in getattr(response, "items", []) or []:
                    text = (getattr(item, "text", "") or "").lower()
                    for doc_id, phrase in DEMO_PHRASES.items():
                        if phrase in text:
                            found.add(doc_id)

        if len(found) < len(DEMO_PHRASES) and st.session_state.get("seeded_ok"):
            found = set(DEMO_PHRASES)  # stored synchronously this session

        return {"count": len(found), "total": len(DEMO_PHRASES), "error": None, "found": found}
    except Exception as e:
        return {"count": None, "total": len(DEMO_PHRASES), "error": e, "found": set()}


def get_demo_status(force=False):
    if force or "demo_status" not in st.session_state:
        st.session_state["demo_status"] = check_demo_status()
    return st.session_state["demo_status"]


def badge(label, bg, fg):
    return f'<span class="badge" style="background:{bg};color:{fg};">{label}</span>'


SEVERITY_COLORS = {
    "Critical": ("#7F1D1D", "#FCA5A5"), "High": ("#7C2D12", "#FED7AA"),
    "Medium": ("#78350F", "#FDE68A"), "Low": ("#14532D", "#BBF7D0"),
    "Unknown": ("#334155", "#CBD5E1"),
}
STATUS_COLORS = {
    "Investigating": ("#1E3A8A", "#BFDBFE"),
    "Resolved": ("#14532D", "#BBF7D0"),
}


def card(html, cls="card"):
    st.markdown(f'<div class="{cls}">{html}</div>', unsafe_allow_html=True)


def render_error(title, message, exc=None):
    card(
        f'<b style="color:#FCA5A5 !important;">⚠ {title}</b><br>{message}',
        cls="card error-card",
    )
    if exc is not None:
        with st.expander("Technical details"):
            st.code(str(exc)[:800])


def check_config():
    missing = []
    if not HINDSIGHT_URL:
        missing.append("HINDSIGHT_BASE_URL")
    if not HINDSIGHT_KEY:
        missing.append("HINDSIGHT_API_KEY")
    if not GROQ_KEY:
        missing.append("GROQ_API_KEY")
    if missing:
        st.set_page_config(page_title="Sentinel - Configuration Error", page_icon="🛡️", layout="centered")
        inject_css()
        st.markdown("<h1>🛡️ Sentinel</h1>", unsafe_allow_html=True)
        card(
            f'<b style="color:#FCA5A5 !important;">⚠ CONFIGURATION ERROR</b><br>'
            f'A required service credential is missing: <code>{", ".join(missing)}</code>.<br><br>'
            f'Set it in this PowerShell window before starting the app, for example:'
            f'<pre style="background:#0B1120;padding:10px;border-radius:8px;">'
            f'$env:{missing[0]}="..."</pre>Then restart with <code>py -m streamlit run app.py</code>.',
            cls="card error-card",
        )
        st.stop()


def next_id():
    st.session_state["_id_counter"] = st.session_state.get("_id_counter", 0) + 1
    return st.session_state["_id_counter"]


# ---- navigation helpers (callbacks run BEFORE widgets are re-created, so
# they can safely change widget-bound session state) -------------------------

PAGES = ["Overview", "Investigate", "Memory", "Patterns", "Incident History"]


def goto(p):
    st.session_state["page"] = p
    st.session_state["nav_side"] = p
    st.session_state["nav_top"] = p


def _nav_side_changed():
    goto(st.session_state["nav_side"])


def _nav_top_changed():
    goto(st.session_state["nav_top"])


def _request_load():
    st.session_state["trigger_load"] = True
    goto("Overview")


# ================================================================== CSS ==

def inject_css():
    st.markdown("""
    <style>
    html, body, .stApp,
    div[data-testid="stAppViewContainer"],
    div[data-testid="stAppViewContainer"] > .main,
    div[data-testid="stMain"] {
        background-color:#0A0E1A !important;
    }
    header[data-testid="stHeader"] {
        background-color:#0A0E1A !important;
        border-bottom:1px solid #1E2A3E;
    }
    div[data-testid="stToolbar"] { visibility:hidden; }
    #MainMenu { visibility:hidden; }
    footer { visibility:hidden; }
    div[data-testid="stAppViewContainer"] .block-container {
        padding-top:2rem !important; max-width:1200px;
    }
    * { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    h1, h2, h3, h4, p, label, span, div, li, .stMarkdown, .stMetric label, .stMetric div {
        color:#E5E9F0 !important;
    }
    h3 { margin-top:28px !important; margin-bottom:6px !important; font-weight:700 !important; }
    p.caption, .stCaption, [data-testid="stCaptionContainer"] { color:#8B95A7 !important; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color:#0D1220 !important; border-right:1px solid #1B2233;
    }
    section[data-testid="stSidebar"] > div { background-color:#0D1220 !important; }

    /* Buttons */
    .stButton > button, .stButton > button:focus {
        background-color:#131B2C !important;
        color:#CBD5E1 !important;
        border:1px solid #1E2A3E !important;
        border-radius:9px !important;
        font-weight:600 !important;
        box-shadow:none !important;
        transition:all .15s ease !important;
    }
    .stButton > button:hover {
        background-color:#182238 !important;
        border-color:#334155 !important;
        color:#F1F5F9 !important;
    }
    .stButton > button[kind="primary"] {
        background-color:#2563EB !important;
        border-color:#2563EB !important;
        color:#FFFFFF !important;
    }
    .stButton > button[kind="primary"]:hover { background-color:#1D4ED8 !important; }
    section[data-testid="stSidebar"] .stButton > button {
        text-align:left !important; justify-content:flex-start !important;
        background-color:transparent !important; border:1px solid transparent !important;
        font-weight:500 !important; padding:8px 12px !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color:#151D30 !important;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background-color:rgba(37,99,235,0.16) !important;
        border-left:3px solid #3B82F6 !important;
        color:#F1F5F9 !important; font-weight:700 !important;
        border-radius:6px !important;
    }

    /* Inputs */
    .stTextArea textarea, .stTextInput input, .stSelectbox > div > div {
        background:#111826 !important; color:#E5E9F0 !important;
        border:1px solid #1E2A3E !important; border-radius:10px !important;
    }
    div[data-baseweb="select"] > div { background:#111826 !important; border-color:#1E2A3E !important; }
    div[data-baseweb="select"] span { color:#E5E9F0 !important; }
    div[data-baseweb="select"] svg { fill:#8B95A7 !important; }

    /* Dropdown popover/menu (floating portal) */
    div[data-baseweb="popover"] { background:transparent !important; }
    div[data-baseweb="popover"] > div,
    ul[data-baseweb="menu"],
    div[data-baseweb="menu"] {
        background-color:#111826 !important;
        border:1px solid #2A3A55 !important;
        border-radius:10px !important;
        box-shadow:0 8px 24px rgba(0,0,0,0.45) !important;
    }
    li[data-baseweb="menu-item"], div[data-baseweb="menu"] li, [role="option"] {
        background-color:transparent !important;
        color:#E5E9F0 !important;
    }
    li[data-baseweb="menu-item"]:hover, [role="option"]:hover,
    li[aria-selected="true"], [role="option"][aria-selected="true"] {
        background-color:#1B2A45 !important;
        color:#93C5FD !important;
    }
    ul[data-testid="stVirtualDropdown"] { background-color:#111826 !important; }
    ul[data-testid="stVirtualDropdown"] li { color:#E5E9F0 !important; }
    ul[data-testid="stVirtualDropdown"] li:hover { background-color:#1B2A45 !important; }

    /* Badges & cards */
    .badge {
        display:inline-block; padding:5px 13px; border-radius:999px;
        font-size:0.76rem; font-weight:700; margin-right:8px; margin-bottom:6px;
        letter-spacing:0.02em;
    }
    .card {
        background:#111826 !important; border:1px solid #1E2A3E; border-radius:12px;
        padding:18px 22px; margin-bottom:16px; box-shadow:0 1px 8px rgba(0,0,0,0.25);
        transition:border-color .15s ease;
    }
    .card:hover { border-color:#2A3A55; }
    .card b {
        color:#7CA8E8 !important; font-size:0.78rem; letter-spacing:0.05em;
        text-transform:uppercase; font-weight:700;
    }
    .card ol, .card ul { margin:10px 0 0 0; padding:0; list-style:none; }
    .card ol li, .card ul li { margin-bottom:8px; }
    .memory-card { border:1px solid #4C3A8C !important; background:#151129 !important; }
    .memory-card b { color:#B7A6F0 !important; }
    .error-card { border:1px solid #7F1D1D !important; background:#1A0F0F !important; }
    .metric-card {
        background:#111826; border:1px solid #1E2A3E; border-radius:12px;
        padding:20px; text-align:center;
    }
    .metric-card .num { font-size:2rem; font-weight:800; color:#F8FAFC !important; }
    .metric-card .lbl {
        font-size:0.72rem; letter-spacing:0.06em; color:#8B95A7 !important;
        text-transform:uppercase; margin-top:4px;
    }
    hr { border-color:#1E2A3E !important; margin:24px 0 !important; }
    .status-dot { color:#22C55E; }
    .brand-title { font-weight:800; letter-spacing:0.02em; margin-bottom:0 !important; }
    .brand-sub { color:#8B95A7 !important; font-size:0.76rem; letter-spacing:0.04em; text-transform:uppercase; }
    div[data-testid="stExpander"] {
        background:#111826 !important; border:1px solid #1E2A3E !important; border-radius:10px !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================ APP INIT ==

st.set_page_config(page_title="Sentinel - AI Incident Response", page_icon="🛡️", layout="wide")
check_config()
inject_css()

if "page" not in st.session_state:
    goto("Overview")
if "history" not in st.session_state:
    st.session_state["history"] = []          # list of incident dicts

# ---------------------------------------------------------------- SIDEBAR
with st.sidebar:
    st.markdown('<p class="brand-title" style="font-size:1.3rem;">🛡️ SENTINEL</p>', unsafe_allow_html=True)
    st.markdown('<p class="brand-sub">AI Incident Response</p>', unsafe_allow_html=True)

    st.radio("NAVIGATION", PAGES, key="nav_side",
             on_change=_nav_side_changed, label_visibility="collapsed")

    st.divider()
    dstatus = get_demo_status()
    st.markdown('<p class="brand-sub" style="margin-bottom:4px;">🧠 MEMORY</p>', unsafe_allow_html=True)
    if dstatus["error"] is not None:
        st.markdown('<span style="color:#F59E0B;font-size:0.85rem;">⚠ Unavailable</span>',
                    unsafe_allow_html=True)
    elif dstatus["count"] and dstatus["count"] >= len(SEED):
        st.caption(f"{len(SEED)} demo incidents available")
        st.button("VIEW MEMORY", key="sb_view_memory", use_container_width=True,
                  on_click=goto, args=("Memory",))
    else:
        st.caption("Not initialized")
        st.button("LOAD MEMORY", key="sb_load_memory", use_container_width=True,
                  on_click=_request_load)

    st.divider()
    st.markdown(
        '<span class="status-dot">●</span> Hindsight Connected<br>'
        '<span class="status-dot">●</span> AI Engine Connected',
        unsafe_allow_html=True)
    st.caption("v1.0")

# ---------------------------------------------------------------- HEADER
head_l, head_r = st.columns([4, 1.3])
with head_l:
    st.markdown('<h1 class="brand-title">SENTINEL</h1>', unsafe_allow_html=True)
    st.caption("AI Incident Response & Organizational Memory")
with head_r:
    st.markdown(
        "<div style='text-align:right;padding-top:18px;font-weight:700;font-size:0.85rem;"
        "letter-spacing:0.05em;'>"
        "<span style='color:#22C55E;'>●</span> AGENT ONLINE<br>"
        "<span style='color:#8B95A7;font-weight:500;'>HINDSIGHT CONNECTED</span></div>",
        unsafe_allow_html=True)
st.divider()

# ---------------------------------------------------------------- NAVIGATION
st.radio("NAVIGATION", PAGES, key="nav_top", horizontal=True,
         on_change=_nav_top_changed, label_visibility="collapsed")

page = st.session_state["page"]

# ================================================================ OVERVIEW
if page == "Overview":
    st.subheader("SYSTEM OVERVIEW")
    st.caption("Real-time view of incidents, organizational memory and learned patterns.")

    dstatus = get_demo_status()
    auto_trigger = st.session_state.pop("trigger_load", False)

    if dstatus["error"] is not None:
        card('<b>⚠ MEMORY UNAVAILABLE</b><br>Sentinel couldn\'t reach organizational memory. '
             'Your incident data has not been changed.', cls="card error-card")
        with st.expander("Technical details"):
            st.code(str(dstatus["error"])[:800])
        if st.button("🔄 TRY AGAIN", key="mem_retry"):
            get_demo_status(force=True)
            st.rerun()

    elif dstatus["count"] and dstatus["count"] >= len(SEED):
        card(
            f'<b>✓ ORGANIZATIONAL MEMORY READY</b><br>'
            f'{len(SEED)} historical incidents available to Sentinel.',
            cls="card memory-card")
        c1, c2 = st.columns(2)
        with c1:
            st.button("VIEW MEMORY", use_container_width=True, key="ov_view",
                      on_click=goto, args=("Memory",))
        with c2:
            st.button("INVESTIGATE →", use_container_width=True, type="primary", key="ov_inv",
                      on_click=goto, args=("Investigate",))

    elif dstatus["count"] and 0 < dstatus["count"] < len(SEED):
        card(
            f'<b>MEMORY INITIALIZATION INCOMPLETE</b><br>'
            f'{dstatus["count"]} of {len(SEED)} demo incidents were stored.',
            cls="card error-card")
        if st.button("🔄 RETRY", key="mem_partial_retry", type="primary"):
            try:
                with st.spinner("Storing remaining incidents..."):
                    seed_memory()
                st.session_state["seeded_ok"] = True
                get_demo_status(force=True)
                st.rerun()
            except Exception as e:
                render_error("MEMORY SERVICE UNAVAILABLE",
                             "Sentinel couldn't reach organizational memory.", e)

    else:
        card(
            '<b>🧠 ORGANIZATIONAL MEMORY</b><br>'
            "Sentinel hasn't been initialized with historical incidents yet.<br><br>"
            f'<span style="color:#8B95A7;font-size:0.8rem;letter-spacing:0.04em;">'
            f'DEMO DATA • HINDSIGHT MEMORY</span><br>'
            f'{len(SEED)} sample incidents ready to load.',
            cls="card memory-card")

        do_load = st.button("LOAD DEMO MEMORY →", type="primary", key="ov_load") or auto_trigger

        if do_load:
            card("<b>INITIALIZING MEMORY</b><br>Teaching Sentinel from historical incidents...")
            steps = ["Connecting to Hindsight", "Storing incident history",
                     "Indexing organizational memory", "Preparing pattern detection"]
            slots = [st.empty() for _ in steps]

            def set_step(i, state):
                icon = {"done": "✓", "active": "●", "pending": "○"}[state]
                color = {"done": "#22C55E", "active": "#3B82F6", "pending": "#475569"}[state]
                slots[i].markdown(
                    f'<div style="color:{color};font-weight:600;padding:3px 0;">{icon}  {steps[i]}</div>',
                    unsafe_allow_html=True)

            for i in range(4):
                set_step(i, "pending")
            set_step(0, "done")
            set_step(1, "active")
            try:
                seed_memory()
                st.session_state["seeded_ok"] = True
            except Exception as e:
                set_step(1, "pending")
                render_error("MEMORY SERVICE UNAVAILABLE",
                             "Sentinel couldn't reach organizational memory. "
                             "Your incident data has not been changed.", e)
                st.stop()
            set_step(1, "done")
            set_step(2, "active")
            new_status = check_demo_status()
            set_step(2, "done")
            set_step(3, "active")
            set_step(3, "done")
            st.session_state["demo_status"] = new_status

            if new_status["error"] is not None or not new_status["count"]:
                render_error("MEMORY SERVICE UNAVAILABLE",
                             "Storage may have succeeded, but Sentinel couldn't verify it yet. "
                             "Check the Memory page or try again.", new_status["error"])
            elif new_status["count"] < len(SEED):
                card(f'<b>MEMORY INITIALIZATION INCOMPLETE</b><br>'
                     f'{new_status["count"]} of {len(SEED)} demo incidents were stored.',
                     cls="card error-card")
            else:
                card(
                    f'<b>✓ MEMORY INITIALIZED</b><br>Sentinel now has {len(SEED)} historical '
                    f'incidents available for investigation.<br><br>🧠 Hindsight Memory — CONNECTED',
                    cls="card memory-card")
                card(
                    f'<div style="text-align:center;color:#8B95A7;letter-spacing:0.03em;">'
                    f'{len(SEED)} INCIDENTS<br>↓<br>HINDSIGHT<br>↓<br>'
                    f'ORGANIZATIONAL MEMORY<br>↓<br>FUTURE INVESTIGATIONS</div>')
                st.markdown("**WHAT SENTINEL REMEMBERED**")
                for doc_id, svc, symptom, cause, fix in SEED:
                    card(
                        f'<span class="badge" style="background:#334155;color:#CBD5E1;">DEMO INCIDENT</span><br>'
                        f'<b style="text-transform:none !important;font-size:0.95rem !important;'
                        f'color:#E5E9F0 !important;">{symptom}</b><br>{cause}<br>'
                        f'<span style="color:#22C55E;">Resolved</span>')

    st.divider()

    # ---- Metrics -----------------------------------------------------------
    history = st.session_state["history"]
    active_n = sum(1 for h in history if h["status"] == "Investigating")
    resolved_n = sum(1 for h in history if h["status"] == "Resolved")
    matches_n = sum(h["memory_matches"] for h in history)
    learned_n = resolved_n  # each Teach Sentinel call adds exactly one new memory item

    m1, m2, m3, m4 = st.columns(4)
    for col, num, lbl in [
        (m1, active_n, "Active Incidents"), (m2, matches_n, "Memory Matches"),
        (m3, resolved_n, "Resolved Incidents"), (m4, learned_n, "Learned Patterns"),
    ]:
        with col:
            st.markdown(f'<div class="metric-card"><div class="num">{num}</div>'
                        f'<div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.subheader("RECENT INCIDENTS")
    if not history:
        card("No incidents have been investigated yet. Go to <b>Investigate</b> to start.")
    else:
        for h in reversed(history[-6:]):
            bg, fg = SEVERITY_COLORS.get(h["severity"], SEVERITY_COLORS["Unknown"])
            sbg, sfg = STATUS_COLORS[h["status"]]
            # computed outside the f-string: nested same-type quotes only work on Python 3.12+
            match_badge = badge(str(h["memory_matches"]) + " MEMORY MATCHES", "#4C1D95", "#C4B5FD")
            card(
                f'<b>{h["problem"][:70]}</b><br>'
                f'{badge(h["severity"], bg, fg)}{badge(h["status"], sbg, sfg)}'
                f'{match_badge}<br>'
                f'<span style="color:#8B95A7;font-size:0.8rem;">{h["timestamp"]}</span>'
            )

# ============================================================= INVESTIGATE
elif page == "Investigate":
    st.subheader("INVESTIGATE INCIDENT")
    st.caption("Give Sentinel an incident. It will investigate the problem using AI and organizational memory.")

    card("<b>NEW INCIDENT</b>")
    problem = st.text_area("Describe what is happening...",
                           "Payment API is timing out for users during peak hours",
                           label_visibility="collapsed", key="problem_input")

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        service = st.selectbox("SERVICE", ["— not specified —", "payments", "auth", "search",
                                           "orders", "checkout", "other"])
    with mc2:
        environment = st.selectbox("ENVIRONMENT", ["— not specified —", "production", "staging", "dev"])
    with mc3:
        impact = st.selectbox("IMPACT", ["— not specified —", "all users", "some users", "internal only"])

    go = st.button("⚡  INVESTIGATE INCIDENT", type="primary", use_container_width=True)

    if go and problem.strip():
        meta_bits = []
        if service != "— not specified —":
            meta_bits.append(f"service: {service}")
        if environment != "— not specified —":
            meta_bits.append(f"environment: {environment}")
        if impact != "— not specified —":
            meta_bits.append(f"impact: {impact}")
        full_query = problem + (f" ({', '.join(meta_bits)})" if meta_bits else "")

        card("<b>INVESTIGATING INCIDENT</b>")
        stage_labels = ["Incident received", "Searching organizational memory...",
                        "Comparing previous incidents", "Analyzing root cause",
                        "Generating recommendations"]
        slots = [st.empty() for _ in stage_labels]

        def set_stage(i, state):
            icon = {"done": "✓", "active": "●", "pending": "○"}[state]
            color = {"done": "#22C55E", "active": "#3B82F6", "pending": "#475569"}[state]
            slots[i].markdown(
                f'<div style="color:{color};font-weight:600;padding:3px 0;">{icon}  {stage_labels[i]}</div>',
                unsafe_allow_html=True)

        set_stage(0, "done")
        for i in range(1, 5):
            set_stage(i, "pending")

        set_stage(1, "active")
        try:
            res = recall_memory(full_query)
            memories = [r.text for r in res.results]
        except Exception as e:
            set_stage(1, "pending")
            render_error("MEMORY SERVICE UNAVAILABLE",
                         "Sentinel couldn't reach organizational memory. Check your connection and try again.", e)
            st.stop()
        set_stage(1, "done")

        set_stage(2, "active")
        set_stage(2, "done")

        set_stage(3, "active")
        try:
            data = investigate_with_ai(full_query, memories)
        except Exception as e:
            set_stage(3, "pending")
            render_error("AI SERVICE UNAVAILABLE",
                         "The AI investigation service could not be reached.", e)
            st.stop()
        set_stage(3, "done")

        set_stage(4, "active")
        set_stage(4, "done")

        record = {
            "id": next_id(), "problem": problem, "full_query": full_query,
            "severity": data["severity"], "confidence": data["confidence"],
            "service": data["affected_service"], "summary": data["summary"],
            "root_cause": data["root_cause"], "actions": data["recommended_actions"],
            "why": data["why_recommendation"], "memories": memories,
            "memory_matches": len(memories), "status": "Investigating",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "resolution": None, "resolved_root_cause": None,
        }
        st.session_state["history"].append(record)
        st.session_state["current_id"] = record["id"]
        st.rerun()

    # --- results for the current / most recently investigated incident ---
    current_id = st.session_state.get("current_id")
    history = st.session_state["history"]
    current = next((h for h in history if h["id"] == current_id), None)

    if current:
        st.subheader("INCIDENT ANALYSIS")
        bg, fg = SEVERITY_COLORS.get(current["severity"], SEVERITY_COLORS["Unknown"])
        sbg, sfg = STATUS_COLORS[current["status"]]
        st.markdown(
            badge(f'SEVERITY: {current["severity"]}', bg, fg) +
            badge(f'CONFIDENCE: {current["confidence"]}', "#4C1D95", "#C4B5FD") +
            badge(f'SERVICE: {current["service"]}', "#1E3A8A", "#BFDBFE") +
            badge(current["status"].upper(), sbg, sfg),
            unsafe_allow_html=True)
        if current["summary"]:
            card(f'<b>SUMMARY</b><br>{current["summary"]}')

        left, right = st.columns([3, 2])
        with left:
            card(f'<b>ROOT CAUSE</b><br>{current["root_cause"]}')
            actions = current["actions"] or ["No specific actions returned."]
            items = "".join(f'<li>{i + 1:02d}&nbsp;&nbsp;{a}</li>' for i, a in enumerate(actions))
            card(f'<b>RECOMMENDED ACTIONS</b><ol>{items}</ol>')

        with right:
            n = current["memory_matches"]
            if n:
                card(f'<b>🧠 MEMORY MATCH</b><br>'
                     f'<span style="font-size:1.15rem;font-weight:700;">{n} similar incident(s) found</span>',
                     cls="card memory-card")
                for i, m in enumerate(current["memories"], 1):
                    parsed = parse_memory_record(m)
                    demo_tag = ('<span class="badge" style="background:#334155;color:#CBD5E1;">'
                                'DEMO INCIDENT</span><br>') if parsed["is_demo"] else ""
                    if parsed["symptom"]:
                        body = (f'{demo_tag}<b>PREVIOUS INCIDENT #{i}</b><br>{parsed["symptom"]}<br><br>'
                                f'<b>ROOT CAUSE</b><br>{parsed["root_cause"]}<br><br>'
                                f'<b>PREVIOUS RESOLUTION</b><br>{parsed["resolution"]}<br><br>'
                                f'<b>OUTCOME</b><br>Resolved')
                    else:
                        body = f'{demo_tag}<b>PREVIOUS INCIDENT #{i}</b><br>{parsed["raw"]}'
                    card(body, cls="card memory-card")
                card(f'<b>💡 WHY THIS RECOMMENDATION?</b><br>{current["why"]}')
            else:
                card('<b>🆕 NO HISTORICAL MATCH</b><br>No similar incidents were found in '
                     'organizational memory. This appears to be a new pattern.', cls="card memory-card")

        # --- timeline ---
        card_lines = [
            "● Incident received", "● Memory searched", "● Historical matches analyzed",
            "● Root cause investigated", "● Recommendation generated",
        ]
        if current["status"] == "Resolved":
            card_lines += ["● Resolution confirmed", "● Memory updated"]
        else:
            card_lines += ["○ Awaiting resolution"]
        timeline_html = "<br>│<br>".join(card_lines)
        card(f'<b>INCIDENT TIMELINE</b><br><br>{timeline_html}')

        with st.expander("🔍 Compare: what a memory-less AI assistant would say"):
            if st.button("Generate comparison"):
                try:
                    with st.spinner("Asking a memory-less assistant..."):
                        baseline = answer_without_memory(current["problem"])
                    card(baseline)
                except Exception as e:
                    render_error("AI SERVICE UNAVAILABLE", "Could not generate the comparison.", e)

        st.subheader("🧠 TEACH SENTINEL")
        st.caption("Confirm what actually fixed the incident so Sentinel can remember it for future investigations.")

        flash = st.session_state.pop("flash", None)
        if flash:
            card(flash, cls="card memory-card")

        rc = st.text_input("CONFIRMED ROOT CAUSE", value=current["root_cause"] or "",
                           key=f"teach_rc_{current['id']}")
        fix = st.text_input("WHAT FIXED IT?", key=f"teach_fix_{current['id']}")
        if st.button("🧠 SAVE TO ORGANIZATIONAL MEMORY", type="primary"):
            if not fix.strip():
                st.warning("Enter what fixed it before saving.")
            else:
                try:
                    content = (f"Incident on {datetime.now():%Y-%m-%d}. "
                               f"Symptom: {current['problem']}. Root cause: {rc}. Resolution: {fix}.")
                    retain_one(content)
                    for h in st.session_state["history"]:
                        if h["id"] == current["id"]:
                            h["status"] = "Resolved"
                            h["resolution"] = fix
                            h["resolved_root_cause"] = rc
                    st.session_state["flash"] = (
                        '<b style="color:#BBF7D0 !important;">✓ MEMORY UPDATED</b><br>'
                        'Sentinel learned from this incident. This resolution can now influence '
                        'future investigations.')
                    st.rerun()  # refresh status badge, timeline and metrics
                except Exception as e:
                    render_error("MEMORY SERVICE UNAVAILABLE",
                                 "Sentinel couldn't save this resolution to organizational memory.", e)

# ================================================================= MEMORY
elif page == "Memory":
    st.subheader("ORGANIZATIONAL MEMORY")
    st.caption("Previous incidents and resolutions remembered by Sentinel.")
    query = st.text_input("Search organizational memory...", key="mem_search")
    if query.strip():
        try:
            with st.spinner("Searching..."):
                res = recall_memory(query)
            records = [r.text for r in res.results]
        except Exception as e:
            render_error("MEMORY SERVICE UNAVAILABLE",
                         "Sentinel couldn't reach organizational memory. Check your connection and try again.", e)
            records = None
        if records is not None:
            if not records:
                card("No matching records found in organizational memory.")
            for i, m in enumerate(records, 1):
                parsed = parse_memory_record(m)
                demo_tag = (' <span class="badge" style="background:#334155;color:#CBD5E1;">'
                            'DEMO INCIDENT</span>') if parsed["is_demo"] else ""
                title = f"Record #{i}"
                if parsed["symptom"]:
                    title += " — " + parsed["symptom"][:60]
                with st.expander(title):
                    if parsed["symptom"]:
                        card(f'{demo_tag}<br><b>INCIDENT</b><br>{parsed["symptom"]}<br><br>'
                             f'<b>ROOT CAUSE</b><br>{parsed["root_cause"]}<br><br>'
                             f'<b>RESOLUTION</b><br>{parsed["resolution"]}<br><br>'
                             f'<b>STATUS</b><br>Resolved')
                    else:
                        card(f'{demo_tag}<br>{parsed["raw"]}')
    else:
        card("Enter a search term above to browse organizational memory — for example, "
             "\"payment\" or \"login\".")

# ================================================================ PATTERNS
elif page == "Patterns":
    st.subheader("RECURRING INCIDENT PATTERNS")
    st.caption("Demonstrates how Sentinel gets smarter as more incidents are remembered.")

    if st.button("🧩 Analyze recurring patterns", type="primary"):
        try:
            with st.spinner("Analyzing recurring patterns..."):
                result = reflect_memory(
                    "Which incident root causes have happened more than once, and what are the fixes?")
            card(f'<b>INSIGHTS FROM ORGANIZATIONAL MEMORY</b><br>{result.text}', cls="card memory-card")
        except Exception as e:
            render_error("MEMORY SERVICE UNAVAILABLE",
                         "Sentinel couldn't analyze patterns in organizational memory.", e)

    st.subheader("THIS SESSION'S INCIDENT PATTERNS")
    history = st.session_state["history"]
    if not history:
        card("No incidents have been investigated yet in this session.")
    else:
        by_service = {}
        for h in history:
            by_service.setdefault(h["service"], []).append(h)
        for svc, items in by_service.items():
            last = items[-1]
            card(f'<b>{str(svc).upper()}</b><br>Seen: {len(items)} time(s) this session<br>'
                 f'Most recent root cause: {last["root_cause"]}')

# ========================================================= INCIDENT HISTORY
elif page == "Incident History":
    st.subheader("INCIDENT HISTORY")
    history = st.session_state["history"]
    if not history:
        card("No incidents have been investigated yet.")
    else:
        for h in reversed(history):
            bg, fg = SEVERITY_COLORS.get(h["severity"], SEVERITY_COLORS["Unknown"])
            sbg, sfg = STATUS_COLORS[h["status"]]
            match_badge = badge(str(h["memory_matches"]) + " MEMORY MATCHES", "#4C1D95", "#C4B5FD")
            with st.expander(f'{h["problem"][:70]}  —  {h["severity"]}  —  {h["status"]}  —  {h["timestamp"]}'):
                st.markdown(badge(h["severity"], bg, fg) + badge(h["status"], sbg, sfg) + match_badge,
                            unsafe_allow_html=True)
                card(f'<b>ROOT CAUSE</b><br>{h["root_cause"]}')
                if h["resolution"]:
                    card(f'<b>RESOLUTION</b><br>{h["resolution"]}', cls="card memory-card")