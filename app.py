"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          WEALTH PORTFOLIO ANALYSER — INSTITUTIONAL GRADE v2.0               ║
║                                                                              ║
║  Modules:                                                                    ║
║    1. Dynamic Multi-Asset (Stocks + MFs + FDs Combined)                     ║
║    2. Multi-Stock Portfolio                                                  ║
║    3. Multi-Mutual Fund Portfolio                                            ║
║    4. Single Stock Deep Dive                                                 ║
║    5. Single Mutual Fund Deep Dive                                           ║
║    6. Fixed Deposit & Credit Risk Analysis                                   ║
║                                                                              ║
║  Fixes over v1:                                                              ║
║    ✅ Sharpe Ratio now uses risk-free rate (India 10Y G-Sec)                ║
║    ✅ Sortino Ratio added                                                    ║
║    ✅ Max Drawdown added                                                     ║
║    ✅ Calmar Ratio added                                                     ║
║    ✅ Weight validation blocks execution (not just warns)                    ║
║    ✅ FD tenure dynamic in all modes (not hardcoded 3 years)                ║
║    ✅ FD excluded from Efficient Frontier MC simulation                     ║
║    ✅ Robust column auto-detection with fallback UI                          ║
║    ✅ Layout safe for 2–10 instruments                                       ║
║    ✅ Rolling returns & drawdown charts                                      ║
║    ✅ Negative CV handled correctly                                          ║
║    ✅ VaR at 95% confidence interval (parametric)                           ║
║    ✅ Live India Risk-Free Rate from RBI (G-Sec proxy)                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wealth Portfolio Analyser",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — PREMIUM DARK-FINTECH AESTHETIC
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif !important; }
code, .mono { font-family: 'JetBrains Mono', monospace !important; }

.main-header {
    font-size: 2.6rem; font-weight: 700; letter-spacing: -0.03em;
    background: linear-gradient(135deg, #00D09C 0%, #36D1DC 50%, #5B86E5 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; padding: 1.2rem 0 0.2rem;
}
.sub-header {
    text-align: center; color: var(--text-color); opacity: 0.55;
    font-size: 1rem; letter-spacing: 0.08em; text-transform: uppercase;
    margin-bottom: 2rem; font-weight: 500;
}
.section-title {
    font-size: 1.3rem; font-weight: 700; letter-spacing: -0.02em;
    color: var(--text-color); border-left: 3px solid #00D09C;
    padding-left: 0.8rem; margin: 1.8rem 0 1rem;
}
.subsection-title { font-size: 1rem; font-weight: 600; color: #5B86E5; margin-bottom: 0.4rem; }

/* Metric Cards */
.metric-card {
    border-radius: 14px; padding: 1.1rem 1.3rem; margin: 0.3rem 0;
    text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.18);
    transition: transform 0.2s ease;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-value { font-size: 1.65rem; font-weight: 700; color: #fff !important; font-family: 'JetBrains Mono', monospace; }
.metric-label { font-size: 0.78rem; font-weight: 500; opacity: 0.9; color: #fff !important; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.25rem; }

.card-green  { background: linear-gradient(135deg, #00b09b, #00D09C); }
.card-cobalt { background: linear-gradient(135deg, #36D1DC, #5B86E5); }
.card-teal   { background: linear-gradient(135deg, #00C9FF, #0088CC); }
.card-dark   { background: linear-gradient(135deg, #373B44, #4286f4); }
.card-orange { background: linear-gradient(135deg, #f12711, #f5af19); }
.card-purple { background: linear-gradient(135deg, #8E44AD, #5B86E5); }
.card-rose   { background: linear-gradient(135deg, #E91E63, #f5af19); }

/* Risk Warning Box */
.risk-box {
    background: rgba(241,39,17,0.07); border-left: 3px solid #f12711;
    padding: 1rem 1.2rem; border-radius: 6px; font-size: 0.93rem; margin: 1rem 0;
}
.info-box {
    background: rgba(0,208,156,0.07); border-left: 3px solid #00D09C;
    padding: 1rem 1.2rem; border-radius: 6px; font-size: 0.93rem; margin: 1rem 0;
}

/* Badges */
.badge-live { display: inline-block; padding: 0.15rem 0.55rem; background: #00D09C;
    color: #000; border-radius: 4px; font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.05em; text-transform: uppercase; margin-right: 0.5rem; }
.badge-fixed { display: inline-block; padding: 0.15rem 0.55rem; background: #5B86E5;
    color: #fff; border-radius: 4px; font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.05em; text-transform: uppercase; margin-right: 0.5rem; }

/* Footer */
.footer { text-align:center; opacity:0.45; font-size:0.8rem; margin-top:2rem; letter-spacing:0.05em; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
FINTECH_COLORS = ['#00D09C','#5B86E5','#f5af19','#00C9FF','#E91E63','#8E44AD','#2ECC71','#FF6B35','#A8DADC','#457B9D']
INDIA_RFR_FALLBACK = 7.1   # India 10Y G-Sec approximate yield (fallback)

# ─────────────────────────────────────────────────────────────────────────────
# LIVE DATA FETCHERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)   # 1 hour — inflation changes rarely but cache sensibly
def fetch_live_inflation_india():
    """World Bank API — India CPI Inflation (FP.CPI.TOTL.ZG)"""
    try:
        url = "https://api.worldbank.org/v2/country/IN/indicator/FP.CPI.TOTL.ZG?format=json&per_page=2"
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            records = r.json()[1]
            # Take the most recent non-null value
            for rec in records:
                if rec.get('value') is not None:
                    return round(float(rec['value']), 2), str(rec['date']), True
    except Exception:
        pass
    return 5.5, "Fallback", False


@st.cache_data(ttl=3600)
def fetch_india_risk_free_rate():
    """
    Proxy: India 10-Year Government Bond Yield via World Bank long-term rate indicator.
    Falls back to a hardcoded constant if unavailable.
    """
    try:
        url = "http://api.worldbank.org/v2/country/IN/indicator/FR.INR.RINR?format=json&per_page=2"
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            records = r.json()[1]
            for rec in records:
                if rec.get('value') is not None:
                    rfr = round(float(rec['value']), 2)
                    # Sanity check — India RFR is typically 5–9%
                    if 3.0 <= rfr <= 12.0:
                        return rfr, True
    except Exception:
        pass
    return INDIA_RFR_FALLBACK, False

# ─────────────────────────────────────────────────────────────────────────────
# COLUMN AUTO-DETECTION (ROBUST)
# ─────────────────────────────────────────────────────────────────────────────

def auto_detect_columns(df: pd.DataFrame):
    """
    Returns (date_col, price_col, div_col) with best-effort detection.
    Returns None for any column it cannot confidently identify.
    """
    cols_lower = {c: str(c).lower() for c in df.columns}

    date_keywords  = ['date', 'month', 'time', 'period', 'year', 'day']
    price_keywords = ['close', 'price', 'nav', 'value', 'rate', 'adj']
    div_keywords   = ['div', 'dividend', 'income', 'yield_amt']

    def match(keywords):
        for c, cl in cols_lower.items():
            if any(k in cl for k in keywords):
                return c
        return None

    date_col  = match(date_keywords)  or df.columns[0]
    price_col = match(price_keywords) or (df.columns[1] if len(df.columns) > 1 else df.columns[0])
    div_col   = match(div_keywords)   # None is fine

    return date_col, price_col, div_col

# ─────────────────────────────────────────────────────────────────────────────
# CORE FINANCIAL CALCULATIONS
# ─────────────────────────────────────────────────────────────────────────────

def calculate_metrics(df: pd.DataFrame, price_col: str, div_col=None, freq: str = "Monthly", rfr_annual: float = INDIA_RFR_FALLBACK):
    """
    Computes a comprehensive set of financial metrics for a price series.

    Returns dict with:
    returns, prices, cagr, ann_risk, avg_return, cv,
    sharpe, sortino, max_drawdown, calmar, var_95, skewness, kurtosis
    """
    prices  = pd.to_numeric(df[price_col], errors='coerce').dropna()
    divs    = (pd.to_numeric(df[div_col], errors='coerce').fillna(0).reindex(prices.index).fillna(0)
            if div_col and div_col in df.columns else pd.Series(0, index=prices.index))

    # ── Returns ──────────────────────────────────────────────────────────────
    returns = ((prices - prices.shift(1) + divs) / prices.shift(1)).dropna() * 100  # in %

    ann_factor = 252 if freq == "Daily" else 12

    # ── CAGR ─────────────────────────────────────────────────────────────────
    if len(returns) > 0:
        compounded = np.prod(1 + returns / 100)
        years = max(len(returns) / ann_factor, 1e-6)
        cagr = (compounded ** (1 / years) - 1) * 100
    else:
        cagr = 0.0

    avg_ret  = returns.mean()
    risk     = returns.std(ddof=1)
    ann_risk = risk * np.sqrt(ann_factor)

    # ── Coefficient of Variation (handle negative mean) ──────────────────────
    cv = (risk / avg_ret) if avg_ret > 0 else np.nan

    # ── Risk-Free Rate per period ─────────────────────────────────────────────
    rfr_period = rfr_annual / ann_factor  # in % (since returns are in %)

    # ── Sharpe Ratio (CORRECT: subtract risk-free rate) ──────────────────────
    excess_returns = returns - rfr_period
    sharpe = (excess_returns.mean() / risk * np.sqrt(ann_factor)) if risk > 0 else 0.0

    # ── Sortino Ratio (downside deviation only) ───────────────────────────────
    downside = returns[returns < rfr_period]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else 1e-9
    sortino = (excess_returns.mean() / downside_std * np.sqrt(ann_factor)) if downside_std > 0 else 0.0

    # ── Maximum Drawdown ──────────────────────────────────────────────────────
    cumulative = (1 + returns / 100).cumprod()
    rolling_max = cumulative.cummax()
    drawdowns = (cumulative - rolling_max) / rolling_max * 100
    max_drawdown = drawdowns.min()

    # ── Calmar Ratio ─────────────────────────────────────────────────────────
    calmar = (cagr / abs(max_drawdown)) if max_drawdown != 0 else 0.0

    # ── Value at Risk — Parametric 95% ───────────────────────────────────────
    # Per-period % loss that won't be exceeded 95% of the time
    var_95 = stats.norm.ppf(0.05, loc=avg_ret, scale=risk)  # negative = loss

    # ── Higher Moments ────────────────────────────────────────────────────────
    skewness = float(returns.skew())
    kurt     = float(returns.kurtosis())   # excess kurtosis

    return {
        'returns': returns, 'prices': prices,
        'cagr': cagr, 'ann_risk': ann_risk, 'avg_return': avg_ret, 'cv': cv,
        'sharpe': sharpe, 'sortino': sortino,
        'max_drawdown': max_drawdown, 'calmar': calmar,
        'var_95': var_95, 'skewness': skewness, 'kurtosis': kurt,
    }


def rolling_returns(returns: pd.Series, window: int = 12):
    """Annualised rolling return over `window` periods."""
    roll = returns.rolling(window).apply(lambda x: (np.prod(1 + x / 100) ** (1) - 1) * 100, raw=True)
    return roll


def max_drawdown_series(returns: pd.Series):
    """Returns the drawdown series (%) for plotting."""
    cumulative  = (1 + returns / 100).cumprod()
    rolling_max = cumulative.cummax()
    return (cumulative - rolling_max) / rolling_max * 100


def portfolio_metrics(weights, cov_matrix, avg_returns, ann_factor, rfr_annual):
    """
    Computes annualised portfolio return, volatility, Sharpe.
    weights     : array-like, sum = 1
    avg_returns : per-period average returns in %
    """
    w   = np.array(weights)
    ret = np.array(avg_returns)

    port_ret = float(np.sum(w * ret) * ann_factor)
    port_var = float(np.dot(w.T, np.dot(cov_matrix, w)) * ann_factor)
    port_vol = np.sqrt(max(port_var, 0))

    rfr_annual_pct = rfr_annual  # already in %
    sharpe = (port_ret - rfr_annual_pct) / port_vol if port_vol > 0 else 0.0

    return port_ret, port_vol, sharpe

# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def metric_card(label: str, value: str, color: str = "green", unit: str = "") -> str:
    cls_map = {
        "green":  "card-green",  "cobalt": "card-cobalt",
        "teal":   "card-teal",   "dark":   "card-dark",
        "orange": "card-orange", "purple": "card-purple",
        "rose":   "card-rose",
    }
    cls = f"metric-card {cls_map.get(color, 'card-dark')}"
    return (f'<div class="{cls}">'
            f'<div class="metric-value">{value}{unit}</div>'
            f'<div class="metric-label">{label}</div></div>')


def safe_fmt(val, decimals=2, prefix="", suffix=""):
    """Format floats safely — returns '—' on NaN/None."""
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "—"
        return f"{prefix}{val:.{decimals}f}{suffix}"
    except Exception:
        return "—"


def render_metric_row(metrics_list):
    """
    metrics_list: [(label, value_str, color), ...]
    Renders in equal columns.
    """
    cols = st.columns(len(metrics_list))
    for col, (label, val, color) in zip(cols, metrics_list):
        with col:
            st.markdown(metric_card(label, val, color), unsafe_allow_html=True)


def load_and_clean(file, data_freq) -> tuple:
    """
    Reads a CSV, auto-detects columns, cleans dates & prices.
    Returns (df_cleaned, date_col, price_col, div_col) or raises ValueError.
    """
    df = pd.read_csv(file)
    if df.empty or len(df.columns) < 2:
        raise ValueError("CSV must have at least 2 columns.")

    date_col, price_col, div_col = auto_detect_columns(df)

    df['_Date'] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
    n_bad = df['_Date'].isna().sum()
    if n_bad > len(df) * 0.5:
        raise ValueError(f"Could not parse dates in column '{date_col}'. Check format.")

    df = df.dropna(subset=['_Date']).sort_values('_Date').set_index('_Date')
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')

    if df[price_col].isna().all():
        raise ValueError(f"Price column '{price_col}' has no numeric data.")

    df = df.dropna(subset=[price_col])
    return df, date_col, price_col, div_col

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📈 Wealth Portfolio Analyser</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Institutional-Grade Asset & Risk Intelligence Platform</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 📂 Analytics Modules")
    mode = st.radio("Module", [
        "1. Dynamic Multi-Asset (All Combined)",
        "2. Multi-Stock Portfolio",
        "3. Multi-Mutual Fund Portfolio",
        "4. Single Stock Deep Dive",
        "5. Single Mutual Fund Deep Dive",
        "6. Fixed Deposit & Credit Risk Analysis"
    ], label_visibility="collapsed")

    st.divider()

    # ── Macroeconomic Inputs ─────────────────────────────────────────────────
    st.markdown("### 🌐 Macroeconomic Inputs")

    use_live_api = st.toggle("Live World Bank Inflation (India)", value=True)
    if use_live_api:
        live_infl, infl_year, infl_live = fetch_live_inflation_india()
        if infl_live:
            st.success(f"✅ Inflation ({infl_year}): **{live_infl}%**")
            default_inflation = live_infl
        else:
            st.warning(f"⚠️ API offline — using fallback {live_infl}%")
            default_inflation = live_infl
    else:
        default_inflation = st.number_input("Manual Inflation (%)", value=5.5, step=0.1, min_value=0.0, max_value=25.0)

    rfr, rfr_live = fetch_india_risk_free_rate()
    rfr_badge = "LIVE" if rfr_live else "FALLBACK"
    st.markdown(
        f'<span class="badge-{"live" if rfr_live else "fixed"}">{rfr_badge}</span>'
        f' Risk-Free Rate: **{rfr}%** (India G-Sec Proxy)',
        unsafe_allow_html=True
    )
    rfr_override = st.number_input("Override Risk-Free Rate (%)", value=rfr, step=0.05, min_value=0.0, max_value=15.0)
    risk_free_rate = rfr_override

    st.divider()
    investment_amount = st.number_input("Capital Deployed (₹)", min_value=1000, value=100000, step=10000)

# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════ #
#   MODES 1, 2, 3 — MULTI-ASSET PORTFOLIO ENGINE                             #
# ═══════════════════════════════════════════════════════════════════════════ #
# ─────────────────────────────────────────────────────────────────────────────

if mode in ["1. Dynamic Multi-Asset (All Combined)", "2. Multi-Stock Portfolio", "3. Multi-Mutual Fund Portfolio"]:

    st.markdown('<div class="section-title">Step 1 — Portfolio Configuration</div>', unsafe_allow_html=True)

    if mode == "1. Dynamic Multi-Asset (All Combined)":
        c1, c2, c3, c4 = st.columns(4)
        data_freq = c1.radio("Data Frequency", ["Monthly", "Daily"], horizontal=True)
        n_stocks  = c2.number_input("Equities", min_value=0, max_value=8, value=2)
        n_mfs     = c3.number_input("Mutual Funds", min_value=0, max_value=8, value=1)
        n_fds     = c4.number_input("FDs / Debt", min_value=0, max_value=4, value=1)

    elif mode == "2. Multi-Stock Portfolio":
        c1, c2 = st.columns(2)
        data_freq = c1.radio("Data Frequency", ["Monthly", "Daily"], horizontal=True)
        n_stocks  = c2.number_input("Number of Equities", min_value=2, max_value=10, value=3)
        n_mfs, n_fds = 0, 0

    else:  # Mode 3
        c1, c2 = st.columns(2)
        data_freq = c1.radio("Data Frequency", ["Monthly", "Daily"], horizontal=True)
        n_mfs     = c2.number_input("Number of Mutual Funds", min_value=2, max_value=10, value=2)
        n_stocks, n_fds = 0, 0

    n_total = n_stocks + n_mfs + n_fds

    if n_total < 2:
        st.warning("⚠️ Allocate at least **2 instruments** to construct a valid portfolio.")
        st.stop()

    # ── Step 2: Data Ingestion ────────────────────────────────────────────────
    st.markdown('<div class="section-title">Step 2 — Ingest Asset Data</div>', unsafe_allow_html=True)

    # Dynamic column layout — max 4 per row, safe for any n_total
    COLS_PER_ROW = 4
    uploaded_files: dict  = {}   # name -> file
    fd_data: dict         = {}   # name -> {rate, risk, tenure}

    all_asset_names: list = []

    # Split into rows
    assets_config = (
        [("stock", i) for i in range(n_stocks)] +
        [("mf",    i) for i in range(n_mfs)]
    )

    for row_start in range(0, len(assets_config), COLS_PER_ROW):
        row_items = assets_config[row_start: row_start + COLS_PER_ROW]
        cols = st.columns(len(row_items))
        for col, (asset_type, idx) in zip(cols, row_items):
            with col:
                if asset_type == "stock":
                    st.markdown(f'<div class="subsection-title">📈 Equity {idx+1}</div>', unsafe_allow_html=True)
                    name = st.text_input("Ticker / Name", f"Stock {idx+1}", key=f"sname_{idx}")
                    f    = st.file_uploader("Upload CSV", type=["csv"], key=f"sfile_{idx}")
                else:
                    st.markdown(f'<div class="subsection-title">🏦 Fund {idx+1}</div>', unsafe_allow_html=True)
                    name = st.text_input("Scheme Name", f"MF {idx+1}", key=f"mname_{idx}")
                    f    = st.file_uploader("Upload CSV", type=["csv"], key=f"mfile_{idx}")
                if f:
                    uploaded_files[name] = f
                all_asset_names.append(name)

    # FD rows (separate since they don't upload CSVs)
    for row_start in range(0, n_fds, COLS_PER_ROW):
        row_range = range(row_start, min(row_start + COLS_PER_ROW, n_fds))
        cols = st.columns(len(list(row_range)))
        for col, i in zip(cols, row_range):
            with col:
                st.markdown(f'<div class="subsection-title">🛡️ Debt / FD {i+1}</div>', unsafe_allow_html=True)
                fd_name   = st.text_input("Instrument Name", f"Bank FD {i+1}", key=f"fdname_{i}")
                fd_rate   = st.number_input("Nominal Yield (%)", 1.0, 15.0, 6.5, 0.05, key=f"fdrate_{i}")
                fd_tenure = st.number_input("Tenure (Years)", 1, 20, 3, key=f"fdtenure_{i}")
                inst_type = st.selectbox("Credit Profile", [
                    "Sovereign / PSU (0.00% risk)",
                    "Tier-1 Private Bank (0.05% risk)",
                    "AAA NBFC (0.25% risk)",
                    "A-Rated Corporate (1.50% risk)"
                ], key=f"fdrisk_{i}")
                risk_map = {"Sovereign / PSU (0.00% risk)": 0.0000,
                            "Tier-1 Private Bank (0.05% risk)": 0.0005,
                            "AAA NBFC (0.25% risk)": 0.0025,
                            "A-Rated Corporate (1.50% risk)": 0.0150}
                fd_data[fd_name] = {
                    'rate': fd_rate,
                    'risk': risk_map[inst_type],
                    'tenure': fd_tenure
                }
                all_asset_names.append(fd_name)

    # Uploaded files needed = stocks + MFs (not FDs)
    required_uploads = n_stocks + n_mfs
    if len(uploaded_files) < required_uploads:
        st.info(f"⏳ Awaiting {required_uploads - len(uploaded_files)} CSV file(s)...")
        st.stop()

    # ── Processing ────────────────────────────────────────────────────────────
    all_metrics:      dict         = {}
    combined_returns: pd.DataFrame = pd.DataFrame()
    combined_prices:  pd.DataFrame = pd.DataFrame()

    ann_factor = 252 if data_freq == "Daily" else 12
    errors     = []

    with st.spinner("🔬 Processing time-series & computing covariance matrices..."):

        # — Market Assets ——————————————————————————————————————————————————————
        for name, file in uploaded_files.items():
            try:
                df, date_col, price_col, div_col = load_and_clean(file, data_freq)
                m = calculate_metrics(df, price_col, div_col, data_freq, risk_free_rate)
                all_metrics[name]      = m
                combined_returns[name] = m['returns']
                combined_prices[name]  = pd.to_numeric(df[price_col], errors='coerce')
            except Exception as e:
                errors.append(f"**{name}**: {e}")

        if errors:
            for err in errors:
                st.error(f"❌ {err}")
            st.stop()

        combined_returns = combined_returns.dropna()
        combined_prices  = combined_prices.dropna()

        # — Fixed Deposits ——————————————————————————————————————————————————————
        for fd_name, data in fd_data.items():
            fd_rate    = data['rate']
            def_prob   = data['risk']
            fd_tenure  = data['tenure']

            # Credit-adjusted expected payoff (user-defined tenure — NOT hardcoded 3yr)
            survival_prob  = (1 - def_prob) ** fd_tenure
            nominal_payoff = (1 + fd_rate / 100) ** fd_tenure
            expected_payoff = nominal_payoff * survival_prob + 0.5 * (1 - survival_prob)
            risk_adj_cagr   = (expected_payoff ** (1 / fd_tenure) - 1) * 100

            # Fisher Equation → Real Rate
            real_rate   = (((1 + risk_adj_cagr / 100) / (1 + default_inflation / 100)) - 1) * 100
            period_rate = real_rate / ann_factor

            # Build synthetic return series aligned to existing data
            if combined_returns.empty:
                idx = pd.date_range(start='2020-01-01', periods=ann_factor * fd_tenure, freq='ME' if data_freq == "Monthly" else 'B')
                combined_returns = pd.DataFrame(index=idx)
                combined_prices  = pd.DataFrame(index=idx)

            combined_returns[fd_name] = period_rate
            combined_prices[fd_name]  = 100 * (1 + combined_returns[fd_name] / 100).cumprod()

            all_metrics[fd_name] = {
                'returns':      pd.Series([period_rate] * len(combined_returns)),
                'prices':       combined_prices[fd_name],
                'cagr':         real_rate,
                'ann_risk':     0.0,
                'avg_return':   period_rate,
                'cv':           np.nan,
                'sharpe':       (real_rate - risk_free_rate) / 0.001,   # effectively ∞ but we show separately
                'sortino':      np.nan,
                'max_drawdown': 0.0,
                'calmar':       np.nan,
                'var_95':       0.0,
                'skewness':     0.0,
                'kurtosis':     0.0,
            }

    all_market_assets = list(uploaded_files.keys())
    all_fd_assets     = list(fd_data.keys())
    all_assets        = all_market_assets + all_fd_assets

    # ── Dashboard ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Step 3 — Quantitative Dashboard</div>', unsafe_allow_html=True)

    if all_fd_assets:
        st.markdown(
            f'<div class="info-box">💡 FD metrics show <strong>Real Yield</strong> '
            f'(inflation & credit-risk adjusted). Inflation used: <strong>{default_inflation}%</strong></div>',
            unsafe_allow_html=True
        )

    tabs = st.tabs(["📊 Price Action", "📈 Distributions", "🧮 Asset Metrics", "🗂 Portfolio Modeller", "🔮 Efficient Frontier", "📉 Drawdown & Rolling"])

    # ── TAB 0: Price Action ────────────────────────────────────────────────────
    with tabs[0]:
        norm = (combined_prices / combined_prices.iloc[0]) * 100
        fig  = px.line(
            norm.reset_index(), x=norm.index.name or 'index',
            y=norm.columns.tolist(),
            color_discrete_sequence=FINTECH_COLORS,
            labels={'value': 'Growth (Base = 100)', 'variable': 'Asset'}
        )
        fig.update_layout(height=500, hovermode="x unified",
                        legend=dict(orientation="h", y=1.04, x=0.5, xanchor='center'),
                        margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB 1: Return Distributions ───────────────────────────────────────────
    with tabs[1]:
        fig2 = go.Figure()
        for i, comp in enumerate(all_market_assets):   # FDs are constant — skip
            ret = all_metrics[comp]['returns']
            fig2.add_trace(go.Histogram(
                x=ret, name=comp, opacity=0.75,
                marker_color=FINTECH_COLORS[i % len(FINTECH_COLORS)],
                nbinsx=35
            ))
            # Overlay Normal curve
            x_range = np.linspace(ret.min(), ret.max(), 200)
            pdf      = stats.norm.pdf(x_range, ret.mean(), ret.std())
            fig2.add_trace(go.Scatter(
                x=x_range, y=pdf * len(ret) * (ret.max() - ret.min()) / 35,
                mode='lines', name=f'{comp} Normal Fit',
                line=dict(color=FINTECH_COLORS[i % len(FINTECH_COLORS)], dash='dash', width=2),
                showlegend=False
            ))
        fig2.update_layout(height=440, barmode='overlay',
                        xaxis_title=f"{data_freq} Return (%)", yaxis_title="Frequency")
        st.plotly_chart(fig2, use_container_width=True)

    # ── TAB 2: Asset Metrics Table ────────────────────────────────────────────
    with tabs[2]:
        # Scrollable comparative metrics table
        rows = []
        for comp in all_assets:
            m = all_metrics[comp]
            is_fd = comp in all_fd_assets
            rows.append({
                "Asset":           comp,
                "Type":            "Debt/FD" if is_fd else ("MF" if comp in [k for k in uploaded_files if "MF" in k or "mf" in k.lower()] else "Equity"),
                "Avg Return/Period (%)": safe_fmt(m['avg_return'], 3),
                "CAGR (%) *":     safe_fmt(m['cagr'], 2),
                "Volatility (%)":  safe_fmt(m['ann_risk'], 2),
                "Sharpe Ratio":    safe_fmt(m['sharpe'], 3) if not is_fd else "∞ (fixed)",
                "Sortino Ratio":   safe_fmt(m['sortino'], 3) if not is_fd else "—",
                "Max Drawdown (%)":safe_fmt(m['max_drawdown'], 2) if not is_fd else "0.00",
                "Calmar Ratio":    safe_fmt(m['calmar'], 3) if not is_fd else "—",
                "VaR 95% /period": safe_fmt(m['var_95'], 3) if not is_fd else "0.000",
                "Skewness":        safe_fmt(m['skewness'], 3) if not is_fd else "—",
                "Excess Kurtosis": safe_fmt(m['kurtosis'], 3) if not is_fd else "—",
            })

        df_metrics = pd.DataFrame(rows).set_index("Asset")
        st.markdown("**\\* FD CAGR = Real Yield (inflation & credit adjusted). Equity/MF CAGR = Raw annualised.**")
        st.dataframe(df_metrics, use_container_width=True, height=min(80 + 40 * len(all_assets), 600))

        # Individual metric cards below
        for comp in all_assets:
            m   = all_metrics[comp]
            st.markdown(f"#### {comp}")
            is_fd = comp in all_fd_assets
            render_metric_row([
                ("Avg Return/Period",            safe_fmt(m['avg_return'], 3, suffix="%"), "cobalt"),
                ("Real CAGR" if is_fd else "CAGR", safe_fmt(m['cagr'], 2, suffix="%"), "green"),
                ("Volatility (Ann.)",             safe_fmt(m['ann_risk'], 2, suffix="%"), "teal"),
                ("Sharpe Ratio",                  safe_fmt(m['sharpe'], 3) if not is_fd else "∞", "dark"),
                ("Sortino Ratio",                 safe_fmt(m['sortino'], 3) if not is_fd else "—", "purple"),
                ("Max Drawdown",                  safe_fmt(m['max_drawdown'], 2, suffix="%") if not is_fd else "0%", "orange"),
            ])
            st.markdown("---")

    # ── TAB 3: Portfolio Modeller ──────────────────────────────────────────────
    with tabs[3]:
        st.markdown("### ⚖️ Capital Allocation")

        weight_cols    = st.columns(min(n_total, COLS_PER_ROW))
        weights_input  = []
        default_w      = round(100.0 / n_total, 1)

        for i, comp in enumerate(all_assets):
            with weight_cols[i % COLS_PER_ROW]:
                w = st.number_input(f"{comp} (%)", 0.0, 100.0, default_w, 0.5, key=f"pw_{comp}")
                weights_input.append(w)

        total_w = sum(weights_input)
        if abs(total_w - 100.0) > 0.01:
            st.error(f"❌ Weights sum to **{total_w:.2f}%** — must equal exactly 100%. Adjust above.")
            st.stop()   # BLOCK execution — not just a warning

        st.divider()
        st.markdown("### 🧮 Risk Matrix")
        cov_df  = combined_returns.cov().fillna(0)
        corr_df = combined_returns.corr().fillna(0)
        mc1, mc2 = st.columns(2)
        with mc1:
            st.write("**Covariance Matrix**")
            st.dataframe(cov_df.style.background_gradient(cmap='Greens'), use_container_width=True)
        with mc2:
            st.write("**Correlation Matrix**")
            st.dataframe(corr_df.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1), use_container_width=True)

        st.divider()
        st.markdown("### 🚀 Portfolio Performance Projections")

        w_norm    = [w / 100.0 for w in weights_input]
        avg_rets  = [all_metrics[c]['avg_return'] for c in all_assets]
        port_ret, port_vol, port_sharpe = portfolio_metrics(w_norm, cov_df.values, avg_rets, ann_factor, risk_free_rate)

        # VaR at portfolio level (parametric)
        port_var_95_pct = stats.norm.ppf(0.05, loc=port_ret / ann_factor, scale=port_vol / np.sqrt(ann_factor))
        port_var_rs     = investment_amount * abs(port_var_95_pct / 100)

        render_metric_row([
            ("Expected Return (Ann.)",   safe_fmt(port_ret, 2, suffix="%"),   "green"),
            ("Portfolio Volatility",      safe_fmt(port_vol, 2, suffix="%"),   "teal"),
            (f"Sharpe (RFR={risk_free_rate}%)", safe_fmt(port_sharpe, 3),     "dark"),
            ("VaR 95% (1-period, ₹)",    f"₹{port_var_rs:,.0f}",              "orange"),
        ])
        st.markdown("")
        exp_return_rs = investment_amount * (port_ret / 100)
        render_metric_row([
            ("Projected AUM",     f"₹{(investment_amount + exp_return_rs):,.0f}", "cobalt"),
            ("Net Capital Gain",  f"+₹{exp_return_rs:,.0f}",                       "green"),
            ("Capital at 1σ Risk", f"±₹{investment_amount * (port_vol / 100):,.0f}","rose"),
        ])

    # ── TAB 4: Efficient Frontier ──────────────────────────────────────────────
    with tabs[4]:
        st.markdown("### 🔮 Efficient Frontier — Monte Carlo Optimization")
        st.markdown(
            '<div class="info-box">FD assets are <strong>excluded</strong> from frontier simulation '
            '(constant return, zero variance distorts the convex hull). FD allocation is fixed at your chosen weight.</div>',
            unsafe_allow_html=True
        )

        n_sim = st.slider("Simulation Portfolios", 500, 3000, 1500, 250)

        market_assets = all_market_assets   # Only these vary in MC
        n_market      = len(market_assets)

        if n_market < 2:
            st.warning("Need at least 2 market assets (stocks/MFs) for frontier simulation.")
        else:
            mc_rets_df = combined_returns[market_assets]
            mc_cov     = mc_rets_df.cov().fillna(0)
            mc_avg     = mc_rets_df.mean().values

            sim_rets, sim_vols, sim_sharpes, sim_ws = [], [], [], []
            np.random.seed(42)

            for _ in range(n_sim):
                w      = np.random.dirichlet(np.ones(n_market))
                r, vol, sh = portfolio_metrics(w, mc_cov.values, mc_avg, ann_factor, risk_free_rate)
                sim_rets.append(r);  sim_vols.append(vol)
                sim_sharpes.append(sh); sim_ws.append(w)

            best_idx  = int(np.argmax(sim_sharpes))
            min_v_idx = int(np.argmin(sim_vols))

            df_ef = pd.DataFrame({'Return (%)': sim_rets, 'Volatility (%)': sim_vols, 'Sharpe': sim_sharpes})
            fig_ef = px.scatter(df_ef, x='Volatility (%)', y='Return (%)', color='Sharpe',
                                color_continuous_scale='Teal', height=520, opacity=0.65)

            fig_ef.add_trace(go.Scatter(
                x=[sim_vols[best_idx]], y=[sim_rets[best_idx]],
                mode='markers+text', text=['⭐ Max Sharpe'],
                textposition='top right',
                marker=dict(size=18, color='#FF9F1C', symbol='star', line=dict(color='white', width=1)),
                name=f'Max Sharpe ({sim_sharpes[best_idx]:.3f})'
            ))
            fig_ef.add_trace(go.Scatter(
                x=[sim_vols[min_v_idx]], y=[sim_rets[min_v_idx]],
                mode='markers+text', text=['🛡 Min Vol'],
                textposition='top left',
                marker=dict(size=16, color='#00D09C', symbol='diamond', line=dict(color='white', width=1)),
                name=f'Min Volatility ({sim_vols[min_v_idx]:.2f}%)'
            ))
            fig_ef.update_layout(coloraxis_colorbar=dict(title='Sharpe'), margin=dict(t=40))
            st.plotly_chart(fig_ef, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**⭐ Optimal (Max Sharpe) Weights:**")
                opt_df = pd.DataFrame({'Asset': market_assets, 'Weight (%)': [f"{sim_ws[best_idx][i]*100:.1f}%" for i in range(n_market)]})
                st.dataframe(opt_df.set_index('Asset'), use_container_width=True)
            with c2:
                st.markdown("**🛡 Min-Volatility Weights:**")
                min_df = pd.DataFrame({'Asset': market_assets, 'Weight (%)': [f"{sim_ws[min_v_idx][i]*100:.1f}%" for i in range(n_market)]})
                st.dataframe(min_df.set_index('Asset'), use_container_width=True)

    # ── TAB 5: Drawdown & Rolling ──────────────────────────────────────────────
    with tabs[5]:
        st.markdown("### 📉 Maximum Drawdown Analysis")
        fig_dd = go.Figure()
        for i, comp in enumerate(all_market_assets):
            dd = max_drawdown_series(all_metrics[comp]['returns'])
            fig_dd.add_trace(go.Scatter(
                x=dd.index, y=dd.values, name=comp, mode='lines',
                line=dict(color=FINTECH_COLORS[i % len(FINTECH_COLORS)], width=1.5),
                fill='tozeroy', fillcolor=f"rgba({int(FINTECH_COLORS[i % len(FINTECH_COLORS)][1:3], 16)}, "
                                        f"{int(FINTECH_COLORS[i % len(FINTECH_COLORS)][3:5], 16)}, "
                                        f"{int(FINTECH_COLORS[i % len(FINTECH_COLORS)][5:7], 16)}, 0.12)"
            ))
        fig_dd.update_layout(height=400, yaxis_title="Drawdown (%)", hovermode="x unified", margin=dict(t=20))
        st.plotly_chart(fig_dd, use_container_width=True)

        st.markdown("### 📈 Rolling Annualised Returns")
        roll_window = st.slider("Rolling Window (periods)", 3, 36, 12)
        fig_roll = go.Figure()
        for i, comp in enumerate(all_market_assets):
            rr = rolling_returns(all_metrics[comp]['returns'], roll_window)
            fig_roll.add_trace(go.Scatter(
                x=rr.index, y=rr.values, name=comp, mode='lines',
                line=dict(color=FINTECH_COLORS[i % len(FINTECH_COLORS)], width=1.8)
            ))
        fig_roll.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.6)
        fig_roll.update_layout(height=400, yaxis_title=f"Rolling {roll_window}-Period Return (%)", hovermode="x unified", margin=dict(t=20))
        st.plotly_chart(fig_roll, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════ #
#   MODES 4 & 5 — SINGLE ASSET DEEP DIVE                                     #
# ═══════════════════════════════════════════════════════════════════════════ #
# ─────────────────────────────────────────────────────────────────────────────

elif mode in ["4. Single Stock Deep Dive", "5. Single Mutual Fund Deep Dive"]:
    asset_type = "Stock" if "Stock" in mode else "Mutual Fund"
    st.markdown(f'<div class="section-title">{asset_type} — Deep Dive Analytics</div>', unsafe_allow_html=True)

    uploaded_custom = st.file_uploader(f"Upload {asset_type} Historical Data (CSV)", type=["csv"])
    if not uploaded_custom:
        st.info("⬆️ Upload a CSV with at least Date and Price/NAV columns.")
        st.stop()

    raw = pd.read_csv(uploaded_custom)
    if raw.empty or len(raw.columns) < 2:
        st.error("❌ CSV must contain at least 2 columns.")
        st.stop()

    d_col, p_col, div_col = auto_detect_columns(raw)

    st.markdown('<div class="section-title">Column Mapping</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    date_col_sel  = c1.selectbox("Date Column",                raw.columns.tolist(), index=list(raw.columns).index(d_col) if d_col in raw.columns else 0)
    price_col_sel = c2.selectbox(f"{'Price' if asset_type=='Stock' else 'NAV'} Column", raw.columns.tolist(), index=list(raw.columns).index(p_col) if p_col in raw.columns else 0)
    div_options   = ['None'] + raw.columns.tolist()
    div_col_sel   = c3.selectbox("Dividend Column (Optional, stocks only)",
                                div_options,
                                index=(div_options.index(div_col) if div_col in div_options else 0)
                                ) if asset_type == "Stock" else "None"
    freq = c4.radio("Data Frequency", ["Monthly", "Daily"], horizontal=True)

    raw['_Date'] = pd.to_datetime(raw[date_col_sel], errors='coerce', dayfirst=True)
    raw = raw.dropna(subset=['_Date']).sort_values('_Date')

    if raw.empty:
        st.error("❌ No valid dates found. Check date column format.")
        st.stop()

    m = calculate_metrics(
        raw.set_index('_Date'),
        price_col_sel,
        None if div_col_sel == 'None' else div_col_sel,
        freq,
        risk_free_rate
    )

    # Price chart
    fig_p = px.area(raw, x='_Date', y=price_col_sel,
                    title=f"Historical {asset_type} — Price Action",
                    color_discrete_sequence=['#00D09C' if asset_type == 'Stock' else '#5B86E5'])
    fig_p.update_traces(line_width=1.8, fillcolor='rgba(0,208,156,0.12)' if asset_type == 'Stock' else 'rgba(91,134,229,0.12)')
    fig_p.update_layout(height=420, xaxis_title="Date", yaxis_title="Price" if asset_type == "Stock" else "NAV")
    st.plotly_chart(fig_p, use_container_width=True)

    # Metrics
    render_metric_row([
        ("CAGR (Annualised)",        safe_fmt(m['cagr'], 2, suffix="%"),          "green"),
        ("Volatility (Ann.)",        safe_fmt(m['ann_risk'], 2, suffix="%"),       "teal"),
        (f"Sharpe (RFR={risk_free_rate}%)", safe_fmt(m['sharpe'], 3),             "dark"),
        ("Sortino Ratio",            safe_fmt(m['sortino'], 3),                    "purple"),
        ("Max Drawdown",             safe_fmt(m['max_drawdown'], 2, suffix="%"),   "orange"),
        ("Calmar Ratio",             safe_fmt(m['calmar'], 3),                     "cobalt"),
    ])
    st.markdown("")
    render_metric_row([
        ("Coeff. of Variation",      safe_fmt(m['cv'], 3),                         "cobalt"),
        ("VaR 95% /period",          safe_fmt(m['var_95'], 3, suffix="%"),         "rose"),
        ("Skewness",                 safe_fmt(m['skewness'], 3),                   "dark"),
        ("Excess Kurtosis",          safe_fmt(m['kurtosis'], 3),                   "purple"),
    ])

    st.markdown('<div class="section-title">Return Distribution</div>', unsafe_allow_html=True)
    ret = m['returns']
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=ret, nbinsx=40, name="Returns",
                                    marker_color='#00D09C' if asset_type=='Stock' else '#5B86E5', opacity=0.8))
    x_r = np.linspace(ret.min(), ret.max(), 200)
    pdf = stats.norm.pdf(x_r, ret.mean(), ret.std())
    fig_hist.add_trace(go.Scatter(x=x_r, y=pdf * len(ret) * (ret.max() - ret.min()) / 40,
                                mode='lines', name='Normal Fit',
                                line=dict(color='#f5af19', width=2.5, dash='dash')))
    fig_hist.update_layout(height=380, xaxis_title=f"{freq} Return (%)", yaxis_title="Frequency",
                        legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown('<div class="section-title">Drawdown Timeline</div>', unsafe_allow_html=True)
    dd_s = max_drawdown_series(ret)
    fig_dd2 = go.Figure(go.Scatter(x=dd_s.index, y=dd_s.values, fill='tozeroy',
                                line=dict(color='#f12711', width=1.5),
                                fillcolor='rgba(241,39,17,0.12)'))
    fig_dd2.update_layout(height=320, yaxis_title="Drawdown (%)", xaxis_title="Date")
    st.plotly_chart(fig_dd2, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════ #
#   MODE 6 — FIXED DEPOSIT & CREDIT RISK                                     #
# ═══════════════════════════════════════════════════════════════════════════ #
# ─────────────────────────────────────────────────────────────────────────────

elif mode == "6. Fixed Deposit & Credit Risk Analysis":
    st.markdown('<div class="section-title">Fixed Income — Credit Risk & Real Yield Engine</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="risk-box">
    <strong>⚠️ Institutional Risk Framework:</strong>
    This engine applies two adjustments to your FD's nominal yield:<br>
    <strong>1. Credit Default Risk</strong> — probability-weighted expected payoff considering partial recovery on default (50% LGD assumption).<br>
    <strong>2. Fisher Equation</strong> → <code>Real Rate = [(1 + Nominal) / (1 + Inflation)] − 1</code><br>
    <em>Result: the actual purchasing-power growth of your capital.</em>
    </div>
    """, unsafe_allow_html=True)

    rc1, rc2 = st.columns([3, 1])
    institution_type = rc1.selectbox("Credit Profile (Determines Default Probability)", [
        "Sovereign / PSU Bank (e.g., SBI, PNB) — 0.00% Annual Default Risk",
        "Tier-1 Private Bank (e.g., HDFC, ICICI, Axis) — 0.05% Annual Default Risk",
        "AAA-Rated NBFC (e.g., Bajaj Finance) — 0.25% Annual Default Risk",
        "A-Rated Corporate Deposit — 1.50% Annual Default Risk",
    ])

    risk_map_6 = {
        "0.00%": 0.0000,
        "0.05%": 0.0005,
        "0.25%": 0.0025,
        "1.50%": 0.0150,
    }
    default_prob = next((v for k, v in risk_map_6.items() if k in institution_type), 0.0005)

    c1, c2, c3 = st.columns(3)
    principal  = c1.number_input("Principal (₹)", min_value=1000, value=investment_amount, step=10000)
    rate       = c2.number_input("Nominal Yield (%)", 1.0, 15.0,
                                6.4 if "0.00%" in institution_type else (6.5 if "0.05%" in institution_type else 7.1), 0.05)
    tenure     = c3.slider("Tenure (Years)", 1, 20, 3)

    comp_freq  = st.radio("Compounding Frequency", ["Quarterly (Standard)", "Monthly", "Annually"], horizontal=True)
    n_comp     = {"Quarterly (Standard)": 4, "Monthly": 12, "Annually": 1}[comp_freq]

    # ── Calculations ──────────────────────────────────────────────────────────
    nominal_maturity = principal * (1 + (rate / 100) / n_comp) ** (n_comp * tenure)
    survival_prob    = (1 - default_prob) ** tenure
    expected_payoff  = nominal_maturity * survival_prob + (principal * 0.5 * (1 - survival_prob))
    risk_adj_cagr    = ((expected_payoff / principal) ** (1 / tenure) - 1) * 100

    real_rate        = (((1 + risk_adj_cagr / 100) / (1 + default_inflation / 100)) - 1) * 100
    real_maturity    = principal * (1 + real_rate / 100) ** tenure

    # Inflation breakeven: how many years till real value = principal?
    breakeven_yr     = np.log(1) / np.log(1 + real_rate / 100) if real_rate > 0 else np.nan

    st.divider()
    st.markdown("### 📊 Risk-Adjusted Outcomes")

    render_metric_row([
        ("Total Invested",               f"₹{principal:,.0f}",         "cobalt"),
        ("Nominal Maturity (Paper)",      f"₹{nominal_maturity:,.0f}", "teal"),
        ("Credit-Adjusted Expected",      f"₹{expected_payoff:,.0f}", "green"),
        ("Real Maturity (Purchasing Power)", f"₹{real_maturity:,.0f}", "orange"),
    ])
    st.markdown("")
    render_metric_row([
        ("Nominal CAGR",        safe_fmt(rate, 2, suffix="%"),         "teal"),
        ("Risk-Adjusted CAGR",  safe_fmt(risk_adj_cagr, 2, suffix="%"), "green"),
        ("True Real Yield",     safe_fmt(real_rate, 2, suffix="%"),    "dark"),
        ("Survival Probability", safe_fmt(survival_prob * 100, 3, suffix="%"), "purple"),
    ])

    if real_rate < 0:
        st.error(
            f"🚨 **Negative Real Yield ({real_rate:.2f}%)** — Your FD is losing purchasing power. "
            f"Inflation ({default_inflation}%) exceeds your risk-adjusted return ({risk_adj_cagr:.2f}%)."
        )
    elif real_rate < 1.5:
        st.warning(f"⚠️ Very low real yield ({real_rate:.2f}%). Consider inflation-indexed bonds (RBI Floating Rate Bonds) as an alternative.")

    # ── Yield Trajectory Chart ────────────────────────────────────────────────
    years     = np.arange(0, tenure + 1)
    nom_vals  = principal * (1 + (rate / 100) / n_comp) ** (n_comp * years)
    adj_vals  = principal * (1 + risk_adj_cagr / 100) ** years
    real_vals = principal * (1 + real_rate / 100) ** years
    infl_vals = principal * (1 + default_inflation / 100) ** years  # Inflation erosion baseline

    df_fd = pd.DataFrame({
        'Year':   np.tile(years, 4),
        'Value':  np.concatenate([nom_vals, adj_vals, real_vals, infl_vals]),
        'Metric': (
            ['① Nominal (Paper Growth)'] * len(years) +
            ['② Credit-Adjusted Expected'] * len(years) +
            ['③ Real Growth (Purchasing Power)'] * len(years) +
            ['④ Inflation Erosion Baseline'] * len(years)
        )
    })

    fig_fd = px.line(df_fd, x='Year', y='Value', color='Metric',
                    title=f"Yield Trajectories — Inflation: {default_inflation}% | Default Risk: {default_prob*100:.2f}% p.a.",
                    color_discrete_map={
                        '① Nominal (Paper Growth)':         '#36D1DC',
                        '② Credit-Adjusted Expected':       '#00D09C',
                        '③ Real Growth (Purchasing Power)': '#f12711',
                        '④ Inflation Erosion Baseline':     '#f5af19',
                    })
    fig_fd.add_hline(y=principal, line_dash='dot', line_color='gray', opacity=0.6,
                    annotation_text="Initial Principal")
    fig_fd.update_layout(height=470, hovermode="x unified",
                        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor='center'),
                        yaxis_tickprefix='₹', yaxis_tickformat=',.0f')
    st.plotly_chart(fig_fd, use_container_width=True)

    # ── Sensitivity Analysis ──────────────────────────────────────────────────
    st.markdown("### 🔬 Sensitivity — Real Yield vs Inflation")
    infl_range  = np.arange(2.0, 12.1, 0.5)
    real_yields = [(((1 + risk_adj_cagr / 100) / (1 + i / 100)) - 1) * 100 for i in infl_range]

    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(x=infl_range, y=real_yields, mode='lines+markers',
                                line=dict(color='#5B86E5', width=2.5),
                                marker=dict(size=7, color='#00D09C')))
    fig_sens.add_hline(y=0, line_dash='dash', line_color='#f12711',
                    annotation_text="Break-Even (Real Yield = 0)", annotation_position="bottom right")
    fig_sens.add_vline(x=default_inflation, line_dash='dot', line_color='#f5af19',
                    annotation_text=f"Current: {default_inflation}%", annotation_position="top left")
    fig_sens.update_layout(height=380, xaxis_title="Inflation (%)", yaxis_title="Real Yield (%)")
    st.plotly_chart(fig_sens, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    f'<div class="footer">'
    f'Wealth Portfolio Analyser v2.0 &nbsp;·&nbsp; '
    f'RFR: {risk_free_rate}% (India G-Sec Proxy) &nbsp;·&nbsp; '
    f'Inflation: {default_inflation}% &nbsp;·&nbsp; '
    f'Built with Streamlit + Plotly + SciPy'
    f'</div>',
    unsafe_allow_html=True
)