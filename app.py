"""
Alpha Research AI — Explainable Stock & ETF Research App
=========================================================
A free, self-hostable research app for stocks and ETFs across India, Asia,
the UK and the US. Built with Streamlit + yfinance (both free).

Implements the requirements in:
  explainable_stock_etf_research_app_requirements_and_code.md
- Search -> select -> generate report flow
- Explainable metric cards (definition, why it matters, calculation,
  benchmark bands, interpretation, score impact)
- Category scorecard with weighted overall score, confidence, risk
- Price trend / score breakdown / ETF exposure charts
- Bull & bear cases, data-source notes, refresh, print-to-PDF

Run locally:   streamlit run app.py
Deploy free:   Streamlit Community Cloud (see README_DEPLOYMENT.md)
"""

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------------------------
# Page setup & styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Alpha Research AI — Explainable Stock & ETF Research",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; max-width: 1200px;}
      div[data-testid="stMetric"] {
          background: #f8fafc; border: 1px solid #e2e8f0;
          border-radius: 14px; padding: 12px 16px;
      }
      .ara-badge {
          display:inline-block; padding:2px 10px; margin-right:6px;
          border-radius:999px; font-size:0.75rem; font-weight:600;
          border:1px solid #cbd5e1; color:#334155; background:#f8fafc;
      }
      .ara-status-excellent {background:#d1fae5;color:#065f46;border-color:#a7f3d0;}
      .ara-status-good      {background:#dbeafe;color:#1e40af;border-color:#bfdbfe;}
      .ara-status-acceptable{background:#fef3c7;color:#92400e;border-color:#fde68a;}
      .ara-status-weak      {background:#fee2e2;color:#991b1b;border-color:#fecaca;}
      .ara-status-na        {background:#f1f5f9;color:#475569;border-color:#e2e8f0;}
      @media print {
        header, footer, [data-testid="stSidebar"], .stButton, .no-print {display:none !important;}
        .block-container {max-width:100% !important; padding:0 !important;}
      }
    </style>
    """,
    unsafe_allow_html=True,
)

RISK_FREE_RATE = 0.04  # stated assumption used in Sharpe/Sortino calculations

# ---------------------------------------------------------------------------
# Region / symbol helpers
# ---------------------------------------------------------------------------

REGION_OPTIONS = ["All markets", "India", "United Kingdom", "Asia (ex-India)", "United States", "Other"]

SUFFIX_REGION = {
    ".NS": "India", ".BO": "India",
    ".L": "United Kingdom", ".IL": "United Kingdom",
    ".T": "Asia (ex-India)", ".HK": "Asia (ex-India)", ".SS": "Asia (ex-India)",
    ".SZ": "Asia (ex-India)", ".KS": "Asia (ex-India)", ".KQ": "Asia (ex-India)",
    ".TW": "Asia (ex-India)", ".SI": "Asia (ex-India)", ".BK": "Asia (ex-India)",
    ".JK": "Asia (ex-India)", ".KL": "Asia (ex-India)",
}

EXAMPLE_CHIPS = [
    ("RELIANCE.NS", "Reliance Industries — NSE"),
    ("TCS.NS", "Tata Consultancy — NSE"),
    ("NIFTYBEES.NS", "Nippon Nifty 50 ETF — NSE"),
    ("SHEL.L", "Shell plc — LSE"),
    ("VUSA.L", "Vanguard S&P 500 ETF — LSE"),
    ("7203.T", "Toyota — Tokyo"),
    ("9988.HK", "Alibaba — Hong Kong"),
    ("MSFT", "Microsoft — NASDAQ"),
    ("VOO", "Vanguard S&P 500 — NYSE"),
]


def region_for_symbol(symbol: str, exchange: str = "") -> str:
    s = (symbol or "").upper()
    for suffix, region in SUFFIX_REGION.items():
        if s.endswith(suffix):
            return region
    ex = (exchange or "").lower()
    if any(k in ex for k in ("nse", "bse", "bombay", "national stock")):
        return "India"
    if any(k in ex for k in ("lse", "london")):
        return "United Kingdom"
    if any(k in ex for k in ("tokyo", "hong kong", "hkse", "shanghai", "shenzhen",
                             "korea", "taiwan", "singapore", "jakarta", "thailand", "kuala")):
        return "Asia (ex-India)"
    if any(k in ex for k in ("nasdaq", "nyse", "amex", "cboe", "bats")):
        return "United States"
    return "Other"


# ---------------------------------------------------------------------------
# Data access (all server-side -> no CORS problems)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner=False)
def search_instruments(query: str) -> list[dict]:
    """Search Yahoo Finance for matching stocks/ETFs. Two strategies with fallback."""
    results = []
    # Strategy 1: yfinance built-in search
    try:
        s = yf.Search(query, max_results=15, news_count=0)
        for q in (s.quotes or []):
            results.append(q)
    except Exception:
        pass
    # Strategy 2: direct public endpoint (fallback)
    if not results:
        try:
            r = requests.get(
                "https://query2.finance.yahoo.com/v1/finance/search",
                params={"q": query, "quotesCount": 15, "newsCount": 0, "enableFuzzyQuery": "true"},
                headers={"User-Agent": "Mozilla/5.0 (research-app)"},
                timeout=10,
            )
            if r.ok:
                results = r.json().get("quotes", [])
        except Exception:
            pass

    cleaned = []
    for q in results:
        sym = q.get("symbol")
        if not sym:
            continue
        qtype = (q.get("quoteType") or "").upper()
        if qtype not in ("EQUITY", "ETF", "MUTUALFUND", "INDEX"):
            continue
        cleaned.append({
            "symbol": sym,
            "name": q.get("longname") or q.get("shortname") or sym,
            "type": "ETF" if qtype == "ETF" else ("Index" if qtype == "INDEX" else "Stock"),
            "exchange": q.get("exchDisp") or q.get("exchange") or "—",
            "region": region_for_symbol(sym, q.get("exchDisp") or q.get("exchange") or ""),
        })
    return cleaned


@st.cache_data(ttl=900, show_spinner=False)
def load_instrument(symbol: str) -> dict:
    """Fetch quote fundamentals + 1y history + ETF fund data for a symbol."""
    t = yf.Ticker(symbol)
    out = {"symbol": symbol, "info": {}, "history": None,
           "sector_weights": None, "top_holdings": None, "errors": []}
    try:
        out["info"] = t.info or {}
    except Exception as e:
        out["errors"].append(f"Fundamentals unavailable: {e}")
    try:
        hist = t.history(period="1y", auto_adjust=True)
        if hist is not None and not hist.empty:
            out["history"] = hist
    except Exception as e:
        out["errors"].append(f"Price history unavailable: {e}")
    # ETF holdings / sector exposure (best effort)
    qtype = (out["info"].get("quoteType") or "").upper()
    if qtype == "ETF":
        try:
            fd = t.funds_data
            sw = getattr(fd, "sector_weightings", None)
            if sw:
                out["sector_weights"] = dict(sw)
            th = getattr(fd, "top_holdings", None)
            if th is not None and hasattr(th, "empty") and not th.empty:
                out["top_holdings"] = th
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# Derived quantitative helpers
# ---------------------------------------------------------------------------

def compute_rsi(close: pd.Series, window: int = 14):
    if close is None or len(close) < window + 1:
        return None
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    val = rsi.dropna()
    return float(val.iloc[-1]) if len(val) else None


def annualised_volatility(close: pd.Series):
    if close is None or len(close) < 30:
        return None
    ret = close.pct_change().dropna()
    return float(ret.std() * math.sqrt(252) * 100)


def sharpe_ratio(close: pd.Series, rf: float = RISK_FREE_RATE):
    if close is None or len(close) < 60:
        return None
    ret = close.pct_change().dropna()
    ann_ret = ret.mean() * 252
    ann_vol = ret.std() * math.sqrt(252)
    if not ann_vol:
        return None
    return float((ann_ret - rf) / ann_vol)


def max_drawdown(close: pd.Series):
    if close is None or len(close) < 30:
        return None
    roll_max = close.cummax()
    dd = close / roll_max - 1
    return float(dd.min() * 100)


def period_return(close: pd.Series):
    if close is None or len(close) < 2:
        return None
    return float((close.iloc[-1] / close.iloc[0] - 1) * 100)


def dividend_yield_pct(info: dict):
    """yfinance has changed the units of dividendYield between versions.
    Compute from rate/price when possible; otherwise normalise heuristically."""
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    rate = info.get("dividendRate") or info.get("trailingAnnualDividendRate")
    if rate and price:
        return float(rate) / float(price) * 100
    dy = info.get("dividendYield")
    if dy is None:
        return None
    dy = float(dy)
    return dy * 100 if dy < 0.5 else dy  # 0.032 -> 3.2%, 3.2 stays 3.2%


# ---------------------------------------------------------------------------
# Explainable metric engine
# ---------------------------------------------------------------------------
# Each metric = value + full explanation. Scoring uses banded thresholds so
# every score is traceable to a stated rule (the "explainable" principle).

def banded(value, bands, higher_is_better=True):
    """bands = [excellent_cut, good_cut, acceptable_cut] on the better->worse axis.
    Returns (score, status)."""
    if value is None:
        return None, "N/A"
    e, g, a = bands
    if higher_is_better:
        if value >= e:
            return 90, "Excellent"
        if value >= g:
            return 72, "Good"
        if value >= a:
            return 52, "Acceptable"
        return 32, "Weak"
    else:
        if value <= e:
            return 90, "Excellent"
        if value <= g:
            return 72, "Good"
        if value <= a:
            return 52, "Acceptable"
        return 32, "Weak"


def fmt(value, kind="num", currency=""):
    if value is None:
        return "Not available"
    try:
        if kind == "pct":
            return f"{value:,.2f}%"
        if kind == "x":
            return f"{value:,.2f}×"
        if kind == "money":
            return f"{currency} {value:,.2f}".strip()
        if kind == "compact":
            for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
                if abs(value) >= div:
                    return f"{value/div:,.2f}{unit}"
            return f"{value:,.0f}"
        return f"{value:,.2f}"
    except Exception:
        return str(value)


def metric(category, name, value_display, score, status, definition, why,
           calculation, bands, benchmark, interpretation, impact):
    return dict(category=category, name=name, value=value_display, score=score,
                status=status, definition=definition, why=why, calculation=calculation,
                bands=bands, benchmark=benchmark, interpretation=interpretation, impact=impact)


def interp(name, value_display, status, better_when):
    if status == "N/A":
        return (f"{name} was not published by the free data source for this instrument. "
                f"It is excluded from scoring and lowers the report confidence instead of distorting the result.")
    return (f"The current value of {value_display} sits in the '{status}' band. "
            f"For this metric, {better_when}.")


# ----------------------------- Stock metrics ------------------------------

def build_stock_metrics(info: dict, hist, currency: str) -> list[dict]:
    m = []
    close = hist["Close"] if hist is not None and "Close" in hist else None
    price = info.get("currentPrice") or info.get("regularMarketPrice") or \
            (float(close.iloc[-1]) if close is not None and len(close) else None)

    # --- Valuation ---
    pe = info.get("trailingPE")
    s, st_ = banded(pe, (15, 25, 40), higher_is_better=False)
    m.append(metric("Valuation", "P/E Ratio (trailing)", fmt(pe, "x"), s, st_,
        "Price divided by the last twelve months of earnings per share. It tells you how many years of current profit you are paying for.",
        "It is the fastest sanity check on whether a share price is demanding: a high P/E means the market has already priced in strong future growth.",
        "Share price ÷ trailing 12-month EPS, as reported by Yahoo Finance.",
        [("Excellent", "≤ 15×"), ("Good", "15–25×"), ("Acceptable", "25–40×"), ("Weak", "> 40×")],
        "Compare against the sector median — Indian IT and US tech typically trade at 25–35×, while UK energy and banks often trade below 12×. A 'high' P/E in one sector can be normal in another.",
        interp("P/E", fmt(pe, "x"), st_, "lower is generally better because you pay less per unit of profit; but very low P/E can also signal the market expects earnings to fall"),
        "Feeds the Valuation category (20% of overall stock score)."))

    fpe = info.get("forwardPE")
    s, st_ = banded(fpe, (14, 22, 35), higher_is_better=False)
    m.append(metric("Valuation", "Forward P/E", fmt(fpe, "x"), s, st_,
        "Price divided by the consensus analyst forecast of next year's earnings.",
        "If forward P/E is well below trailing P/E, analysts expect earnings to grow — the stock is cheaper than it looks on trailing numbers.",
        "Share price ÷ consensus forward EPS estimate.",
        [("Excellent", "≤ 14×"), ("Good", "14–22×"), ("Acceptable", "22–35×"), ("Weak", "> 35×")],
        "Read alongside trailing P/E: forward < trailing implies expected growth; forward > trailing implies expected decline.",
        interp("Forward P/E", fmt(fpe, "x"), st_, "lower is better, provided the forecasts are credible"),
        "Feeds the Valuation category (20% weight)."))

    peg = info.get("trailingPegRatio") or info.get("pegRatio")
    s, st_ = banded(peg, (1.0, 1.5, 2.5), higher_is_better=False)
    m.append(metric("Valuation", "PEG Ratio", fmt(peg, "x"), s, st_,
        "P/E divided by expected earnings growth rate. It adjusts the P/E for how fast profits are growing.",
        "A P/E of 30 is expensive for a company growing 5% a year but cheap for one growing 40%. PEG normalises that.",
        "Trailing P/E ÷ expected annual EPS growth (%).",
        [("Excellent", "≤ 1.0"), ("Good", "1.0–1.5"), ("Acceptable", "1.5–2.5"), ("Weak", "> 2.5")],
        "Peter Lynch's classic rule of thumb: PEG ≈ 1 is fair value. Growth stocks worldwide often trade at 1.5–2.0.",
        interp("PEG", fmt(peg, "x"), st_, "lower is better — you pay less per unit of growth"),
        "Feeds the Valuation category (20% weight)."))

    ps = info.get("priceToSalesTrailing12Months")
    s, st_ = banded(ps, (2, 4, 8), higher_is_better=False)
    m.append(metric("Valuation", "Price-to-Sales", fmt(ps, "x"), s, st_,
        "Market capitalisation divided by trailing twelve-month revenue.",
        "Useful when earnings are volatile or negative — revenue is harder to manipulate than profit.",
        "Market cap ÷ trailing 12-month revenue.",
        [("Excellent", "≤ 2×"), ("Good", "2–4×"), ("Acceptable", "4–8×"), ("Weak", "> 8×")],
        "Software businesses sustain higher P/S (5–10×) than retailers or refiners (0.3–1×) because of margin differences. Always compare within sector.",
        interp("P/S", fmt(ps, "x"), st_, "lower is better for the same margin profile"),
        "Feeds the Valuation category (20% weight)."))

    pb = info.get("priceToBook")
    s, st_ = banded(pb, (1.5, 3, 6), higher_is_better=False)
    m.append(metric("Valuation", "Price-to-Book", fmt(pb, "x"), s, st_,
        "Share price divided by book value (net assets) per share.",
        "Especially relevant for banks, insurers and asset-heavy businesses where the balance sheet drives value.",
        "Market cap ÷ shareholders' equity.",
        [("Excellent", "≤ 1.5×"), ("Good", "1.5–3×"), ("Acceptable", "3–6×"), ("Weak", "> 6×")],
        "Indian private banks often trade at 2–4× book when ROE is high; UK banks nearer 0.5–1×. High P/B is justified only by high return on equity.",
        interp("P/B", fmt(pb, "x"), st_, "lower is better unless a high return on equity justifies the premium"),
        "Feeds the Valuation category (20% weight)."))

    ev_ebitda = info.get("enterpriseToEbitda")
    s, st_ = banded(ev_ebitda, (10, 15, 25), higher_is_better=False)
    m.append(metric("Valuation", "EV / EBITDA", fmt(ev_ebitda, "x"), s, st_,
        "Enterprise value (market cap + net debt) divided by earnings before interest, tax, depreciation and amortisation.",
        "It compares companies regardless of how they are financed — the preferred multiple for takeovers and cross-border comparisons.",
        "(Market cap + total debt − cash) ÷ trailing EBITDA.",
        [("Excellent", "≤ 10×"), ("Good", "10–15×"), ("Acceptable", "15–25×"), ("Weak", "> 25×")],
        "Historical market averages sit around 10–12×. Capital-light software runs higher; commodity producers lower.",
        interp("EV/EBITDA", fmt(ev_ebitda, "x"), st_, "lower is better on a like-for-like basis"),
        "Feeds the Valuation category (20% weight)."))

    mcap = info.get("marketCap")
    fcf = info.get("freeCashflow")
    fcf_yield = (fcf / mcap * 100) if fcf and mcap else None
    s, st_ = banded(fcf_yield, (6, 4, 2), higher_is_better=True)
    m.append(metric("Valuation", "Free Cash Flow Yield", fmt(fcf_yield, "pct"), s, st_,
        "Free cash flow (cash from operations minus capital expenditure) as a percentage of market capitalisation.",
        "Cash is the hardest number to fake. A high FCF yield means the business generates real cash relative to its price.",
        "Trailing free cash flow ÷ market cap × 100.",
        [("Excellent", "≥ 6%"), ("Good", "4–6%"), ("Acceptable", "2–4%"), ("Weak", "< 2%")],
        "Compare against the local 10-year government bond yield: an FCF yield above it means the equity out-earns the 'risk-free' alternative before any growth.",
        interp("FCF yield", fmt(fcf_yield, "pct"), st_, "higher is better — more cash generated per unit of price"),
        "Feeds the Valuation category (20% weight)."))

    dy = dividend_yield_pct(info)
    s, st_ = banded(dy, (4, 2.5, 1), higher_is_better=True)
    m.append(metric("Valuation", "Dividend Yield", fmt(dy, "pct"), s, st_,
        "Annual dividends per share divided by the share price.",
        "It is the cash return you receive while holding, independent of price movement.",
        "Trailing annual dividend rate ÷ current price × 100 (computed directly to avoid provider unit inconsistencies).",
        [("Excellent", "≥ 4%"), ("Good", "2.5–4%"), ("Acceptable", "1–2.5%"), ("Weak", "< 1% or nil")],
        "FTSE 100 averages ~3.5–4%; Nifty 50 ~1.2–1.5%; S&P 500 ~1.3%. A yield far above the market average can be a distress signal, so check payout sustainability.",
        interp("Dividend yield", fmt(dy, "pct"), st_, "higher is better for income, provided the payout is covered by free cash flow"),
        "Feeds the Valuation category (20% weight). Growth companies legitimately score low here — read together with growth metrics."))

    # --- Growth ---
    rev_g = info.get("revenueGrowth")
    rev_g = rev_g * 100 if rev_g is not None else None
    s, st_ = banded(rev_g, (15, 8, 3), higher_is_better=True)
    m.append(metric("Growth", "Revenue Growth (yoy)", fmt(rev_g, "pct"), s, st_,
        "Year-on-year change in total revenue for the most recent reported period.",
        "Revenue growth is the raw material of all future profit growth — margins can only be squeezed so far.",
        "(Latest period revenue ÷ same period last year − 1) × 100.",
        [("Excellent", "≥ 15%"), ("Good", "8–15%"), ("Acceptable", "3–8%"), ("Weak", "< 3%")],
        "Compare against nominal GDP growth of the home market (~10–11% for India, ~3–4% for UK/US). Growing slower than nominal GDP means losing share of the economy.",
        interp("Revenue growth", fmt(rev_g, "pct"), st_, "higher is better, especially if achieved without margin erosion"),
        "Feeds the Growth category (20% weight)."))

    eps_g = info.get("earningsGrowth")
    eps_g = eps_g * 100 if eps_g is not None else None
    s, st_ = banded(eps_g, (18, 10, 4), higher_is_better=True)
    m.append(metric("Growth", "EPS Growth (yoy)", fmt(eps_g, "pct"), s, st_,
        "Year-on-year change in earnings per share.",
        "Over long periods, share prices track EPS growth more closely than any other single variable.",
        "(Latest EPS ÷ prior-year EPS − 1) × 100.",
        [("Excellent", "≥ 18%"), ("Good", "10–18%"), ("Acceptable", "4–10%"), ("Weak", "< 4%")],
        "Quality compounders worldwide sustain 10–15% EPS growth across cycles. One-off spikes from base effects should be discounted.",
        interp("EPS growth", fmt(eps_g, "pct"), st_, "higher is better if it comes from operations rather than buybacks alone"),
        "Feeds the Growth category (20% weight)."))

    # --- Profitability ---
    gm = info.get("grossMargins")
    gm = gm * 100 if gm is not None else None
    s, st_ = banded(gm, (50, 35, 20), higher_is_better=True)
    m.append(metric("Profitability", "Gross Margin", fmt(gm, "pct"), s, st_,
        "Revenue minus cost of goods sold, as a percentage of revenue.",
        "It measures pricing power. High gross margins give a company room to invest, absorb shocks and out-spend competitors.",
        "Gross profit ÷ revenue × 100 (trailing twelve months).",
        [("Excellent", "≥ 50%"), ("Good", "35–50%"), ("Acceptable", "20–35%"), ("Weak", "< 20%")],
        "Software: 70–90%. Consumer brands: 40–60%. Autos: 15–25%. Refining/commodities: <15%. Judge within the sector's normal range.",
        interp("Gross margin", fmt(gm, "pct"), st_, "higher is better — it signals durable pricing power"),
        "Feeds the Profitability category (20% weight)."))

    om = info.get("operatingMargins")
    om = om * 100 if om is not None else None
    s, st_ = banded(om, (25, 15, 8), higher_is_better=True)
    m.append(metric("Profitability", "Operating Margin", fmt(om, "pct"), s, st_,
        "Operating profit (before interest and tax) as a percentage of revenue.",
        "It shows how efficiently the whole operating model converts sales into profit, after all running costs.",
        "Operating income ÷ revenue × 100 (trailing twelve months).",
        [("Excellent", "≥ 25%"), ("Good", "15–25%"), ("Acceptable", "8–15%"), ("Weak", "< 8%")],
        "A stable or rising operating margin over several years is a stronger signal than a single high reading.",
        interp("Operating margin", fmt(om, "pct"), st_, "higher and more stable is better"),
        "Feeds the Profitability category (20% weight)."))

    nm = info.get("profitMargins")
    nm = nm * 100 if nm is not None else None
    s, st_ = banded(nm, (18, 10, 5), higher_is_better=True)
    m.append(metric("Profitability", "Net Margin", fmt(nm, "pct"), s, st_,
        "Bottom-line profit as a percentage of revenue, after everything including tax and interest.",
        "The final measure of how much of each rupee, pound or dollar of sales becomes shareholder profit.",
        "Net income ÷ revenue × 100 (trailing twelve months).",
        [("Excellent", "≥ 18%"), ("Good", "10–18%"), ("Acceptable", "5–10%"), ("Weak", "< 5%")],
        "Global large-cap average is roughly 8–11%. Persistently above 15% usually indicates a genuine moat.",
        interp("Net margin", fmt(nm, "pct"), st_, "higher is better"),
        "Feeds the Profitability category (20% weight)."))

    roe = info.get("returnOnEquity")
    roe = roe * 100 if roe is not None else None
    s, st_ = banded(roe, (20, 13, 8), higher_is_better=True)
    m.append(metric("Profitability", "Return on Equity", fmt(roe, "pct"), s, st_,
        "Net income divided by shareholders' equity — the profit generated per unit of owners' capital.",
        "Warren Buffett's favourite quality test: businesses that compound at high ROE without excess debt create the most long-term value.",
        "Net income ÷ average shareholders' equity × 100.",
        [("Excellent", "≥ 20%"), ("Good", "13–20%"), ("Acceptable", "8–13%"), ("Weak", "< 8%")],
        "Check ROE against debt-to-equity: an ROE inflated by heavy leverage is lower quality than the same ROE achieved with a clean balance sheet.",
        interp("ROE", fmt(roe, "pct"), st_, "higher is better when it is not manufactured through leverage"),
        "Feeds the Profitability category (20% weight)."))

    roa = info.get("returnOnAssets")
    roa = roa * 100 if roa is not None else None
    s, st_ = banded(roa, (10, 6, 3), higher_is_better=True)
    m.append(metric("Profitability", "Return on Assets", fmt(roa, "pct"), s, st_,
        "Net income divided by total assets.",
        "It strips out financing structure entirely — a pure measure of how productively the asset base is used.",
        "Net income ÷ total assets × 100.",
        [("Excellent", "≥ 10%"), ("Good", "6–10%"), ("Acceptable", "3–6%"), ("Weak", "< 3%")],
        "Asset-light businesses (software, consultancies) naturally score higher than utilities or telecoms. Compare within business model.",
        interp("ROA", fmt(roa, "pct"), st_, "higher is better"),
        "Feeds the Profitability category (20% weight)."))

    # --- Financial Health ---
    de = info.get("debtToEquity")  # yfinance reports this as a percentage figure
    s, st_ = banded(de, (40, 80, 150), higher_is_better=False)
    m.append(metric("Financial Health", "Debt-to-Equity", fmt(de, "pct"), s, st_,
        "Total debt as a percentage of shareholders' equity.",
        "Leverage amplifies both good and bad outcomes. High debt turns a rough year into an existential one.",
        "Total debt ÷ shareholders' equity × 100 (as reported by the provider).",
        [("Excellent", "≤ 40%"), ("Good", "40–80%"), ("Acceptable", "80–150%"), ("Weak", "> 150%")],
        "Capital-intensive sectors (utilities, telecoms, real estate) run structurally higher leverage; banks are excluded from this rule entirely as leverage is their business model.",
        interp("Debt-to-equity", fmt(de, "pct"), st_, "lower is better — it buys resilience in downturns"),
        "Feeds the Financial Health category (15% weight)."))

    cr = info.get("currentRatio")
    s, st_ = banded(cr, (2.0, 1.5, 1.0), higher_is_better=True)
    m.append(metric("Financial Health", "Current Ratio", fmt(cr, "x"), s, st_,
        "Current assets divided by current liabilities — can the company pay its bills over the next 12 months?",
        "The classic short-term solvency test. Below 1.0 means near-term obligations exceed near-term resources.",
        "Current assets ÷ current liabilities.",
        [("Excellent", "≥ 2.0"), ("Good", "1.5–2.0"), ("Acceptable", "1.0–1.5"), ("Weak", "< 1.0")],
        "Fast-turnover retailers can safely run below 1.0 because inventory converts to cash quickly; manufacturers should not.",
        interp("Current ratio", fmt(cr, "x"), st_, "higher is safer, though far above 3 can indicate lazy capital"),
        "Feeds the Financial Health category (15% weight)."))

    cash = info.get("totalCash")
    cash_pct = (cash / mcap * 100) if cash and mcap else None
    s, st_ = banded(cash_pct, (15, 8, 3), higher_is_better=True)
    m.append(metric("Financial Health", "Cash Position (% of market cap)", fmt(cash_pct, "pct"), s, st_,
        "Total cash and equivalents as a percentage of market capitalisation.",
        "Cash is optionality: buybacks, acquisitions, surviving downturns without dilution.",
        "Total cash ÷ market cap × 100.",
        [("Excellent", "≥ 15%"), ("Good", "8–15%"), ("Acceptable", "3–8%"), ("Weak", "< 3%")],
        "Read together with debt: net cash (cash > debt) is the strongest position of all.",
        interp("Cash position", fmt(cash_pct, "pct"), st_, "higher is better as a resilience buffer"),
        "Feeds the Financial Health category (15% weight)."))

    # --- Risk ---
    beta = info.get("beta")
    s, st_ = banded(beta, (0.9, 1.15, 1.5), higher_is_better=False)
    m.append(metric("Risk", "Beta (5y monthly)", fmt(beta, "x"), s, st_,
        "How much the stock moves relative to its market index. Beta 1.2 means it typically moves 12% when the market moves 10%.",
        "It quantifies market-linked risk: high-beta names fall hardest in corrections.",
        "Regression slope of the stock's monthly returns against the benchmark index over 5 years.",
        [("Excellent (defensive)", "≤ 0.9"), ("Good", "0.9–1.15"), ("Acceptable", "1.15–1.5"), ("Weak (high risk)", "> 1.5")],
        "Consumer staples and utilities cluster near 0.5–0.8; high-growth tech and small caps at 1.3–2.0. Neither is 'wrong' — it depends on your risk appetite.",
        interp("Beta", fmt(beta, "x"), st_, "lower means smoother — this app scores lower beta as safer, not as 'better returns'"),
        "Feeds the Risk category (15% weight)."))

    vol = annualised_volatility(close)
    s, st_ = banded(vol, (20, 30, 45), higher_is_better=False)
    m.append(metric("Risk", "Volatility (1y, annualised)", fmt(vol, "pct"), s, st_,
        "The standard deviation of daily returns over the last year, scaled to an annual figure.",
        "It tells you how bumpy the ride has actually been — a direct, backward-looking measure of price risk.",
        "Std-dev of daily % returns × √252, computed from the fetched 1-year price history.",
        [("Excellent", "≤ 20%"), ("Good", "20–30%"), ("Acceptable", "30–45%"), ("Weak", "> 45%")],
        "Large-cap indices typically run 12–20% annualised; individual large caps 20–35%; small caps and turnarounds higher.",
        interp("Volatility", fmt(vol, "pct"), st_, "lower is calmer; make sure you can hold through the swings the number implies"),
        "Feeds the Risk category (15% weight)."))

    mdd = max_drawdown(close)
    s, st_ = banded(mdd, (-15, -25, -40), higher_is_better=True)
    m.append(metric("Risk", "Max Drawdown (1y)", fmt(mdd, "pct"), s, st_,
        "The largest peak-to-trough fall in the share price over the past year.",
        "This is the loss you would have felt buying at the worst moment — a visceral, real-world risk measure.",
        "Minimum of (price ÷ running peak − 1) over the 1-year history.",
        [("Excellent", "shallower than −15%"), ("Good", "−15% to −25%"), ("Acceptable", "−25% to −40%"), ("Weak", "deeper than −40%")],
        "Even world-class companies routinely draw down 20–30% within a year. What matters is whether the business recovered and why it fell.",
        interp("Max drawdown", fmt(mdd, "pct"), st_, "shallower is better"),
        "Feeds the Risk category (15% weight)."))

    # --- Technical Trend ---
    ma50 = info.get("fiftyDayAverage") or (float(close.rolling(50).mean().iloc[-1]) if close is not None and len(close) >= 50 else None)
    ma200 = info.get("twoHundredDayAverage") or (float(close.rolling(200).mean().iloc[-1]) if close is not None and len(close) >= 200 else None)
    above50 = (price / ma50 - 1) * 100 if price and ma50 else None
    s, st_ = banded(above50, (3, 0, -5), higher_is_better=True)
    m.append(metric("Technical Trend", "Price vs 50-day MA", fmt(above50, "pct"), s, st_,
        "How far the current price sits above or below its 50-day moving average.",
        "The 50-day MA is the market's medium-term trend line; trading above it indicates positive momentum.",
        f"(Current price ÷ 50-day average − 1) × 100. Current 50-day MA: {fmt(ma50, 'money', currency)}.",
        [("Excellent", "> +3%"), ("Good", "0 to +3%"), ("Acceptable", "0 to −5%"), ("Weak", "< −5%")],
        "Trend-followers treat a decisive break below the 50-day MA as an early caution flag.",
        interp("Price vs 50-day MA", fmt(above50, "pct"), st_, "above the average signals momentum; far above can signal short-term over-extension"),
        "Feeds the Technical Trend category (10% weight)."))

    above200 = (price / ma200 - 1) * 100 if price and ma200 else None
    s, st_ = banded(above200, (5, 0, -8), higher_is_better=True)
    m.append(metric("Technical Trend", "Price vs 200-day MA", fmt(above200, "pct"), s, st_,
        "How far the current price sits above or below its 200-day moving average — the classic long-term trend measure.",
        "Institutions widely use the 200-day MA as the bull/bear dividing line for an individual security.",
        f"(Current price ÷ 200-day average − 1) × 100. Current 200-day MA: {fmt(ma200, 'money', currency)}.",
        [("Excellent", "> +5%"), ("Good", "0 to +5%"), ("Acceptable", "0 to −8%"), ("Weak", "< −8%")],
        "Golden cross (50-day rising above 200-day) is a well-known bullish structure; a death cross is the reverse.",
        interp("Price vs 200-day MA", fmt(above200, "pct"), st_, "above the line = long-term uptrend intact"),
        "Feeds the Technical Trend category (10% weight)."))

    rsi = compute_rsi(close)
    if rsi is None:
        s, st_ = None, "N/A"
    elif 45 <= rsi <= 62:
        s, st_ = 85, "Positive"
    elif 35 <= rsi < 45 or 62 < rsi <= 70:
        s, st_ = 65, "Neutral"
    elif rsi < 30:
        s, st_ = 50, "Oversold"
    elif rsi > 70:
        s, st_ = 45, "Overbought"
    else:
        s, st_ = 55, "Neutral"
    m.append(metric("Technical Trend", "RSI (14-day)", fmt(rsi), s, st_,
        "Relative Strength Index — a 0–100 oscillator measuring the speed and size of recent price moves.",
        "It flags stretched conditions: persistent readings above 70 suggest over-buying; below 30, capitulation.",
        "Computed from the fetched price history using 14-day average gains vs losses (standard Wilder-style formulation).",
        [("Oversold", "< 30"), ("Neutral", "30–45 or 62–70"), ("Positive", "45–62"), ("Overbought", "> 70")],
        "RSI works best as a timing overlay, never as a standalone buy/sell signal. Strong uptrends can stay 'overbought' for months.",
        (f"RSI of {fmt(rsi)} is in the '{st_}' zone." if rsi is not None else "RSI could not be computed — insufficient price history."),
        "Feeds the Technical Trend category (10% weight)."))

    return m


# ------------------------------ ETF metrics --------------------------------

def build_etf_metrics(info: dict, hist, data: dict) -> list[dict]:
    m = []
    close = hist["Close"] if hist is not None and "Close" in hist else None

    er = info.get("netExpenseRatio") or info.get("annualReportExpenseRatio")
    if er is not None:
        er = float(er)
        er = er * 100 if er < 0.05 else er  # normalise fraction vs percent
    s, st_ = banded(er, (0.20, 0.50, 1.00), higher_is_better=False)
    m.append(metric("Cost", "Expense Ratio", fmt(er, "pct"), s, st_,
        "The annual fee the fund charges, deducted automatically from returns.",
        "Cost is the single most reliable predictor of long-run fund performance — every basis point compounds against you, every year.",
        "Published total expense ratio (TER) from the fund provider, via Yahoo Finance.",
        [("Excellent", "≤ 0.20%"), ("Good", "0.20–0.50%"), ("Acceptable", "0.50–1.00%"), ("Weak", "> 1.00%")],
        "Broad index ETFs now cost 0.03–0.20% (e.g. large S&P 500 or Nifty 50 trackers). Niche/thematic and active ETFs run 0.4–0.95%. Anything near 1% needs strong justification.",
        interp("Expense ratio", fmt(er, "pct"), st_, "lower is always better — this is the one metric where cheapest genuinely wins"),
        "Feeds the Cost category (20% of overall ETF score)."))

    aum = info.get("totalAssets")
    s, st_ = banded(aum, (1e9, 2e8, 5e7), higher_is_better=True)
    m.append(metric("Liquidity", "Assets Under Management", fmt(aum, "compact"), s, st_,
        "The total market value of everything the fund holds.",
        "Small funds risk closure (forcing a taxable exit at a time not of your choosing) and tend to have wider trading spreads.",
        "Published fund AUM via Yahoo Finance, in the fund's base currency.",
        [("Excellent", "≥ 1B"), ("Good", "200M–1B"), ("Acceptable", "50–200M"), ("Weak", "< 50M")],
        "Funds below ~50M in AUM run a materially higher closure risk; providers regularly cull them.",
        interp("AUM", fmt(aum, "compact"), st_, "bigger is safer and usually cheaper to trade"),
        "Feeds the Liquidity category (15% weight)."))

    adv = info.get("averageVolume")
    s, st_ = banded(adv, (500_000, 100_000, 20_000), higher_is_better=True)
    m.append(metric("Liquidity", "Average Daily Volume", fmt(adv, "compact"), s, st_,
        "The average number of fund units traded per day.",
        "Higher volume means tighter bid-ask spreads — you lose less money simply entering and exiting.",
        "Trailing average daily unit volume via Yahoo Finance.",
        [("Excellent", "≥ 500K"), ("Good", "100–500K"), ("Acceptable", "20–100K"), ("Weak", "< 20K")],
        "For ETFs, underlying-basket liquidity matters more than on-screen volume, but thin on-screen volume still widens spreads for retail-size orders.",
        interp("Average volume", fmt(adv, "compact"), st_, "higher is better for cheap execution"),
        "Feeds the Liquidity category (15% weight)."))

    r1y = period_return(close)
    s, st_ = banded(r1y, (15, 8, 0), higher_is_better=True)
    m.append(metric("Performance", "1-Year Return", fmt(r1y, "pct"), s, st_,
        "Total price change over the last twelve months (dividends reinvested where the data source adjusts for them).",
        "Recent performance in context — useful, but the least predictive number in this report on its own.",
        "(Latest close ÷ close 12 months ago − 1) × 100, from adjusted price history.",
        [("Excellent", "≥ 15%"), ("Good", "8–15%"), ("Acceptable", "0–8%"), ("Weak", "negative")],
        "Always compare with the fund's own benchmark index, not with cash. A −5% year when the index fell 8% is good management.",
        interp("1-year return", fmt(r1y, "pct"), st_, "higher is better, but one year proves little — check 3–5 year consistency"),
        "Feeds the Performance category (20% weight)."))

    r3y = info.get("threeYearAverageReturn")
    r3y = r3y * 100 if r3y is not None and abs(r3y) < 2 else r3y
    s, st_ = banded(r3y, (12, 7, 3), higher_is_better=True)
    m.append(metric("Performance", "3-Year Average Annual Return", fmt(r3y, "pct"), s, st_,
        "The annualised average return over the past three years.",
        "Three years smooths out single-year noise and covers at least one meaningful market wobble.",
        "Fund-reported 3-year annualised return via Yahoo Finance.",
        [("Excellent", "≥ 12%"), ("Good", "7–12%"), ("Acceptable", "3–7%"), ("Weak", "< 3%")],
        "Global equities have returned ~7–10% annualised over long horizons; sustained double digits usually reflects a strong market regime, not magic.",
        interp("3-year return", fmt(r3y, "pct"), st_, "higher is better, judged against the fund's benchmark"),
        "Feeds the Performance category (20% weight)."))

    vol = annualised_volatility(close)
    s, st_ = banded(vol, (15, 22, 32), higher_is_better=False)
    m.append(metric("Risk", "Volatility (1y, annualised)", fmt(vol, "pct"), s, st_,
        "Standard deviation of the fund's daily returns over the last year, annualised.",
        "It measures how bumpy the fund's ride is — critical for judging whether you can hold it through a downturn.",
        "Std-dev of daily % returns × √252, computed from the fetched 1-year history.",
        [("Excellent", "≤ 15%"), ("Good", "15–22%"), ("Acceptable", "22–32%"), ("Weak", "> 32%")],
        "Broad developed-market equity ETFs: 12–18%. Single-country emerging or thematic tech ETFs: 20–35%. Bond ETFs: 3–8%.",
        interp("Volatility", fmt(vol, "pct"), st_, "lower is calmer for the same return"),
        "Feeds the Risk category (20% weight)."))

    shp = sharpe_ratio(close)
    s, st_ = banded(shp, (1.0, 0.6, 0.2), higher_is_better=True)
    m.append(metric("Risk", "Sharpe Ratio (1y)", fmt(shp), s, st_,
        "Return earned above the risk-free rate, per unit of volatility taken.",
        "It answers the real question: were you paid enough for the risk? Two funds with equal returns are not equal if one was twice as volatile.",
        f"(Annualised return − {RISK_FREE_RATE:.0%} assumed risk-free rate) ÷ annualised volatility, from the 1-year history.",
        [("Excellent", "≥ 1.0"), ("Good", "0.6–1.0"), ("Acceptable", "0.2–0.6"), ("Weak", "< 0.2")],
        "Above 1.0 over a full cycle is genuinely good; above 2.0 is rare and usually regime-dependent.",
        interp("Sharpe", fmt(shp), st_, "higher is better — more return per unit of risk"),
        "Feeds the Risk category (20% weight)."))

    mdd = max_drawdown(close)
    s, st_ = banded(mdd, (-10, -18, -30), higher_is_better=True)
    m.append(metric("Risk", "Max Drawdown (1y)", fmt(mdd, "pct"), s, st_,
        "The largest peak-to-trough fall over the past year.",
        "The most honest risk number: it is the loss an unlucky buyer actually experienced.",
        "Minimum of (price ÷ running peak − 1) over the 1-year history.",
        [("Excellent", "shallower than −10%"), ("Good", "−10% to −18%"), ("Acceptable", "−18% to −30%"), ("Weak", "deeper than −30%")],
        "Equity index funds routinely see −10% to −20% intra-year drawdowns even in positive years.",
        interp("Max drawdown", fmt(mdd, "pct"), st_, "shallower is better"),
        "Feeds the Risk category (20% weight)."))

    b3 = info.get("beta3Year") or info.get("beta")
    s, st_ = banded(b3, (0.95, 1.1, 1.35), higher_is_better=False)
    m.append(metric("Risk", "Beta (3y)", fmt(b3, "x"), s, st_,
        "The fund's sensitivity to its reference market index.",
        "Tells you whether this fund amplifies or dampens market swings in your portfolio.",
        "Regression of fund returns against the market index over 3 years, via Yahoo Finance.",
        [("Excellent (defensive)", "≤ 0.95"), ("Good", "0.95–1.1"), ("Acceptable", "1.1–1.35"), ("Weak (amplified)", "> 1.35")],
        "A plain index tracker should sit very close to 1.0 by construction; large deviation signals leverage, concentration or a different exposure than the name implies.",
        interp("Beta", fmt(b3, "x"), st_, "closer to 1.0 is expected for trackers; below 1.0 is defensive"),
        "Feeds the Risk category (20% weight)."))

    # Holdings quality
    sw = data.get("sector_weights") or {}
    top_sector_pct = max(sw.values()) * 100 if sw else None
    s, st_ = banded(top_sector_pct, (25, 35, 50), higher_is_better=False)
    m.append(metric("Holdings Quality", "Top Sector Concentration", fmt(top_sector_pct, "pct"), s, st_,
        "The weight of the fund's single largest sector.",
        "Concentration is hidden risk: a '500 stock' fund with 45% in one sector behaves like a sector bet in a shock.",
        "Largest single sector weighting from the fund's published composition.",
        [("Excellent", "≤ 25%"), ("Good", "25–35%"), ("Acceptable", "35–50%"), ("Weak", "> 50%")],
        "Broad global trackers keep top sectors near 20–30%. Thematic funds are deliberately concentrated — score this in light of what the fund promises to be.",
        interp("Top sector weight", fmt(top_sector_pct, "pct"), st_, "lower means more genuine diversification"),
        "Feeds the Holdings Quality category (15% weight)."))

    th = data.get("top_holdings")
    top10 = None
    try:
        if th is not None:
            col = [c for c in th.columns if "holding" in c.lower() or "percent" in c.lower() or "weight" in c.lower()]
            if col:
                v = float(th[col[0]].head(10).sum())
                top10 = v * 100 if v <= 1.5 else v
    except Exception:
        pass
    s, st_ = banded(top10, (25, 40, 60), higher_is_better=False)
    m.append(metric("Holdings Quality", "Top-10 Holdings Weight", fmt(top10, "pct"), s, st_,
        "The combined weight of the fund's ten largest positions.",
        "It reveals how much the fund really depends on a handful of names, whatever the total holding count says.",
        "Sum of the top-10 position weights from the fund's published holdings.",
        [("Excellent", "≤ 25%"), ("Good", "25–40%"), ("Acceptable", "40–60%"), ("Weak", "> 60%")],
        "Cap-weighted mega-cap indices (Nasdaq-100, Nifty 50) legitimately run 45–60% in the top ten today — know that you own concentration when you buy them.",
        interp("Top-10 weight", fmt(top10, "pct"), st_, "lower means broader diversification"),
        "Feeds the Holdings Quality category (15% weight)."))

    yld = dividend_yield_pct(info)
    s, st_ = banded(yld, (3, 1.8, 0.8), higher_is_better=True)
    m.append(metric("Performance", "Distribution Yield", fmt(yld, "pct"), s, st_,
        "The income the fund pays out annually as a percentage of its price.",
        "For income investors this is the point of the fund; for growth investors it indicates the style tilt.",
        "Trailing 12-month distributions ÷ current price × 100.",
        [("Excellent", "≥ 3%"), ("Good", "1.8–3%"), ("Acceptable", "0.8–1.8%"), ("Weak / n.a.", "< 0.8% (accumulating funds legitimately show ~0)")],
        "Accumulating (Acc) share classes reinvest internally and show near-zero yield by design — check the share class before judging.",
        interp("Distribution yield", fmt(yld, "pct"), st_, "higher is better for income mandates only"),
        "Feeds the Performance category (20% weight)."))

    return m


# ---------------------------------------------------------------------------
# Scoring, rating, confidence, risk, bull/bear
# ---------------------------------------------------------------------------

STOCK_WEIGHTS = {"Valuation": 0.20, "Growth": 0.20, "Profitability": 0.20,
                 "Financial Health": 0.15, "Risk": 0.15, "Technical Trend": 0.10}
ETF_WEIGHTS = {"Cost": 0.20, "Liquidity": 0.15, "Performance": 0.20,
               "Risk": 0.20, "Holdings Quality": 0.15}


def summarise(metrics: list[dict], weights: dict):
    cat_scores = {}
    for cat in weights:
        vals = [x["score"] for x in metrics if x["category"] == cat and x["score"] is not None]
        if vals:
            cat_scores[cat] = round(float(np.mean(vals)))
    if cat_scores:
        wsum = sum(weights[c] for c in cat_scores)
        overall = round(sum(cat_scores[c] * weights[c] for c in cat_scores) / wsum)
    else:
        overall = None
    scored = sum(1 for x in metrics if x["score"] is not None)
    confidence = round(scored / len(metrics) * 100) if metrics else 0
    return cat_scores, overall, confidence


def rating_label(overall):
    if overall is None:
        return "Insufficient data"
    if overall >= 78:
        return "Strong profile"
    if overall >= 63:
        return "Positive profile"
    if overall >= 48:
        return "Neutral / Watch"
    return "Cautious"


def risk_label(metrics):
    risk_scores = [x["score"] for x in metrics if x["category"] == "Risk" and x["score"] is not None]
    if not risk_scores:
        return "Unknown"
    avg = np.mean(risk_scores)
    if avg >= 75:
        return "Low"
    if avg >= 55:
        return "Moderate"
    return "Elevated"


def bull_bear(metrics):
    scored = [x for x in metrics if x["score"] is not None]
    bulls = sorted(scored, key=lambda x: -x["score"])[:4]
    bears = sorted(scored, key=lambda x: x["score"])[:4]
    bull_pts = [f"**{b['name']}** at {b['value']} ({b['status']}): {b['why']}" for b in bulls if b["score"] >= 65]
    bear_pts = [f"**{b['name']}** at {b['value']} ({b['status']}): {b['interpretation']}" for b in bears if b["score"] <= 60]
    if not bull_pts:
        bull_pts = ["No metric currently reaches the 'Good' band — the positive case rests on factors outside this quantitative screen."]
    if not bear_pts:
        bear_pts = ["No scored metric currently sits in the weak bands — the main risks are qualitative (execution, regulation, competition)."]
    return bull_pts, bear_pts


# ---------------------------------------------------------------------------
# UI — search panel
# ---------------------------------------------------------------------------

if "selected" not in st.session_state:
    st.session_state.selected = None
if "results" not in st.session_state:
    st.session_state.results = []
if "report_symbol" not in st.session_state:
    st.session_state.report_symbol = None

col_logo, col_badge = st.columns([3, 1])
with col_logo:
    st.title("📊 Alpha Research AI")
    st.caption("Explainable stock & ETF research — India · Asia · UK · US · free data · print-ready reports")
with col_badge:
    st.markdown('<div style="text-align:right;padding-top:1.4rem;"><span class="ara-badge">Free data</span><span class="ara-badge">PDF ready</span></div>', unsafe_allow_html=True)

st.divider()

with st.container(border=True):
    st.subheader("Search a stock or ETF")
    c1, c2, c3 = st.columns([4, 2, 1])
    with c1:
        query = st.text_input("Name or ticker", placeholder="e.g. Reliance, TCS.NS, Shell, SHEL.L, Toyota, 7203.T, VOO",
                              label_visibility="collapsed")
    with c2:
        market = st.selectbox("Market", REGION_OPTIONS, label_visibility="collapsed")
    with c3:
        do_search = st.button("🔍 Search", use_container_width=True, type="primary")

    st.caption("Quick examples: " + " · ".join(f"`{s}`" for s, _ in EXAMPLE_CHIPS))
    st.caption("Indian tickers use **.NS** (NSE) or **.BO** (BSE) — e.g. `RELIANCE.NS`, `INFY.BO`. "
               "London uses **.L**, Tokyo **.T**, Hong Kong **.HK**.")

    if do_search and query.strip():
        with st.spinner("Searching global markets…"):
            res = search_instruments(query.strip())
        if market != "All markets":
            res = [r for r in res if r["region"] == market]
        st.session_state.results = res
        st.session_state.selected = None
        if not res:
            st.warning("No match found. Try the exchange suffix directly — `RELIANCE.NS`, `SHEL.L`, `7203.T`, `9988.HK` — "
                       "or type the exact ticker and generate anyway below.")

    if st.session_state.results:
        options = {f"{r['symbol']}  ·  {r['name']}  ·  {r['type']}  ·  {r['exchange']}  ·  {r['region']}": r
                   for r in st.session_state.results}
        choice = st.radio("Select the exact instrument:", list(options.keys()), index=None)
        if choice:
            st.session_state.selected = options[choice]

    # Direct-ticker escape hatch
    with st.expander("Or enter an exact ticker directly"):
        direct = st.text_input("Exact ticker (with suffix)", placeholder="e.g. HDFCBANK.NS")
        if direct.strip():
            st.session_state.selected = {"symbol": direct.strip().upper(), "name": direct.strip().upper(),
                                         "type": "Unknown", "exchange": "—",
                                         "region": region_for_symbol(direct.strip().upper())}

    gen_col1, gen_col2 = st.columns([5, 1])
    with gen_col2:
        if st.button("📄 Generate report", type="primary", use_container_width=True,
                     disabled=st.session_state.selected is None):
            st.session_state.report_symbol = st.session_state.selected["symbol"]

# ---------------------------------------------------------------------------
# UI — report
# ---------------------------------------------------------------------------

def status_html(status):
    s = str(status).lower()
    cls = ("ara-status-excellent" if "excellent" in s or "positive" in s or "low" in s.split()
           else "ara-status-good" if "good" in s
           else "ara-status-acceptable" if any(k in s for k in ("acceptable", "neutral", "oversold", "moderate"))
           else "ara-status-weak" if any(k in s for k in ("weak", "overbought", "elevated", "high"))
           else "ara-status-na")
    return f'<span class="ara-badge {cls}">{status}</span>'


if st.session_state.report_symbol:
    symbol = st.session_state.report_symbol
    with st.spinner(f"Building report for {symbol} — fetching quote, fundamentals and 1-year history…"):
        data = load_instrument(symbol)

    info = data["info"]
    hist = data["history"]

    if not info and hist is None:
        st.error(f"No data could be retrieved for **{symbol}**. Check the ticker suffix "
                 "(`.NS` NSE · `.BO` BSE · `.L` London · `.T` Tokyo · `.HK` Hong Kong) and try again. "
                 "Free sources occasionally rate-limit; wait a minute and press Refresh.")
    else:
        qtype = (info.get("quoteType") or "").upper()
        is_etf = qtype == "ETF"
        name = info.get("longName") or info.get("shortName") or symbol
        currency = info.get("currency") or ""
        exchange = info.get("fullExchangeName") or info.get("exchange") or "—"
        sector = info.get("sector") or info.get("category") or ("ETF" if is_etf else "—")
        price = info.get("currentPrice") or info.get("regularMarketPrice") or \
                (float(hist["Close"].iloc[-1]) if hist is not None else None)
        region = region_for_symbol(symbol, exchange)

        metrics = build_etf_metrics(info, hist, data) if is_etf else build_stock_metrics(info, hist, currency)
        weights = ETF_WEIGHTS if is_etf else STOCK_WEIGHTS
        cat_scores, overall, confidence = summarise(metrics, weights)
        rating = rating_label(overall)
        risk = risk_label(metrics)
        bull_pts, bear_pts = bull_bear(metrics)
        refreshed = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")

        st.divider()

        # --- Actions ---
        a1, a2, a3 = st.columns([4, 1, 1])
        with a1:
            st.subheader("Generated research report")
            st.caption("Web-based report with latest free data, explainable metrics and print-ready layout.")
        with a2:
            if st.button("🔄 Refresh data", use_container_width=True):
                load_instrument.clear()
                st.rerun()
        with a3:
            st.markdown('<div class="no-print" style="font-size:0.8rem;color:#64748b;padding-top:0.4rem;">'
                        '🖨️ <b>Print / PDF:</b> press <kbd>Ctrl/Cmd&nbsp;+&nbsp;P</kbd> and choose "Save as PDF".</div>',
                        unsafe_allow_html=True)

        # --- Header card ---
        with st.container(border=True):
            h1, h2 = st.columns([3, 2])
            with h1:
                st.markdown(
                    status_html("ETF" if is_etf else "Stock") + status_html(exchange) +
                    status_html(region) + (status_html(currency) if currency else ""),
                    unsafe_allow_html=True)
                st.markdown(f"## {name}")
                st.markdown(f"**{symbol}** · {sector}")
                summary_bits = []
                if overall is not None:
                    summary_bits.append(f"This {'fund' if is_etf else 'company'} scores **{overall}/100** on the weighted framework "
                                        f"({', '.join(f'{c} {w:.0%}' for c, w in weights.items())}).")
                summary_bits.append(f"Data coverage confidence is **{confidence}%** — unscored metrics are excluded rather than guessed.")
                summary_bits.append(f"Quantitative risk reads as **{risk}**.")
                st.markdown(" ".join(summary_bits))
                st.caption(f"Last refreshed: {refreshed}")
            with h2:
                k1, k2 = st.columns(2)
                k1.metric("Overall view", rating)
                k2.metric("Score", f"{overall}/100" if overall is not None else "—")
                k3, k4 = st.columns(2)
                k3.metric("Confidence", f"{confidence}%")
                k4.metric("Latest price", f"{currency} {price:,.2f}" if price else "—")

        # --- Scorecard + price chart ---
        sc1, sc2 = st.columns([1, 2])
        with sc1:
            with st.container(border=True):
                st.markdown("#### 🛡️ Scorecard")
                for cat, sc in cat_scores.items():
                    st.markdown(f"**{cat}** — {sc}/100 &nbsp;<span style='color:#64748b;font-size:0.8rem;'>(weight {weights[cat]:.0%})</span>",
                                unsafe_allow_html=True)
                    st.progress(sc / 100)
                missing = [c for c in weights if c not in cat_scores]
                if missing:
                    st.caption("Not scored (no data): " + ", ".join(missing) + ". Weights re-normalised across scored categories.")
                st.caption("Every score below is traceable to a stated benchmark band — open any metric card for the rule.")
        with sc2:
            with st.container(border=True):
                st.markdown("#### 📈 Price trend (1 year)")
                if hist is not None:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines",
                                             name="Close", line=dict(color="#2563eb", width=2),
                                             fill="tozeroy", fillcolor="rgba(37,99,235,0.08)"))
                    if len(hist) >= 50:
                        fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"].rolling(50).mean(),
                                                 name="50-day MA", line=dict(color="#f97316", width=1.4, dash="dot")))
                    if len(hist) >= 200:
                        fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"].rolling(200).mean(),
                                                 name="200-day MA", line=dict(color="#16a34a", width=1.4, dash="dash")))
                    fig.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10),
                                      legend=dict(orientation="h", y=1.08),
                                      yaxis_title=currency, xaxis_title=None)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Price history unavailable from the free source for this instrument.")

        # --- Tabs ---
        tab_m, tab_c, tab_bb, tab_s = st.tabs(["🧠 Explainable metrics", "📊 Charts", "🐂 Bull / 🐻 Bear", "🗄️ Sources & notes"])

        with tab_m:
            st.markdown("**Explainable metric interpretation guide** — every metric states what it means, why it matters, "
                        "how it is calculated, what good looks like (with reasoning), and how it moves the score.")
            for cat in weights:
                cat_metrics = [x for x in metrics if x["category"] == cat]
                extra = [x for x in metrics if x["category"] not in weights]
                if not cat_metrics:
                    continue
                st.markdown(f"##### {cat}")
                for mt in cat_metrics:
                    header = f"{mt['name']}  —  {mt['value']}  ·  {mt['status']}" + \
                             (f"  ·  score {mt['score']}/100" if mt["score"] is not None else "")
                    with st.expander(header):
                        st.markdown(status_html(mt["status"]), unsafe_allow_html=True)
                        st.markdown(f"**What it means** — {mt['definition']}")
                        st.markdown(f"**Why it matters** — {mt['why']}")
                        st.markdown(f"**How it is calculated** — {mt['calculation']}")
                        st.markdown("**What good looks like (benchmark bands)**")
                        bcols = st.columns(len(mt["bands"]))
                        for bc, (label, rng) in zip(bcols, mt["bands"]):
                            bc.markdown(f"<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;"
                                        f"padding:8px 10px;'><div style='font-size:0.7rem;color:#64748b;text-transform:uppercase;'>{label}</div>"
                                        f"<div style='font-size:0.85rem;font-weight:600;'>{rng}</div></div>",
                                        unsafe_allow_html=True)
                        st.markdown(f"**Sector / category context** — {mt['benchmark']}")
                        st.markdown(f"**Interpretation of the current value** — {mt['interpretation']}")
                        st.markdown(f"**Impact on the score** — {mt['impact']}")

        with tab_c:
            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown("#### Score breakdown")
                if cat_scores:
                    df = pd.DataFrame({"Category": list(cat_scores.keys()), "Score": list(cat_scores.values())})
                    fig = px.bar(df, x="Category", y="Score", range_y=[0, 100], text="Score",
                                 color_discrete_sequence=["#111827"])
                    fig.update_traces(textposition="outside")
                    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig, use_container_width=True)
            with cc2:
                if is_etf and data.get("sector_weights"):
                    st.markdown("#### Sector exposure")
                    sw = data["sector_weights"]
                    df = pd.DataFrame({"Sector": [k.replace("_", " ").title() for k in sw],
                                       "Weight": [v * 100 for v in sw.values()]}).sort_values("Weight", ascending=False)
                    fig = px.pie(df, names="Sector", values="Weight", hole=0.45)
                    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig, use_container_width=True)
                elif hist is not None:
                    st.markdown("#### Monthly return profile")
                    monthly = hist["Close"].resample("ME").last().pct_change().dropna() * 100
                    df = pd.DataFrame({"Month": monthly.index.strftime("%b %y"), "Return %": monthly.values})
                    fig = px.bar(df, x="Month", y="Return %",
                                 color=df["Return %"] > 0, color_discrete_map={True: "#16a34a", False: "#dc2626"})
                    fig.update_layout(height=340, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig, use_container_width=True)
            if is_etf and data.get("top_holdings") is not None:
                st.markdown("#### Top holdings")
                st.dataframe(data["top_holdings"], use_container_width=True)

        with tab_bb:
            b1, b2 = st.columns(2)
            with b1:
                st.success("#### 🐂 Bull case")
                for p in bull_pts:
                    st.markdown(f"- {p}")
            with b2:
                st.warning("#### 🐻 Bear case")
                for p in bear_pts:
                    st.markdown(f"- {p}")
            st.caption("Bull/bear points are generated transparently from the highest- and lowest-scoring metrics above — "
                       "no hidden judgement is applied.")

        with tab_s:
            s1, s2, s3 = st.columns(3)
            with s1:
                with st.container(border=True):
                    st.markdown("**Quote & fundamentals source**")
                    st.caption("Yahoo Finance public data via the open-source `yfinance` library (free; unofficial).")
            with s2:
                with st.container(border=True):
                    st.markdown("**Price history source**")
                    st.caption("Yahoo Finance adjusted daily prices, 1-year window.")
            with s3:
                with st.container(border=True):
                    st.markdown("**Latest refresh**")
                    st.caption(refreshed + " · cache TTL 15 min")
            if data["errors"]:
                st.warning("Partial data issues: " + " | ".join(data["errors"]))
            st.info("⚠️ Free web data can be delayed (often 15–20 min for LSE/NSE), rate-limited or incomplete for some "
                    "exchanges. Some fields (e.g. ETF expense ratios on non-US listings) are not always published. "
                    "This report is an educational research aid, **not investment advice**. Verify key figures against "
                    "the company's filings or the fund provider's factsheet before acting.")

else:
    with st.container(border=True):
        st.markdown("### 📄 No report generated yet")
        st.markdown("Search for a stock or ETF above, select the exact result, then click **Generate report**. "
                    "Try `RELIANCE.NS`, `TCS.NS`, `NIFTYBEES.NS`, `SHEL.L`, `VUSA.L`, `7203.T`, `9988.HK`, `MSFT` or `VOO`.")
