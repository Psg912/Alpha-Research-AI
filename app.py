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

import json
import math
import os
import re
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


# --- Sector-adjusted benchmark bands -------------------------------------
# Base thresholds are market-wide norms; SECTOR_OVERRIDES shift them to the
# sector's own typical range (yfinance sector names). Every override is a
# deliberate, documented judgement — e.g. banks are *supposed* to be
# leveraged, tech is *supposed* to have fat gross margins.

BASE_THRESHOLDS = {
    "pe": (15, 25, 40), "fpe": (14, 22, 35), "ps": (2, 4, 8), "pb": (1.5, 3, 6),
    "ev_ebitda": (10, 15, 25), "dy": (4, 2.5, 1), "rev_g": (15, 8, 3),
    "gm": (50, 35, 20), "om": (25, 15, 8), "nm": (18, 10, 5),
    "roe": (20, 13, 8), "de": (40, 80, 150),
}

SECTOR_OVERRIDES = {
    "Technology": {"pe": (20, 32, 48), "fpe": (18, 28, 42), "ps": (4, 8, 14), "pb": (4, 8, 15),
                   "ev_ebitda": (14, 20, 30), "dy": (1.5, 0.8, 0.3), "rev_g": (18, 10, 4),
                   "gm": (65, 50, 35), "om": (30, 20, 12), "nm": (22, 14, 8),
                   "roe": (25, 16, 10), "de": (30, 60, 120)},
    "Financial Services": {"pe": (10, 16, 25), "fpe": (9, 14, 22), "pb": (1.0, 1.8, 3.0),
                           "dy": (4.5, 3, 1.5), "rev_g": (12, 7, 3), "om": (35, 25, 15),
                           "nm": (25, 16, 10), "roe": (15, 11, 7), "de": (200, 400, 800)},
    "Energy": {"pe": (8, 14, 22), "fpe": (7, 12, 20), "ps": (0.8, 1.5, 3), "pb": (1.0, 1.8, 3.0),
               "ev_ebitda": (5, 8, 12), "dy": (5, 3.5, 2), "rev_g": (10, 4, 0),
               "gm": (30, 18, 8), "om": (18, 10, 5), "nm": (10, 6, 3),
               "roe": (18, 12, 7), "de": (45, 90, 160)},
    "Utilities": {"pe": (12, 18, 28), "fpe": (11, 16, 25), "ps": (1.5, 2.5, 4), "pb": (1.2, 2.0, 3.2),
                  "ev_ebitda": (9, 12, 16), "dy": (5, 3.5, 2), "rev_g": (6, 3, 1),
                  "om": (20, 14, 8), "nm": (12, 8, 4), "roe": (12, 9, 6), "de": (100, 180, 300)},
    "Healthcare": {"pe": (18, 28, 42), "fpe": (16, 25, 38), "ev_ebitda": (12, 17, 26),
                   "gm": (60, 45, 30), "om": (22, 14, 8), "nm": (15, 9, 5)},
    "Consumer Defensive": {"pe": (18, 26, 38), "fpe": (16, 23, 34), "ps": (1, 2, 4),
                           "ev_ebitda": (11, 15, 22), "dy": (3.5, 2.5, 1.2), "rev_g": (8, 5, 2),
                           "gm": (35, 25, 15), "om": (14, 9, 5), "nm": (9, 6, 3),
                           "roe": (25, 16, 10), "de": (60, 110, 190)},
    "Consumer Cyclical": {"ps": (0.8, 1.8, 3.5), "gm": (35, 25, 15), "om": (12, 8, 4), "nm": (8, 5, 2.5)},
    "Basic Materials": {"pe": (9, 15, 24), "fpe": (8, 13, 21), "ps": (1, 2, 3.5), "pb": (1.2, 2.2, 3.8),
                        "ev_ebitda": (6, 9, 13), "gm": (28, 18, 10), "om": (18, 11, 6), "nm": (12, 7, 3)},
    "Industrials": {"gm": (32, 22, 14), "om": (15, 10, 6), "nm": (10, 6, 3), "de": (60, 110, 190)},
    "Communication Services": {"de": (70, 120, 200)},
    "Real Estate": {"pb": (0.9, 1.5, 2.5), "dy": (5, 3.5, 2), "de": (90, 160, 280)},
}


def get_bands(key, sector):
    """Return (thresholds, sector_name_if_adjusted_else_None) for a metric."""
    ov = SECTOR_OVERRIDES.get(sector or "", {})
    if key in ov:
        return ov[key], sector
    return BASE_THRESHOLDS[key], None


def band_ranges(th, higher, kind):
    """Generate the four band cards' display ranges from numeric thresholds."""
    e, g, a = th
    suf = "×" if kind == "x" else "%" if kind == "pct" else ""
    b = lambda v: f"{v:g}{suf}"
    if higher:
        return [("Excellent", f"≥ {b(e)}"), ("Good", f"{b(g)}–{b(e)}"),
                ("Acceptable", f"{b(a)}–{b(g)}"), ("Weak", f"< {b(a)}")]
    return [("Excellent", f"≤ {b(e)}"), ("Good", f"{b(e)}–{b(g)}"),
            ("Acceptable", f"{b(g)}–{b(a)}"), ("Weak", f"> {b(a)}")]


def metric(category, name, value_display, score, status, definition, why,
           calculation, bands, benchmark, interpretation, impact,
           raw=None, thresholds=None, higher=True, kind="num", adjusted=None):
    return dict(category=category, name=name, value=value_display, score=score,
                status=status, definition=definition, why=why, calculation=calculation,
                bands=bands, benchmark=benchmark, interpretation=interpretation, impact=impact,
                raw=raw, thresholds=thresholds, higher=higher, kind=kind, adjusted=adjusted)


def band_of(raw, thresholds, higher):
    """Return the band index (0=Excellent … 3=Weak) the raw value falls into."""
    e, g, a = thresholds
    if higher:
        return 0 if raw >= e else 1 if raw >= g else 2 if raw >= a else 3
    return 0 if raw <= e else 1 if raw <= g else 2 if raw <= a else 3


def position_summary(mt):
    """Plain-English statement of where the current value sits vs the numeric
    band cutoffs, and the exact distance to the next better band."""
    raw, th = mt.get("raw"), mt.get("thresholds")
    if raw is None or th is None:
        return None
    higher, kind = mt.get("higher", True), mt.get("kind", "num")
    labels = [b[0] for b in mt["bands"]]
    idx = band_of(raw, th, higher)
    f = lambda v: fmt(v, kind)
    sign = "≥" if higher else "≤"
    cuts = list(zip(labels[:3], th))
    parts = [f"Current value **{f(raw)}** falls in the **{labels[idx]}** band."]
    if idx == 0:
        parts.append("This is the strongest band for this metric — "
                     f"it clears the {labels[0]} cutoff of {sign} {f(th[0])} with room to spare.")
    else:
        nxt_label, nxt_cut = cuts[idx - 1]
        gap = abs(nxt_cut - raw)
        move = "fall" if higher is False else "rise"
        parts.append(f"To reach **{nxt_label}** ({sign} {f(nxt_cut)}), the value would need to {move} by **{f(gap)}**.")
        if idx >= 2:
            top_label, top_cut = cuts[0]
            parts.append(f"The {top_label} cutoff of {sign} {f(top_cut)} is **{f(abs(top_cut - raw))}** away.")
    return " ".join(parts)


def interp(name, value_display, status, better_when):
    if status == "N/A":
        return (f"{name} was not published by the free data source for this instrument. "
                f"It is excluded from scoring and lowers the report confidence instead of distorting the result.")
    return (f"The current value of {value_display} sits in the '{status}' band. "
            f"For this metric, {better_when}.")


# ----------------------------- Stock metrics ------------------------------

def build_stock_metrics(info: dict, hist, currency: str, sector: str = None, adjust: bool = True) -> list[dict]:
    bsec = sector if adjust else None  # None => market-wide base bands
    m = []
    close = hist["Close"] if hist is not None and "Close" in hist else None
    price = info.get("currentPrice") or info.get("regularMarketPrice") or \
            (float(close.iloc[-1]) if close is not None and len(close) else None)

    # --- Valuation ---
    pe = info.get("trailingPE")
    th_, adj_ = get_bands("pe", bsec)
    s, st_ = banded(pe, th_, higher_is_better=False)
    m.append(metric("Valuation", "P/E Ratio (trailing)", fmt(pe, "x"), s, st_,
        "Price divided by the last twelve months of earnings per share. It tells you how many years of current profit you are paying for.",
        "It is the fastest sanity check on whether a share price is demanding: a high P/E means the market has already priced in strong future growth.",
        "Share price ÷ trailing 12-month EPS, as reported by Yahoo Finance.",
        band_ranges(th_, False, "x"),
        "Compare against the sector median — Indian IT and US tech typically trade at 25–35×, while UK energy and banks often trade below 12×. A 'high' P/E in one sector can be normal in another.",
        interp("P/E", fmt(pe, "x"), st_, "lower is generally better because you pay less per unit of profit; but very low P/E can also signal the market expects earnings to fall"),
        "Feeds the Valuation category (20% of overall stock score).", raw=pe, thresholds=th_, higher=False, kind="x", adjusted=adj_))

    fpe = info.get("forwardPE")
    th_, adj_ = get_bands("fpe", bsec)
    s, st_ = banded(fpe, th_, higher_is_better=False)
    m.append(metric("Valuation", "Forward P/E", fmt(fpe, "x"), s, st_,
        "Price divided by the consensus analyst forecast of next year's earnings.",
        "If forward P/E is well below trailing P/E, analysts expect earnings to grow — the stock is cheaper than it looks on trailing numbers.",
        "Share price ÷ consensus forward EPS estimate.",
        band_ranges(th_, False, "x"),
        "Read alongside trailing P/E: forward < trailing implies expected growth; forward > trailing implies expected decline.",
        interp("Forward P/E", fmt(fpe, "x"), st_, "lower is better, provided the forecasts are credible"),
        "Feeds the Valuation category (20% weight).", raw=fpe, thresholds=th_, higher=False, kind="x", adjusted=adj_))

    peg = info.get("trailingPegRatio") or info.get("pegRatio")
    s, st_ = banded(peg, (1.0, 1.5, 2.5), higher_is_better=False)
    m.append(metric("Valuation", "PEG Ratio", fmt(peg, "x"), s, st_,
        "P/E divided by expected earnings growth rate. It adjusts the P/E for how fast profits are growing.",
        "A P/E of 30 is expensive for a company growing 5% a year but cheap for one growing 40%. PEG normalises that.",
        "Trailing P/E ÷ expected annual EPS growth (%).",
        [("Excellent", "≤ 1.0"), ("Good", "1.0–1.5"), ("Acceptable", "1.5–2.5"), ("Weak", "> 2.5")],
        "Peter Lynch's classic rule of thumb: PEG ≈ 1 is fair value. Growth stocks worldwide often trade at 1.5–2.0.",
        interp("PEG", fmt(peg, "x"), st_, "lower is better — you pay less per unit of growth"),
        "Feeds the Valuation category (20% weight).", raw=peg, thresholds=(1.0, 1.5, 2.5), higher=False, kind="x"))

    ps = info.get("priceToSalesTrailing12Months")
    th_, adj_ = get_bands("ps", bsec)
    s, st_ = banded(ps, th_, higher_is_better=False)
    m.append(metric("Valuation", "Price-to-Sales", fmt(ps, "x"), s, st_,
        "Market capitalisation divided by trailing twelve-month revenue.",
        "Useful when earnings are volatile or negative — revenue is harder to manipulate than profit.",
        "Market cap ÷ trailing 12-month revenue.",
        band_ranges(th_, False, "x"),
        "Software businesses sustain higher P/S (5–10×) than retailers or refiners (0.3–1×) because of margin differences. Always compare within sector.",
        interp("P/S", fmt(ps, "x"), st_, "lower is better for the same margin profile"),
        "Feeds the Valuation category (20% weight).", raw=ps, thresholds=th_, higher=False, kind="x", adjusted=adj_))

    pb = info.get("priceToBook")
    th_, adj_ = get_bands("pb", bsec)
    s, st_ = banded(pb, th_, higher_is_better=False)
    m.append(metric("Valuation", "Price-to-Book", fmt(pb, "x"), s, st_,
        "Share price divided by book value (net assets) per share.",
        "Especially relevant for banks, insurers and asset-heavy businesses where the balance sheet drives value.",
        "Market cap ÷ shareholders' equity.",
        band_ranges(th_, False, "x"),
        "Indian private banks often trade at 2–4× book when ROE is high; UK banks nearer 0.5–1×. High P/B is justified only by high return on equity.",
        interp("P/B", fmt(pb, "x"), st_, "lower is better unless a high return on equity justifies the premium"),
        "Feeds the Valuation category (20% weight).", raw=pb, thresholds=th_, higher=False, kind="x", adjusted=adj_))

    ev_ebitda = info.get("enterpriseToEbitda")
    th_, adj_ = get_bands("ev_ebitda", bsec)
    s, st_ = banded(ev_ebitda, th_, higher_is_better=False)
    m.append(metric("Valuation", "EV / EBITDA", fmt(ev_ebitda, "x"), s, st_,
        "Enterprise value (market cap + net debt) divided by earnings before interest, tax, depreciation and amortisation.",
        "It compares companies regardless of how they are financed — the preferred multiple for takeovers and cross-border comparisons.",
        "(Market cap + total debt − cash) ÷ trailing EBITDA.",
        band_ranges(th_, False, "x"),
        "Historical market averages sit around 10–12×. Capital-light software runs higher; commodity producers lower.",
        interp("EV/EBITDA", fmt(ev_ebitda, "x"), st_, "lower is better on a like-for-like basis"),
        "Feeds the Valuation category (20% weight).", raw=ev_ebitda, thresholds=th_, higher=False, kind="x", adjusted=adj_))

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
        "Feeds the Valuation category (20% weight).", raw=fcf_yield, thresholds=(6, 4, 2), higher=True, kind="pct"))

    dy = dividend_yield_pct(info)
    th_, adj_ = get_bands("dy", bsec)
    s, st_ = banded(dy, th_, higher_is_better=True)
    m.append(metric("Valuation", "Dividend Yield", fmt(dy, "pct"), s, st_,
        "Annual dividends per share divided by the share price.",
        "It is the cash return you receive while holding, independent of price movement.",
        "Trailing annual dividend rate ÷ current price × 100 (computed directly to avoid provider unit inconsistencies).",
        band_ranges(th_, True, "pct"),
        "FTSE 100 averages ~3.5–4%; Nifty 50 ~1.2–1.5%; S&P 500 ~1.3%. A yield far above the market average can be a distress signal, so check payout sustainability.",
        interp("Dividend yield", fmt(dy, "pct"), st_, "higher is better for income, provided the payout is covered by free cash flow"),
        "Feeds the Valuation category (20% weight). Growth companies legitimately score low here — read together with growth metrics.", raw=dy, thresholds=th_, higher=True, kind="pct", adjusted=adj_))

    # --- Growth ---
    rev_g = info.get("revenueGrowth")
    rev_g = rev_g * 100 if rev_g is not None else None
    th_, adj_ = get_bands("rev_g", bsec)
    s, st_ = banded(rev_g, th_, higher_is_better=True)
    m.append(metric("Growth", "Revenue Growth (yoy)", fmt(rev_g, "pct"), s, st_,
        "Year-on-year change in total revenue for the most recent reported period.",
        "Revenue growth is the raw material of all future profit growth — margins can only be squeezed so far.",
        "(Latest period revenue ÷ same period last year − 1) × 100.",
        band_ranges(th_, True, "pct"),
        "Compare against nominal GDP growth of the home market (~10–11% for India, ~3–4% for UK/US). Growing slower than nominal GDP means losing share of the economy.",
        interp("Revenue growth", fmt(rev_g, "pct"), st_, "higher is better, especially if achieved without margin erosion"),
        "Feeds the Growth category (20% weight).", raw=rev_g, thresholds=th_, higher=True, kind="pct", adjusted=adj_))

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
        "Feeds the Growth category (20% weight).", raw=eps_g, thresholds=(18, 10, 4), higher=True, kind="pct"))

    # --- Profitability ---
    gm = info.get("grossMargins")
    gm = gm * 100 if gm is not None else None
    th_, adj_ = get_bands("gm", bsec)
    s, st_ = banded(gm, th_, higher_is_better=True)
    m.append(metric("Profitability", "Gross Margin", fmt(gm, "pct"), s, st_,
        "Revenue minus cost of goods sold, as a percentage of revenue.",
        "It measures pricing power. High gross margins give a company room to invest, absorb shocks and out-spend competitors.",
        "Gross profit ÷ revenue × 100 (trailing twelve months).",
        band_ranges(th_, True, "pct"),
        "Software: 70–90%. Consumer brands: 40–60%. Autos: 15–25%. Refining/commodities: <15%. Judge within the sector's normal range.",
        interp("Gross margin", fmt(gm, "pct"), st_, "higher is better — it signals durable pricing power"),
        "Feeds the Profitability category (20% weight).", raw=gm, thresholds=th_, higher=True, kind="pct", adjusted=adj_))

    om = info.get("operatingMargins")
    om = om * 100 if om is not None else None
    th_, adj_ = get_bands("om", bsec)
    s, st_ = banded(om, th_, higher_is_better=True)
    m.append(metric("Profitability", "Operating Margin", fmt(om, "pct"), s, st_,
        "Operating profit (before interest and tax) as a percentage of revenue.",
        "It shows how efficiently the whole operating model converts sales into profit, after all running costs.",
        "Operating income ÷ revenue × 100 (trailing twelve months).",
        band_ranges(th_, True, "pct"),
        "A stable or rising operating margin over several years is a stronger signal than a single high reading.",
        interp("Operating margin", fmt(om, "pct"), st_, "higher and more stable is better"),
        "Feeds the Profitability category (20% weight).", raw=om, thresholds=th_, higher=True, kind="pct", adjusted=adj_))

    nm = info.get("profitMargins")
    nm = nm * 100 if nm is not None else None
    th_, adj_ = get_bands("nm", bsec)
    s, st_ = banded(nm, th_, higher_is_better=True)
    m.append(metric("Profitability", "Net Margin", fmt(nm, "pct"), s, st_,
        "Bottom-line profit as a percentage of revenue, after everything including tax and interest.",
        "The final measure of how much of each rupee, pound or dollar of sales becomes shareholder profit.",
        "Net income ÷ revenue × 100 (trailing twelve months).",
        band_ranges(th_, True, "pct"),
        "Global large-cap average is roughly 8–11%. Persistently above 15% usually indicates a genuine moat.",
        interp("Net margin", fmt(nm, "pct"), st_, "higher is better"),
        "Feeds the Profitability category (20% weight).", raw=nm, thresholds=th_, higher=True, kind="pct", adjusted=adj_))

    roe = info.get("returnOnEquity")
    roe = roe * 100 if roe is not None else None
    th_, adj_ = get_bands("roe", bsec)
    s, st_ = banded(roe, th_, higher_is_better=True)
    m.append(metric("Profitability", "Return on Equity", fmt(roe, "pct"), s, st_,
        "Net income divided by shareholders' equity — the profit generated per unit of owners' capital.",
        "Warren Buffett's favourite quality test: businesses that compound at high ROE without excess debt create the most long-term value.",
        "Net income ÷ average shareholders' equity × 100.",
        band_ranges(th_, True, "pct"),
        "Check ROE against debt-to-equity: an ROE inflated by heavy leverage is lower quality than the same ROE achieved with a clean balance sheet.",
        interp("ROE", fmt(roe, "pct"), st_, "higher is better when it is not manufactured through leverage"),
        "Feeds the Profitability category (20% weight).", raw=roe, thresholds=th_, higher=True, kind="pct", adjusted=adj_))

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
        "Feeds the Profitability category (20% weight).", raw=roa, thresholds=(10, 6, 3), higher=True, kind="pct"))

    # --- Financial Health ---
    de = info.get("debtToEquity")  # yfinance reports this as a percentage figure
    th_, adj_ = get_bands("de", bsec)
    s, st_ = banded(de, th_, higher_is_better=False)
    m.append(metric("Financial Health", "Debt-to-Equity", fmt(de, "pct"), s, st_,
        "Total debt as a percentage of shareholders' equity.",
        "Leverage amplifies both good and bad outcomes. High debt turns a rough year into an existential one.",
        "Total debt ÷ shareholders' equity × 100 (as reported by the provider).",
        band_ranges(th_, False, "pct"),
        "Capital-intensive sectors (utilities, telecoms, real estate) run structurally higher leverage; banks are excluded from this rule entirely as leverage is their business model.",
        interp("Debt-to-equity", fmt(de, "pct"), st_, "lower is better — it buys resilience in downturns"),
        "Feeds the Financial Health category (15% weight).", raw=de, thresholds=th_, higher=False, kind="pct", adjusted=adj_))

    cr = info.get("currentRatio")
    s, st_ = banded(cr, (2.0, 1.5, 1.0), higher_is_better=True)
    m.append(metric("Financial Health", "Current Ratio", fmt(cr, "x"), s, st_,
        "Current assets divided by current liabilities — can the company pay its bills over the next 12 months?",
        "The classic short-term solvency test. Below 1.0 means near-term obligations exceed near-term resources.",
        "Current assets ÷ current liabilities.",
        [("Excellent", "≥ 2.0"), ("Good", "1.5–2.0"), ("Acceptable", "1.0–1.5"), ("Weak", "< 1.0")],
        "Fast-turnover retailers can safely run below 1.0 because inventory converts to cash quickly; manufacturers should not.",
        interp("Current ratio", fmt(cr, "x"), st_, "higher is safer, though far above 3 can indicate lazy capital"),
        "Feeds the Financial Health category (15% weight).", raw=cr, thresholds=(2.0, 1.5, 1.0), higher=True, kind="x"))

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
        "Feeds the Financial Health category (15% weight).", raw=cash_pct, thresholds=(15, 8, 3), higher=True, kind="pct"))

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
        "Feeds the Risk category (15% weight).", raw=beta, thresholds=(0.9, 1.15, 1.5), higher=False, kind="x"))

    vol = annualised_volatility(close)
    s, st_ = banded(vol, (20, 30, 45), higher_is_better=False)
    m.append(metric("Risk", "Volatility (1y, annualised)", fmt(vol, "pct"), s, st_,
        "The standard deviation of daily returns over the last year, scaled to an annual figure.",
        "It tells you how bumpy the ride has actually been — a direct, backward-looking measure of price risk.",
        "Std-dev of daily % returns × √252, computed from the fetched 1-year price history.",
        [("Excellent", "≤ 20%"), ("Good", "20–30%"), ("Acceptable", "30–45%"), ("Weak", "> 45%")],
        "Large-cap indices typically run 12–20% annualised; individual large caps 20–35%; small caps and turnarounds higher.",
        interp("Volatility", fmt(vol, "pct"), st_, "lower is calmer; make sure you can hold through the swings the number implies"),
        "Feeds the Risk category (15% weight).", raw=vol, thresholds=(20, 30, 45), higher=False, kind="pct"))

    mdd = max_drawdown(close)
    s, st_ = banded(mdd, (-15, -25, -40), higher_is_better=True)
    m.append(metric("Risk", "Max Drawdown (1y)", fmt(mdd, "pct"), s, st_,
        "The largest peak-to-trough fall in the share price over the past year.",
        "This is the loss you would have felt buying at the worst moment — a visceral, real-world risk measure.",
        "Minimum of (price ÷ running peak − 1) over the 1-year history.",
        [("Excellent", "shallower than −15%"), ("Good", "−15% to −25%"), ("Acceptable", "−25% to −40%"), ("Weak", "deeper than −40%")],
        "Even world-class companies routinely draw down 20–30% within a year. What matters is whether the business recovered and why it fell.",
        interp("Max drawdown", fmt(mdd, "pct"), st_, "shallower is better"),
        "Feeds the Risk category (15% weight).", raw=mdd, thresholds=(-15, -25, -40), higher=True, kind="pct"))

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
        "Feeds the Technical Trend category (10% weight).", raw=above50, thresholds=(3, 0, -5), higher=True, kind="pct"))

    above200 = (price / ma200 - 1) * 100 if price and ma200 else None
    s, st_ = banded(above200, (5, 0, -8), higher_is_better=True)
    m.append(metric("Technical Trend", "Price vs 200-day MA", fmt(above200, "pct"), s, st_,
        "How far the current price sits above or below its 200-day moving average — the classic long-term trend measure.",
        "Institutions widely use the 200-day MA as the bull/bear dividing line for an individual security.",
        f"(Current price ÷ 200-day average − 1) × 100. Current 200-day MA: {fmt(ma200, 'money', currency)}.",
        [("Excellent", "> +5%"), ("Good", "0 to +5%"), ("Acceptable", "0 to −8%"), ("Weak", "< −8%")],
        "Golden cross (50-day rising above 200-day) is a well-known bullish structure; a death cross is the reverse.",
        interp("Price vs 200-day MA", fmt(above200, "pct"), st_, "above the line = long-term uptrend intact"),
        "Feeds the Technical Trend category (10% weight).", raw=above200, thresholds=(5, 0, -8), higher=True, kind="pct"))

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
        "Feeds the Cost category (20% of overall ETF score).", raw=er, thresholds=(0.20, 0.50, 1.00), higher=False, kind="pct"))

    aum = info.get("totalAssets")
    s, st_ = banded(aum, (1e9, 2e8, 5e7), higher_is_better=True)
    m.append(metric("Liquidity", "Assets Under Management", fmt(aum, "compact"), s, st_,
        "The total market value of everything the fund holds.",
        "Small funds risk closure (forcing a taxable exit at a time not of your choosing) and tend to have wider trading spreads.",
        "Published fund AUM via Yahoo Finance, in the fund's base currency.",
        [("Excellent", "≥ 1B"), ("Good", "200M–1B"), ("Acceptable", "50–200M"), ("Weak", "< 50M")],
        "Funds below ~50M in AUM run a materially higher closure risk; providers regularly cull them.",
        interp("AUM", fmt(aum, "compact"), st_, "bigger is safer and usually cheaper to trade"),
        "Feeds the Liquidity category (15% weight).", raw=aum, thresholds=(1e9, 2e8, 5e7), higher=True, kind="compact"))

    adv = info.get("averageVolume")
    s, st_ = banded(adv, (500_000, 100_000, 20_000), higher_is_better=True)
    m.append(metric("Liquidity", "Average Daily Volume", fmt(adv, "compact"), s, st_,
        "The average number of fund units traded per day.",
        "Higher volume means tighter bid-ask spreads — you lose less money simply entering and exiting.",
        "Trailing average daily unit volume via Yahoo Finance.",
        [("Excellent", "≥ 500K"), ("Good", "100–500K"), ("Acceptable", "20–100K"), ("Weak", "< 20K")],
        "For ETFs, underlying-basket liquidity matters more than on-screen volume, but thin on-screen volume still widens spreads for retail-size orders.",
        interp("Average volume", fmt(adv, "compact"), st_, "higher is better for cheap execution"),
        "Feeds the Liquidity category (15% weight).", raw=adv, thresholds=(500_000, 100_000, 20_000), higher=True, kind="compact"))

    r1y = period_return(close)
    s, st_ = banded(r1y, (15, 8, 0), higher_is_better=True)
    m.append(metric("Performance", "1-Year Return", fmt(r1y, "pct"), s, st_,
        "Total price change over the last twelve months (dividends reinvested where the data source adjusts for them).",
        "Recent performance in context — useful, but the least predictive number in this report on its own.",
        "(Latest close ÷ close 12 months ago − 1) × 100, from adjusted price history.",
        [("Excellent", "≥ 15%"), ("Good", "8–15%"), ("Acceptable", "0–8%"), ("Weak", "negative")],
        "Always compare with the fund's own benchmark index, not with cash. A −5% year when the index fell 8% is good management.",
        interp("1-year return", fmt(r1y, "pct"), st_, "higher is better, but one year proves little — check 3–5 year consistency"),
        "Feeds the Performance category (20% weight).", raw=r1y, thresholds=(15, 8, 0), higher=True, kind="pct"))

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
        "Feeds the Performance category (20% weight).", raw=r3y, thresholds=(12, 7, 3), higher=True, kind="pct"))

    vol = annualised_volatility(close)
    s, st_ = banded(vol, (15, 22, 32), higher_is_better=False)
    m.append(metric("Risk", "Volatility (1y, annualised)", fmt(vol, "pct"), s, st_,
        "Standard deviation of the fund's daily returns over the last year, annualised.",
        "It measures how bumpy the fund's ride is — critical for judging whether you can hold it through a downturn.",
        "Std-dev of daily % returns × √252, computed from the fetched 1-year history.",
        [("Excellent", "≤ 15%"), ("Good", "15–22%"), ("Acceptable", "22–32%"), ("Weak", "> 32%")],
        "Broad developed-market equity ETFs: 12–18%. Single-country emerging or thematic tech ETFs: 20–35%. Bond ETFs: 3–8%.",
        interp("Volatility", fmt(vol, "pct"), st_, "lower is calmer for the same return"),
        "Feeds the Risk category (20% weight).", raw=vol, thresholds=(15, 22, 32), higher=False, kind="pct"))

    shp = sharpe_ratio(close)
    s, st_ = banded(shp, (1.0, 0.6, 0.2), higher_is_better=True)
    m.append(metric("Risk", "Sharpe Ratio (1y)", fmt(shp), s, st_,
        "Return earned above the risk-free rate, per unit of volatility taken.",
        "It answers the real question: were you paid enough for the risk? Two funds with equal returns are not equal if one was twice as volatile.",
        f"(Annualised return − {RISK_FREE_RATE:.0%} assumed risk-free rate) ÷ annualised volatility, from the 1-year history.",
        [("Excellent", "≥ 1.0"), ("Good", "0.6–1.0"), ("Acceptable", "0.2–0.6"), ("Weak", "< 0.2")],
        "Above 1.0 over a full cycle is genuinely good; above 2.0 is rare and usually regime-dependent.",
        interp("Sharpe", fmt(shp), st_, "higher is better — more return per unit of risk"),
        "Feeds the Risk category (20% weight).", raw=shp, thresholds=(1.0, 0.6, 0.2), higher=True, kind="num"))

    mdd = max_drawdown(close)
    s, st_ = banded(mdd, (-10, -18, -30), higher_is_better=True)
    m.append(metric("Risk", "Max Drawdown (1y)", fmt(mdd, "pct"), s, st_,
        "The largest peak-to-trough fall over the past year.",
        "The most honest risk number: it is the loss an unlucky buyer actually experienced.",
        "Minimum of (price ÷ running peak − 1) over the 1-year history.",
        [("Excellent", "shallower than −10%"), ("Good", "−10% to −18%"), ("Acceptable", "−18% to −30%"), ("Weak", "deeper than −30%")],
        "Equity index funds routinely see −10% to −20% intra-year drawdowns even in positive years.",
        interp("Max drawdown", fmt(mdd, "pct"), st_, "shallower is better"),
        "Feeds the Risk category (20% weight).", raw=mdd, thresholds=(-10, -18, -30), higher=True, kind="pct"))

    b3 = info.get("beta3Year") or info.get("beta")
    s, st_ = banded(b3, (0.95, 1.1, 1.35), higher_is_better=False)
    m.append(metric("Risk", "Beta (3y)", fmt(b3, "x"), s, st_,
        "The fund's sensitivity to its reference market index.",
        "Tells you whether this fund amplifies or dampens market swings in your portfolio.",
        "Regression of fund returns against the market index over 3 years, via Yahoo Finance.",
        [("Excellent (defensive)", "≤ 0.95"), ("Good", "0.95–1.1"), ("Acceptable", "1.1–1.35"), ("Weak (amplified)", "> 1.35")],
        "A plain index tracker should sit very close to 1.0 by construction; large deviation signals leverage, concentration or a different exposure than the name implies.",
        interp("Beta", fmt(b3, "x"), st_, "closer to 1.0 is expected for trackers; below 1.0 is defensive"),
        "Feeds the Risk category (20% weight).", raw=b3, thresholds=(0.95, 1.1, 1.35), higher=False, kind="x"))

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
        "Feeds the Holdings Quality category (15% weight).", raw=top_sector_pct, thresholds=(25, 35, 50), higher=False, kind="pct"))

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
        "Feeds the Holdings Quality category (15% weight).", raw=top10, thresholds=(25, 40, 60), higher=False, kind="pct"))

    yld = dividend_yield_pct(info)
    s, st_ = banded(yld, (3, 1.8, 0.8), higher_is_better=True)
    m.append(metric("Performance", "Distribution Yield", fmt(yld, "pct"), s, st_,
        "The income the fund pays out annually as a percentage of its price.",
        "For income investors this is the point of the fund; for growth investors it indicates the style tilt.",
        "Trailing 12-month distributions ÷ current price × 100.",
        [("Excellent", "≥ 3%"), ("Good", "1.8–3%"), ("Acceptable", "0.8–1.8%"), ("Weak / n.a.", "< 0.8% (accumulating funds legitimately show ~0)")],
        "Accumulating (Acc) share classes reinvest internally and show near-zero yield by design — check the share class before judging.",
        interp("Distribution yield", fmt(yld, "pct"), st_, "higher is better for income mandates only"),
        "Feeds the Performance category (20% weight).", raw=yld, thresholds=(3, 1.8, 0.8), higher=True, kind="pct"))

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
    bull_pts = [f"**{b['name']}** at {b['value']}: {plain_words(b) or b['why']}" for b in bulls if b["score"] >= 65]
    bear_pts = [f"**{b['name']}** at {b['value']}: {plain_words(b) or b['interpretation']}" for b in bears if b["score"] <= 60]
    if not bull_pts:
        bull_pts = ["Nothing stands out as strong in the numbers alone — any case for buying rests on things "
                    "the numbers can't see, like new products, management quality or industry change."]
    if not bear_pts:
        bear_pts = ["No number looks worryingly weak right now — the remaining risks are the kind numbers "
                    "can't show, like competition, regulation or management missteps."]
    return bull_pts, bear_pts


# ---------------------------------------------------------------------------
# Peer comparison (free Yahoo similar-symbols endpoint)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_peer_symbols(symbol: str) -> list[str]:
    try:
        r = requests.get(
            f"https://query2.finance.yahoo.com/v6/finance/recommendationsbysymbol/{symbol}",
            headers={"User-Agent": "Mozilla/5.0 (research-app)"}, timeout=10)
        if r.ok:
            res = r.json().get("finance", {}).get("result", [])
            if res:
                return [x["symbol"] for x in res[0].get("recommendedSymbols", [])][:4]
    except Exception:
        pass
    return []


@st.cache_data(ttl=1800, show_spinner=False)
def peer_snapshot(symbol: str) -> dict:
    try:
        return yf.Ticker(symbol).info or {}
    except Exception:
        return {}


def peer_row(sym: str, inf: dict, is_etf: bool, star: bool = False) -> dict:
    label = ("★ " if star else "") + sym
    nm = (inf.get("shortName") or inf.get("longName") or sym)[:30]
    if is_etf:
        er = inf.get("netExpenseRatio") or inf.get("annualReportExpenseRatio")
        if er is not None:
            er = float(er)
            er = er * 100 if er < 0.05 else er
        r3 = inf.get("threeYearAverageReturn")
        r3 = r3 * 100 if r3 is not None and abs(r3) < 2 else r3
        return {"Symbol": label, "Name": nm, "Expense %": er, "AUM": inf.get("totalAssets"),
                "Yield %": dividend_yield_pct(inf), "3y return %": r3,
                "Beta": inf.get("beta3Year") or inf.get("beta")}
    pct = lambda k: inf.get(k) * 100 if inf.get(k) is not None else None
    return {"Symbol": label, "Name": nm, "P/E": inf.get("trailingPE"),
            "Fwd P/E": inf.get("forwardPE"), "P/B": inf.get("priceToBook"),
            "EV/EBITDA": inf.get("enterpriseToEbitda"), "ROE %": pct("returnOnEquity"),
            "Net margin %": pct("profitMargins"), "Rev growth %": pct("revenueGrowth"),
            "Div yield %": dividend_yield_pct(inf), "Beta": inf.get("beta"),
            "Mkt cap": inf.get("marketCap")}


# ---------------------------------------------------------------------------
# Watchlist & score history (auto-logged on every generate/refresh)
# ---------------------------------------------------------------------------

HISTORY_FILE = "score_history.csv"


def log_score(symbol, name, itype, overall, confidence, risk, cat_scores, adjusted):
    row = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "symbol": symbol, "name": name, "type": itype, "overall": overall,
           "confidence": confidence, "risk": risk, "sector_adjusted": bool(adjusted),
           "categories": json.dumps(cat_scores)}
    try:
        if os.path.exists(HISTORY_FILE):
            prev = pd.read_csv(HISTORY_FILE)
            last = prev[prev["symbol"] == symbol].tail(1)
            if not last.empty:  # dedupe: same score within 10 minutes
                age = (pd.Timestamp.now(tz="UTC") - pd.to_datetime(last["timestamp"].iloc[0], utc=True)).total_seconds()
                if age < 600 and int(last["overall"].iloc[0]) == int(overall):
                    return
            df = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(HISTORY_FILE, index=False)
    except Exception:
        pass  # history must never break the report


def load_history() -> pd.DataFrame:
    try:
        return pd.read_csv(HISTORY_FILE, parse_dates=["timestamp"])
    except Exception:
        return pd.DataFrame()


def render_history(current_symbol=None):
    hist_df = load_history()
    if hist_df.empty:
        st.info("No score history yet — every generated or refreshed report is logged here automatically, "
                "so you can track how an instrument's score moves over time.")
        return
    st.markdown("**Watchlist — latest score per instrument** (Δ compares with your previous report of the same instrument)")
    latest_rows = []
    for sym, g in hist_df.sort_values("timestamp").groupby("symbol"):
        last = g.iloc[-1]
        prev = g.iloc[-2] if len(g) > 1 else None
        delta = (last["overall"] - prev["overall"]) if prev is not None else None
        latest_rows.append({"Symbol": sym, "Name": str(last["name"])[:30], "Type": last["type"],
                            "Score": int(last["overall"]),
                            "Δ": (f"{delta:+.0f}" if delta is not None else "—"),
                            "Risk": last["risk"], "Confidence %": int(last["confidence"]),
                            "Last report": pd.to_datetime(last["timestamp"]).strftime("%d %b %Y %H:%M")})
    st.dataframe(pd.DataFrame(latest_rows).sort_values("Score", ascending=False),
                 hide_index=True, use_container_width=True)
    syms = sorted(hist_df["symbol"].unique())
    default = [current_symbol] if current_symbol in syms else syms[:3]
    pick = st.multiselect("Chart score history for:", syms, default=default)
    if pick:
        sub = hist_df[hist_df["symbol"].isin(pick)]
        fig = px.line(sub, x="timestamp", y="overall", color="symbol", markers=True, range_y=[0, 100])
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                          yaxis_title="Overall score", xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
    hc1, hc2 = st.columns(2)
    with hc1:
        st.download_button("⬇️ Download history backup (CSV)",
                           hist_df.to_csv(index=False).encode(), "score_history.csv", "text/csv")
    with hc2:
        up = st.file_uploader("Restore / merge a history CSV", type="csv")
        if up is not None:
            try:
                merged = (pd.concat([hist_df, pd.read_csv(up, parse_dates=["timestamp"])])
                          .drop_duplicates(subset=["timestamp", "symbol"]).sort_values("timestamp"))
                merged.to_csv(HISTORY_FILE, index=False)
                st.success(f"Merged — history now holds {len(merged)} entries. Reload to see it.")
            except Exception as e:
                st.error(f"Could not merge that file: {e}")
    st.caption("⚠️ On free hosting the history file resets whenever the app redeploys or restarts — "
               "download a backup periodically and re-upload it here to restore.")


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
            st.session_state.pending_log = True

# ---------------------------------------------------------------------------
# UI — report
# ---------------------------------------------------------------------------

# --- Investment-horizon views ---------------------------------------------
# Same category scores, re-weighted for the holding period. Long-term (5y+)
# emphasises quality and survivability; mid-term (~3y) emphasises entry
# valuation and trend, because there is less time for quality to compound
# past a poor entry price. Transparent, rule-based — no hidden model.

HORIZON_WEIGHTS = {
    "stock": {
        "long": {"Profitability": 0.30, "Financial Health": 0.20, "Growth": 0.20,
                 "Valuation": 0.15, "Risk": 0.15},
        "mid": {"Valuation": 0.25, "Growth": 0.20, "Profitability": 0.15,
                "Technical Trend": 0.15, "Risk": 0.15, "Financial Health": 0.10},
    },
    "etf": {
        "long": {"Cost": 0.30, "Holdings Quality": 0.20, "Risk": 0.20,
                 "Performance": 0.15, "Liquidity": 0.15},
        "mid": {"Performance": 0.30, "Risk": 0.25, "Cost": 0.15,
                "Liquidity": 0.15, "Holdings Quality": 0.15},
    },
}


def horizon_view(metrics, cat_scores, wmap):
    """Return (score, emoji, label, confidence%) for one horizon.
    Confidence = share of the horizon's weighted inputs actually covered by data."""
    total_w = sum(wmap.values())
    conf_acc, covered = 0.0, {}
    for c, w in wmap.items():
        cat_ms = [x for x in metrics if x["category"] == c]
        if cat_ms:
            conf_acc += w * sum(1 for x in cat_ms if x["score"] is not None) / len(cat_ms)
        if c in cat_scores:
            covered[c] = w
    if not covered:
        return None, "⚪", "Insufficient data", 0
    score = round(sum(cat_scores[c] * w for c, w in covered.items()) / sum(covered.values()))
    conf = round(conf_acc / total_w * 100)
    if score >= 72:
        emoji, label = "🟢", "Favourable"
    elif score >= 58:
        emoji, label = "🟡", "Moderately favourable"
    elif score >= 45:
        emoji, label = "🟠", "Neutral / mixed"
    else:
        emoji, label = "🔴", "Unfavourable"
    return score, emoji, label, conf


def fmt_weights(d):
    return ", ".join(f"{c} {w:.0%}" for c, w in d.items())


def business_summary(info, max_sentences=2, max_chars=360):
    """First couple of sentences of the company/fund description."""
    txt = (info.get("longBusinessSummary") or "").strip()
    if not txt:
        fam, cat = info.get("fundFamily"), info.get("category")
        if fam or cat:
            return f"{fam or 'Fund'} — {cat or 'exchange-traded fund'}."
        return None
    sents = re.split(r"(?<=[.!?])\s+", txt)
    out = " ".join(sents[:max_sentences])
    if len(out) > max_chars:
        out = out[:max_chars].rsplit(" ", 1)[0] + "…"
    return out


# ---------------------------------------------------------------------------
# Plain-language layer — every metric explained without finance vocabulary
# ---------------------------------------------------------------------------

def _pw(with_value, without_value):
    """Build a plain-words function: uses the live number when available,
    falls back to a generic sentence when the data source didn't publish it."""
    def f(mt):
        raw = mt.get("raw")
        try:
            return with_value(raw, mt) if raw is not None else without_value
        except Exception:
            return without_value
    return f


PLAIN_WORDS = {
    "P/E Ratio (trailing)": _pw(
        lambda r, m: f"You're paying about {r:,.0f} years of today's profit for this share — like paying {r:,.0f} years' rent upfront to buy the house.",
        "This counts how many years of today's profit you'd pay for the share — like working out how many years' rent it would take to buy the house."),
    "Forward P/E": _pw(
        lambda r, m: f"Same idea, but using *next year's expected* profit: about {r:,.0f} years' worth at the forecast.",
        "Same idea as P/E, but using next year's expected profit instead of last year's."),
    "PEG Ratio": _pw(
        lambda r, m: f"This checks whether the price is fair *for how fast profits are growing*. Around 1 is fair; {r:,.1f} means you're paying {'a discount' if r < 1 else 'roughly fair value' if r <= 1.3 else 'a premium'} for the growth.",
        "This checks whether the price is fair for how fast profits are growing — around 1 is considered fair."),
    "Price-to-Sales": _pw(
        lambda r, m: f"The whole company is priced at about {r:,.1f} years of its sales — before any costs are taken out.",
        "This prices the whole company against its yearly sales — useful when profits are lumpy."),
    "Price-to-Book": _pw(
        lambda r, m: f"The share costs about {r:,.1f}× what the company's possessions minus its debts are worth on paper.",
        "This compares the share price with what the company's possessions minus debts are worth on paper."),
    "EV / EBITDA": _pw(
        lambda r, m: f"A takeover-style check: buying the whole company, debts included, would cost about {r:,.0f} years of its core cash profits.",
        "A takeover-style check: how many years of core cash profits it would take to pay for the whole company, debts included."),
    "Free Cash Flow Yield": _pw(
        lambda r, m: f"For every 100 you'd pay for the whole company, it generates about {r:,.1f} in spare cash each year — like a house whose rent covers {r:,.1f}% of its price annually.",
        "How much spare cash the business generates each year compared with its price — like checking whether a house's rent justifies its price."),
    "Dividend Yield": _pw(
        lambda r, m: f"The company pays you back about {r:,.1f}% of your money each year in cash — a bit like interest on savings, on top of any change in the share price.",
        "The cash the company pays you each year for holding the share — a bit like interest on savings. Many growing companies deliberately pay little or none."),
    "Revenue Growth (yoy)": _pw(
        lambda r, m: f"The company sold about {abs(r):,.0f}% {'more' if r >= 0 else 'less'} than a year ago — {'the shop is busier than before' if r >= 0 else 'the shop is quieter than before'}.",
        "Whether the company is selling more than it did a year ago — the raw material of all future profit."),
    "EPS Growth (yoy)": _pw(
        lambda r, m: f"Profit per share is {abs(r):,.0f}% {'higher' if r >= 0 else 'lower'} than a year ago.",
        "Whether the profit behind each share is growing — over long periods, share prices tend to follow this."),
    "Gross Margin": _pw(
        lambda r, m: f"For every 100 of sales, about {r:,.0f} is left after paying for the product itself — that's the room it has for everything else.",
        "How much of each sale is left after paying for the product itself — the company's basic pricing power."),
    "Operating Margin": _pw(
        lambda r, m: f"After all the day-to-day running costs, about {r:,.0f} of every 100 in sales remains as profit.",
        "How much of each sale is left after all the day-to-day running costs."),
    "Net Margin": _pw(
        lambda r, m: f"At the very end — after everything, including tax — the company keeps about {r:,.0f} of every 100 it sells.",
        "How much of each sale the company finally keeps after absolutely everything, including tax."),
    "Return on Equity": _pw(
        lambda r, m: f"For every 100 of the owners' money in the business, it earned about {r:,.0f} this year — think of it as the interest rate the business pays its owners.",
        "How hard the owners' money works — think of it as the interest rate the business earns on the money shareholders have put in."),
    "Return on Assets": _pw(
        lambda r, m: f"For every 100 of things the company owns — factories, stock, cash — it earned about {r:,.1f} this year.",
        "How much profit the company squeezes out of everything it owns — factories, stock, cash."),
    "Debt-to-Equity": _pw(
        lambda r, m: f"For every 100 the owners have put in, the company has borrowed about {r:,.0f}. {'A light load.' if r <= 50 else 'A moderate load.' if r <= 120 else 'A heavy load — borrowing magnifies both good and bad years.'}",
        "How much the company has borrowed compared with the owners' own money — borrowing magnifies both good years and bad ones."),
    "Current Ratio": _pw(
        lambda r, m: f"For every 1 of bills due within a year, the company has about {r:,.1f} readily available to pay them.",
        "Whether the company can comfortably pay the bills landing in the next twelve months."),
    "Cash Position (% of market cap)": _pw(
        lambda r, m: f"About {r:,.0f}% of the company's entire price tag is sitting there as ready cash — its rainy-day fund.",
        "How big the company's rainy-day cash fund is compared with its price tag."),
    "Beta (5y monthly)": _pw(
        lambda r, m: f"When the whole market moves 10%, this typically moves about {abs(r) * 10:,.0f}% — {'more jumpy than' if r > 1.1 else 'calmer than' if r < 0.9 else 'about the same as'} the market overall.",
        "How much this tends to move when the whole market moves — above 1 means jumpier than average, below 1 means calmer."),
    "Beta (3y)": _pw(
        lambda r, m: f"When the whole market moves 10%, this typically moves about {abs(r) * 10:,.0f}% — {'more jumpy than' if r > 1.1 else 'calmer than' if r < 0.9 else 'about the same as'} the market overall.",
        "How much this tends to move when the whole market moves — above 1 means jumpier than average, below 1 means calmer."),
    "Volatility (1y, annualised)": _pw(
        lambda r, m: f"In a typical year the price swings roughly {r:,.0f}% up or down — that's how bumpy the ride has actually been.",
        "How bumpy the ride has been — the size of the typical up-and-down swings over a year."),
    "Max Drawdown (1y)": _pw(
        lambda r, m: f"If you'd bought at the worst possible moment in the past year, you'd have been down about {abs(r):,.0f}% at the lowest point.",
        "The deepest fall from a peak in the past year — the loss an unlucky buyer actually felt."),
    "Price vs 50-day MA": _pw(
        lambda r, m: f"The price is {abs(r):,.1f}% {'above' if r >= 0 else 'below'} its average of the last two months — {'recent momentum is positive' if r >= 0 else 'it has been drifting down recently'}.",
        "Whether the price is above or below its average of the last two months — a quick check on recent direction."),
    "Price vs 200-day MA": _pw(
        lambda r, m: f"The price is {abs(r):,.1f}% {'above' if r >= 0 else 'below'} its average of the last ten months — the long-run direction {'still points up' if r >= 0 else 'currently points down'}.",
        "Whether the price is above or below its average of the last ten months — the classic long-run direction check."),
    "RSI (14-day)": _pw(
        lambda r, m: f"A 0–100 'speedometer' of recent buying vs selling — currently {r:,.0f}. Around 50 is calm; above 70 suggests overheated buying, below 30 heavy selling.",
        "A 0–100 'speedometer' of recent buying vs selling. Around 50 is calm; above 70 suggests overheated buying, below 30 heavy selling."),
    # --- ETF metrics ---
    "Expense Ratio": _pw(
        lambda r, m: f"The fund keeps {r:,.2f}% of your money every single year as its fee. The lower this is, the more of the growth stays yours.",
        "The yearly fee the fund quietly takes from your money — the single most reliable predictor of long-run results: lower wins."),
    "Assets Under Management": _pw(
        lambda r, m: f"The fund looks after {fmt(r, 'compact')} in total. Big funds are cheaper to trade and far less likely to be shut down.",
        "How much money the fund looks after in total — very small funds risk being closed, forcing you out at a bad time."),
    "Average Daily Volume": _pw(
        lambda r, m: f"About {fmt(r, 'compact')} units change hands each day — busier trading generally means you get a fairer price when buying or selling.",
        "How busily the fund trades each day — busier means fairer prices when you buy or sell."),
    "1-Year Return": _pw(
        lambda r, m: f"100 invested a year ago would be roughly {100 * (1 + r / 100):,.0f} today.",
        "What 100 invested a year ago would be worth today."),
    "3-Year Average Annual Return": _pw(
        lambda r, m: f"Over the last three years it has grown about {r:,.1f}% per year on average — a fairer test than any single year.",
        "How much it grew per year, averaged over the last three years — a fairer test than any single year."),
    "Sharpe Ratio (1y)": _pw(
        lambda r, m: f"How well you were paid for the bumps: above 1 means the return justified the ride; here it's {r:,.2f}.",
        "How well the return paid you for the bumpiness endured — above 1 means the ride was worth it."),
    "Top Sector Concentration": _pw(
        lambda r, m: f"About {r:,.0f}% of the fund sits in its single biggest industry — {'a well-spread basket' if r <= 30 else 'quite a lot of eggs in one basket' if r <= 45 else 'most of the eggs are in one basket'}.",
        "How much of the fund depends on one single industry — the more it does, the more it behaves like a bet on that industry."),
    "Top-10 Holdings Weight": _pw(
        lambda r, m: f"The 10 biggest investments make up about {r:,.0f}% of the entire fund — the rest is spread across everything else it holds.",
        "How much of the fund rides on just its 10 biggest investments."),
    "Distribution Yield": _pw(
        lambda r, m: f"The fund pays out about {r:,.1f}% of its price per year as income to you.",
        "The income the fund pays out each year. Some funds reinvest instead of paying out — near zero can be by design."),
}


def plain_words(mt):
    fn = PLAIN_WORDS.get(mt["name"])
    return fn(mt) if fn else None


CAT_SUBTITLES = {
    "Valuation": "Are you paying a fair price?",
    "Growth": "Is the business getting bigger?",
    "Profitability": "Does it make good money?",
    "Financial Health": "Can it pay its bills?",
    "Risk": "How bumpy is the ride?",
    "Technical Trend": "Which way is the price heading?",
    "Cost": "How much of your money the fund keeps as fees",
    "Liquidity": "How easy it is to buy and sell",
    "Performance": "How well has it done?",
    "Holdings Quality": "How spread out are its investments?",
}

STOCK_PLAIN = {
    "Valuation": ("the price looks reasonable for what you get",
                  "the price is on the fuller side",
                  "you're paying a premium price for it"),
    "Growth": ("the business is growing at a healthy clip",
               "the business is growing, but only modestly",
               "the business is barely growing"),
    "Profitability": ("it makes good money on what it sells",
                      "its profits are decent but not standout",
                      "it struggles to turn sales into real profit"),
    "Financial Health": ("it carries little debt and can comfortably pay its bills",
                         "its debt load looks manageable",
                         "it carries a heavy debt load"),
    "Risk": ("the share price has been relatively calm",
             "the share price has been moderately bumpy",
             "the share price has been a rough ride"),
    "Technical Trend": ("the price has been heading upward lately",
                        "the price has been moving sideways lately",
                        "the price has been drifting down lately"),
}

ETF_PLAIN = {
    "Cost": ("it's cheap to own",
             "its yearly fee is about average",
             "its yearly fee is high and quietly eats into returns"),
    "Liquidity": ("it's big and easy to trade",
                  "it's reasonably easy to trade",
                  "it's small and thinly traded"),
    "Performance": ("returns have been strong",
                    "returns have been okay",
                    "returns have been weak"),
    "Risk": ("the ride has been fairly smooth",
             "the ride has been moderately bumpy",
             "the ride has been very bumpy"),
    "Holdings Quality": ("its investments are nicely spread out",
                         "its investments are somewhat concentrated",
                         "a lot of it rides on just a few bets"),
}


def _join(ps):
    return ps[0] if len(ps) == 1 else ", ".join(ps[:-1]) + " and " + ps[-1]


def plain_verdict(cat_scores, is_etf):
    """2–3 template sentences saying in everyday words what the lights show.
    Never says anything the category scores don't already say."""
    table = ETF_PLAIN if is_etf else STOCK_PLAIN
    good, mid, bad = [], [], []
    for c, s in cat_scores.items():
        if c not in table:
            continue
        g, m, b = table[c]
        (good if s >= 70 else mid if s >= 50 else bad).append(g if s >= 70 else m if s >= 50 else b)
    bits = []
    if good:
        bits.append("On the plus side, " + _join(good) + ".")
    if bad:
        bits.append(("The main concern is that " if len(bad) == 1 else "The main concerns: ") + _join(bad) + ".")
    if mid:
        bits.append("In between: " + _join(mid) + ".")
    return " ".join(bits) if bits else None


def light_for(score):
    """Map a metric score to a RAYG traffic light. Legend shown on the dashboard."""
    if score is None:
        return "⚪"
    if score >= 80:
        return "🟢"
    if score >= 62:
        return "🟡"
    if score >= 45:
        return "🟠"
    return "🔴"


def render_metric_detail(mt):
    """Full explainable card for one metric — used in the dashboard pop-ups
    and the metric deep-dive tab."""
    st.markdown(status_html(mt["status"]), unsafe_allow_html=True)
    plain = plain_words(mt)
    if plain:
        st.markdown(f"<div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;"
                    f"padding:10px 12px;margin:6px 0 10px 0;'>💬 <b>In plain words</b> — {plain}</div>",
                    unsafe_allow_html=True)
    st.markdown(f"**What it means** — {mt['definition']}")
    st.markdown(f"**Why it matters** — {mt['why']}")
    st.markdown(f"**How it is calculated** — {mt['calculation']}")
    st.markdown("**What good looks like (benchmark bands)**")
    raw_v, th_v = mt.get("raw"), mt.get("thresholds")
    if raw_v is not None and th_v:
        active = band_of(raw_v, th_v, mt.get("higher", True))
    else:  # non-monotonic metrics (e.g. RSI): match on status label
        stat = str(mt["status"]).lower()
        active = next((i for i, (lb, _) in enumerate(mt["bands"])
                       if stat != "n/a" and (stat in lb.lower() or lb.lower().startswith(stat))), None)
    bcols = st.columns(len(mt["bands"]))
    for i, (bc, (label, rng)) in enumerate(zip(bcols, mt["bands"])):
        if i == active:
            style = "background:#eff6ff;border:2px solid #1d4ed8;"
            marker = (f"<div style='margin-top:5px;font-size:0.72rem;color:#1d4ed8;"
                      f"font-weight:700;'>◉ {mt['value']} — you are here</div>")
        else:
            style, marker = "background:#f8fafc;border:1px solid #e2e8f0;", ""
        bc.markdown(f"<div style='{style}border-radius:10px;padding:8px 10px;'>"
                    f"<div style='font-size:0.7rem;color:#64748b;text-transform:uppercase;'>{label}</div>"
                    f"<div style='font-size:0.85rem;font-weight:600;'>{rng}</div>{marker}</div>",
                    unsafe_allow_html=True)
    pos = position_summary(mt)
    if pos:
        st.markdown(f"📍 **Where the current value sits** — {pos}")
    st.markdown(f"**Sector / category context** — {mt['benchmark']}")
    st.markdown(f"**Interpretation of the current value** — {mt['interpretation']}")
    st.markdown(f"**Impact on the score** — {mt['impact']}")
    if mt.get("adjusted"):
        st.caption(f"ℹ️ Benchmark bands above are adjusted to **{mt['adjusted']}** sector norms. "
                   "Use the toggle at the top of the report to switch back to market-wide bands.")


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

        st.divider()

        # --- Actions ---
        a1, a2, a3 = st.columns([4, 1, 1])
        with a1:
            st.subheader("Generated research report")
            st.caption("Web-based report with latest free data, explainable metrics and print-ready layout.")
        with a2:
            if st.button("🔄 Refresh data", use_container_width=True):
                load_instrument.clear()
                st.session_state.pending_log = True
                st.rerun()
        with a3:
            st.markdown('<div class="no-print" style="font-size:0.8rem;color:#64748b;padding-top:0.4rem;">'
                        '🖨️ <b>Print / PDF:</b> press <kbd>Ctrl/Cmd&nbsp;+&nbsp;P</kbd> and choose "Save as PDF".</div>',
                        unsafe_allow_html=True)
        if not is_etf:
            adjust_bands = st.toggle(
                "Sector-adjusted benchmark bands", value=True,
                help="On: thresholds shift to the sector's own norms (e.g. banks are judged on P/B and allowed "
                     "structural leverage; tech is allowed higher P/E but expected to deliver fatter margins). "
                     "Off: one market-wide yardstick for all sectors.")
        else:
            adjust_bands = False

        metrics = (build_etf_metrics(info, hist, data) if is_etf
                   else build_stock_metrics(info, hist, currency,
                                            sector=info.get("sector"), adjust=adjust_bands))
        weights = ETF_WEIGHTS if is_etf else STOCK_WEIGHTS
        cat_scores, overall, confidence = summarise(metrics, weights)
        rating = rating_label(overall)
        risk = risk_label(metrics)
        bull_pts, bear_pts = bull_bear(metrics)
        refreshed = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")

        if st.session_state.get("pending_log") and overall is not None:
            log_score(symbol, name, "ETF" if is_etf else "Stock", overall,
                      confidence, risk, cat_scores, adjust_bands)
            st.session_state.pending_log = False

        with st.expander("🧭 First time here? How to read this report in 30 seconds"):
            st.markdown(
                "1. **The top box** tells you what this company or fund actually is, its latest price (in brackets), "
                "and an overall score out of 100.\n"
                "2. **Long-term vs mid-term** — two quick verdicts: one for 'buy and hold 5+ years', one for "
                "'hold around 3 years', each with a confidence level.\n"
                "3. **🚦 Scorecard tab** shows every measure as a traffic light: 🟢 great · 🟡 good · 🟠 so-so · "
                "🔴 weak · ⚪ no data. **Click any measure** for a plain-words explanation — no finance background needed.\n"
                "4. **👍 For / ⚠️ Watch-outs** sums up the strongest points for and against in everyday language.\n"
                "5. This is an educational screen, **not** financial advice — it helps you ask better questions, "
                "not skip them.")

        # --- Header card ---
        hw = HORIZON_WEIGHTS["etf" if is_etf else "stock"]
        lt = horizon_view(metrics, cat_scores, hw["long"])
        md = horizon_view(metrics, cat_scores, hw["mid"])
        with st.container(border=True):
            h1, h2 = st.columns([3, 2])
            with h1:
                st.markdown(
                    status_html("ETF" if is_etf else "Stock") + status_html(exchange) +
                    status_html(region) + (status_html(currency) if currency else ""),
                    unsafe_allow_html=True)
                price_tag = (f"&nbsp;<span style='font-size:1.15rem;color:#334155;font-weight:600;"
                             f"white-space:nowrap;'>({currency} {price:,.2f})</span>" if price else "")
                st.markdown(f"<h2 style='margin:0.3rem 0 0.2rem 0;'>{name}{price_tag}</h2>",
                            unsafe_allow_html=True)
                st.markdown(f"**{symbol}** · {sector}")
                desc = business_summary(info)
                if desc:
                    st.markdown(f"<span style='color:#334155;'>{desc}</span>", unsafe_allow_html=True)
                else:
                    st.caption("No business description available from the free data source.")
                st.caption(f"Quantitative risk: **{risk}** · data coverage: **{confidence}%** "
                           f"(unscored metrics are excluded rather than guessed) · last refreshed: {refreshed}")
            with h2:
                with st.container(border=True):
                    ov_txt = f"{overall}/100" if overall is not None else "—"
                    st.markdown(
                        f"<div style='font-size:0.75rem;color:#64748b;text-transform:uppercase;'>Overall score</div>"
                        f"<div style='font-size:1.7rem;font-weight:700;line-height:1.2;'>{ov_txt} "
                        f"<span style='font-size:1rem;font-weight:600;color:#475569;'>· {rating}</span></div>",
                        unsafe_allow_html=True)
                with st.container(border=True):
                    for title, (hs, hemoji, hlabel, hconf) in (("Long-term (5y+)", lt), ("Mid-term (~3y)", md)):
                        tail = f" — **{hs}/100** · confidence {hconf}%" if hs is not None else ""
                        st.markdown(f"{hemoji} **{title}:** {hlabel}{tail}")
                    with st.popover("ⓘ How the horizon views are calculated"):
                        st.markdown(
                            "Each horizon re-weights the same category scores from this report to match what "
                            "matters over that holding period — same data, different emphasis, no hidden model.\n\n"
                            f"**Long-term (5y+):** {fmt_weights(hw['long'])}. Quality, balance-sheet strength and "
                            "durable growth dominate, because they decide whether the business compounds; "
                            "today's price chart matters little over five years.\n\n"
                            f"**Mid-term (~3y):** {fmt_weights(hw['mid'])}. Entry valuation and trend carry more "
                            "weight, because over three years there is less time for quality to compound past a "
                            "poor entry price.\n\n"
                            "**Confidence** = the share of that horizon's weighted inputs actually covered by "
                            "data for this instrument.")
                    st.caption("Rule-based screen derived from the scores in this report — an educational aid, "
                               "**not** personal investment advice.")

        # --- Tabs (dashboard first) ---
        tab_dash, tab_m, tab_c, tab_p, tab_bb, tab_h, tab_s = st.tabs(
            ["🚦 Scorecard", "🧠 Metric deep-dive", "📊 Charts", "👥 Peer comparison",
             "👍 For / ⚠️ Watch-outs", "⭐ Watchlist & history", "🗄️ Sources & notes"])

        with tab_dash:
            verdict_txt = plain_verdict(cat_scores, is_etf)
            if verdict_txt:
                st.markdown(f"<div style='background:#f0f9ff;border:1px solid #bae6fd;border-radius:12px;"
                            f"padding:12px 14px;margin-bottom:8px;font-size:1.02rem;'>💬 <b>In plain words</b> — "
                            f"{verdict_txt}</div>", unsafe_allow_html=True)
            counts = {"🟢": 0, "🟡": 0, "🟠": 0, "🔴": 0, "⚪": 0}
            for x in metrics:
                counts[light_for(x["score"])] += 1
            st.markdown(
                f"**At a glance:** {' · '.join(f'{k} {v}' for k, v in counts.items() if v)}"
                f" &nbsp;&nbsp;|&nbsp;&nbsp; **Legend:** 🟢 Excellent · 🟡 Good · 🟠 Acceptable · "
                f"🔴 Weak · ⚪ No data &nbsp;—&nbsp; click any metric for its full explanation.")
            dash_cats = [c for c in weights if any(x["category"] == c for x in metrics)]
            dcols = st.columns(3)
            for i, cat in enumerate(dash_cats):
                with dcols[i % 3]:
                    with st.container(border=True):
                        sc = cat_scores.get(cat)
                        st.markdown(f"##### {cat} — {sc}/100" if sc is not None
                                    else f"##### {cat} — no data")
                        st.caption(CAT_SUBTITLES.get(cat, ""))
                        st.progress((sc or 0) / 100,
                                    text=f"Weight in overall score: {weights[cat]:.0%}")
                        for mt in [x for x in metrics if x["category"] == cat]:
                            with st.popover(f"{light_for(mt['score'])} {mt['name']}  ·  {mt['value']}",
                                            use_container_width=True):
                                st.markdown(f"#### {mt['name']} — {mt['value']}")
                                render_metric_detail(mt)
            missing = [c for c in weights if c not in cat_scores]
            if missing:
                st.caption("Not scored (no data): " + ", ".join(missing) +
                           ". Weights are re-normalised across the scored categories.")
            st.caption("Every light is traceable to a stated benchmark band — nothing is a black box.")

        with tab_m:
            st.markdown("**Explainable metric interpretation guide** — every metric states what it means, why it matters, "
                        "how it is calculated, what good looks like (with reasoning), and how it moves the score.")
            for cat in weights:
                cat_metrics = [x for x in metrics if x["category"] == cat]
                extra = [x for x in metrics if x["category"] not in weights]
                if not cat_metrics:
                    continue
                st.markdown(f"##### {cat}")
                st.caption(CAT_SUBTITLES.get(cat, ""))
                for mt in cat_metrics:
                    header = f"{light_for(mt['score'])} {mt['name']}  —  {mt['value']}  ·  {mt['status']}" + \
                             (f"  ·  score {mt['score']}/100" if mt["score"] is not None else "")
                    with st.expander(header):
                        render_metric_detail(mt)

        with tab_p:
            st.markdown(f"**Peer comparison** — nearest peers to **{symbol}** per Yahoo Finance's free "
                        "similarity engine, on like-for-like metrics. Bands tell you 'good in absolute terms'; "
                        "this tab tells you 'good *for this kind of company*'.")
            with st.spinner("Finding peers and fetching their fundamentals…"):
                peer_syms = fetch_peer_symbols(symbol)
                rows = [peer_row(symbol, info, is_etf, star=True)]
                for p in peer_syms:
                    snap = peer_snapshot(p)
                    if snap:
                        rows.append(peer_row(p, snap, is_etf))
            if len(rows) == 1:
                st.info("No peer suggestions were available from the free source for this symbol — "
                        "coverage is strongest for US large caps, patchier for some Asian and UK listings.")
            else:
                pdf_ = pd.DataFrame(rows)
                med = pdf_.iloc[1:].median(numeric_only=True)
                med_row = {c: med.get(c) for c in pdf_.columns}
                med_row["Symbol"], med_row["Name"] = "— Peer median —", ""
                pdf_ = pd.concat([pdf_, pd.DataFrame([med_row])], ignore_index=True)
                disp = pdf_.copy()
                for col in disp.columns:
                    if col in ("Mkt cap", "AUM"):
                        disp[col] = disp[col].map(lambda v: fmt(v, "compact") if pd.notna(v) else "—")
                    elif disp[col].dtype.kind == "f":
                        disp[col] = disp[col].map(lambda v: f"{v:,.2f}" if pd.notna(v) else "—")
                st.dataframe(disp, hide_index=True, use_container_width=True)
                verdict = []
                checks = ([("Expense %", True), ("Yield %", False), ("3y return %", False)] if is_etf
                          else [("P/E", True), ("ROE %", False), ("Net margin %", False),
                                ("Rev growth %", False), ("Div yield %", False)])
                for col, lower_better in checks:
                    v, mv = pdf_.loc[0, col] if col in pdf_.columns else None, med.get(col)
                    if pd.notna(v) and pd.notna(mv):
                        good = v < mv if lower_better else v > mv
                        verdict.append(f"{'✅' if good else '⚠️'} **{col}**: {v:,.2f} vs peer median {mv:,.2f} "
                                       f"({'better' if good else 'worse'})")
                if verdict:
                    st.markdown("**vs peer median:**")
                    st.markdown("  \n".join(verdict))
                st.caption("★ = the instrument in this report. Peer set is Yahoo's similarity suggestion — "
                           "sanity-check that the peers really are comparable before drawing conclusions.")

        with tab_c:
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
                st.success("#### 👍 What's going for it")
                for p in bull_pts:
                    st.markdown(f"- {p}")
            with b2:
                st.warning("#### ⚠️ What to watch out for")
                for p in bear_pts:
                    st.markdown(f"- {p}")
            st.caption("These points are generated from the highest- and lowest-scoring measures in this "
                       "report — nothing hidden, no opinions added.")

        with tab_h:
            render_history(current_symbol=symbol)

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
    if not load_history().empty:
        st.divider()
        st.subheader("⭐ Your watchlist & score history")
        render_history()
