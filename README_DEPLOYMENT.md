# Alpha Research AI — Free Deployment Guide

An explainable stock & ETF research app (India · Asia · UK · US) built with
Streamlit + yfinance. **Total running cost: £0.** No credit card required
anywhere in this guide.

---

## Why the original React prototype could not stand alone

The React code called Yahoo Finance and Stooq **directly from the browser**.
Both block cross-origin browser requests (CORS), and Yahoo's quote endpoint
now requires session cookies — so the prototype silently fell back to its
7-instrument demo dataset. The fix is to fetch data **server-side**, which is
exactly what this Streamlit app does: Python runs on the server, so there is
no CORS, and the `yfinance` library handles Yahoo's cookie/crumb handshake
for you. It also covers NSE/BSE (India), LSE, Tokyo, Hong Kong, Shanghai,
Korea, Taiwan, Singapore and the US.

---

## Option A (recommended): Streamlit Community Cloud — free forever tier

**What you need:** a free GitHub account. That's all.

### Step 1 — Run it locally first (5 minutes)
```bash
pip install -r requirements.txt
streamlit run app.py
```
Your browser opens at `http://localhost:8501`. Test with `RELIANCE.NS`,
`SHEL.L`, `7203.T`, `VOO`.

### Step 2 — Put the two files on GitHub
1. Create a free account at https://github.com (if you don't have one).
2. Click **New repository** → name it e.g. `alpha-research-ai` → Public → Create.
3. Click **Add file → Upload files** and upload:
   - `app.py`
   - `requirements.txt`
4. Commit.

### Step 3 — Deploy on Streamlit Community Cloud
1. Go to https://share.streamlit.io and sign in **with your GitHub account**.
2. Click **Create app** (or "New app").
3. Select your repository, branch `main`, main file `app.py`.
4. Click **Deploy**. First build takes 2–3 minutes.

You get a permanent public URL like
`https://<your-app-name>.streamlit.app` — open it on any phone, tablet or PC.

### Free-tier behaviour to know about
- Apps **sleep after ~12 hours of no visitors** and wake automatically
  (~30–60 s) when someone opens the link. Fine for personal research use.
- Resource limit is ~1 GB RAM — this app uses a fraction of that.
- To keep the app private, you can restrict viewers to specific email
  addresses in the app's settings (still free).

### Updating the app later
Edit `app.py` in GitHub (or push from your machine) — Streamlit Cloud
redeploys automatically within a minute.

---

## Option B: Hugging Face Spaces (free CPU tier)

1. Free account at https://huggingface.co
2. **New Space** → SDK: *Streamlit* → hardware: *CPU basic (free)*.
3. Upload `app.py` and `requirements.txt`.
4. URL: `https://huggingface.co/spaces/<username>/<space-name>`.

Same sleep-and-wake behaviour. Good fallback if Streamlit Cloud is busy.

## Option C: Run only on your own machine (100% private, zero hosting)

```bash
pip install -r requirements.txt
streamlit run app.py
```
Bookmark `http://localhost:8501`. To reach it from your phone on the same
Wi-Fi: `streamlit run app.py --server.address 0.0.0.0` then browse to
`http://<your-PC-ip>:8501`.

---

## Data sources, coverage and honest limitations (all free)

| Item | Source | Notes |
|---|---|---|
| Search, quotes, fundamentals | Yahoo Finance via `yfinance` | Free, unofficial library scraping public endpoints. Occasional rate limits — the app caches for 15 min and shows partial-data warnings. |
| Price history (1y daily) | Yahoo Finance adjusted closes | Basis for volatility, drawdown, Sharpe, RSI, moving averages. |
| ETF sector weights & top holdings | Yahoo fund data via `yfinance` | Coverage best for US-listed ETFs; UK/India listings sometimes omit expense ratio or holdings — the app marks these "Not available" and lowers confidence rather than guessing. |

- **India:** use `.NS` (NSE) or `.BO` (BSE) — `RELIANCE.NS`, `TCS.NS`,
  `HDFCBANK.NS`, `NIFTYBEES.NS` (Nifty 50 ETF).
- **UK:** `.L` — `SHEL.L`, `VOD.L`, `VUSA.L`. Note LSE prices come in pence (GBp).
- **Asia:** `.T` Tokyo, `.HK` Hong Kong, `.SS/.SZ` China, `.KS` Korea, `.SI` Singapore.
- Free quotes are typically delayed 15–20 minutes for LSE/NSE.
- Yahoo's terms permit personal use; for anything commercial, move to a
  licensed provider — the requirements doc's Section 5 production notes still apply.

**This app is an educational research aid, not investment advice.**

---

## Optional free upgrades (later)

- **Alpha Vantage** (free key, 25 requests/day) as a second quote source.
- **Financial Modeling Prep** free tier for deeper fundamentals on US stocks.
- **UptimeRobot** free plan pinging your URL every 30 min keeps the
  Streamlit app awake during your active hours.
- Add `st.download_button` CSV export of the metric table (10 lines of code).
