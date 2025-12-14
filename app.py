import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize

# -----------------------------
# App config & intro
# -----------------------------
st.set_page_config(page_title="Multi-Asset Risk Optimization Dashboard", layout="wide")
st.title("📊 Multi-Asset Risk Optimization Dashboard")

# Sidebar: Historical period selector
st.sidebar.header("📅 Historical Period")
period_option = st.sidebar.selectbox(
    "Select historical period for weekly averages:",
    options=["3 months", "6 months", "1 year"],
    index=2  # Default to "1 year"
)
period_map = {"3 months": "3mo", "6 months": "6mo", "1 year": "1y"}
selected_period = period_map[period_option]

st.write(f"""
This dashboard helps you make smarter investment decisions across currencies, commodities, equities, and indices.

✅ Use the **ticker validation tool** in the sidebar to check which assets are currently supported.  
📈 Compare weekly trade performance using the **view mode toggle** to switch between the latest week and the **average of weekly trades over the selected historical period** (*currently: {period_option}*).  
🔄 Use the **scenario sliders** to apply your personal risk constraints and explore how different thresholds impact your optimal asset allocation.  
🎯 Receive **prescriptive portfolio recommendations** based on return, drawdown, CR/DD, and Sharpe ratio.
""")

# -----------------------------
# Data prep: fetch and resample to 4h OHLC
# -----------------------------
def fetch_ohlc(ticker, period="1y"):
    try:
        df = yf.download(ticker, period=period, interval="1h", auto_adjust=False, progress=False)
        granularity = "4h"

        if df.empty or not {"Open","High","Low","Close"}.issubset(df.columns):
            df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
            granularity = "daily"

        if df.empty:
            raise ValueError(f"No data for {ticker}")

        if granularity == "4h":
            df_resampled = df.resample("4h").agg({
                "Open":"first","High":"max","Low":"min","Close":"last"
            }).dropna()
        else:
            df_resampled = df[["Open","High","Low","Close"]].dropna()

        return df_resampled, granularity
    except Exception as e:
        # Return empty so caller can skip
        return pd.DataFrame(), None

# -----------------------------
# Directional weekly metrics (dominant trade selection)
# -----------------------------
def compute_directional_week(wk, capital=10000, pip_size=None, pip_value=None, point_value=1.0):
    wk = wk.sort_index()
    monday_open = wk.iloc[0]["Open"]

    def price_diff_to_dollars(diff):
        if pip_size and pip_value:
            pips = diff / pip_size
            return pips * pip_value
        else:
            return diff * point_value

    candidates = []
    for ts, row in wk.iterrows():
        close_val = row["Close"]
        if isinstance(close_val, pd.Series):
            close_val = close_val.iloc[0]

        try:
            long_ret = float(price_diff_to_dollars(close_val - monday_open) / capital)
            short_ret = float(price_diff_to_dollars(monday_open - close_val) / capital)
        except Exception:
            continue  # skip this row if there's a type or math error

        candidates.append(("up", ts, long_ret))
        candidates.append(("down", ts, short_ret))

    if not candidates:
        return {
            "direction": None,
            "end_day": None,
            "return_cap (%)": 0.0,
            "drawdown_cap (%)": 0.0,
            "CR/DD": None
        }

    direction, end_day, best_ret = max(candidates, key=lambda x: x[2])
    best_ret = abs(best_ret)  # ensure positive return

    if direction == "up":
        lows = wk.loc[wk.index <= end_day, "Low"]
        dd_cap = float(price_diff_to_dollars(monday_open - lows.min()) / capital * 100)
    else:
        highs = wk.loc[wk.index <= end_day, "High"]
        dd_cap = float(price_diff_to_dollars(highs.max() - monday_open) / capital * 100)

    dd_cap = abs(dd_cap)  # ensure positive drawdown
    cr_dd = float(best_ret / (dd_cap / 100)) if dd_cap != 0 else None

    return {
        "direction": direction,
        "end_day": end_day,
        "return_cap (%)": round(best_ret * 100, 2),
        "drawdown_cap (%)": round(dd_cap, 2),
        "CR/DD": round(cr_dd, 2) if cr_dd is not None else None
    }

# -----------------------------
# Weekly metrics wrapper with Sharpe ratio
# -----------------------------
def compute_weekly_metrics(df_ohlc, capital=10000, pip_size=None, pip_value=None, point_value=1.0):
    weekly_rows = []
    weekly_returns = []

    for week_key, wk in df_ohlc.groupby(df_ohlc.index.to_period("W")):
        res = compute_directional_week(
            wk, capital=capital, pip_size=pip_size, pip_value=pip_value, point_value=point_value
        )
        weekly_rows.append({
            "week": week_key.start_time,
            "direction": res["direction"],
            "end_day": res["end_day"],
            "return_cap (%)": res["return_cap (%)"],
            "drawdown_cap (%)": res["drawdown_cap (%)"],
            "CR/DD": res["CR/DD"]
        })
        weekly_returns.append(res["return_cap (%)"]/100.0)

    df_weekly = pd.DataFrame(weekly_rows).set_index("week")

    if weekly_returns:
        mean_ret = np.mean(weekly_returns)
        std_ret = np.std(weekly_returns)
        sharpe = mean_ret / std_ret if std_ret != 0 else None
    else:
        sharpe = None

    df_weekly["Sharpe"] = sharpe
    return df_weekly

# -----------------------------
# Expanded Asset registry
# -----------------------------
ASSETS = [
    # FX majors
    {"name":"EUR/USD","ticker":"EURUSD=X","class":"FX","pip_size":0.0001,"pip_value":10},
    {"name":"GBP/JPY","ticker":"GBPJPY=X","class":"FX","pip_size":0.01,"pip_value":10},
    {"name":"USD/JPY","ticker":"USDJPY=X","class":"FX","pip_size":0.01,"pip_value":10},
    {"name":"USD/CHF","ticker":"USDCHF=X","class":"FX","pip_size":0.0001,"pip_value":10},
    {"name":"EUR/CHF","ticker":"EURCHF=X","class":"FX","pip_size":0.0001,"pip_value":10},
    {"name":"AUD/JPY","ticker":"AUDJPY=X","class":"FX","pip_size":0.01,"pip_value":10},
    {"name":"CAD/JPY","ticker":"CADJPY=X","class":"FX","pip_size":0.01,"pip_value":10},
    {"name":"NZD/JPY","ticker":"NZDJPY=X","class":"FX","pip_size":0.01,"pip_value":10},
    {"name":"EUR/CAD","ticker":"EURCAD=X","class":"FX","pip_size":0.0001,"pip_value":10},
    {"name":"GBP/CHF","ticker":"GBPCHF=X","class":"FX","pip_size":0.0001,"pip_value":10},

    # US Indices
    {"name":"S&P 500","ticker":"^GSPC","class":"Index","point_value":1.0},
    {"name":"Nasdaq","ticker":"^IXIC","class":"Index","point_value":1.0},
    {"name":"Dow Jones","ticker":"^DJI","class":"Index","point_value":1.0},

    # US Equities
    {"name":"AAPL","ticker":"AAPL","class":"Equity","point_value":1.0},
    {"name":"MSFT","ticker":"MSFT","class":"Equity","point_value":1.0},
    {"name":"AMZN","ticker":"AMZN","class":"Equity","point_value":1.0},
    {"name":"NVDA","ticker":"NVDA","class":"Equity","point_value":1.0},
    {"name":"JPM","ticker":"JPM","class":"Equity","point_value":1.0},
    {"name":"XOM","ticker":"XOM","class":"Equity","point_value":1.0},
    {"name":"JNJ","ticker":"JNJ","class":"Equity","point_value":1.0},
    {"name":"TSLA","ticker":"TSLA","class":"Equity","point_value":1.0},

    # Commodities (continuous futures tickers on Yahoo Finance)
    {"name":"Gold","ticker":"GC=F","class":"Commodity","point_value":1.0},
    {"name":"Crude Oil","ticker":"CL=F","class":"Commodity","point_value":1.0},
    {"name":"Natural Gas","ticker":"NG=F","class":"Commodity","point_value":1.0},
]

# -----------------------------
# Metrics explainer
# -----------------------------
with st.expander("📘 What do these metrics mean?"):
    st.markdown("""
**Metric Definitions**

- **direction:** Whether the dominant trade for the week was long ("up") or short ("down"), based on the highest return relative to Monday’s open.
- **end_day:** The day the dominant trade ended — the last candle used to compute return and drawdown.
- **return_cap (%):** Percentage gain on capital from Monday’s open to the end of the dominant move (always positive).
- **drawdown_cap (%):** Maximum adverse movement before reaching the end of the dominant move, measured as a percentage of capital (always positive).
- **CR/DD:** Capital Return divided by Drawdown — a measure of trade efficiency. Higher is better.
- **Sharpe:** Risk-adjusted return across all weeks. Higher values indicate more consistent performance.
- **granularity:** The data frequency used (either 4h or daily), depending on availability.
""")

# -----------------------------
# Build metrics_dict
# -----------------------------
def build_metrics_dict(assets=ASSETS, capital=10000, period="1y"):
    metrics_dict = {}
    for a in assets:
        df_ohlc, granularity = fetch_ohlc(a["ticker"], period=period)
        if df_ohlc.empty or granularity is None:
            continue

        # Skip assets missing required metadata
        if a["class"] == "FX" and ("pip_size" not in a or "pip_value" not in a):
            continue
        if a["class"] != "FX" and "point_value" not in a:
            continue

        if a["class"] == "FX":
            wdf = compute_weekly_metrics(
                df_ohlc, capital=capital,
                pip_size=a["pip_size"], pip_value=a["pip_value"]
            )
        else:
            wdf = compute_weekly_metrics(
                df_ohlc, capital=capital,
                pip_size=None, pip_value=None, point_value=a.get("point_value", 1.0)
            )
        wdf["granularity"] = granularity
        metrics_dict[a["name"]] = wdf
    return metrics_dict

# -----------------------------
# Optimization and decision intelligence helpers
# -----------------------------
def summarize_asset_metrics(df):
    avg_ret = float(df["return_cap (%)"].mean() / 100.0) if "return_cap (%)" in df else 0.0
    avg_dd = float(df["drawdown_cap (%)"].mean() / 100.0) if "drawdown_cap (%)" in df else 0.0
    avg_crdd = float(df["CR/DD"].mean()) if "CR/DD" in df else None
    sharpe = float(df["Sharpe"].mean()) if "Sharpe" in df else None
    return avg_ret, avg_dd, avg_crdd, sharpe

def filter_assets_by_multiplier(selected_assets, metrics_dict, multiplier):
    qualified = []
    excluded = []
    for asset in selected_assets:
        df = metrics_dict[asset]
        avg_ret, avg_dd, _, _ = summarize_asset_metrics(df)
        if avg_ret >= multiplier * avg_dd:
            qualified.append(asset)
        else:
            excluded.append((asset, avg_ret, avg_dd))
    return qualified, excluded

def optimize_portfolio(selected_assets, metrics_dict, max_portfolio_dd):
    returns = []
    drawdowns = []
    for asset in selected_assets:
        avg_ret, avg_dd, _, _ = summarize_asset_metrics(metrics_dict[asset])
        returns.append(avg_ret)
        drawdowns.append(avg_dd)

    returns = np.array(returns)
    drawdowns = np.array(drawdowns)
    n = len(selected_assets)
    if n == 0:
        return np.array([])

    def objective(weights):
        port_ret = float(np.dot(weights, returns))
        port_dd = float(np.dot(weights, drawdowns))
        if port_dd == 0:
            return -port_ret
        return -(port_ret / port_dd)

    cons = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'ineq', 'fun': lambda w: max_portfolio_dd - float(np.dot(w, drawdowns))}
    ]
    bounds = [(0.0, 1.0) for _ in range(n)]
    init_guess = np.ones(n) / n

    result = minimize(objective, init_guess, bounds=bounds, constraints=cons)
    return result.x if result.success else init_guess

def decision_intelligence_recommendation(metrics_dict, assets):
    best_asset = None
    best_score = -np.inf
    rec = None

    for asset in assets:
        df = metrics_dict[asset]
        latest = df.tail(1).iloc[0]
        crdd = latest.get("CR/DD", None)
        direction = latest.get("direction", None)
        end_day = latest.get("end_day", None)

        if crdd is not None and crdd > best_score:
            best_score = crdd
            best_asset = asset
            rec = {
                "asset": asset,
                "direction": direction,
                "exit_day": end_day,
                "crdd": crdd,
                "granularity": df["granularity"].iloc[0]
            }

    return rec

# -----------------------------
# Streamlit UI
# -----------------------------

# Ticker validation
st.sidebar.header("✅ Ticker Validation")

@st.cache_data(show_spinner=False)
def validate_tickers(asset_list):
    valid_assets = []
    invalid_assets = []
    for a in asset_list:
        try:
            df = yf.download(a["ticker"], period="5d", interval="1d", progress=False)
            if df.empty:
                invalid_assets.append(a["name"])
            else:
                valid_assets.append(a["name"])
        except:
            invalid_assets.append(a["name"])
    return valid_assets, invalid_assets

if st.sidebar.checkbox("Run ticker validation"):
    with st.spinner("Validating tickers..."):
        valid_assets, invalid_assets = validate_tickers(ASSETS)
        st.sidebar.success(f"✅ {len(valid_assets)} valid tickers")
        st.sidebar.write(", ".join(valid_assets))
        if invalid_assets:
            st.sidebar.error(f"⚠️ {len(invalid_assets)} invalid tickers")
            st.sidebar.write(", ".join(invalid_assets))

# Build metrics dictionary using selected period
metrics_dict = build_metrics_dict(period=selected_period)

st.sidebar.header("Asset Selection")
asset_names = list(metrics_dict.keys())
selected_assets = st.sidebar.multiselect("Choose assets to compare", asset_names, default=asset_names[:3])

view_mode = st.sidebar.radio("View mode", ["Latest week", "Average across weeks"])

# Scenario toggles (your personal constraints)
st.sidebar.header("Scenario toggles")
max_dd_pct = st.sidebar.slider("Max Portfolio Drawdown (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
min_return_dd_mult = st.sidebar.slider("Minimum Return-to-Drawdown Ratio (x)", min_value=1.0, max_value=5.0, value=3.0, step=0.5)

# Show analytics table(s)
if selected_assets:
    for asset in selected_assets:
        df = metrics_dict[asset]
        granularity = df["granularity"].iloc[0]

        if view_mode == "Latest week":
            latest = df.tail(1).T
            latest.columns = [f"{asset} ({granularity})"]
            st.subheader(f"{asset} ({granularity} data) — Latest Week")
            st.dataframe(latest)
        else:
            avg = df.drop(columns=["granularity"]).mean(numeric_only=True).to_frame()
            avg.columns = [f"{asset} ({granularity})"]
            st.subheader(f"{asset} ({granularity} data) — Average Across Weeks")
            st.dataframe(avg)
else:
    st.warning("Please select at least one asset from the sidebar.")

st.markdown("---")

# -----------------------------
# Prescriptive Optimization & Recommendations
# -----------------------------
if selected_assets:
    st.header("🔮 Prescriptive optimization and risk recommendations")

    # 1) Filter assets by return-to-drawdown multiplier
    qualified_assets, excluded_assets = filter_assets_by_multiplier(selected_assets, metrics_dict, min_return_dd_mult)

    # Show qualification status
    st.subheader("Asset qualification under current scenario")
    if qualified_assets:
        st.write("Qualified assets (return ≥ multiplier × drawdown):")
        st.write(", ".join(qualified_assets))
    else:
        st.warning("No assets qualify under the current return-to-drawdown threshold. Consider lowering the multiplier or selecting different assets.")

    if excluded_assets:
        st.write("Excluded assets and reasons:")
        ex_df = pd.DataFrame(excluded_assets, columns=["Asset", "Avg Return (fraction)", "Avg Drawdown (fraction)"])
        ex_df["Return-to-Drawdown (x)"] = ex_df["Avg Return (fraction)"] / ex_df["Avg Drawdown (fraction)"]
        st.dataframe(ex_df)

    # 2) Optimize portfolio on qualified assets subject to max drawdown
    st.subheader("Optimal allocation under constraints")
    if qualified_assets:
        weights = optimize_portfolio(qualified_assets, metrics_dict, max_dd_pct / 100.0)

        alloc_df = pd.DataFrame({"Asset": qualified_assets, "Weight (%)": (weights * 100.0)})
        st.dataframe(alloc_df.set_index("Asset"))
        st.bar_chart(alloc_df.set_index("Asset"))

        rets, dds = [], []
        for asset in qualified_assets:
            avg_ret, avg_dd, avg_crdd, sharpe = summarize_asset_metrics(metrics_dict[asset])
            rets.append(avg_ret)
            dds.append(avg_dd)
        port_ret = float(np.dot(weights, np.array(rets)))
        port_dd = float(np.dot(weights, np.array(dds)))
        port_eff = (port_ret / port_dd) if port_dd != 0 else None

        st.markdown(f"**Recommendation:** Allocate capital as above to maximize risk-adjusted return under a maximum portfolio drawdown of {max_dd_pct:.2f}%.")
        st.markdown(f"**Estimated portfolio return:** {port_ret*100:.2f}% per (average) week")
        st.markdown(f"**Estimated portfolio drawdown:** {port_dd*100:.2f}%")
        if port_eff is not None:
            st.markdown(f"**Efficiency (Return/Drawdown):** {port_eff:.2f}x")

        with st.expander("Model summary (objective, constraints, decision variables)"):
            st.markdown(f"""
**Decision variables:** Weights \\(w_i\\) for each qualified asset (non-negative).  
**Objective:** Maximize \\( \\frac{{\\sum_i w_i \\cdot r_i}}{{\\sum_i w_i \\cdot d_i}} \\) where \\(r_i\\) is average return (fraction) and \\(d_i\\) is average drawdown (fraction).  
**Constraints:**  
- Sum of weights: \\( \\sum_i w_i = 1 \\)  
- Non-negativity: \\( w_i \\ge 0 \\)  
- Portfolio drawdown cap: \\( \\sum_i w_i \\cdot d_i \\le {max_dd_pct/100.0:.4f} \\)  
**Pre-filter rule:** Asset qualifies only if \\( r_i \\ge {min_return_dd_mult:.1f} \\cdot d_i \\).
""")
    else:
        st.info("No optimization performed because no assets qualified under the current scenario.")

    # 3) Decision intelligence: trade-level suggestion
    st.subheader("Trade-level suggestion (decision intelligence)")
    di_rec = decision_intelligence_recommendation(metrics_dict, qualified_assets if qualified_assets else selected_assets)
    if di_rec:
        st.markdown(f"""
**Suggested trade:** {di_rec['asset']} — go **{di_rec['direction']}** this week.  
**Exit target:** {di_rec['exit_day']}  
**Rationale:** Highest recent CR/DD ({di_rec['crdd']}) among the considered assets.  
**Data granularity:** {di_rec['granularity']}
        """)
    else:
        st.info("No trade suggestion available due to missing CR/DD or direction metrics.")
