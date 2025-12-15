# 📊 Multi-Asset Risk Optimization Dashboard

This interactive Streamlit dashboard helps investors and analysts make smarter, risk-aware decisions across multiple asset classes — including currencies, commodities, equities, and indices.

---

## 🚀 Try It Live

👉 [Launch the App](https://asset-allocation-optimizer-steqldnhgosh7jezsyzhhw.streamlit.app)

---

## 🎬 Demo Video

[Watch the 3-minute walkthrough →](https://www.loom.com/share/8ee64f5f1e474b60a533634f988d1a48)

This video covers:

- **The Problem (60 sec):** Why risk-aware asset selection matters in volatile markets.
- **Your Approach (60 sec):** How the dashboard uses prescriptive analytics to filter and optimize.
- **Live Demo (90–120 sec):** A walkthrough of the app, including asset selection, scenario tuning, and portfolio recommendations.
- **What You Learned (30–60 sec):** Key takeaways from building a real-world optimization tool.


## 🔍 Features

- ✅ **Ticker Validation**: Instantly check which assets are currently supported via live data.
- 📅 **Historical Period Selector**: Choose between 3 months, 6 months, or 1 year of historical data.
- 📈 **Weekly Trade Metrics**: Analyze dominant weekly trades with return, drawdown, CR/DD, and Sharpe ratio.
- 🔄 **View Mode Toggle**: Switch between the latest week or the average across the selected period.
- 🎯 **Prescriptive Portfolio Optimization**: Allocate capital based on your risk constraints and return-to-drawdown preferences.
- 🤖 **Trade-Level Recommendation**: Get a weekly suggestion based on the highest CR/DD asset.

---

## 🧠 How It Works

1. **Data Fetching**: Uses `yfinance` to pull OHLC data for selected assets.
2. **Weekly Analysis**: Computes directional trades, drawdowns, and CR/DD for each week.
3. **Optimization**: Solves a constrained optimization problem to maximize return/drawdown efficiency.
4. **Decision Intelligence**: Recommends a trade based on the most efficient recent asset.

---

## 📂 Asset Coverage

- **FX Majors**: EUR/USD, GBP/JPY, USD/JPY, etc.
- **US Indices**: S&P 500, Nasdaq, Dow Jones
- **Equities**: AAPL, MSFT, TSLA, etc.
- **Commodities**: Gold, Crude Oil, Natural Gas

---

## 🛠 Tech Stack

- Python
- Streamlit
- yfinance
- NumPy, Pandas, SciPy

---

## ▶️ Getting Started Locally

```bash
git clone https://github.com/solomonnjenga11-ai/Asset-Allocation-Optimizer.git
cd Asset-Allocation-Optimizer
pip install -r requirements.txt
streamlit run app.py
