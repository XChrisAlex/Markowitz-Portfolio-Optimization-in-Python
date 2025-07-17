import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta
from scipy.optimize import minimize
import matplotlib.pyplot as plt

#======================== IT NEEDS A COUPLE OF MINUTES TO RUN ========================# 


# ========================
# CONFIGURATION PARAMETERS
# ========================



# Training on Jan–Sep 2024
DATA_START_DATE = "2023-06-30" #it has to be greater or equal to "PERFORMANCE_START_DATE"
DATA_END_DATE = "2025-06-30" #it has to be greater or equal to "PERFORMANCE END DATE".Note:We wont be training data before the Performance date before

# Evaluation on Oct–Dec 2024
PERFORMANCE_START_DATE = "2024-12-31"
PERFORMANCE_END_DATE = "2025-04-30"

# Rebalancing frequency (in calendar days)
REBALANCE_FREQ_DAYS = 31
# Portfolio construction settings - Use this or edit the tickers list below adjusted to you preferences
TOP_SECTORS = 5
TOP_COMPANIES_PER_SECTOR = 4
ROLLING_WINDOW_DAYS = 365*2 # this is later used as calendar dates (aproxim. into 252 rows)
RISK_FREE_RATE = 0.03 

# ==========================
# 1. LOAD S&P 500 COMPANY DATA
# ==========================

url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
sp500_table = pd.read_html(url, header=0)[0]

sp500 = sp500_table[['Symbol', 'Security', 'GICS Sector', 'GICS Sub-Industry']].copy()
sp500.rename(columns={
    'Symbol': 'Ticker',
    'Security': 'Company',
    'GICS Sector': 'Sector',
    'GICS Sub-Industry': 'Industry'
}, inplace=True)
sp500['Ticker'] = sp500['Ticker'].str.replace('.', '-', regex=False)

# ==========================
# 2. FETCH MARKET CAPITALIZATIONS
# ==========================

tickers = sp500['Ticker'].tolist()
market_caps = []

for ticker in tickers:
    try:
        info = yf.Ticker(ticker).info
        market_caps.append(info.get('marketCap', None))
    except:
        market_caps.append(None)

sp500 = sp500.loc[:len(tickers) - 1].copy()
sp500['MarketCap'] = market_caps
sp500 = sp500.dropna(subset=['MarketCap']).reset_index(drop=True)

# ==========================
# 3. SELECT TOP SECTORS AND COMPANIES
# ==========================

sector_caps = sp500.groupby("Sector")["MarketCap"].sum().sort_values(ascending=False)
top_sectors = sector_caps.head(TOP_SECTORS).index.tolist()

print("Top Sectors by Total Market Cap:")
for sector in top_sectors:
    print("-", sector)

top_sector_companies = sp500[sp500["Sector"].isin(top_sectors)]

top_companies = (
    top_sector_companies
    .sort_values(["Sector", "MarketCap"], ascending=[True, False])
    .groupby("Sector")
    .head(TOP_COMPANIES_PER_SECTOR)
    .reset_index(drop=True)
)

print("\nTop Companies per Sector:")
print(top_companies.head(25))

# ==========================
# 4. DOWNLOAD PRICE DATA
# ==========================

tickers = top_companies['Ticker'].unique().tolist()
price_data = yf.download(tickers, start=DATA_START_DATE, end=DATA_END_DATE)["Close"]

print("\nSample of Adjusted Prices:")
print(price_data.head())

daily_returns = price_data.pct_change().dropna()

# ==========================
# 5. PORTFOLIO OPTIMIZATION FUNCTIONS
# ==========================

def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)) * 252)
    return returns, std

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_return, p_std = portfolio_performance(weights, mean_returns, cov_matrix)
    return - (p_return - risk_free_rate) / p_std

def get_constraints(num_assets):
    return ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

def get_bounds(num_assets):
    return tuple((0, 1) for _ in range(num_assets))

def optimize_weights(returns_window, risk_free_rate=RISK_FREE_RATE):
    mean_returns = returns_window.mean() * 252
    cov_matrix = returns_window.cov()
    num_assets = len(mean_returns)
    init_guess = num_assets * [1. / num_assets]

    result = minimize(
        neg_sharpe_ratio,
        init_guess,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=get_bounds(num_assets),
        constraints=get_constraints(num_assets)
    )

    return result.x

# ==========================
# 6. REBALANCING ROUTINE
# ==========================

rebalance_dates = pd.date_range(
    start=PERFORMANCE_START_DATE, 
    end=PERFORMANCE_END_DATE, 
    freq=f'{REBALANCE_FREQ_DAYS}D'
)

rebalance_results = []

for rebalance_date in rebalance_dates:
    start_window = rebalance_date - timedelta(days=ROLLING_WINDOW_DAYS)
    returns_window = daily_returns.loc[start_window:rebalance_date]

    print(f"Rebalancing on {rebalance_date.date()} using data from {returns_window.index[0].date()} to {returns_window.index[-1].date()}")
    
    weights = optimize_weights(returns_window)
    rebalance_results.append({
        "Date": rebalance_date,
        "Weights": weights
    })


rebalance_df = pd.DataFrame(rebalance_results)
weights_matrix = pd.DataFrame(
    rebalance_df["Weights"].tolist(),
    index=rebalance_df["Date"],
    columns=daily_returns.columns
)

print("\nPortfolio Weights Over Time:")
print(weights_matrix.head())

# ==========================
# 7. PORTFOLIO PERFORMANCE TRACKING
# ==========================

print("Tracking date range:", PERFORMANCE_START_DATE, "to", PERFORMANCE_END_DATE)
print("Available return dates:", daily_returns.index.min().date(), "to", daily_returns.index.max().date())


spy_data = yf.download("SPY", start=PERFORMANCE_START_DATE, end=PERFORMANCE_END_DATE)["Close"]
spy_values = spy_data / spy_data.iloc[0]

tracking_dates = daily_returns.loc[PERFORMANCE_START_DATE:PERFORMANCE_END_DATE].index
portfolio_values = pd.Series(index=tracking_dates, dtype='float64')
portfolio_values.iloc[0] = 1

current_weights = weights_matrix.loc[rebalance_dates[0]].values
current_rebalance_index = 0

for i in range(1, len(tracking_dates)):
    current_date = tracking_dates[i]
    
    if current_rebalance_index + 1 < len(rebalance_dates) and current_date >= rebalance_dates[current_rebalance_index + 1]:
        current_rebalance_index += 1
        current_weights = weights_matrix.loc[rebalance_dates[current_rebalance_index]].values

    returns_today = daily_returns.loc[current_date].values
    portfolio_values.iloc[i] = portfolio_values.iloc[i - 1] * (1 + np.dot(current_weights, returns_today))

spy_values_aligned = spy_values.reindex(tracking_dates).ffill()

# ==========================
# 8. PLOT PERFORMANCE
# ==========================

plt.figure(figsize=(10, 5))
plt.plot(portfolio_values, label="Optimized Portfolio", linewidth=2)
plt.plot(spy_values_aligned, label="S&P 500 (SPY)", linestyle='--')

plt.title("Portfolio vs. S&P 500 (SPY) - Starting Value: $1")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
