# Markowitz-Portfolio-Optimization-in-Python
This project constructs an optimized equity portfolio from S&amp;P 500 companies using the Markowitz framework. The portfolio is rebalanced at fixed calendar intervals using historical daily returns and compared to the performance of the S&amp;P 500 index (SPY).


##  Features

- Automatically fetches S&P 500 constituents and market capitalizations
- Selects top sectors and top companies by market cap
- Downloads historical adjusted closing prices via Yahoo Finance
- Optimizes portfolio weights by maximizing the Sharpe ratio
- Rebalances periodically using a rolling return window
- Compares portfolio performance to SPY (benchmark)

##  Configuration

You can configure:

- Date range for training and evaluation
- Number of top sectors and companies per sector
- Rolling window length (in calendar days)
- Rebalancing frequency
- Risk-free rate

All parameters can be adjusted in the script under the `CONFIGURATION PARAMETERS` section.

##  Installation

First, clone the repository and create a virtual environment:

```bash
git clone https://github.com/yourusername/portfolio-optimization.git
cd portfolio-optimization
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
