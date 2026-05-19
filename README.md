# Nifty 500 Backtesting Framework

A Python-based backtesting framework for testing trading strategies on 50 representative stocks from the Nifty 500 index. This project includes data generation, strategy implementation, and detailed performance analysis with visualizations.

## Overview

This framework allows you to:
- Generate synthetic or real historical OHLCV (Open, High, Low, Close, Volume) data for Indian stocks
- Test multiple trading strategies on historical data
- Generate detailed backtest reports and performance charts
- Compare strategy performance across different stocks

## Project Structure

```
├── backtest.py              # Main backtesting engine
├── generate_data.py         # Synthetic data generation script
├── data/                    # Historical stock data (CSV files)
├── backtest_results/        # Output folder for backtest results
└── README.md               # This file
```

## Available Strategies

### 1. **SMA Crossover**
Buy when the 50-day Simple Moving Average crosses above the 200-day SMA; sell on cross below.

### 2. **RSI Mean Reversion**
Buy when the RSI(14) drops below 30; sell when RSI exceeds 70.

### 3. **Momentum**
Buy the top-N% stocks by 12-month return each month; rebalance monthly.

## Quick Start

### Generate Data
```bash
python generate_data.py
```
Creates synthetic OHLCV data for all stocks in the `data/` directory.

### Run Backtest

**Default (all stocks, SMA strategy):**
```bash
python backtest.py
```

**Specific strategy:**
```bash
python backtest.py --strategy sma      # SMA Crossover
python backtest.py --strategy rsi      # RSI Mean Reversion
python backtest.py --strategy momentum # Momentum
```

**Specific tickers:**
```bash
python backtest.py --tickers TCS INFY HDFCBANK --strategy sma
```

## Output

Each backtest generates:
- **backtest_summary.csv** - Performance metrics (returns, Sharpe ratio, drawdown, etc.)
- **{TICKER}_equity.csv** - Daily portfolio value for each stock
- **{TICKER}_chart.png** - Price chart with buy/sell signals and moving averages

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib

## Data

The `data/` directory contains historical or synthetic OHLCV data for stocks including:
- IT: TCS, INFY, WIPRO, TECHM, HCLTECH, MPHASIS
- Banking: HDFCBANK, ICICIBANK, KOTAKBANK, AXISBANK, SBIN, and more
- NBFC/Insurance: BAJFINANCE, BAJAJFINSV, HDFCAMC, SBILIFE
- Energy: RELIANCE, ONGC, BPCL, POWERGRID
- FMCG: HINDUNILVR, ITC, NESTLEIND, BRITANNIA
- And others across various sectors

## Notes

- Results are saved to `backtest_results/` by default
- Use different result directories for comparison (backtest_results_test, backtest_results_test2, etc.)
- All prices are in INR
- Data spans 10 years (2015-2024)
