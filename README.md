# Nifty 500 Backtesting Framework

A Python-based backtesting framework for Indian equity strategies, including a paper-driven PF01 dual momentum implementation (V0-V5).

## Overview

This framework allows you to:
- Generate synthetic OHLCV data for quick local testing
- Run classic strategies (SMA, RSI, simple momentum)
- Run PF01 dual momentum variants V0 to V5 from the paper specification
- Export variant-level metrics and calibration/holdout summaries

## Project Structure

```
├── backtest.py               # Main backtesting engine and CLI
├── dual_momentum_pf01.py     # PF01 strategy logic (V0-V5)
├── pf01_data_loader.py       # PF01 paper input CSV loaders
├── generate_data.py          # Synthetic data generation script
├── data/                     # Per-ticker OHLCV files (Date, Open, High, Low, Close, Volume)
├── paper_inputs/             # PF01 paper-grade CSV inputs (TRI, RF, PTI, etc.)
├── backtest_results/         # Output folder for backtest results
└── README.md                 # This file
```

## Available Strategies

### 1. **SMA Crossover**
Buy when the 50-day Simple Moving Average crosses above the 200-day SMA; sell on cross below.

### 2. **RSI Mean Reversion**
Buy when the RSI(14) drops below 30; sell when RSI exceeds 70.

### 3. **Momentum**
Buy the top-N% stocks by 12-month return each month; rebalance monthly.

### 4. **PF01 Dual Momentum (Paper Strategy)**
Implements all variants from the paper:
- `V0`: baseline raw 12-1 momentum
- `V1`: volatility-adjusted composite (M6_adj, M12_adj, WH52)
- `V2`: V1 + per-stock absolute filter
- `V3`: V0 + EMA(21) gate
- `V4`: V1 + EMA(21) gate
- `V5`: V1 + per-stock absolute filter + EMA(21) gate

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
python backtest.py --strategy pf01 --variant V5
```

**Specific tickers:**
```bash
python backtest.py --tickers TCS INFY HDFCBANK --strategy sma
```

### Run PF01 for all variants
```bash
python backtest.py --strategy pf01 --pf01-run-all-variants --out backtest_results_pf01
```

### PF01 with paper inputs
```bash
python backtest.py \
  --strategy pf01 \
  --pf01-run-all-variants \
  --pf01-inputs-dir paper_inputs \
  --pf01-calibration-end 2017-12-31 \
  --tc-roundtrip 0.0032 \
  --rf-annual 0.06 \
  --out backtest_results_pf01
```

## Output

Standard (SMA/RSI/Momentum) backtests generate:
- **backtest_summary.csv** - Performance metrics (returns, Sharpe ratio, drawdown, etc.)
- **{TICKER}_equity.csv** - Daily portfolio value for each stock
- **{TICKER}_chart.png** - Price chart with buy/sell signals and moving averages

PF01 generates:
- **pf01_V{n}_equity.csv** - Daily portfolio return, equity, turnover, regime, gate state
- **pf01_V{n}_summary.csv** - Variant-level metrics
- **pf01_variants_summary.csv** - All variant metrics in one file
- **pf01_calibration_holdout_summary.csv** - Split-window metrics using `--pf01-calibration-end`

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- yfinance (optional fallback data download)

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
- Use different output folders for clean experiment tracking
- PF01 paper-grade run expects CSVs in `paper_inputs/` (see `paper_inputs/README.md`)
- If paper input files are missing, PF01 falls back to internal proxies and logs warnings
