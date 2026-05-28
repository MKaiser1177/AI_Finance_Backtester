# PF01 Paper Input Files

Put real data CSV files in this folder to run paper-grade PF01 tests.

## Required for full compliance

- `nifty500_tri.csv`
  - Columns: `Date,TRI`
- `rbi_91d_tbill.csv`
  - Columns: `Date,AnnualRate`
  - `AnnualRate` must be decimal (example `0.0675`)
- `constituents_pti.csv`
  - Columns: `Date,Symbol`
  - One row per symbol per point-in-time rebalance date
- `liquidbees.csv`
  - Columns: `Date,Close`

## Strongly recommended

- `sectors.csv`
  - Columns: `Symbol,Sector`
- `eligibility_flags.csv`
  - Columns: `Date,Symbol,PledgePct,OnCaution,TradeToTrade`
  - `PledgePct` decimal percentage (0 means passes)
  - `OnCaution`, `TradeToTrade` are 0/1 flags

If any file is missing, the backtest falls back to internal proxies and prints a warning.
