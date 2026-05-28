"""
Utilities to load paper-grade PF01 input datasets from local CSV files.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import pandas as pd


def _read_csv(path: str) -> Optional[pd.DataFrame]:
    if not path or not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_nifty500_tri(data_dir: str) -> Optional[pd.Series]:
    """
    Expected CSV: nifty500_tri.csv
    Columns: Date, TRI
    """
    path = os.path.join(data_dir, "nifty500_tri.csv")
    df = _read_csv(path)
    if df is None:
        return None
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.sort_values("Date").set_index("Date")
    return out["TRI"].astype(float)


def load_liquidbees(data_dir: str) -> Optional[pd.Series]:
    """
    Expected CSV: liquidbees.csv
    Columns: Date, Close
    """
    path = os.path.join(data_dir, "liquidbees.csv")
    df = _read_csv(path)
    if df is None:
        return None
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.sort_values("Date").set_index("Date")
    return out["Close"].astype(float)


def load_rf_annual_series(data_dir: str) -> Optional[pd.Series]:
    """
    Expected CSV: rbi_91d_tbill.csv
    Columns: Date, AnnualRate
    AnnualRate should be decimal (e.g. 0.0675 for 6.75%).
    """
    path = os.path.join(data_dir, "rbi_91d_tbill.csv")
    df = _read_csv(path)
    if df is None:
        return None
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.sort_values("Date").set_index("Date")
    return out["AnnualRate"].astype(float)


def load_constituents_pti(data_dir: str) -> Optional[Dict[pd.Timestamp, List[str]]]:
    """
    Expected CSV: constituents_pti.csv
    Columns: Date, Symbol
    """
    path = os.path.join(data_dir, "constituents_pti.csv")
    df = _read_csv(path)
    if df is None:
        return None
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out["Symbol"] = out["Symbol"].astype(str).str.strip()
    grouped = out.groupby("Date")["Symbol"].apply(list)
    return {k: v for k, v in grouped.items()}


def load_sector_map(data_dir: str) -> Dict[str, str]:
    """
    Expected CSV: sectors.csv
    Columns: Symbol, Sector
    """
    path = os.path.join(data_dir, "sectors.csv")
    df = _read_csv(path)
    if df is None:
        return {}
    out = df.copy()
    out["Symbol"] = out["Symbol"].astype(str).str.strip()
    out["Sector"] = out["Sector"].astype(str).str.strip()
    return dict(zip(out["Symbol"], out["Sector"]))


def load_eligibility_flags(data_dir: str) -> Optional[pd.DataFrame]:
    """
    Optional CSV: eligibility_flags.csv
    Columns:
      Date, Symbol, PledgePct, OnCaution, TradeToTrade
    """
    path = os.path.join(data_dir, "eligibility_flags.csv")
    df = _read_csv(path)
    if df is None:
        return None
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out["Symbol"] = out["Symbol"].astype(str).str.strip()
    for c in ["PledgePct", "OnCaution", "TradeToTrade"]:
        if c not in out.columns:
            out[c] = 0
    out["PledgePct"] = pd.to_numeric(out["PledgePct"], errors="coerce").fillna(0.0)
    out["OnCaution"] = pd.to_numeric(out["OnCaution"], errors="coerce").fillna(0).astype(int)
    out["TradeToTrade"] = pd.to_numeric(out["TradeToTrade"], errors="coerce").fillna(0).astype(int)
    return out.sort_values(["Date", "Symbol"])

