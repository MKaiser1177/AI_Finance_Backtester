"""
PF01 dual momentum backtest engine (V0-V5).

This module implements the strategy logic from:
"Dual Momentum in Indian Equities: Design, Variants, and Implementation".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


VARIANTS = {"V0", "V1", "V2", "V3", "V4", "V5"}


@dataclass
class PF01Config:
    variant: str = "V5"
    top_n: int = 25
    sector_cap: float = 0.20
    adtv_floor: float = 5e7
    rf_annual_default: float = 0.06
    round_trip_cost: float = 0.0032
    ema_span: int = 21
    ema_exit_buffer: float = 0.99


def _is_composite_variant(variant: str) -> bool:
    return variant in {"V1", "V2", "V4", "V5"}


def _has_stock_abs_filter(variant: str) -> bool:
    return variant in {"V2", "V5"}


def _has_ema_gate(variant: str) -> bool:
    return variant in {"V3", "V4", "V5"}


def _annualized_vol(returns: pd.Series) -> float:
    sd = returns.std()
    if pd.isna(sd) or sd == 0:
        return np.nan
    return float(sd * np.sqrt(252))


def _sharpe(returns: pd.Series, rf_annual: float) -> float:
    excess = returns - (rf_annual / 252)
    sd = excess.std()
    if pd.isna(sd) or sd == 0:
        return np.nan
    return float(np.sqrt(252) * excess.mean() / sd)


def _sortino(returns: pd.Series, rf_annual: float) -> float:
    excess = returns - (rf_annual / 252)
    downside = excess[excess < 0]
    dd = downside.std()
    if pd.isna(dd) or dd == 0:
        return np.nan
    return float(np.sqrt(252) * excess.mean() / dd)


def _cagr(equity: pd.Series) -> float:
    if len(equity) < 2:
        return np.nan
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return np.nan
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan
    roll_max = equity.cummax()
    dd = (equity / roll_max) - 1.0
    return float(dd.min())


def _safe_rank_desc(values: pd.Series) -> pd.Series:
    return values.rank(ascending=False, method="average")


def _nearest_trading_day(idx: pd.DatetimeIndex, target: pd.Timestamp) -> Optional[pd.Timestamp]:
    loc = idx.get_indexer([target], method="ffill")[0]
    if loc < 0:
        return None
    return idx[loc]


def _compute_rf_12m(
    date: pd.Timestamp,
    rf_series_annual: Optional[pd.Series],
    default_annual: float,
) -> float:
    if rf_series_annual is None or rf_series_annual.empty:
        return (1 + default_annual) ** 1 - 1
    rf = rf_series_annual.reindex(rf_series_annual.index.union([date])).sort_index().ffill().loc[date]
    return float((1 + rf) ** 1 - 1)


def _score_baseline_12_1(closes: pd.DataFrame, rebal_pos: int, universe: List[str]) -> pd.Series:
    p21 = closes.iloc[max(0, rebal_pos - 21)][universe]
    p252 = closes.iloc[max(0, rebal_pos - 252)][universe]
    signal = (p21 / p252) - 1
    return signal.replace([np.inf, -np.inf], np.nan).dropna()


def _score_composite(closes: pd.DataFrame, rebal_pos: int, universe: List[str]) -> pd.Series:
    window252 = closes.iloc[max(0, rebal_pos - 252) : rebal_pos + 1][universe]
    log_ret = np.log(window252 / window252.shift(1))
    vol252 = log_ret.std() * np.sqrt(252)

    p21 = closes.iloc[max(0, rebal_pos - 21)][universe]
    p126 = closes.iloc[max(0, rebal_pos - 126)][universe]
    p252 = closes.iloc[max(0, rebal_pos - 252)][universe]
    p0 = closes.iloc[rebal_pos][universe]
    whmax = closes.iloc[max(0, rebal_pos - 252) : rebal_pos + 1][universe].max()

    m6 = (p21 / p126) - 1
    m12 = (p21 / p252) - 1
    wh52 = p0 / whmax
    m6_adj = m6 / vol252
    m12_adj = m12 / vol252

    frame = pd.DataFrame({"m6": m6_adj, "m12": m12_adj, "wh52": wh52}).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return pd.Series(dtype=float)

    rank_m6 = _safe_rank_desc(frame["m6"])
    rank_m12 = _safe_rank_desc(frame["m12"])
    rank_wh52 = _safe_rank_desc(frame["wh52"])
    return (rank_m6 + rank_m12 + rank_wh52) / 3.0


def _apply_sector_cap(
    selected: List[str],
    ranked: List[str],
    sectors: Dict[str, str],
    cap_weight: float,
    top_n: int,
) -> List[str]:
    if not selected:
        return selected

    max_names_per_sector = int(np.floor(cap_weight * top_n))
    if max_names_per_sector <= 0:
        return selected

    selected = list(selected)
    sector_counts: Dict[str, int] = {}
    for t in selected:
        sec = sectors.get(t, "UNKNOWN")
        sector_counts[sec] = sector_counts.get(sec, 0) + 1

    replaced = True
    while replaced:
        replaced = False
        overflow_sec = None
        for sec, count in sector_counts.items():
            if count > max_names_per_sector:
                overflow_sec = sec
                break
        if overflow_sec is None:
            break

        drop_ticker = None
        for t in reversed(ranked):
            if t in selected and sectors.get(t, "UNKNOWN") == overflow_sec:
                drop_ticker = t
                break
        if drop_ticker is None:
            break

        add_ticker = None
        for t in ranked:
            if t in selected:
                continue
            if sector_counts.get(sectors.get(t, "UNKNOWN"), 0) + 1 > max_names_per_sector:
                continue
            add_ticker = t
            break
        if add_ticker is None:
            break

        selected.remove(drop_ticker)
        selected.append(add_ticker)
        sector_counts[overflow_sec] -= 1
        new_sec = sectors.get(add_ticker, "UNKNOWN")
        sector_counts[new_sec] = sector_counts.get(new_sec, 0) + 1
        replaced = True

    return selected


def run_pf01(
    data: Dict[str, pd.DataFrame],
    variant: str = "V5",
    nifty500_tri: Optional[pd.Series] = None,
    rf_annual_series: Optional[pd.Series] = None,
    constituents_pti: Optional[Dict[pd.Timestamp, List[str]]] = None,
    sectors: Optional[Dict[str, str]] = None,
    liquidbees_close: Optional[pd.Series] = None,
    eligibility_flags: Optional[pd.DataFrame] = None,
    config: Optional[PF01Config] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - daily dataframe with returns/equity/regime state
      - one-row metrics dataframe for the chosen variant
    """
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Expected one of: {sorted(VARIANTS)}")

    cfg = config or PF01Config(variant=variant)
    cfg.variant = variant

    closes = pd.DataFrame({t: df["Close"] for t, df in data.items()}).sort_index()
    volumes = pd.DataFrame({t: df["Volume"] for t, df in data.items()}).sort_index()
    closes = closes.ffill()
    volumes = volumes.fillna(0.0)
    if closes.empty:
        raise ValueError("No close-price data available for PF01 backtest.")

    all_dates = closes.index
    if nifty500_tri is None or nifty500_tri.empty:
        # Practical fallback: use equal-weight universe proxy as market TRI.
        nifty500_tri = closes.mean(axis=1)
    else:
        nifty500_tri = nifty500_tri.reindex(all_dates).ffill().bfill()

    if liquidbees_close is not None and not liquidbees_close.empty:
        liquid_rets = liquidbees_close.reindex(all_dates).ffill().pct_change().fillna(0.0)
    else:
        liquid_rets = pd.Series((cfg.rf_annual_default / 252), index=all_dates)

    month_ends = closes.resample("ME").last().index
    rebalance_days = [_nearest_trading_day(all_dates, d) for d in month_ends]
    rebalance_days = [d for d in rebalance_days if d is not None]
    rebalance_set = set(rebalance_days)

    one_way_cost = cfg.round_trip_cost / 2.0
    sectors = sectors or {}

    weights: Dict[str, float] = {"LIQUIDBEES": 1.0}
    regime = "RISK-OFF"
    gate_composition: Dict[str, float] = {}
    last_monthly_composition: Dict[str, float] = {}

    daily_ret = pd.Series(index=all_dates, dtype=float)
    turnover = pd.Series(0.0, index=all_dates)
    regime_s = pd.Series(index=all_dates, dtype=object)
    gate_state_s = pd.Series(False, index=all_dates)
    nav = pd.Series(index=all_dates, dtype=float)
    nav.iloc[0] = 1.0

    for i, dt in enumerate(all_dates):
        if i == 0:
            regime_s.iloc[i] = regime
            gate_state_s.iloc[i] = regime == "GATE-OFF"
            daily_ret.iloc[i] = 0.0
            continue

        ret_today = 0.0
        px_ret = closes.iloc[i] / closes.iloc[i - 1] - 1.0
        for t, w in weights.items():
            if t == "LIQUIDBEES":
                ret_today += w * float(liquid_rets.iloc[i])
            else:
                rt = float(px_ret.get(t, 0.0))
                if np.isnan(rt):
                    rt = 0.0
                ret_today += w * rt

        daily_ret.iloc[i] = ret_today
        nav.iloc[i] = nav.iloc[i - 1] * (1.0 + ret_today)

        if dt in rebalance_set:
            rebal_pos = i
            rf_12m = _compute_rf_12m(dt, rf_annual_series, cfg.rf_annual_default)
            market_12m = (nifty500_tri.iloc[i] / nifty500_tri.iloc[max(0, i - 252)]) - 1.0

            if market_12m <= rf_12m:
                target_weights = {"LIQUIDBEES": 1.0}
                new_regime = "RISK-OFF"
            else:
                if constituents_pti:
                    keys = sorted(constituents_pti.keys())
                    key = max([k for k in keys if k <= dt], default=keys[0])
                    universe = [t for t in constituents_pti[key] if t in closes.columns]
                else:
                    universe = list(closes.columns)

                eligible = []
                adtv_60 = (
                    closes.iloc[max(0, i - 60) : i + 1][universe]
                    * volumes.iloc[max(0, i - 60) : i + 1][universe]
                ).mean(axis=0)
                hist_ok = closes.iloc[: i + 1][universe].count(axis=0) >= 252
                if eligibility_flags is not None and not eligibility_flags.empty:
                    flags_now = eligibility_flags[eligibility_flags["Date"] <= dt].sort_values("Date")
                    flags_now = flags_now.drop_duplicates(subset=["Symbol"], keep="last").set_index("Symbol")
                else:
                    flags_now = None
                for t in universe:
                    base_ok = bool(hist_ok.get(t, False)) and float(adtv_60.get(t, 0.0)) >= cfg.adtv_floor
                    if not base_ok:
                        continue
                    if flags_now is not None and t in flags_now.index:
                        pledge_ok = float(flags_now.loc[t, "PledgePct"]) <= 0.0
                        caution_ok = int(flags_now.loc[t, "OnCaution"]) == 0
                        t2t_ok = int(flags_now.loc[t, "TradeToTrade"]) == 0
                        if not (pledge_ok and caution_ok and t2t_ok):
                            continue
                    eligible.append(t)

                if _is_composite_variant(variant):
                    comp = _score_composite(closes, rebal_pos, eligible)
                    ranked = comp.sort_values().index.tolist()
                    rank_map = {t: r + 1 for r, t in enumerate(ranked)}
                else:
                    raw = _score_baseline_12_1(closes, rebal_pos, eligible)
                    ranked = raw.sort_values(ascending=False).index.tolist()
                    rank_map = {t: r + 1 for r, t in enumerate(ranked)}

                current_names = [t for t in weights if t != "LIQUIDBEES"]
                final: List[str] = []
                for t in current_names:
                    if len(final) < cfg.top_n and t in rank_map and rank_map[t] <= 30:
                        final.append(t)
                for t in ranked:
                    if len(final) < cfg.top_n and t not in final and rank_map.get(t, 10**9) <= 25:
                        final.append(t)

                if _has_stock_abs_filter(variant):
                    filtered: List[str] = []
                    used = set(final)
                    for t in final:
                        stock_12m = (closes.iloc[max(0, i - 21)][t] / closes.iloc[max(0, i - 252)][t]) - 1.0
                        if stock_12m > rf_12m:
                            filtered.append(t)
                    for t in ranked:
                        if len(filtered) >= cfg.top_n:
                            break
                        if t in used:
                            continue
                        stock_12m = (closes.iloc[max(0, i - 21)][t] / closes.iloc[max(0, i - 252)][t]) - 1.0
                        if stock_12m > rf_12m:
                            filtered.append(t)
                    final = filtered

                final = _apply_sector_cap(final, ranked, sectors, cfg.sector_cap, cfg.top_n)
                if final:
                    eq_weight = 1.0 / cfg.top_n
                    target_weights = {t: eq_weight for t in final[: cfg.top_n]}
                    residual = 1.0 - sum(target_weights.values())
                    if residual > 0:
                        target_weights["LIQUIDBEES"] = residual
                else:
                    target_weights = {"LIQUIDBEES": 1.0}

                last_monthly_composition = dict(target_weights)
                new_regime = "RISK-ON"

            all_names = set(weights) | set(target_weights)
            turn = float(sum(abs(target_weights.get(t, 0.0) - weights.get(t, 0.0)) for t in all_names))
            cost = one_way_cost * turn
            if cost > 0:
                nav.iloc[i] = nav.iloc[i] * (1.0 - cost)
                daily_ret.iloc[i] = (nav.iloc[i] / nav.iloc[i - 1]) - 1.0
            turnover.iloc[i] += turn
            weights = target_weights
            regime = new_regime

        if _has_ema_gate(variant):
            ema21 = nav.iloc[: i + 1].ewm(span=cfg.ema_span, adjust=False).mean().iloc[-1]
            if regime == "RISK-ON" and nav.iloc[i] < (ema21 * cfg.ema_exit_buffer):
                gate_composition = dict(last_monthly_composition)
                target_weights = {"LIQUIDBEES": 1.0}
                all_names = set(weights) | set(target_weights)
                turn = float(sum(abs(target_weights.get(t, 0.0) - weights.get(t, 0.0)) for t in all_names))
                cost = one_way_cost * turn
                if cost > 0:
                    nav.iloc[i] = nav.iloc[i] * (1.0 - cost)
                    daily_ret.iloc[i] = (nav.iloc[i] / nav.iloc[i - 1]) - 1.0
                turnover.iloc[i] += turn
                weights = target_weights
                regime = "GATE-OFF"
            elif regime == "GATE-OFF" and nav.iloc[i] > ema21:
                future_rebals = [d for d in rebalance_days if d > dt]
                days_to_next = (future_rebals[0] - dt).days if future_rebals else 999
                if days_to_next > 5 and gate_composition:
                    target_weights = dict(gate_composition)
                    all_names = set(weights) | set(target_weights)
                    turn = float(sum(abs(target_weights.get(t, 0.0) - weights.get(t, 0.0)) for t in all_names))
                    cost = one_way_cost * turn
                    if cost > 0:
                        nav.iloc[i] = nav.iloc[i] * (1.0 - cost)
                        daily_ret.iloc[i] = (nav.iloc[i] / nav.iloc[i - 1]) - 1.0
                    turnover.iloc[i] += turn
                    weights = target_weights
                    regime = "RISK-ON"

        regime_s.iloc[i] = regime
        gate_state_s.iloc[i] = regime == "GATE-OFF"

    result = pd.DataFrame(
        {
            "daily_return": daily_ret.fillna(0.0),
            "equity": nav.ffill().fillna(1.0),
            "turnover": turnover.fillna(0.0),
            "regime": regime_s.ffill().fillna("RISK-OFF"),
            "gate_active": gate_state_s.fillna(False).astype(int),
        },
        index=all_dates,
    )

    ret = result["daily_return"]
    equity = result["equity"]
    cagr = _cagr(equity)
    vol = _annualized_vol(ret)
    sr = _sharpe(ret, cfg.rf_annual_default)
    sor = _sortino(ret, cfg.rf_annual_default)
    mdd = _max_drawdown(equity)
    calmar = (cagr / abs(mdd)) if (pd.notna(cagr) and pd.notna(mdd) and mdd < 0) else np.nan
    months_liquid = int((result["regime"] != "RISK-ON").resample("ME").max().sum())
    years = max((all_dates[-1] - all_dates[0]).days / 365.25, 1e-9)
    annual_turnover = float(result["turnover"].sum() / years)

    metrics = pd.DataFrame(
        [
            {
                "Variant": variant,
                "CAGR": cagr,
                "AnnualVol": vol,
                "Sharpe": sr,
                "Sortino": sor,
                "MaxDrawdown": mdd,
                "Calmar": calmar,
                "MonthsRiskOffOrGate": months_liquid,
                "AnnualTurnover": annual_turnover,
            }
        ]
    )
    return result, metrics


def compute_pf01_metrics(result: pd.DataFrame, rf_annual: float, variant: str) -> pd.DataFrame:
    ret = result["daily_return"]
    equity = result["equity"]
    cagr = _cagr(equity)
    vol = _annualized_vol(ret)
    sr = _sharpe(ret, rf_annual)
    sor = _sortino(ret, rf_annual)
    mdd = _max_drawdown(equity)
    calmar = (cagr / abs(mdd)) if (pd.notna(cagr) and pd.notna(mdd) and mdd < 0) else np.nan
    months_liquid = int((result["regime"] != "RISK-ON").resample("ME").max().sum())
    years = max((result.index[-1] - result.index[0]).days / 365.25, 1e-9)
    annual_turnover = float(result["turnover"].sum() / years)
    return pd.DataFrame(
        [
            {
                "Variant": variant,
                "CAGR": cagr,
                "AnnualVol": vol,
                "Sharpe": sr,
                "Sortino": sor,
                "MaxDrawdown": mdd,
                "Calmar": calmar,
                "MonthsRiskOffOrGate": months_liquid,
                "AnnualTurnover": annual_turnover,
            }
        ]
    )
