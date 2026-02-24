# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnusedCallResult=false, reportImplicitStringConcatenation=false
import json
from pathlib import Path
from uuid import uuid4


V2_PATH = Path(r"E:\programs\APEXUSTech_Inter\project4\YiboSun_Project4_26_08_V2.ipynb")
V5_PATH = Path(r"E:\programs\APEXUSTech_Inter\project4\YiboSun_Project4_26_08_V5.ipynb")


def md_cell(source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "id": uuid4().hex[:8],
        "source": source_lines,
    }


def code_cell(source_lines):
    return {
        "cell_type": "code",
        "metadata": {},
        "id": uuid4().hex[:8],
        "execution_count": None,
        "outputs": [],
        "source": source_lines,
    }


def to_lines(text):
    return text.strip("\n").splitlines(keepends=True)


def get_v2_sources():
    raw_obj: object = json.loads(V2_PATH.read_text(encoding="utf-8"))  # pyright: ignore[reportAny]
    if not isinstance(raw_obj, dict):
        return []
    raw_cells = raw_obj.get("cells", [])
    if not isinstance(raw_cells, list):
        return []
    parsed: list[list[str]] = []
    for cell in raw_cells:
        if isinstance(cell, dict):
            src = cell.get("source", [])
            if isinstance(src, list):
                parsed.append([str(x) for x in src])
            else:
                parsed.append([])
    return parsed


def main():
    v2 = get_v2_sources()
    cells = []

    # 1
    cells.append(md_cell(to_lines("""
## Project Summary V5 — Dual-Factor VIX Regime Classification

This V5 notebook upgrades the V4 framework with a sophisticated VIX regime system:

- **Dual-factor regime**: combines VIX time-proportion (noise filter) with VIX/VIX3M term structure inversion
- **Three regimes**: Normal (full exposure), Elevated (reduced), Panic (minimal)
- **Adaptive thresholds**: rolling 252-day 75th percentile with absolute floor of 20
- **Graceful fallback**: degrades safely when VIX3M data unavailable
- Retains V4 improvements: conviction weighting, conditional short leg, lower momentum thresholds
""")))

    # 2
    cells.append(md_cell(to_lines("""
### Data Collection

In this project, we use three datasets: adj_close.csv (Yahoo Finance API), volume.csv (Yahoo Finance API), and VIX.csv proxy from `^VIX` and `^VIX3M` (3-month volatility, for term structure) in Yahoo Finance, with IRX (`^IRX`) for cash return proxy. Benchmarks include both **XLK** and **SPY**.
""")))

    # 3 (copy exact)
    cells.append(code_cell(v2[2]))

    # 4 (copy)
    cells.append(md_cell(v2[3]))

    # 5 copy + benchmark edit
    c5 = "".join(v2[4]).replace("benchmark_tickers = ['XLK']", "benchmark_tickers = ['XLK', 'SPY']")
    c5 = c5.replace("market_indicators = ['^VIX', '^IRX']", "market_indicators = ['^VIX', '^VIX3M', '^IRX']")
    cells.append(code_cell(to_lines(c5)))

    # 6 copy + add SPY section and renumber to 1-6
    c6 = "".join(v2[5])
    c6 = c6.replace("1️⃣ Overall Statistics", "1 Overall Statistics")
    c6 = c6.replace("2️⃣ VIX Data", "2 VIX Data")
    c6 = c6.replace("3️⃣ IRX Data (Risk-free rate proxy)", "3 IRX Data (Risk-free rate proxy)")
    c6 = c6.replace("4️⃣ XLK Benchmark", "4 XLK Benchmark")
    c6 = c6.replace("5️⃣ Missing Data Check", "7 Missing Data Check")
    c6 = c6.replace(
        "# Missing data check\n",
        "# VIX3M statistics\n"
        "print(f\"\\n5 VIX3M Data:\")\n"
        "if '^VIX3M' in adj_close.columns:\n"
        "    vix3m = adj_close['^VIX3M'].dropna()\n"
        "    print(f\"   Valid rows: {len(vix3m)}\")\n"
        "    print(f\"   Range: {vix3m.min():.2f} - {vix3m.max():.2f}\")\n"
        "    print(f\"   Latest: {vix3m.iloc[-1]:.2f}\")\n"
        "else:\n"
        "    print(\"   VIX3M not available - will use single-factor regime\")\n"
        "\n"
        "# SPY statistics\n"
        "print(f\"\\n6 SPY Benchmark:\")\n"
        "if 'SPY' in adj_close.columns:\n"
        "    spy = adj_close['SPY'].dropna()\n"
        "    print(f\"   Valid rows: {len(spy)}\")\n"
        "    print(f\"   Price range: ${spy.min():.2f} - ${spy.max():.2f}\")\n"
        "    print(f\"   Latest: ${spy.iloc[-1]:.2f}\")\n"
        "else:\n"
        "    print(\"   ❌ SPY not available\")\n"
        "\n"
        "# Missing data check\n",
    )
    cells.append(code_cell(to_lines(c6)))

    # 7
    cells.append(md_cell(v2[10]))

    # 8 copy + exclude list update
    c8 = "".join(v2[11]).replace(
        "exclude_symbols = ['XLK', '^VIX', '^IRX']",
        "exclude_symbols = ['XLK', 'SPY', '^VIX', '^VIX3M', '^IRX']",
    )
    cells.append(code_cell(to_lines(c8)))

    # 9
    cells.append(code_cell(v2[12]))

    # 10
    cells.append(md_cell(to_lines("""
### V5 Dual-Factor VIX Regime Classification

V5 replaces the V4 single-point VIX gating with a dual-factor monthly regime classifier:

**Factor 1 — Time-Proportion (Noise Filter)**
- Threshold: `thr = max(20, rolling_252d_75th_percentile(VIX))`
- Metric: proportion of trading days in month where VIX > thr
- Trigger: proportion > 25%

**Factor 2 — Term Structure Inversion (VIX/VIX3M)**
- Normal (contango): VIX < VIX3M, ratio < 1
- Elevated: monthly avg ratio > 0.97
- Panic (backwardation): monthly avg ratio > 1.00

**Three Regimes:**

| Regime | Condition | w_vix |
|--------|-----------|-------|
| Normal | Neither factor triggers | 1.00 |
| Elevated | Exactly one factor triggers | 0.75 |
| Panic | Both factors trigger (high time-proportion AND term structure inversion) | 0.35 |

**Fallback**: If VIX3M unavailable, max regime is Elevated (no Panic).

Position sizing formula (unchanged from V4):
`equity_weight = w_vix * (0.40 * w_mom + 0.35 * w_trend + 0.25 * 1.0)`
""")))

    # 11
    cells.append(code_cell(to_lines("""
def classify_trend(adj_close, signal_date):
    if 'XLK' not in adj_close.columns:
        return 'Mixed'
    idx = adj_close.index[adj_close.index <= signal_date]
    if len(idx) == 0:
        return 'Mixed'
    sdate = idx[-1]

    def lb_ret(months):
        start = sdate - pd.DateOffset(months=months)
        prior_idx = adj_close.index[adj_close.index <= start]
        if len(prior_idx) == 0:
            return np.nan
        p0 = adj_close.at[prior_idx[-1], 'XLK']
        p1 = adj_close.at[sdate, 'XLK']
        if pd.isna(p0) or pd.isna(p1) or p0 == 0:
            return np.nan
        return p1 / p0 - 1

    r3 = lb_ret(3)
    r6 = lb_ret(6)
    r12 = lb_ret(12)
    positives = int(r3 > 0) + int(r6 > 0) + int(r12 > 0)
    if positives == 3:
        return 'Strong Uptrend'
    if positives == 2:
        return 'Uptrend'
    if positives == 1:
        return 'Mixed'
    return 'Downtrend'


def calculate_position_size_v3(vix_level, momentum_signal, trend_strength):
    if pd.isna(vix_level):
        w_vix = 0.75
    elif vix_level < 15:
        w_vix = 1.0
    elif vix_level < 20:
        w_vix = 1.0
    elif vix_level < 25:
        w_vix = 0.75
    elif vix_level < 30:
        w_vix = 0.50
    elif vix_level < 40:
        w_vix = 0.25
    else:
        w_vix = 0.0

    if pd.isna(momentum_signal):
        w_mom = 0.50
    elif momentum_signal > 1.5:
        w_mom = 1.0
    elif momentum_signal > 1.0:
        w_mom = 0.85
    elif momentum_signal > 0.5:
        w_mom = 0.70
    elif momentum_signal > 0.0:
        w_mom = 0.50
    else:
        w_mom = 0.25

    trend_map = {
        'Strong Uptrend': 1.0,
        'Uptrend': 0.80,
        'Mixed': 0.50,
        'Downtrend': 0.25,
    }
    w_trend = trend_map.get(trend_strength, 0.50)

    equity_weight = w_vix * (0.4 * w_mom + 0.4 * w_trend + 0.2 * 1.0)
    equity_weight = float(np.clip(equity_weight, 0.0, 1.0))
    cash_weight = 1.0 - equity_weight

    return {
        'equity_weight': equity_weight,
        'cash_weight': cash_weight,
        'signal_details': {
            'vix_level': float(vix_level) if pd.notna(vix_level) else np.nan,
            'momentum_signal': float(momentum_signal) if pd.notna(momentum_signal) else np.nan,
            'trend_strength': trend_strength,
            'w_vix': float(w_vix),
            'w_mom': float(w_mom),
            'w_trend': float(w_trend),
        },
    }


def compute_monthly_vix_regime(
    vix_daily,
    vix3m_daily,
    signal_date,
    lookback_days=252,
    q=0.75,
    vix_floor=20.0,
    p_thr=0.25,
    r_elev=0.97,
    r_panic=1.00,
    min_days=10,
    prev_regime='Normal',
):
    \"\"\"
    Dual-factor monthly VIX regime classifier.

    Factor 1: Time-proportion of days VIX > adaptive threshold
    Factor 2: VIX/VIX3M term structure ratio

    Returns (regime_str, w_vix_float)
    \"\"\"
    w_map = {'Normal': 1.0, 'Elevated': 0.75, 'Panic': 0.35}

    vix_upto = vix_daily.loc[:signal_date].dropna()
    if len(vix_upto) < 30:
        return prev_regime, w_map.get(prev_regime, 1.0)

    # Adaptive threshold: rolling 252d 75th percentile with floor
    window = vix_upto.iloc[-lookback_days:] if len(vix_upto) > lookback_days else vix_upto
    thr_vix = max(vix_floor, float(window.quantile(q)))

    # This month's daily VIX
    month_period = signal_date.to_period('M')
    month_mask = vix_upto.index.to_period('M') == month_period
    vix_month = vix_upto.loc[month_mask]

    if len(vix_month) < min_days:
        return prev_regime, w_map.get(prev_regime, 1.0)

    # Factor 1: time-proportion
    p_high = float((vix_month > thr_vix).mean())
    high_time = (p_high > p_thr)

    # Factor 2: term structure ratio
    r_avg = None
    if vix3m_daily is not None and len(vix3m_daily) > 0:
        vix3m_upto = vix3m_daily.loc[:signal_date].dropna()
        common = vix_month.index.intersection(vix3m_upto.index)
        if len(common) >= min_days:
            ratio = (vix_month.loc[common] / vix3m_upto.loc[common])
            ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
            if len(ratio) >= min_days:
                r_avg = float(ratio.mean())

    # Classify regime
    if r_avg is None:
        # Fallback: no VIX3M data, cap at Elevated
        regime = 'Elevated' if high_time else 'Normal'
    else:
        inv_panic = (r_avg > r_panic)
        flat_elev = (r_avg > r_elev)

        if high_time and inv_panic:
            regime = 'Panic'
        elif high_time or flat_elev:
            regime = 'Elevated'
        else:
            regime = 'Normal'

    return regime, w_map[regime]


def calculate_position_size_v5(regime, w_vix, momentum_signal, trend_strength):
    \"\"\"V5 position sizing using regime-based VIX weight.\"\"\"
    # Momentum signal weight (same thresholds as V4)
    if pd.isna(momentum_signal):
        w_mom = 0.50
    elif momentum_signal > 0.8:
        w_mom = 1.0
    elif momentum_signal > 0.4:
        w_mom = 0.90
    elif momentum_signal > 0.0:
        w_mom = 0.75
    elif momentum_signal > -0.3:
        w_mom = 0.50
    else:
        w_mom = 0.20

    trend_map = {
        'Strong Uptrend': 1.0,
        'Uptrend': 0.80,
        'Mixed': 0.50,
        'Downtrend': 0.25,
    }
    w_trend = trend_map.get(trend_strength, 0.50)

    # Same formula as V4 but w_vix comes from regime classifier
    equity_weight = w_vix * (0.40 * w_mom + 0.35 * w_trend + 0.25 * 1.0)
    equity_weight = float(np.clip(equity_weight, 0.0, 1.0))
    cash_weight = 1.0 - equity_weight

    return {
        'equity_weight': equity_weight,
        'cash_weight': cash_weight,
        'signal_details': {
            'regime': regime,
            'w_vix': float(w_vix),
            'momentum_signal': float(momentum_signal) if pd.notna(momentum_signal) else np.nan,
            'trend_strength': trend_strength,
            'w_mom': float(w_mom),
            'w_trend': float(w_trend),
        },
    }
""")))

    # 12
    cells.append(md_cell(to_lines("""
### Strategy Backtesting: V5 and V4 Baseline
""")))

    # 13
    cells.append(code_cell(to_lines("""
def backtest_strategy_v3(
    adj_close,
    volume,
    lookback_months=6,
    method='simple',
    decile=0.3,
    tc_bps=5,
    holding_period=1,
    winsor_q=0.01,
):
    daily_returns = adj_close.pct_change(fill_method=None)

    month_ends = adj_close.resample('M').last().index
    valid_month_ends = []
    for me in month_ends:
        month_data = adj_close[(adj_close.index.month == me.month) & (adj_close.index.year == me.year)]
        if len(month_data) > 0:
            valid_month_ends.append(month_data.index[-1])
    valid_month_ends = pd.DatetimeIndex(valid_month_ends).unique().sort_values()

    if len(valid_month_ends) <= lookback_months + 1:
        return pd.DataFrame(columns=[
            'strategy_returns', 'equity_returns', 'cash_returns',
            'equity_weight', 'cash_weight', 'num_holdings', 'turnover'
        ])

    def _get_ret(symbol, start_date, end_date):
        if symbol not in adj_close.columns:
            return np.nan
        try:
            p0 = adj_close.at[start_date, symbol]
            p1 = adj_close.at[end_date, symbol]
            if pd.isna(p0) or pd.isna(p1) or p0 == 0:
                return np.nan
            return p1 / p0 - 1
        except Exception:
            return np.nan

    def _cash_return(prev_date, curr_date):
        if '^IRX' not in adj_close.columns:
            return 0.0
        rf = adj_close['^IRX'].loc[(adj_close.index > prev_date) & (adj_close.index <= curr_date)]
        if len(rf) == 0:
            return 0.0
        daily_rf = pd.to_numeric(rf, errors='coerce') / 100.0 / 252.0
        daily_rf = daily_rf.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return float((1.0 + daily_rf).prod() - 1.0)

    out_rows = []
    out_index = []

    current_holdings = pd.Index([])
    equity_weight = 0.0
    cash_weight = 1.0
    prev_asset_weights = {'CASH': 1.0}

    start_i = lookback_months
    for i in range(start_i, len(valid_month_ends)):
        current_date = valid_month_ends[i]
        previous_date = valid_month_ends[i - 1]

        rebalance_now = ((i - start_i) % holding_period == 0) or (len(current_holdings) == 0)
        turnover = 0.0

        if rebalance_now:
            signal_date = previous_date
            scores = calculate_momentum_scores(
                adj_close,
                volume,
                daily_returns,
                signal_date,
                lookback_months,
                method,
            )

            scores = scores.replace([np.inf, -np.inf], np.nan).dropna()
            if len(scores) > 0:
                top_n = max(1, int(len(scores) * decile))
                selected = scores.sort_values(ascending=False).head(top_n)
                current_holdings = selected.index
                mom_signal = float(selected.mean()) if len(selected) > 0 else 0.0
            else:
                current_holdings = pd.Index([])
                mom_signal = 0.0

            if '^VIX' in adj_close.columns:
                vix_series = adj_close['^VIX'].loc[adj_close.index <= signal_date].dropna()
                vix_level = float(vix_series.iloc[-1]) if len(vix_series) > 0 else np.nan
            else:
                vix_level = np.nan

            trend_strength = classify_trend(adj_close, signal_date)
            sizing = calculate_position_size_v3(vix_level, mom_signal, trend_strength)
            equity_weight = sizing['equity_weight']
            cash_weight = sizing['cash_weight']

            new_asset_weights = {'CASH': cash_weight}
            if len(current_holdings) > 0 and equity_weight > 0:
                per_stock = equity_weight / len(current_holdings)
                for s in current_holdings:
                    new_asset_weights[s] = per_stock

            all_assets = set(prev_asset_weights) | set(new_asset_weights)
            l1 = sum(abs(new_asset_weights.get(a, 0.0) - prev_asset_weights.get(a, 0.0)) for a in all_assets)
            turnover = 0.5 * l1
            prev_asset_weights = new_asset_weights

        stock_rets = []
        for sym in current_holdings:
            r = _get_ret(sym, previous_date, current_date)
            if pd.notna(r) and np.isfinite(r):
                stock_rets.append(r)

        if len(stock_rets) > 0:
            stock_rets = pd.Series(stock_rets)
            if len(stock_rets) > 5:
                lo = stock_rets.quantile(winsor_q)
                hi = stock_rets.quantile(1 - winsor_q)
                stock_rets = stock_rets.clip(lo, hi)
            equity_ret = float(stock_rets.mean())
        else:
            equity_ret = 0.0

        cash_ret = _cash_return(previous_date, current_date)
        tc = (tc_bps / 10000.0) * turnover
        strategy_ret = equity_weight * equity_ret + cash_weight * cash_ret - tc

        out_index.append(current_date)
        out_rows.append({
            'strategy_returns': strategy_ret,
            'equity_returns': equity_ret,
            'cash_returns': cash_ret,
            'equity_weight': equity_weight,
            'cash_weight': cash_weight,
            'num_holdings': int(len(current_holdings)),
            'turnover': turnover,
        })

    out = pd.DataFrame(out_rows, index=pd.DatetimeIndex(out_index))
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def backtest_strategy_v5(
    adj_close,
    volume,
    lookback_months=6,
    method='simple',
    decile=0.3,
    tc_bps=5,
    holding_period=1,
    winsor_q=0.01,
):
    daily_returns = adj_close.pct_change(fill_method=None)

    month_ends = adj_close.resample('M').last().index
    valid_month_ends = []
    for me in month_ends:
        month_data = adj_close[(adj_close.index.month == me.month) & (adj_close.index.year == me.year)]
        if len(month_data) > 0:
            valid_month_ends.append(month_data.index[-1])
    valid_month_ends = pd.DatetimeIndex(valid_month_ends).unique().sort_values()

    if len(valid_month_ends) <= lookback_months + 1:
        return pd.DataFrame(columns=[
            'strategy_returns', 'equity_returns', 'cash_returns',
            'equity_weight', 'cash_weight', 'num_holdings', 'turnover'
        ])

    def _get_ret(symbol, start_date, end_date):
        if symbol not in adj_close.columns:
            return np.nan
        try:
            p0 = adj_close.at[start_date, symbol]
            p1 = adj_close.at[end_date, symbol]
            if pd.isna(p0) or pd.isna(p1) or p0 == 0:
                return np.nan
            return p1 / p0 - 1
        except Exception:
            return np.nan

    def _cash_return(prev_date, curr_date):
        if '^IRX' not in adj_close.columns:
            return 0.0
        rf = adj_close['^IRX'].loc[(adj_close.index > prev_date) & (adj_close.index <= curr_date)]
        if len(rf) == 0:
            return 0.0
        daily_rf = pd.to_numeric(rf, errors='coerce') / 100.0 / 252.0
        daily_rf = daily_rf.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return float((1.0 + daily_rf).prod() - 1.0)

    out_rows = []
    out_index = []

    current_holdings = pd.Index([])
    equity_weight = 0.0
    cash_weight = 1.0
    prev_asset_weights = {'CASH': 1.0}
    current_asset_weights = {'CASH': 1.0}
    prev_regime = 'Normal'

    start_i = lookback_months
    for i in range(start_i, len(valid_month_ends)):
        current_date = valid_month_ends[i]
        previous_date = valid_month_ends[i - 1]

        rebalance_now = ((i - start_i) % holding_period == 0) or (len(current_holdings) == 0)
        turnover = 0.0

        if rebalance_now:
            signal_date = previous_date
            scores = calculate_momentum_scores(
                adj_close,
                volume,
                daily_returns,
                signal_date,
                lookback_months,
                method,
            )

            scores = scores.replace([np.inf, -np.inf], np.nan).dropna()
            if len(scores) > 0:
                top_n = max(1, int(len(scores) * decile))
                selected = scores.sort_values(ascending=False).head(top_n)
                current_holdings = selected.index
                mom_signal = float(selected.mean()) if len(selected) > 0 else 0.0
            else:
                current_holdings = pd.Index([])
                mom_signal = 0.0

            # V5: Dual-factor regime classification
            vix_daily = adj_close['^VIX'] if '^VIX' in adj_close.columns else pd.Series(dtype=float)
            vix3m_daily = adj_close['^VIX3M'] if '^VIX3M' in adj_close.columns else None

            regime, w_vix = compute_monthly_vix_regime(
                vix_daily, vix3m_daily, signal_date,
                prev_regime=prev_regime,
            )
            prev_regime = regime

            trend_strength = classify_trend(adj_close, signal_date)
            sizing = calculate_position_size_v5(regime, w_vix, mom_signal, trend_strength)
            equity_weight = sizing['equity_weight']
            cash_weight = sizing['cash_weight']

            weights = {}

            # V4 Change 4: momentum-weighted stock allocation
            if len(current_holdings) > 0 and equity_weight > 0 and len(scores) > 0:
                held_scores = scores.loc[scores.index.isin(current_holdings)]
                if len(held_scores) > 0:
                    shifted = held_scores - held_scores.min() + 0.01
                    if float(shifted.sum()) > 0:
                        stock_weights = (shifted / shifted.sum()) * equity_weight
                        for ticker in current_holdings:
                            if ticker in stock_weights.index:
                                weights[ticker] = float(stock_weights[ticker])
                            else:
                                weights[ticker] = equity_weight / len(current_holdings)
                    else:
                        per_stock = equity_weight / len(current_holdings)
                        for ticker in current_holdings:
                            weights[ticker] = per_stock
                else:
                    per_stock = equity_weight / len(current_holdings)
                    for ticker in current_holdings:
                        weights[ticker] = per_stock

            # V4 Change 5: conditional short leg in downtrend regimes
            if trend_strength == 'Downtrend' and regime != 'Panic' and len(scores) > 0:
                n_short = max(1, int(len(scores) * 0.10))
                short_tickers = scores.nsmallest(n_short).index.tolist()
                short_weight = 0.15 * cash_weight
                if len(short_tickers) > 0:
                    per_short = short_weight / len(short_tickers)
                    for t in short_tickers:
                        weights[t] = weights.get(t, 0.0) - per_short

            new_asset_weights = {'CASH': cash_weight}
            for k, v in weights.items():
                if pd.notna(v) and np.isfinite(v):
                    new_asset_weights[k] = float(v)

            all_assets = set(prev_asset_weights) | set(new_asset_weights)
            l1 = sum(abs(new_asset_weights.get(a, 0.0) - prev_asset_weights.get(a, 0.0)) for a in all_assets)
            turnover = 0.5 * l1
            prev_asset_weights = new_asset_weights
            current_asset_weights = new_asset_weights

        # Realized returns for all non-cash positions (long and short weights)
        realized = {}
        for sym, w in current_asset_weights.items():
            if sym == 'CASH':
                continue
            r = _get_ret(sym, previous_date, current_date)
            if pd.notna(r) and np.isfinite(r):
                realized[sym] = float(r)

        if len(realized) > 0:
            ret_series = pd.Series(realized)
            if len(ret_series) > 5:
                lo = ret_series.quantile(winsor_q)
                hi = ret_series.quantile(1 - winsor_q)
                ret_series = ret_series.clip(lo, hi)

            common = [s for s in ret_series.index if s in current_asset_weights and s != 'CASH']
            if len(common) > 0:
                w_series = pd.Series({s: current_asset_weights[s] for s in common}, dtype=float)
                weighted_stock_ret = float((w_series * ret_series.loc[common]).sum())
                gross_exposure = float(w_series.abs().sum())
                equity_ret = float(weighted_stock_ret / gross_exposure) if gross_exposure > 0 else 0.0
            else:
                weighted_stock_ret = 0.0
                equity_ret = 0.0
        else:
            weighted_stock_ret = 0.0
            equity_ret = 0.0

        cash_ret = _cash_return(previous_date, current_date)
        tc = (tc_bps / 10000.0) * turnover
        strategy_ret = weighted_stock_ret + cash_weight * cash_ret - tc

        out_index.append(current_date)
        out_rows.append({
            'strategy_returns': strategy_ret,
            'equity_returns': equity_ret,
            'cash_returns': cash_ret,
            'equity_weight': equity_weight,
            'cash_weight': cash_weight,
            'num_holdings': int(len(current_holdings)),
            'turnover': turnover,
        })

    out = pd.DataFrame(out_rows, index=pd.DatetimeIndex(out_index))
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out
""")))

    # 14
    cells.append(code_cell(to_lines("""
# Reload data
adj_close = pd.read_csv('adj_close.csv', index_col=0, parse_dates=True)
volume = pd.read_csv('volume.csv', index_col=0, parse_dates=True)

if adj_close.index.tz is not None:
    adj_close.index = adj_close.index.tz_localize(None)
if volume.index.tz is not None:
    volume.index = volume.index.tz_localize(None)

ret_simple_v5 = backtest_strategy_v5(adj_close, volume, lookback_months=6, method='simple', decile=0.3, holding_period=1)
ret_risk_v5 = backtest_strategy_v5(adj_close, volume, lookback_months=6, method='risk_adjusted', decile=0.3, holding_period=1)
ret_vol_v5 = backtest_strategy_v5(adj_close, volume, lookback_months=6, method='volume_weighted', decile=0.3, holding_period=1)

for name, df_ in [
    ('Simple', ret_simple_v5),
    ('Risk-Adjusted', ret_risk_v5),
    ('Volume-Weighted', ret_vol_v5),
]:
    r = df_['strategy_returns'] if 'strategy_returns' in df_.columns else pd.Series(dtype=float)
    ann_ret = (1 + r.mean()) ** 12 - 1 if len(r) > 0 else 0.0
    ann_vol = r.std() * np.sqrt(12) if len(r) > 1 else np.nan
    sharpe = ann_ret / ann_vol if pd.notna(ann_vol) and ann_vol > 0 else np.nan
    print(f"{name}: months={len(r)}, ann_ret={ann_ret:.2%}, ann_vol={ann_vol:.2%}, sharpe={sharpe:.3f}")
""")))

    # 15
    cells.append(md_cell(to_lines("""
### Holding Period Analysis
""")))

    # 16
    cells.append(code_cell(to_lines("""
def analyze_holding_periods(adj_close, volume, backtest_fn, methods=None, holding_periods=None, lookback_months=6):
    if methods is None:
        methods = ['simple', 'risk_adjusted', 'volume_weighted']
    if holding_periods is None:
        holding_periods = [1, 2, 3, 6]

    train_end = pd.Timestamp('2024-12-31')
    rows = []

    for m in methods:
        for hp in holding_periods:
            bt = backtest_fn(
                adj_close,
                volume,
                lookback_months=lookback_months,
                method=m,
                decile=0.3,
                tc_bps=5,
                holding_period=hp,
            )
            r = bt['strategy_returns']
            r_train = r[r.index <= train_end]
            if len(r_train) < 12:
                continue

            ann_ret = (1 + r_train.mean()) ** 12 - 1
            ann_vol = r_train.std() * np.sqrt(12)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
            max_dd = ((1 + r_train).cumprod() / (1 + r_train).cumprod().cummax() - 1).min()

            rows.append({
                'method': m,
                'holding_period': hp,
                'months': len(r_train),
                'annual_return': ann_ret,
                'annual_volatility': ann_vol,
                'sharpe': sharpe,
                'max_drawdown': max_dd,
            })

    return pd.DataFrame(rows).sort_values('sharpe', ascending=False)
""")))

    # 17
    cells.append(code_cell(to_lines("""
holding_period_results_v5 = analyze_holding_periods(adj_close, volume, backtest_strategy_v5)
holding_period_results_v3 = analyze_holding_periods(adj_close, volume, backtest_strategy_v3)

print("V5 holding period grid:")
print(holding_period_results_v5.round(4))
print("\\nV3 holding period grid:")
print(holding_period_results_v3.round(4))

best_config_v5 = holding_period_results_v5.iloc[0]
best_config_v3 = holding_period_results_v3.iloc[0]

print("\\nBest V5 config:")
print(best_config_v5)
print("\\nBest V3 config:")
print(best_config_v3)

best_strategy_v5_df = backtest_strategy_v5(
    adj_close,
    volume,
    lookback_months=6,
    method=best_config_v5['method'],
    decile=0.3,
    tc_bps=5,
    holding_period=int(best_config_v5['holding_period']),
)

best_strategy_v3_df = backtest_strategy_v3(
    adj_close,
    volume,
    lookback_months=6,
    method=best_config_v3['method'],
    decile=0.3,
    tc_bps=5,
    holding_period=int(best_config_v3['holding_period']),
)

best_v5_returns = best_strategy_v5_df['strategy_returns']
best_v3_returns = best_strategy_v3_df['strategy_returns']
""")))

    # 18
    cells.append(code_cell(to_lines("""
print("Momentum decay analysis (score vs forward returns)")

daily_returns = adj_close.pct_change(fill_method=None)
month_ends = adj_close.resample('M').last().index
valid_month_ends = []
for me in month_ends:
    m = adj_close[(adj_close.index.month == me.month) & (adj_close.index.year == me.year)]
    if len(m) > 0:
        valid_month_ends.append(m.index[-1])
valid_month_ends = pd.DatetimeIndex(valid_month_ends).unique().sort_values()

score_forward_corr = {}
for h in [1, 2, 3, 4, 5, 6]:
    corr_list = []
    for i in range(6, len(valid_month_ends) - h):
        d0 = valid_month_ends[i]
        d1 = valid_month_ends[i + h]
        scores = calculate_momentum_scores(adj_close, volume, daily_returns, d0, 6, method='simple')
        if len(scores) == 0:
            continue

        fwd = (adj_close.loc[d1, scores.index] / adj_close.loc[d0, scores.index] - 1).replace([np.inf, -np.inf], np.nan).dropna()
        common = scores.index.intersection(fwd.index)
        if len(common) >= 10:
            c = scores.loc[common].corr(fwd.loc[common])
            if pd.notna(c):
                corr_list.append(c)

    score_forward_corr[h] = float(np.mean(corr_list)) if len(corr_list) > 0 else np.nan

for h, c in score_forward_corr.items():
    print(f"Forward {h}M correlation: {c:.4f}")

print("\\nAutocorrelation of strategy returns (best V5 config):")
sr = best_strategy_v5_df['strategy_returns']
for lag in [1, 2, 3, 4, 5, 6]:
    print(f"Lag {lag}: {sr.autocorr(lag=lag):.4f}")
""")))

    # 19
    cells.append(md_cell(to_lines("""
### Performance Metrics
""")))

    # 20
    cells.append(code_cell(to_lines("""
def calculate_annual_metrics(monthly_returns, rf_monthly=None):
    monthly_returns = monthly_returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(monthly_returns) == 0:
        return {
            'months': 0,
            'annual_return': np.nan,
            'annual_volatility': np.nan,
            'sharpe_ratio': np.nan,
            'max_drawdown': np.nan,
            'win_rate': np.nan,
        }

    ann_ret = (1 + monthly_returns.mean()) ** 12 - 1
    ann_vol = monthly_returns.std() * np.sqrt(12)

    if rf_monthly is None:
        rf_ann = 0.0
    else:
        rf_aligned = rf_monthly.reindex(monthly_returns.index, method='ffill').fillna(0.0)
        rf_ann = ((1 + rf_aligned.mean()) ** 12 - 1)

    sharpe = (ann_ret - rf_ann) / ann_vol if ann_vol > 0 else np.nan
    wealth = (1 + monthly_returns).cumprod()
    max_dd = (wealth / wealth.cummax() - 1).min()

    return {
        'months': len(monthly_returns),
        'annual_return': ann_ret,
        'annual_volatility': ann_vol,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'win_rate': (monthly_returns > 0).mean(),
    }


rf_daily = adj_close['^IRX'] / 100 / 252 if '^IRX' in adj_close.columns else pd.Series(0.0, index=adj_close.index)
rf_monthly = rf_daily.resample('M').apply(lambda x: (1 + x).prod() - 1 if len(x) > 0 else 0)

xlk_monthly_returns = adj_close['XLK'].resample('M').last().pct_change(fill_method=None).dropna()
spy_monthly_returns = adj_close['SPY'].resample('M').last().pct_change(fill_method=None).dropna()

strategy_map = {
    'V5 Simple (6m, hp1)': ret_simple_v5['strategy_returns'],
    'V5 Risk-Adj (6m, hp1)': ret_risk_v5['strategy_returns'],
    'V5 Volume-Wtd (6m, hp1)': ret_vol_v5['strategy_returns'],
    'V5 Best Holding Config': best_v5_returns,
    'V3 Best Holding Config': best_v3_returns,
    'XLK Benchmark': xlk_monthly_returns,
    'SPY Benchmark': spy_monthly_returns,
}

results_summary = {}
for name, r in strategy_map.items():
    m = calculate_annual_metrics(r, rf_monthly=rf_monthly)
    results_summary[name] = m
    print(
        f"{name}: months={m['months']}, ann_ret={m['annual_return']:.2%}, "
        f"ann_vol={m['annual_volatility']:.2%}, sharpe={m['sharpe_ratio']:.3f}, "
        f"maxDD={m['max_drawdown']:.2%}, win={m['win_rate']:.1%}"
    )
""")))

    # 21
    cells.append(code_cell(to_lines("""
results_df = pd.DataFrame(results_summary).T
print(results_df.round(4))
""")))

    # 22
    cells.append(code_cell(to_lines("""
# V5 vs V3 vs Benchmarks Summary Table
comparison = pd.DataFrame({
    'V5 Best': calculate_annual_metrics(best_v5_returns, rf_monthly=rf_monthly),
    'V3 Best': calculate_annual_metrics(best_v3_returns, rf_monthly=rf_monthly),
    'XLK': calculate_annual_metrics(xlk_monthly_returns, rf_monthly=rf_monthly),
    'SPY': calculate_annual_metrics(spy_monthly_returns, rf_monthly=rf_monthly),
}).T
print("\\n=== V5 vs V3 vs Benchmarks ===")
print(comparison.round(4))
""")))

    # 23
    cells.append(md_cell(to_lines("""
## Market Regime Analysis
""")))

    # 24 adapt from V2 cell 22
    c24 = "".join(v2[21])
    c24 = c24.replace("backtest_strategy(adj_close, volume, lookback_months=lb, method=method)", "backtest_strategy_v5(adj_close, volume, lookback_months=lb, method=method, decile=0.3)")
    c24 = c24.replace("strategy_returns_train = strategy_returns[strategy_returns.index <= train_end]", "strategy_returns = strategy_returns['strategy_returns']\n            strategy_returns_train = strategy_returns[strategy_returns.index <= train_end]")
    c24 = c24.replace("primary_strategy = backtest_strategy(adj_close, volume, lookback_months=6, method='simple')", "primary_strategy = backtest_strategy_v5(adj_close, volume, lookback_months=6, method='simple', decile=0.3)\n    primary_strategy = primary_strategy['strategy_returns']")
    c24 = c24.replace("# XLK monthly returns (our market benchmark)", "# XLK/SPY monthly returns (market benchmarks)")
    c24 = c24.replace("xlk_monthly = adj_close['XLK'].resample('M').last().pct_change().dropna()", "xlk_monthly = adj_close['XLK'].resample('M').last().pct_change(fill_method=None).dropna()")
    c24 = c24.replace("print(f\"✅ XLK monthly returns: {len(xlk_monthly)} months\")", "spy_monthly = adj_close['SPY'].resample('M').last().pct_change(fill_method=None).dropna()\nprint(f\"✅ XLK monthly returns: {len(xlk_monthly)} months\")\nprint(f\"✅ SPY monthly returns: {len(spy_monthly)} months\")")
    c24 = c24.replace("(1 + rf_daily).resample('M').apply", "rf_daily.resample('M').apply")
    c24 += "\n\nprimary_strategy_df = backtest_strategy_v5(adj_close, volume, lookback_months=best_params[1] if best_params else 6, method=best_params[0] if best_params else 'simple', decile=0.3)\nprimary_strategy_df = primary_strategy_df[primary_strategy_df.index > train_end]\n"
    cells.append(code_cell(to_lines(c24)))

    # 25
    cells.append(md_cell(to_lines("""
### Market Regime Classification with VIX
""")))

    # 26 adapt from V2 cell 24
    c26 = "".join(v2[23])
    c26 = c26.replace("strategy_index = primary_strategy.index", "strategy_index = primary_strategy_df.index")
    c26 = c26.replace("'strategy_returns': primary_strategy,", "'strategy_returns': primary_strategy_df['strategy_returns'],\n    'equity_weight': primary_strategy_df['equity_weight'],\n    'cash_weight': primary_strategy_df['cash_weight'],")
    c26 = c26.replace("'xlk_returns': xlk_aligned,", "'xlk_returns': xlk_aligned,\n    'spy_returns': spy_monthly.reindex(strategy_index, method='ffill').fillna(0),")
    cells.append(code_cell(to_lines(c26)))

    # 27
    cells.append(md_cell(to_lines("""
### Strategy Performance by Market Regime
""")))

    # 28 adapt from V2 cell 26 + cash allocation section
    c28 = "".join(v2[25])
    c28 += "\n\n# 9. Cash Allocation by Regime\nprint(\"\\n9️⃣ Cash Allocation by Regime:\")\nif 'equity_weight' in regime_data.columns:\n    cash_by_regime = regime_data.groupby('combined_regime')['equity_weight'].mean().sort_values(ascending=False)\n    print(cash_by_regime.rename('avg_equity_weight'))\n"
    cells.append(code_cell(to_lines(c28)))

    # 29
    cells.append(code_cell(to_lines("""
# V5 VIX Regime Analysis
vix_daily = adj_close['^VIX'] if '^VIX' in adj_close.columns else pd.Series(dtype=float)
vix3m_daily = adj_close['^VIX3M'] if '^VIX3M' in adj_close.columns else None

# Compute regimes for all months
month_ends = adj_close.resample('M').last().index
regimes = []
prev_r = 'Normal'
for me in month_ends:
    r, w = compute_monthly_vix_regime(vix_daily, vix3m_daily, me, prev_regime=prev_r)
    regimes.append({'date': me, 'regime': r, 'w_vix': w})
    prev_r = r

regime_df = pd.DataFrame(regimes).set_index('date')
print("VIX Regime Distribution:")
print(regime_df['regime'].value_counts())
print(f"\\nRegime timeline (last 12 months):")
print(regime_df.tail(12))

# If VIX3M available, show term structure stats
if vix3m_daily is not None and '^VIX3M' in adj_close.columns:
    ratio = (adj_close['^VIX'] / adj_close['^VIX3M']).dropna()
    print(f"\\nVIX/VIX3M ratio stats:")
    print(f"  Mean: {ratio.mean():.3f}")
    print(f"  Median: {ratio.median():.3f}")
    print(f"  Days in backwardation (>1.0): {(ratio > 1.0).sum()} ({(ratio > 1.0).mean():.1%})")
""")))

    # 30
    cells.append(code_cell(to_lines("""
import plotly.graph_objects as go

# Regime color timeline
regime_colors = {'Normal': 'green', 'Elevated': 'orange', 'Panic': 'red'}
fig = go.Figure()

for regime_name, color in regime_colors.items():
    mask = regime_df['regime'] == regime_name
    subset = regime_df[mask]
    if len(subset) > 0:
        fig.add_trace(go.Scatter(
            x=subset.index,
            y=subset['w_vix'],
            mode='markers',
            name=regime_name,
            marker=dict(color=color, size=8),
        ))

# Add VIX line on secondary axis
vix_monthly = adj_close['^VIX'].resample('M').last() if '^VIX' in adj_close.columns else pd.Series(dtype=float)
fig.add_trace(go.Scatter(
    x=vix_monthly.index, y=vix_monthly.values,
    mode='lines', name='VIX (monthly)',
    yaxis='y2', line=dict(color='gray', dash='dot'),
))

fig.update_layout(
    title='V5 VIX Regime Classification Timeline',
    yaxis=dict(title='w_vix (Position Weight)'),
    yaxis2=dict(title='VIX Level', overlaying='y', side='right'),
    template='plotly_white',
)
fig.show()
""")))

    # 31
    cells.append(md_cell(to_lines("""
### Visualizations
""")))

    # 32
    cells.append(code_cell(to_lines("""
import plotly.graph_objects as go
from plotly.subplots import make_subplots

viz_index = best_strategy_v5_df.index.intersection(best_strategy_v3_df.index)
cum_df = pd.DataFrame({
    'V5': best_v5_returns.reindex(viz_index, fill_value=0.0),
    'V3': best_v3_returns.reindex(viz_index, fill_value=0.0),
    'XLK': xlk_monthly_returns.reindex(viz_index, method='ffill').fillna(0.0),
    'SPY': spy_monthly_returns.reindex(viz_index, method='ffill').fillna(0.0),
}).dropna()

cum = (1 + cum_df).cumprod() - 1

fig = go.Figure()
fig.add_trace(go.Scatter(x=cum.index, y=cum['V5'], mode='lines', name='V5 Best'))
fig.add_trace(go.Scatter(x=cum.index, y=cum['V3'], mode='lines', name='V3 Best'))
fig.add_trace(go.Scatter(x=cum.index, y=cum['XLK'], mode='lines', name='XLK'))
fig.add_trace(go.Scatter(x=cum.index, y=cum['SPY'], mode='lines', name='SPY'))
fig.update_layout(title='Cumulative Returns: V5 vs V3 vs XLK vs SPY', yaxis_title='Cumulative Return', template='plotly_white')
fig.show()
""")))

    # 33
    cells.append(code_cell(to_lines("""
roll_12m_v5 = (1 + best_v5_returns).rolling(12).apply(np.prod, raw=True) - 1
roll_12m_v3 = (1 + best_v3_returns).rolling(12).apply(np.prod, raw=True) - 1
roll_12m_xlk = (1 + xlk_monthly_returns.reindex(best_v5_returns.index, method='ffill').fillna(0.0)).rolling(12).apply(np.prod, raw=True) - 1
vix_monthly = adj_close['^VIX'].resample('M').last() if '^VIX' in adj_close.columns else pd.Series(index=best_v5_returns.index, dtype=float)
vix_plot = vix_monthly.reindex(best_v5_returns.index, method='ffill')

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=roll_12m_v5.index, y=roll_12m_v5, mode='lines', name='V5 12M'), secondary_y=False)
fig.add_trace(go.Scatter(x=roll_12m_v3.index, y=roll_12m_v3, mode='lines', name='V3 12M'), secondary_y=False)
fig.add_trace(go.Scatter(x=roll_12m_xlk.index, y=roll_12m_xlk, mode='lines', name='XLK 12M'), secondary_y=False)
fig.add_trace(go.Scatter(x=vix_plot.index, y=vix_plot, mode='lines', name='VIX', line=dict(dash='dot')), secondary_y=True)
fig.update_layout(title='Rolling 12M Returns with VIX Overlay', template='plotly_white')
fig.update_yaxes(title_text='Rolling 12M Return', secondary_y=False)
fig.update_yaxes(title_text='VIX', secondary_y=True)
fig.show()
""")))

    # 34
    cells.append(code_cell(to_lines("""
alloc_df_v5 = best_strategy_v5_df[['equity_weight', 'cash_weight']].copy()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=alloc_df_v5.index,
    y=alloc_df_v5['equity_weight'],
    mode='lines',
    stackgroup='one',
    name='Equity Weight'
))
fig.add_trace(go.Scatter(
    x=alloc_df_v5.index,
    y=alloc_df_v5['cash_weight'],
    mode='lines',
    stackgroup='one',
    name='Cash Weight'
))
fig.update_layout(title='V5 Dynamic Allocation Over Time', yaxis_title='Portfolio Weight', template='plotly_white')
fig.show()
""")))

    # 35
    cells.append(code_cell(to_lines("""
heat_df = holding_period_results_v5.pivot(index='method', columns='holding_period', values='sharpe')

fig = go.Figure(data=go.Heatmap(
    z=heat_df.values,
    x=heat_df.columns.astype(str),
    y=heat_df.index,
    colorscale='RdYlGn',
    colorbar=dict(title='Sharpe')
))
fig.update_layout(title='V5 Holding Period Grid Search Heatmap (Sharpe)', xaxis_title='Holding Period (months)', yaxis_title='Method', template='plotly_white')
fig.show()
""")))

    # 36
    cells.append(code_cell(to_lines("""
alloc_v5 = best_strategy_v5_df[['equity_weight', 'cash_weight']].copy()
alloc_v3 = best_strategy_v3_df[['equity_weight', 'cash_weight']].copy()

common_idx = alloc_v5.index.intersection(alloc_v3.index)
alloc_v5 = alloc_v5.reindex(common_idx).ffill().fillna(0.0)
alloc_v3 = alloc_v3.reindex(common_idx).ffill().fillna(0.0)

fig = make_subplots(rows=1, cols=2, subplot_titles=('V5 Allocation', 'V3 Allocation'))
fig.add_trace(go.Scatter(x=alloc_v5.index, y=alloc_v5['equity_weight'], mode='lines', stackgroup='one', name='V5 Equity', showlegend=True), row=1, col=1)
fig.add_trace(go.Scatter(x=alloc_v5.index, y=alloc_v5['cash_weight'], mode='lines', stackgroup='one', name='V5 Cash', showlegend=True), row=1, col=1)
fig.add_trace(go.Scatter(x=alloc_v3.index, y=alloc_v3['equity_weight'], mode='lines', stackgroup='one', name='V3 Equity', showlegend=True), row=1, col=2)
fig.add_trace(go.Scatter(x=alloc_v3.index, y=alloc_v3['cash_weight'], mode='lines', stackgroup='one', name='V3 Cash', showlegend=True), row=1, col=2)
fig.update_layout(title='V5 vs V3 Allocation Comparison', yaxis_title='Portfolio Weight', template='plotly_white')
fig.show()
""")))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    V5_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote V5 notebook to {V5_PATH}")
    print(f"Total cells: {len(cells)}")


if __name__ == "__main__":
    main()
