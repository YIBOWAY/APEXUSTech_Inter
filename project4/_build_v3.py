# pyright: reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnusedCallResult=false, reportImplicitStringConcatenation=false
import json
from pathlib import Path
from uuid import uuid4


V2_PATH = Path(r"E:\programs\APEXUSTech_Inter\project4\YiboSun_Project4_26_08_V2.ipynb")
V3_PATH = Path(r"E:\programs\APEXUSTech_Inter\project4\YiboSun_Project4_26_08_V3.ipynb")


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
## Project Summary V3

### Long-Only Momentum Strategy with Dynamic Cash Management

This V3 notebook upgrades the prior long-short implementation to a **long-only momentum framework** with adaptive risk allocation.

- Removed short exposure and allocate only to the top **30% momentum** names.
- Added a **dynamic cash management engine** driven by VIX regime, momentum signal quality, and trend strength.
- Included **holding period analysis** (1/2/3/6 month rebalancing) with train/test split.
- Expanded benchmarking to a **dual benchmark** setup: XLK and SPY.
- Added regime-level diagnostics plus visual analytics for equity/cash allocation behavior.
""")))

    # 2
    cells.append(md_cell(to_lines("""
### Data Collection

In this project, we use three datasets: adj_close.csv (Yahoo Finance API), volume.csv (Yahoo Finance API), and VIX.csv proxy from `^VIX` in Yahoo Finance, with IRX (`^IRX`) for cash return proxy. Benchmarks include both **XLK** and **SPY**.
""")))

    # 3 (copy exact)
    cells.append(code_cell(v2[2]))

    # 4 (copy)
    cells.append(md_cell(v2[3]))

    # 5 copy + benchmark edit
    c5 = "".join(v2[4]).replace("benchmark_tickers = ['XLK']", "benchmark_tickers = ['XLK', 'SPY']")
    cells.append(code_cell(to_lines(c5)))

    # 6 copy + add SPY section and renumber to 1-6
    c6 = "".join(v2[5])
    c6 = c6.replace("1️⃣ Overall Statistics", "1 Overall Statistics")
    c6 = c6.replace("2️⃣ VIX Data", "2 VIX Data")
    c6 = c6.replace("3️⃣ IRX Data (Risk-free rate proxy)", "3 IRX Data (Risk-free rate proxy)")
    c6 = c6.replace("4️⃣ XLK Benchmark", "4 XLK Benchmark")
    c6 = c6.replace("5️⃣ Missing Data Check", "6 Missing Data Check")
    c6 = c6.replace(
        "# Missing data check\n",
        "# SPY statistics\n"
        "print(f\"\\n5 SPY Benchmark:\")\n"
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
        "exclude_symbols = ['XLK', 'SPY', '^VIX', '^IRX']",
    )
    cells.append(code_cell(to_lines(c8)))

    # 9
    cells.append(code_cell(v2[12]))

    # 10
    cells.append(md_cell(to_lines("""
### Position Sizing with Dynamic Cash Management

Portfolio exposure is controlled by a multiplicative signal stack:

- **VIX Gate**: hard risk cap from volatility regime.
- **Momentum Signal**: stronger cross-sectional momentum increases equity allocation.
- **Trend Strength**: multi-horizon trend confirmation adjusts conviction.

Final equity allocation is:

`equity_weight = w_vix * (0.4 * w_mom + 0.4 * w_trend + 0.2 * 1.0)`

with clipping to `[0, 1]`; remaining capital is allocated to cash (IRX proxy).
""")))

    # 11
    cells.append(code_cell(to_lines("""
def calculate_position_size(vix_level, momentum_signal, trend_strength):
    # Calculate dynamic equity/cash allocation from risk and signal state.
    # VIX gate
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

    # Momentum signal gate
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

    # Trend gate
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
""")))

    # 12
    cells.append(md_cell(to_lines("""
### Strategy Backtesting: Long-Only with Cash Management
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
    # Long-only momentum backtest with dynamic cash allocation and variable holding periods.
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

    def _trend_strength(signal_date):
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

            trend_strength = _trend_strength(signal_date)
            sizing = calculate_position_size(vix_level, mom_signal, trend_strength)
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

ret_simple_v3 = backtest_strategy_v3(adj_close, volume, lookback_months=6, method='simple', decile=0.3, holding_period=1)
ret_risk_v3 = backtest_strategy_v3(adj_close, volume, lookback_months=6, method='risk_adjusted', decile=0.3, holding_period=1)
ret_vol_v3 = backtest_strategy_v3(adj_close, volume, lookback_months=6, method='volume_weighted', decile=0.3, holding_period=1)

for name, df_ in [
    ('Simple', ret_simple_v3),
    ('Risk-Adjusted', ret_risk_v3),
    ('Volume-Weighted', ret_vol_v3),
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
def analyze_holding_periods(adj_close, volume, methods=None, holding_periods=None, lookback_months=6):
    if methods is None:
        methods = ['simple', 'risk_adjusted', 'volume_weighted']
    if holding_periods is None:
        holding_periods = [1, 2, 3, 6]

    train_end = pd.Timestamp('2024-12-31')
    rows = []

    for m in methods:
        for hp in holding_periods:
            bt = backtest_strategy_v3(
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
holding_period_results = analyze_holding_periods(adj_close, volume)
print(holding_period_results.round(4))

best_config = holding_period_results.iloc[0]
print("\\nBest config:")
print(best_config)

best_strategy_df = backtest_strategy_v3(
    adj_close,
    volume,
    lookback_months=6,
    method=best_config['method'],
    decile=0.3,
    tc_bps=5,
    holding_period=int(best_config['holding_period']),
)
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

print("\\nAutocorrelation of strategy returns (best config):")
sr = best_strategy_df['strategy_returns']
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
    'Simple (6m, hp1)': ret_simple_v3['strategy_returns'],
    'Risk-Adj (6m, hp1)': ret_risk_v3['strategy_returns'],
    'Volume-Wtd (6m, hp1)': ret_vol_v3['strategy_returns'],
    'Best Holding Config': best_strategy_df['strategy_returns'],
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
    cells.append(md_cell(to_lines("""
## Market Regime Analysis
""")))

    # 23 adapt from V2 cell 22
    c23 = "".join(v2[21])
    c23 = c23.replace("backtest_strategy(adj_close, volume, lookback_months=lb, method=method)", "backtest_strategy_v3(adj_close, volume, lookback_months=lb, method=method, decile=0.3)")
    c23 = c23.replace("strategy_returns_train = strategy_returns[strategy_returns.index <= train_end]", "strategy_returns = strategy_returns['strategy_returns']\n            strategy_returns_train = strategy_returns[strategy_returns.index <= train_end]")
    c23 = c23.replace("primary_strategy = backtest_strategy(adj_close, volume, lookback_months=6, method='simple')", "primary_strategy = backtest_strategy_v3(adj_close, volume, lookback_months=6, method='simple', decile=0.3)\n    primary_strategy = primary_strategy['strategy_returns']")
    c23 = c23.replace("primary_strategy = strategies[best_strategy]['returns']", "primary_strategy = strategies[best_strategy]['returns']")
    c23 = c23.replace("# XLK monthly returns (our market benchmark)", "# XLK/SPY monthly returns (market benchmarks)")
    c23 = c23.replace("print(f\"✅ XLK monthly returns: {len(xlk_monthly)} months\")", "spy_monthly = adj_close['SPY'].resample('M').last().pct_change().dropna()\nprint(f\"✅ XLK monthly returns: {len(xlk_monthly)} months\")\nprint(f\"✅ SPY monthly returns: {len(spy_monthly)} months\")")
    c23 = c23.replace("(1 + rf_daily).resample('M').apply", "rf_daily.resample('M').apply")  # fix double-compounding bug
    c23 += "\n\nprimary_strategy_df = backtest_strategy_v3(adj_close, volume, lookback_months=best_params[1] if best_params else 6, method=best_params[0] if best_params else 'simple', decile=0.3)\nprimary_strategy_df = primary_strategy_df[primary_strategy_df.index > train_end]\n"
    cells.append(code_cell(to_lines(c23)))

    # 24
    cells.append(md_cell(to_lines("""
### Market Regime Classification with VIX
""")))

    # 25 adapt from V2 cell 24
    c25 = "".join(v2[23])
    c25 = c25.replace("strategy_index = primary_strategy.index", "strategy_index = primary_strategy_df.index")
    c25 = c25.replace("'strategy_returns': primary_strategy,", "'strategy_returns': primary_strategy_df['strategy_returns'],\n    'equity_weight': primary_strategy_df['equity_weight'],\n    'cash_weight': primary_strategy_df['cash_weight'],")
    c25 = c25.replace("'xlk_returns': xlk_aligned,", "'xlk_returns': xlk_aligned,\n    'spy_returns': spy_monthly.reindex(strategy_index, method='ffill').fillna(0),")
    cells.append(code_cell(to_lines(c25)))

    # 26
    cells.append(md_cell(to_lines("""
### Strategy Performance by Market Regime
""")))

    # 27 adapt from V2 cell 26 + cash allocation section
    c27 = "".join(v2[25])
    c27 += "\n\n# 9. Cash Allocation by Regime\nprint(\"\\n9️⃣ Cash Allocation by Regime:\")\nif 'equity_weight' in regime_data.columns:\n    cash_by_regime = regime_data.groupby('combined_regime')['equity_weight'].mean().sort_values(ascending=False)\n    print(cash_by_regime.rename('avg_equity_weight'))\n"
    cells.append(code_cell(to_lines(c27)))

    # 28
    cells.append(md_cell(to_lines("""
### Visualizations
""")))

    # 29
    cells.append(code_cell(to_lines("""
import plotly.graph_objects as go
from plotly.subplots import make_subplots

cum_df = pd.DataFrame({
    'Strategy': primary_strategy_df['strategy_returns'],
    'XLK': xlk_monthly.reindex(primary_strategy_df.index, method='ffill'),
    'SPY': spy_monthly.reindex(primary_strategy_df.index, method='ffill'),
}).dropna()

cum = (1 + cum_df).cumprod() - 1

fig = go.Figure()
fig.add_trace(go.Scatter(x=cum.index, y=cum['Strategy'], mode='lines', name='Strategy'))
fig.add_trace(go.Scatter(x=cum.index, y=cum['XLK'], mode='lines', name='XLK'))
fig.add_trace(go.Scatter(x=cum.index, y=cum['SPY'], mode='lines', name='SPY'))
fig.update_layout(title='Cumulative Returns: Strategy vs XLK vs SPY', yaxis_title='Cumulative Return', template='plotly_white')
fig.show()
""")))

    # 30
    cells.append(code_cell(to_lines("""
roll_12m_strategy = (1 + primary_strategy_df['strategy_returns']).rolling(12).apply(np.prod, raw=True) - 1
roll_12m_xlk = (1 + xlk_monthly.reindex(primary_strategy_df.index, method='ffill')).rolling(12).apply(np.prod, raw=True) - 1
vix_plot = vix_monthly.reindex(primary_strategy_df.index, method='ffill')

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=roll_12m_strategy.index, y=roll_12m_strategy, mode='lines', name='Strategy 12M'), secondary_y=False)
fig.add_trace(go.Scatter(x=roll_12m_xlk.index, y=roll_12m_xlk, mode='lines', name='XLK 12M'), secondary_y=False)
fig.add_trace(go.Scatter(x=vix_plot.index, y=vix_plot, mode='lines', name='VIX', line=dict(dash='dot')), secondary_y=True)
fig.update_layout(title='Rolling 12M Returns with VIX', template='plotly_white')
fig.update_yaxes(title_text='Rolling 12M Return', secondary_y=False)
fig.update_yaxes(title_text='VIX', secondary_y=True)
fig.show()
""")))

    # 31
    cells.append(code_cell(to_lines("""
alloc_df = primary_strategy_df[['equity_weight', 'cash_weight']].copy()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=alloc_df.index,
    y=alloc_df['equity_weight'],
    mode='lines',
    stackgroup='one',
    name='Equity Weight'
))
fig.add_trace(go.Scatter(
    x=alloc_df.index,
    y=alloc_df['cash_weight'],
    mode='lines',
    stackgroup='one',
    name='Cash Weight'
))
fig.update_layout(title='Dynamic Allocation Over Time', yaxis_title='Portfolio Weight', template='plotly_white')
fig.show()
""")))

    # 32
    cells.append(code_cell(to_lines("""
heat_df = holding_period_results.pivot(index='method', columns='holding_period', values='sharpe')

fig = go.Figure(data=go.Heatmap(
    z=heat_df.values,
    x=heat_df.columns.astype(str),
    y=heat_df.index,
    colorscale='RdYlGn',
    colorbar=dict(title='Sharpe')
))
fig.update_layout(title='Holding Period Grid Search Heatmap (Sharpe)', xaxis_title='Holding Period (months)', yaxis_title='Method', template='plotly_white')
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

    V3_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote V3 notebook to {V3_PATH}")
    print(f"Total cells: {len(cells)}")


if __name__ == "__main__":
    main()
