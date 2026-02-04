import json
from pathlib import Path
from uuid import uuid4


SRC_PATH = Path(r"E:\programs\APEXUSTech_Inter\YiboSun_Project4_26_08.ipynb")
DST_PATH = Path(r"E:\programs\APEXUSTech_Inter\project4\YiboSun_Project4_26_08_V2.ipynb")


def update_data_collection_markdown(nb):
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            text = "".join(cell.get("source", []))
            if text.strip().startswith("### Data Collection"):
                new_text = (
                    "### Data Collection\n\n"
                    "In this project, we use three datasets: adj_close.csv (Tiingo API), "
                    "volume.csv (Tiingo API), and VIX.csv (manually downloaded from investing.com). "
                    "IRX (3‑month Treasury) is fetched via Alpha Vantage. "
                    "An optional Yahoo Finance requests example is included below (no yfinance dependency)."
                )
                cell["source"] = [new_text]
                return


def insert_yahoo_snippet(nb):
    insert_idx = None
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code":
            text = "".join(cell.get("source", []))
            if text.strip().startswith("import pandas as pd"):
                insert_idx = i + 1
                break

    if insert_idx is None:
        return

    yahoo_md = {
        "cell_type": "markdown",
        "metadata": {},
        "id": uuid4().hex[:8],
        "source": ["# Yahoo Finance API - 直接使用requests，无需yfinance库\n"],
    }
    yahoo_code = {
        "cell_type": "code",
        "metadata": {},
        "id": uuid4().hex[:8],
        "execution_count": None,
        "outputs": [],
        "source": [
            "import requests\n",
            "\n",
            "# 设置时间范围：2024-10-01 到 2026-01-20\n",
            "start_date = datetime(2024, 10, 1)\n",
            "end_date = datetime(2026, 1, 20)\n",
            "start_ts = int(start_date.timestamp())\n",
            "end_ts = int(end_date.timestamp())\n",
            "\n",
            "url = f'https://query1.finance.yahoo.com/v8/finance/chart/AAPL?period1={start_ts}&period2={end_ts}&interval=1d'\n",
            "headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}\n",
            "\n",
            "print(\"Fetching data from Yahoo Finance...\")\n",
            "response = requests.get(url, headers=headers)\n",
            "data = response.json()\n",
        ],
    }

    nb["cells"][insert_idx:insert_idx] = [yahoo_md, yahoo_code]


def update_data_download_cell(nb):
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "end_date = datetime.today().strftime('%Y-%m-%d')" in src:
            src = src.replace(
                "end_date = datetime.today().strftime('%Y-%m-%d')",
                "end_date = '2026-01-30'  # updated to 2026-01-30",
            )
        if "adj_close = adj_close.ffill(limit=5).bfill(limit=5)" in src:
            src = src.replace(
                "adj_close = adj_close.ffill(limit=5).bfill(limit=5)",
                "adj_close = adj_close.ffill(limit=5)  # no backfill to avoid look-ahead bias",
            )
        if "volume = volume.ffill(limit=5).bfill(limit=5)" in src:
            src = src.replace(
                "volume = volume.ffill(limit=5).bfill(limit=5)",
                "volume = volume.ffill(limit=5)  # no backfill to avoid look-ahead bias",
            )
        cell["source"] = [src]


def replace_irx_vix_cell(nb):
    new_lines = [
        "# Reload current data\n",
        "adj_close = pd.read_csv('adj_close.csv', index_col=0, parse_dates=True)\n",
        "volume = pd.read_csv('volume.csv', index_col=0, parse_dates=True)\n",
        "\n",
        "print(\"=\"*60)\n",
        "print(\"Replacing synthetic data with real data\")\n",
        "print(\"=\"*60)\n",
        "\n",
        "# 1. Fetch and process IRX data from Alpha Vantage\n",
        "print(\"1. Processing Alpha Vantage IRX data...\")\n",
        "\n",
        "url = f'https://www.alphavantage.co/query?function=TREASURY_YIELD&interval=daily&maturity=3month&apikey={Alpha_api_key}'\n",
        "\n",
        "try:\n",
        "    response = requests.get(url)\n",
        "    if response.status_code == 200:\n",
        "        irx_data = response.json()\n",
        "        \n",
        "        if 'data' in irx_data:\n",
        "            # Convert to DataFrame\n",
        "            irx_df = pd.DataFrame(irx_data['data'])\n",
        "            irx_df['date'] = pd.to_datetime(irx_df['date'])\n",
        "            irx_df.set_index('date', inplace=True)\n",
        "            irx_df['value'] = pd.to_numeric(irx_df['value'], errors='coerce')\n",
        "            \n",
        "            # Filter for our required date range\n",
        "            start_datetime = pd.to_datetime('2019-12-01')\n",
        "            end_datetime = pd.to_datetime('2026-01-30')\n",
        "            irx_df = irx_df[(irx_df.index >= start_datetime) & (irx_df.index <= end_datetime)]\n",
        "            irx_df = irx_df.sort_index()\n",
        "            \n",
        "            print(f\"Number of IRX data points: {len(irx_df)}\")\n",
        "            \n",
        "            # Handle timezone issues - unify to timezone-naive\n",
        "            if adj_close.index.tz is not None:\n",
        "                adj_close.index = adj_close.index.tz_localize(None)\n",
        "            if irx_df.index.tz is not None:\n",
        "                irx_df.index = irx_df.index.tz_localize(None)\n",
        "            \n",
        "            # Vectorized alignment to trading dates\n",
        "            irx_series = irx_df['value'].reindex(adj_close.index, method='ffill')\n",
        "            adj_close['^IRX'] = irx_series\n",
        "            \n",
        "            irx_updated_count = adj_close['^IRX'].notna().sum()\n",
        "            print(f\"✅ IRX real data replacement successful for {irx_updated_count} trading days\")\n",
        "            print(f\"   Data range: {irx_df['value'].min():.3f}% - {irx_df['value'].max():.3f}%\")\n",
        "            print(f\"   Latest value: {adj_close['^IRX'].iloc[-1]:.3f}%\")\n",
        "        else:\n",
        "            print(\"❌ Incorrect Alpha Vantage IRX data format\")\n",
        "    else:\n",
        "        print(f\"❌ Alpha Vantage request failed: {response.status_code}\")\n",
        "        \n",
        "except Exception as e:\n",
        "    print(f\"❌ IRX data processing failed: {e}\")\n",
        "\n",
        "# 2. Process VIX data\n",
        "print(f\"\\n2. Processing VIX data...\")\n",
        "\n",
        "vix_file = 'VIX.csv'\n",
        "vix_success = False\n",
        "\n",
        "if os.path.exists(vix_file):\n",
        "    try:\n",
        "        print(f\"Found VIX file: {vix_file}\")\n",
        "        vix_df = pd.read_csv(vix_file)\n",
        "        \n",
        "        # Adapt for Chinese column names\n",
        "        column_mapping = {\n",
        "            '日期': 'date',\n",
        "            '收盘': 'close',\n",
        "            '开盘': 'open',\n",
        "            '高': 'high',\n",
        "            '低': 'low'\n",
        "        }\n",
        "        \n",
        "        # Rename columns\n",
        "        vix_df = vix_df.rename(columns=column_mapping)\n",
        "        \n",
        "        if 'date' in vix_df.columns and 'close' in vix_df.columns:\n",
        "            # Process dates\n",
        "            vix_df['date'] = pd.to_datetime(vix_df['date'], errors='coerce')\n",
        "            vix_df = vix_df.dropna(subset=['date'])\n",
        "            vix_df.set_index('date', inplace=True)\n",
        "            \n",
        "            # Clean price data\n",
        "            if vix_df['close'].dtype == 'object':\n",
        "                vix_df['close'] = vix_df['close'].astype(str).str.replace(',', '').str.replace('%', '')\n",
        "            vix_df['close'] = pd.to_numeric(vix_df['close'], errors='coerce')\n",
        "            \n",
        "            # Filter date range\n",
        "            start_datetime = pd.to_datetime('2019-12-01')\n",
        "            end_datetime = pd.to_datetime('2026-01-30')\n",
        "            vix_df = vix_df[(vix_df.index >= start_datetime) & (vix_df.index <= end_datetime)]\n",
        "            vix_df = vix_df.sort_index()\n",
        "            \n",
        "            print(f\"Number of VIX data points: {len(vix_df)}\")\n",
        "            \n",
        "            # Handle timezone issues - unify to timezone-naive\n",
        "            if vix_df.index.tz is not None:\n",
        "                vix_df.index = vix_df.index.tz_localize(None)\n",
        "            \n",
        "            # Vectorized alignment to trading dates\n",
        "            vix_series = vix_df['close'].reindex(adj_close.index, method='ffill')\n",
        "            adj_close['^VIX'] = vix_series\n",
        "            \n",
        "            vix_updated_count = adj_close['^VIX'].notna().sum()\n",
        "            print(f\"✅ VIX real data replacement successful for {vix_updated_count} trading days\")\n",
        "            print(f\"   Data range: {vix_df['close'].min():.2f} - {vix_df['close'].max():.2f}\")\n",
        "            print(f\"   Latest value: {adj_close['^VIX'].iloc[-1]:.2f}\")\n",
        "            vix_success = True\n",
        "        else:\n",
        "            print(f\"❌ Could not recognize VIX file column structure\")\n",
        "            \n",
        "    except Exception as e:\n",
        "        print(f\"❌ VIX data processing failed: {e}\")\n",
        "else:\n",
        "    print(\"❌ VIX.csv file not found\")\n",
        "\n",
        "# 3. Remove VIX and IRX from volume data (if they exist)\n",
        "print(f\"\\n3. Cleaning up volume data...\")\n",
        "\n",
        "volume_updated = False\n",
        "if '^VIX' in volume.columns:\n",
        "    volume = volume.drop(columns=['^VIX'])\n",
        "    print(\"✅ Removed VIX from volume data\")\n",
        "    volume_updated = True\n",
        "\n",
        "if '^IRX' in volume.columns:\n",
        "    volume = volume.drop(columns=['^IRX'])\n",
        "    print(\"✅ Removed IRX from volume data\")\n",
        "    volume_updated = True\n",
        "\n",
        "if not volume_updated:\n",
        "    print(\"ℹ️ VIX and IRX are already absent from volume data\")\n",
        "\n",
        "# 4. Save the updated data\n",
        "print(f\"\\n4. Saving updated data...\")\n",
        "\n",
        "adj_close.to_csv('adj_close.csv')\n",
        "volume.to_csv('volume.csv')\n",
        "\n",
        "print(\"✅ adj_close.csv has been updated (with real IRX and VIX data)\")\n",
        "print(\"✅ volume.csv has been updated (VIX and IRX volume data removed)\")\n",
        "\n",
        "# 5. Data quality check\n",
        "print(f\"\\n5. Statistics for updated data:\")\n",
        "print(f\"   adj_close shape: {adj_close.shape}\")\n",
        "print(f\"   volume shape: {volume.shape}\")\n",
        "\n",
        "if '^IRX' in adj_close.columns:\n",
        "    irx_real_count = adj_close['^IRX'].notna().sum()\n",
        "    irx_latest = adj_close['^IRX'].iloc[-1]\n",
        "    irx_mean = adj_close['^IRX'].mean()\n",
        "    irx_std = adj_close['^IRX'].std()\n",
        "    print(f\"   IRX real data: {irx_real_count} points\")\n",
        "    print(f\"       Latest value: {irx_latest:.3f}%\")\n",
        "    print(f\"       Average value: {irx_mean:.3f}%\")\n",
        "    print(f\"       Standard deviation: {irx_std:.3f}%\")\n",
        "\n",
        "if '^VIX' in adj_close.columns:\n",
        "    vix_real_count = adj_close['^VIX'].notna().sum()\n",
        "    vix_latest = adj_close['^VIX'].iloc[-1]\n",
        "    vix_mean = adj_close['^VIX'].mean()\n",
        "    vix_std = adj_close['^VIX'].std()\n",
        "    print(f\"   VIX real data: {vix_real_count} points\")\n",
        "    print(f\"       Latest value: {vix_latest:.2f}\")\n",
        "    print(f\"       Average value: {vix_mean:.2f}\")\n",
        "    print(f\"       Standard deviation: {vix_std:.2f}\")\n",
        "\n",
        "# 6. Verify data changes\n",
        "print(f\"\\n6. Verifying data changes:\")\n",
        "print(\"   First 5 VIX values:\", adj_close['^VIX'].head().tolist())\n",
        "print(\"   First 5 IRX values:\", adj_close['^IRX'].head().tolist())\n",
    ]

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "Replacing synthetic data with real data" in src:
            cell["source"] = new_lines
            return


def replace_backtest_strategy(nb):
    new_lines = [
        "def backtest_strategy(adj_close, volume, lookback_months=6, method='simple', decile=0.2, tc_bps=5, winsor_q=0.01):\n",
        "    \"\"\"\n",
        "    Backtests a long-short momentum strategy with monthly rebalancing.\n",
        "    \n",
        "    Parameters:\n",
        "    - adj_close: DataFrame of adjusted close prices\n",
        "    - volume: DataFrame of trading volumes\n",
        "    - lookback_months: Lookback period (3, 6, 12)\n",
        "    - method: 'simple', 'risk_adjusted', or 'volume_weighted'\n",
        "    - decile: The top/bottom quantile to use (0.2 = 20%)\n",
        "    - tc_bps: Transaction cost per side in basis points\n",
        "    - winsor_q: winsorization quantile for cross-sectional monthly returns\n",
        "    \n",
        "    Returns:\n",
        "    - A Series of monthly strategy returns (indexed by month-end dates)\n",
        "    \"\"\"\n",
        "    # Fix FutureWarning - explicitly specify fill_method=None\n",
        "    daily_returns = adj_close.pct_change(fill_method=None)\n",
        "    \n",
        "    # Get month-end dates (resample to the last trading day of the month), ensuring actual trading days are used.\n",
        "    month_ends = adj_close.resample('ME').last().index  # 'ME' for month end\n",
        "    \n",
        "    # Filter for month-end dates that actually exist in the data\n",
        "    valid_month_ends = []\n",
        "    for me in month_ends:\n",
        "        month_data = adj_close.loc[adj_close.index.month == me.month]\n",
        "        month_data = month_data.loc[month_data.index.year == me.year]\n",
        "        if len(month_data) > 0:\n",
        "            actual_month_end = month_data.index[-1]\n",
        "            valid_month_ends.append(actual_month_end)\n",
        "    \n",
        "    valid_month_ends = pd.DatetimeIndex(valid_month_ends).unique()\n",
        "    \n",
        "    # Initialize returns Series, starting after the lookback period\n",
        "    portfolio_returns = pd.Series(index=valid_month_ends[lookback_months:], dtype=float, name='Strategy Returns')\n",
        "    \n",
        "    # Monthly rebalancing loop\n",
        "    for i in range(lookback_months, len(valid_month_ends)):\n",
        "        current_date = valid_month_ends[i]\n",
        "        previous_date = valid_month_ends[i-1]\n",
        "        \n",
        "        # Ensure both dates are in the data\n",
        "        if current_date not in adj_close.index:\n",
        "            print(f\"Warning: {current_date} not in data, skipping\")\n",
        "            portfolio_returns[current_date] = 0.0\n",
        "            continue\n",
        "        if previous_date not in adj_close.index:\n",
        "            print(f\"Warning: {previous_date} not in data, skipping\")\n",
        "            portfolio_returns[current_date] = 0.0\n",
        "            continue\n",
        "        \n",
        "        # Calculate scores from the previous month-end\n",
        "        scores = calculate_momentum_scores(adj_close, volume, daily_returns, previous_date, lookback_months, method)\n",
        "        \n",
        "        if scores.empty:\n",
        "            portfolio_returns[current_date] = 0.0\n",
        "            continue\n",
        "        \n",
        "        # Rank and select top/bottom stocks\n",
        "        num_stocks = len(scores)\n",
        "        top_n = max(1, int(num_stocks * decile))\n",
        "        bottom_n = max(1, int(num_stocks * decile))\n",
        "        \n",
        "        ranks = scores.rank(ascending=False)  # Rank in descending order (1=highest)\n",
        "        long_stocks = scores[ranks <= top_n].index\n",
        "        short_stocks = scores[ranks > num_stocks - bottom_n].index\n",
        "        \n",
        "        # Next month's return (from previous to current)\n",
        "        try:\n",
        "            ret_next = (adj_close.loc[current_date, scores.index] / adj_close.loc[previous_date, scores.index] - 1)\n",
        "            ret_next = ret_next.replace([np.inf, -np.inf], np.nan).dropna()\n",
        "        except KeyError as e:\n",
        "            print(f\"KeyError for dates {current_date} or {previous_date}: {e}\")\n",
        "            portfolio_returns[current_date] = 0.0\n",
        "            continue\n",
        "        \n",
        "        # Winsorize cross-sectional returns to reduce outlier impact (no hard caps)\n",
        "        if len(ret_next) > 5:\n",
        "            lower = ret_next.quantile(winsor_q)\n",
        "            upper = ret_next.quantile(1 - winsor_q)\n",
        "            ret_next = ret_next.clip(lower, upper)\n",
        "        \n",
        "        # Check if we have enough stocks for both long and short positions\n",
        "        available_long = [stock for stock in long_stocks if stock in ret_next.index]\n",
        "        available_short = [stock for stock in short_stocks if stock in ret_next.index]\n",
        "        \n",
        "        if len(available_long) == 0 or len(available_short) == 0:\n",
        "            portfolio_returns[current_date] = 0.0\n",
        "            continue\n",
        "        \n",
        "        long_ret = ret_next[available_long].mean()\n",
        "        short_ret = ret_next[available_short].mean()\n",
        "        \n",
        "        # Additional safety check for valid returns\n",
        "        if pd.isna(long_ret) or pd.isna(short_ret):\n",
        "            portfolio_returns[current_date] = 0.0\n",
        "            continue\n",
        "            \n",
        "        strategy_ret = long_ret - short_ret\n",
        "        \n",
        "        # Transaction cost: Assume full turnover, 2 sides (long + short) * tc_bps\n",
        "        tc = (tc_bps / 10000) * 2  # bps to decimal, *2 for long/short\n",
        "        strategy_ret -= tc\n",
        "        \n",
        "        portfolio_returns[current_date] = strategy_ret\n",
        "    \n",
        "    return portfolio_returns.dropna()\n",
    ]

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if src.strip().startswith("def backtest_strategy"):
            cell["source"] = new_lines
            return


def update_regime_analysis(nb):
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "Market Regime Analysis - Data Preparation" in src:
            insert_text = (
                "lookbacks = [3, 6, 12]\n\n"
                "# Train/Test split to avoid data snooping\n"
                "train_end = pd.Timestamp('2024-12-31')\n"
                "print(f\"Train/Test split: train <= {train_end.date()}, test > {train_end.date()}\")\n"
            )
            src = src.replace("lookbacks = [3, 6, 12]\n", insert_text)
            src = src.replace(
                "if len(strategy_returns) > 24:",
                "strategy_returns_train = strategy_returns[strategy_returns.index <= train_end]\n            if len(strategy_returns_train) > 24:",
            )
            src = src.replace(
                "returns_clean = strategy_returns.copy()",
                "returns_clean = strategy_returns_train.copy()",
            )
            src = src.replace(
                "primary_strategy = strategies[best_strategy]['returns']",
                "primary_strategy = strategies[best_strategy]['returns']\n    primary_strategy = primary_strategy[primary_strategy.index > train_end]",
            )
            src = src.replace(
                "primary_strategy = backtest_strategy(adj_close, volume, lookback_months=6, method='simple')",
                "primary_strategy = backtest_strategy(adj_close, volume, lookback_months=6, method='simple')\n    primary_strategy = primary_strategy[primary_strategy.index > train_end]",
            )
            cell["source"] = [src]
            return


def update_test_date(nb):
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "test_date = pd.Timestamp('2025-07-31')" in src:
            src = src.replace(
                "test_date = pd.Timestamp('2025-07-31')",
                "test_date = pd.Timestamp('2026-01-30')",
            )
            cell["source"] = [src]
            return


def main():
    nb = json.loads(SRC_PATH.read_text(encoding="utf-8"))
    update_data_collection_markdown(nb)
    insert_yahoo_snippet(nb)
    update_data_download_cell(nb)
    replace_irx_vix_cell(nb)
    replace_backtest_strategy(nb)
    update_regime_analysis(nb)
    update_test_date(nb)

    nb["metadata"] = nb.get("metadata", {})
    DST_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote V2 notebook to {DST_PATH}")


if __name__ == "__main__":
    main()
