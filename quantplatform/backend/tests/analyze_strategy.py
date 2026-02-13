"""
策略归因分析脚本 — 诊断多头/空头各自的贡献与亏损来源

分析内容:
1. 多头 vs 空头 各自的月度收益率
2. 纯多头策略 vs 多空策略 vs XLK 基准 对比
3. 空头造成的累计亏损金额
4. 被做空后大涨的"杀伤性"股票排名
5. 被做空的大蓝筹（NVDA, MSFT 等）统计
6. 月度归因热力图：哪些月份空头亏最多
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from datetime import datetime

from app.config import settings
from app.services.momentum_strategy import (
    calculate_momentum_scores,
    EXCLUDE_SYMBOLS,
)

# ============================================================
# 配置
# ============================================================

METHODS = ["risk_adjusted", "simple", "volume_weighted"]
LOOKBACK = 6
DECILE = 0.2
TC_BPS = 5
WINSOR_Q = 0.01
START_VALUE = 10000.0

# 大蓝筹列表（做空这些股票风险特别大）
MEGA_CAPS = ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "CSCO", "ACN", "ADBE"]


def load_data():
    """从 CSV 文件加载数据（使用 notebook 已下载的数据）"""
    # 尝试从 project4 目录读取
    project4_dir = os.path.join(os.path.dirname(__file__), "..", "..", "project4")
    adj_close_path = os.path.join(project4_dir, "adj_close.csv")
    volume_path = os.path.join(project4_dir, "volume.csv")

    if os.path.exists(adj_close_path):
        print(f"从 {adj_close_path} 加载数据...")
        adj_close = pd.read_csv(adj_close_path, index_col=0, parse_dates=True)
        volume = pd.read_csv(volume_path, index_col=0, parse_dates=True)
        return adj_close, volume

    print("未找到 CSV 数据文件。请先在 project4 目录运行 notebook 下载数据。")
    print(f"期望路径: {adj_close_path}")
    sys.exit(1)


def run_decomposed_backtest(adj_close, volume, method="risk_adjusted", lookback_months=6):
    """
    运行回测并分解多头/空头各自的贡献。

    返回 DataFrame，列:
    - long_ret: 多头月度收益率
    - short_ret: 空头月度收益率（注意：做空亏损表现为负的 -short_ret）
    - long_minus_short: 策略收益 = long_ret - short_ret
    - benchmark_ret: XLK 月度收益率
    - long_stocks: 做多的股票列表
    - short_stocks: 做空的股票列表
    - short_individual: 每只被做空股票的收益率 dict
    """
    daily_returns = adj_close.pct_change(fill_method=None)

    # 月末日期
    month_ends = adj_close.resample("ME").last().index
    valid_month_ends = []
    for me in month_ends:
        month_data = adj_close.loc[
            (adj_close.index.month == me.month) & (adj_close.index.year == me.year)
        ]
        if len(month_data) > 0:
            valid_month_ends.append(month_data.index[-1])
    valid_month_ends = pd.DatetimeIndex(valid_month_ends).unique()

    records = []

    for i in range(lookback_months, len(valid_month_ends)):
        current_date = valid_month_ends[i]
        previous_date = valid_month_ends[i - 1]

        if current_date not in adj_close.index or previous_date not in adj_close.index:
            continue

        scores = calculate_momentum_scores(
            adj_close, volume, daily_returns, previous_date, lookback_months, method
        )

        if scores.empty:
            continue

        num_stocks = len(scores)
        top_n = max(1, int(num_stocks * DECILE))
        bottom_n = max(1, int(num_stocks * DECILE))

        ranks = scores.rank(ascending=False)
        long_stocks = scores[ranks <= top_n].index.tolist()
        short_stocks = scores[ranks > num_stocks - bottom_n].index.tolist()

        # 下月收益
        try:
            ret_next = (
                adj_close.loc[current_date, scores.index]
                / adj_close.loc[previous_date, scores.index]
                - 1
            )
            ret_next = ret_next.replace([np.inf, -np.inf], np.nan).dropna()
        except KeyError:
            continue

        # Winsorize
        if len(ret_next) > 5:
            lower = ret_next.quantile(WINSOR_Q)
            upper = ret_next.quantile(1 - WINSOR_Q)
            ret_next = ret_next.clip(lower, upper)

        available_long = [s for s in long_stocks if s in ret_next.index]
        available_short = [s for s in short_stocks if s in ret_next.index]

        if len(available_long) == 0 or len(available_short) == 0:
            continue

        long_ret = ret_next[available_long].mean()
        short_ret = ret_next[available_short].mean()

        # 空头每只股票的收益率（正值=股票涨了=做空亏了）
        short_individual = {s: float(ret_next[s]) for s in available_short}

        # Benchmark
        benchmark_ret = 0.0
        if "XLK" in adj_close.columns:
            try:
                benchmark_ret = (
                    adj_close.loc[current_date, "XLK"]
                    / adj_close.loc[previous_date, "XLK"]
                    - 1
                )
            except KeyError:
                pass

        tc = (TC_BPS / 10000) * 2

        records.append({
            "date": current_date,
            "long_ret": float(long_ret),
            "short_ret": float(short_ret),  # 空头组合的实际涨跌（正=涨了=做空亏了）
            "short_cost": float(short_ret),  # 做空的成本 = 空头组合涨幅
            "long_minus_short": float(long_ret - short_ret - tc),
            "long_only_net": float(long_ret - tc / 2),  # 纯多头（只扣一半TC）
            "benchmark_ret": float(benchmark_ret),
            "long_stocks": available_long,
            "short_stocks": available_short,
            "short_individual": short_individual,
            "long_individual": {s: float(ret_next[s]) for s in available_long},
            "num_long": len(available_long),
            "num_short": len(available_short),
        })

    return pd.DataFrame(records).set_index("date")


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def analyze_method(adj_close, volume, method, lookback=6):
    """对单个方法进行全面分析"""
    print_section(f"策略: {method.upper()} (Lookback={lookback}M)")

    df = run_decomposed_backtest(adj_close, volume, method, lookback)

    if df.empty:
        print("  无回测数据！")
        return df

    # ----------------------------------------------------------
    # 1. 多头 vs 空头 总览
    # ----------------------------------------------------------
    print_section("1. 多头 vs 空头 收益率总览")

    total_months = len(df)

    # 累计收益（复利）
    cum_long = (1 + df["long_only_net"]).prod() - 1
    cum_short_cost = (1 + df["short_cost"]).prod() - 1  # 空头组合累计涨了多少
    cum_strategy = (1 + df["long_minus_short"]).prod() - 1
    cum_benchmark = (1 + df["benchmark_ret"]).prod() - 1

    print(f"  回测期间: {df.index[0].strftime('%Y-%m')} ~ {df.index[-1].strftime('%Y-%m')} ({total_months} 个月)")
    print(f"")
    print(f"  {'纯多头策略累计收益:':<30} {cum_long:>+10.1%}")
    print(f"  {'多空策略累计收益:':<30} {cum_strategy:>+10.1%}")
    print(f"  {'XLK 基准累计收益:':<30} {cum_benchmark:>+10.1%}")
    print(f"  {'空头组合累计涨幅:':<30} {cum_short_cost:>+10.1%}  ← 做空亏损来源")
    print(f"")

    # 月均
    print(f"  {'多头月均收益:':<30} {df['long_ret'].mean():>+8.2%}  (年化 {((1+df['long_ret'].mean())**12-1):>+.1%})")
    print(f"  {'空头月均涨幅(=做空成本):':<30} {df['short_ret'].mean():>+8.2%}  (年化 {((1+df['short_ret'].mean())**12-1):>+.1%})")
    print(f"  {'策略月均收益:':<30} {df['long_minus_short'].mean():>+8.2%}")
    print(f"  {'基准月均收益:':<30} {df['benchmark_ret'].mean():>+8.2%}")

    # ----------------------------------------------------------
    # 2. 空头造成亏损的月份分析
    # ----------------------------------------------------------
    print_section("2. 空头造成亏损最大的月份 (空头组合大涨 = 做空巨亏)")

    # 空头组合涨幅最大的月份（做空亏损最大）
    worst_short_months = df.nlargest(10, "short_ret")
    print(f"  {'月份':<12} {'空头涨幅':>10} {'多头收益':>10} {'策略净收益':>12} {'基准':>10}")
    print(f"  {'-'*56}")
    for dt, row in worst_short_months.iterrows():
        print(f"  {dt.strftime('%Y-%m'):<12} {row['short_ret']:>+10.2%} {row['long_ret']:>+10.2%} {row['long_minus_short']:>+12.2%} {row['benchmark_ret']:>+10.2%}")

    # ----------------------------------------------------------
    # 3. 多头 vs 空头胜率
    # ----------------------------------------------------------
    print_section("3. 多头 vs 空头胜率对比")

    long_win = (df["long_ret"] > 0).sum()
    short_win = (df["short_ret"] < 0).sum()  # 空头组合跌了 = 做空赚了
    strategy_win = (df["long_minus_short"] > 0).sum()

    print(f"  多头盈利月份: {long_win}/{total_months} ({long_win/total_months:.0%})")
    print(f"  空头盈利月份: {short_win}/{total_months} ({short_win/total_months:.0%})  ← 空头组合下跌=做空赚钱")
    print(f"  策略盈利月份: {strategy_win}/{total_months} ({strategy_win/total_months:.0%})")

    # ----------------------------------------------------------
    # 4. 被做空后涨幅最大的个股（杀伤力排名）
    # ----------------------------------------------------------
    print_section("4. 被做空后涨幅最大的个股 (杀伤力排名)")

    # 收集所有空头个股记录
    short_records = []
    for dt, row in df.iterrows():
        for ticker, ret in row["short_individual"].items():
            short_records.append({
                "date": dt,
                "ticker": ticker,
                "return": ret,  # 正=涨了=做空亏了
                "is_mega_cap": ticker in MEGA_CAPS,
            })

    short_df = pd.DataFrame(short_records)

    if not short_df.empty:
        # 单次做空亏损最大的记录
        print("\n  [单月做空亏损最大 Top 15] (股票涨幅 = 做空亏损)")
        print(f"  {'月份':<12} {'股票':>8} {'涨幅':>10} {'大蓝筹':>8}")
        print(f"  {'-'*40}")
        worst_singles = short_df.nlargest(15, "return")
        for _, row in worst_singles.iterrows():
            mega = "  是" if row["is_mega_cap"] else ""
            print(f"  {row['date'].strftime('%Y-%m'):<12} {row['ticker']:>8} {row['return']:>+10.1%}{mega}")

        # 累计被做空次数和平均亏损
        print("\n  [被做空次数最多的股票 & 平均涨幅]")
        ticker_stats = short_df.groupby("ticker").agg(
            times_shorted=("return", "count"),
            avg_return=("return", "mean"),
            total_return=("return", "sum"),
            max_return=("return", "max"),
        ).sort_values("times_shorted", ascending=False)

        print(f"  {'股票':>8} {'做空次数':>8} {'平均涨幅':>10} {'累计涨幅':>10} {'最大单月涨幅':>12} {'蓝筹':>6}")
        print(f"  {'-'*58}")
        for ticker, row in ticker_stats.head(20).iterrows():
            mega = "  是" if ticker in MEGA_CAPS else ""
            print(f"  {ticker:>8} {row['times_shorted']:>8.0f} {row['avg_return']:>+10.2%} {row['total_return']:>+10.2%} {row['max_return']:>+12.2%}{mega}")

    # ----------------------------------------------------------
    # 5. 大蓝筹被做空分析
    # ----------------------------------------------------------
    print_section("5. 大蓝筹被做空分析")

    if not short_df.empty:
        mega_shorts = short_df[short_df["is_mega_cap"]]
        if len(mega_shorts) > 0:
            print(f"  大蓝筹被做空总次数: {len(mega_shorts)} (占全部做空的 {len(mega_shorts)/len(short_df):.0%})")
            print(f"  大蓝筹做空平均涨幅: {mega_shorts['return'].mean():+.2%} (涨=亏)")
            print(f"  非蓝筹做空平均涨幅: {short_df[~short_df['is_mega_cap']]['return'].mean():+.2%}")
            print(f"")

            mega_by_ticker = mega_shorts.groupby("ticker").agg(
                count=("return", "count"),
                avg_ret=("return", "mean"),
                max_ret=("return", "max"),
            ).sort_values("count", ascending=False)

            print(f"  {'股票':>8} {'做空次数':>8} {'平均涨幅':>10} {'最大单月涨':>12}")
            print(f"  {'-'*42}")
            for ticker, row in mega_by_ticker.iterrows():
                print(f"  {ticker:>8} {row['count']:>8.0f} {row['avg_ret']:>+10.2%} {row['max_ret']:>+12.2%}")
        else:
            print("  本策略未做空任何大蓝筹。")

    # ----------------------------------------------------------
    # 6. 纯多头 vs 多空 vs 基准 净值曲线
    # ----------------------------------------------------------
    print_section("6. 净值曲线对比 (起始 $10,000)")

    equity_long = (1 + df["long_only_net"]).cumprod() * START_VALUE
    equity_strategy = (1 + df["long_minus_short"]).cumprod() * START_VALUE
    equity_bench = (1 + df["benchmark_ret"]).cumprod() * START_VALUE

    # 年末快照
    print(f"  {'年份':<8} {'纯多头':>12} {'多空策略':>12} {'XLK基准':>12} {'空头拖累':>12}")
    print(f"  {'-'*58}")
    for year in sorted(df.index.year.unique()):
        year_data = df[df.index.year == year]
        if len(year_data) > 0:
            last_date = year_data.index[-1]
            e_long = equity_long.loc[last_date]
            e_strat = equity_strategy.loc[last_date]
            e_bench = equity_bench.loc[last_date]
            drag = e_long - e_strat  # 空头拖累金额
            print(f"  {year:<8} ${e_long:>10,.0f} ${e_strat:>10,.0f} ${e_bench:>10,.0f} ${drag:>10,.0f}")

    final_long = equity_long.iloc[-1]
    final_strat = equity_strategy.iloc[-1]
    final_bench = equity_bench.iloc[-1]
    print(f"  {'最终':<8} ${final_long:>10,.0f} ${final_strat:>10,.0f} ${final_bench:>10,.0f} ${final_long - final_strat:>10,.0f}")

    # ----------------------------------------------------------
    # 7. 核心结论
    # ----------------------------------------------------------
    print_section("7. 核心结论")

    short_drag_pct = (final_long - final_strat) / final_strat * 100
    vs_benchmark = (final_strat - final_bench) / final_bench * 100

    print(f"  空头操作拖累策略收益: ${final_long - final_strat:,.0f} ({short_drag_pct:+.1f}%)")
    print(f"  多空策略 vs XLK 基准:  {vs_benchmark:+.1f}%")
    if final_long > final_bench:
        print(f"  纯多头策略 vs XLK 基准: +{(final_long - final_bench)/final_bench*100:.1f}%  ← 多头选股有效!")
    else:
        print(f"  纯多头策略 vs XLK 基准: {(final_long - final_bench)/final_bench*100:+.1f}%")

    # 空头月均拖累
    avg_drag = df["short_ret"].mean()
    if avg_drag > 0:
        print(f"\n  空头组合月均上涨 {avg_drag:+.2%}，在科技股牛市中做空持续亏损。")
        print(f"  建议: 考虑纯多头策略，或加入做空止损机制。")

    return df


def main():
    print("=" * 70)
    print("  QuantPlatform 策略归因分析")
    print("  诊断多头/空头各自的收益贡献")
    print("=" * 70)

    adj_close, volume = load_data()
    print(f"  数据: {len(adj_close)} 个交易日, {len(adj_close.columns)} 只股票")
    print(f"  区间: {adj_close.index[0].strftime('%Y-%m-%d')} ~ {adj_close.index[-1].strftime('%Y-%m-%d')}")

    all_results = {}
    for method in METHODS:
        result = analyze_method(adj_close, volume, method, LOOKBACK)
        all_results[method] = result

    # ----------------------------------------------------------
    # 跨策略汇总
    # ----------------------------------------------------------
    print_section("跨策略汇总对比")

    print(f"  {'策略':<25} {'多空收益':>10} {'纯多头收益':>12} {'空头拖累':>10} {'XLK基准':>10}")
    print(f"  {'-'*70}")
    for method, df in all_results.items():
        if df.empty:
            continue
        cum_ls = (1 + df["long_minus_short"]).prod() - 1
        cum_lo = (1 + df["long_only_net"]).prod() - 1
        cum_bench = (1 + df["benchmark_ret"]).prod() - 1
        drag = cum_lo - cum_ls
        print(f"  {method:<25} {cum_ls:>+10.1%} {cum_lo:>+12.1%} {drag:>+10.1%} {cum_bench:>+10.1%}")

    print_section("关于止损")
    print("  当前策略没有任何止损机制:")
    print("  - 持仓周期固定 1 个月，月末强制换仓")
    print("  - Winsorization 只是裁剪极端收益率（前后1%），不是止损")
    print("  - 做空无保护：如果某只被做空的股票月内涨 50%，亏损直接计入")
    print("")
    print("  建议的止损方案:")
    print("  - 方案 A: 空头月度止损 (如单只空头股票涨 >15% 则平仓)")
    print("  - 方案 B: 去掉空头，改为纯多头策略")
    print("  - 方案 C: 空头改为做空 XLK ETF (对冲 beta，不做个股空头)")


if __name__ == "__main__":
    main()
