# Project3_v2 使用文档

本文件说明如何使用 `project3_v2.ipynb`（优化版回测 Notebook），并列出技术栈与运行步骤。

---

## 1. 适用范围

- 事件驱动回测框架（优化版）
- 商品期货跨期价差（Calendar Spread）Z-score 均值回归策略
- 关注：速度、可复现、结构化模块化、指标一致性

---

## 2. 技术栈

**语言与环境**
- Python 3.x

**核心依赖**
- `pandas`：数据加载/处理
- `numpy`：数值计算
- `matplotlib`：绘图

**项目模块**
- `project3/optimized_backtest.py`：优化版事件驱动回测框架
- `project3/optimized_integration.py`：Notebook 集成工具（运行/绘图/摘要）

---

## 3. 数据格式要求

CSV 必须满足：
- Index 为日期（可解析为 datetime）
- 至少包含两列：`NEAR` 与 `FAR`

示例：

| Date | NEAR | FAR |
|------|------|-----|
| 2022-01-03 | 3200.0 | 3230.0 |

---

## 4. 快速开始

1. 打开 `project3_v2.ipynb`
2. 运行 **Setup & Reproducibility** cell（固定随机种子）
3. 在 **Configuration** cell 设置参数
4. 运行数据验证与回测
5. 观察输出指标与图表

---

## 5. 关键参数说明

| 参数 | 说明 | 建议范围 |
|------|------|----------|
| `lookback_window` | Z-score 回看窗口 | 20–120 |
| `z_threshold` | 入场阈值 | 1.0–3.0 |
| `quantity` | 每腿合约数量 | 1–20 |
| `commission` | 单次交易佣金 | 2.5–7.5 |
| `slippage` | 单次交易滑点 | 0.005–0.02 |

---

## 6. 指标输出

Notebook 输出包括：
- Total Return
- Sharpe Ratio
- Maximum Drawdown
- Equity Curve
- Spread + Rolling Mean
- Z-Score 曲线

---

## 7. Baseline 指标一致性

`project3_v2.ipynb` 提供 baseline 对比单元：
```python
baseline_total_return = None
baseline_sharpe_ratio = None
baseline_max_drawdown = None
```
填写原始 notebook 的指标后，自动判断是否在容差范围内。

---

## 8. 常见问题

**Q1: 没找到 CSV 文件？**
- 请确认 `real_akshare_spread_data.csv` 或 `demo_spread_data.csv` 等文件在同目录下。

**Q2: 提示缺少 NEAR/FAR 列？**
- 确认数据列名正确，必要时重命名。

**Q3: 结果不可复现？**
- 确认已设置随机种子，并保证使用相同数据文件。

---

## 9. 参考资料

- Zipline：事件驱动处理与可复现配置
  https://zipline-trader.readthedocs.io/en/latest/beginner-tutorial.html
- QuantConnect：数据验证流程建议
  https://www.quantconnect.com/docs/v2/lean-engine/contributions/datasets/testing-data-models
- Backtrader：滑点与佣金建模
  https://www.backtrader.com/docu/slippage/slippage/  
  https://www.backtrader.com/docu/commission-schemes/commission-schemes/
