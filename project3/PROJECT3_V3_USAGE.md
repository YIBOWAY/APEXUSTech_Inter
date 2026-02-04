# Project3_v3 使用文档

本文件说明如何使用 `project3_v3.ipynb`（v3 迭代版），并说明新增 Phase1/Phase2 优化项。

---

## 1. v3 相比 v2 的新增能力

**Phase1：执行与风控增强**
- 信号在 t 生成，订单在 t+1 成交（next‑bar fill，v3 专用引擎实现）
- 风控规则：
  - 现金不足不下单
  - 最大持仓/杠杆限制
  - 最大回撤触发停止交易

**Phase2：实验与诊断增强**
- 参数扫描（lookback / z_threshold / quantity / commission / slippage）
- 输出扫描结果表与排序（如 Sharpe）

---

## 2. 技术栈

**语言与环境**
- Python 3.x

**核心依赖**
- `pandas`
- `numpy`
- `matplotlib`

**项目模块**
- `project3/optimized_backtest.py`（v2）
- `project3/optimized_integration.py`（v2）
- `project3/optimized_backtest_v3.py`（v3 专用引擎）
- `project3/optimized_integration_v3.py`（v3 集成层）

---

## 3. 数据格式要求

CSV 必须满足：
- Index 为日期
- 列包含 `NEAR` 与 `FAR`

---

## 4. 快速开始

1. 打开 `project3_v3.ipynb`
2. 运行 Setup cell（固定种子）
3. 运行配置/数据校验
4. 执行 v3 回测
5. 查看指标与图表
6. 执行参数扫描

---

## 5. v3 新增配置项

| 参数 | 说明 | 默认 |
|------|------|------|
| `execution_delay` | 成交延迟（bar 数） | 1 |
| `max_leverage` | 最大杠杆 | 2.0 |
| `max_position_per_leg` | 单腿最大持仓 | 2 × quantity |
| `max_drawdown_pct` | 最大回撤阈值 | 10% |

---

## 6. 参数扫描默认范围

| 参数 | 默认扫描范围 |
|------|---------------|
| `lookback_window` | 20, 30, 40, 60, 90, 120 |
| `z_threshold` | 1.0, 1.5, 2.0, 2.5, 3.0 |
| `quantity` | 1, 5, 10, 20 |
| `commission` | 2.5, 5.0, 7.5 |
| `slippage` | 0.005, 0.01, 0.02 |

---

## 7. 风控规则解释

**A. 现金不足不下单**
- 订单成本 + 佣金 > 可用现金 → 不下单

**B. 最大杠杆/持仓限制**
- 约束总敞口，避免过度持仓

**D. 最大回撤停止交易**
- 回撤超过阈值后停止新交易（可选是否强制平仓）

---

## 8. 常见问题

**Q1: 运行时出现空结果？**
- 检查 CSV 是否存在 NEAR/FAR
- 检查 lookback_window 是否过大

**Q2: 指标明显变化？**
- v3 使用 next‑bar fill，指标与 v2 不完全一致属于合理现象

---

## 9. 参考资料

- 执行延迟与回测滑点建模（IBKR）
  https://www.interactivebrokers.com/campus/ibkr-quant-news/slippage-in-model-backtesting/
- Backtrader 滑点模型
  https://www.backtrader.com/docu/slippage/slippage/
- Backtrader 佣金设置案例
  https://www.slingacademy.com/article/handling-commission-and-slippage-in-backtrader/
- Execution delay 与延迟建模
  https://support.eareview.net/support/solutions/articles/19000052275-using-execution-delay-slippage
