# QuantPlatform V2 - 项目全面指南

> V2 基于 V1 MVP 进行了代码质量修复、策略逻辑优化和单元测试覆盖。

---

## 目录

1. [项目概览](#1-项目概览)
2. [技术栈说明](#2-技术栈说明)
3. [目录结构与文件说明](#3-目录结构与文件说明)
4. [数据库详解](#4-数据库详解)
5. [后端架构](#5-后端架构)
6. [前端架构](#6-前端架构)
7. [前后端如何连接](#7-前后端如何连接)
8. [一次回测的完整数据流](#8-一次回测的完整数据流)
9. [数据缓存机制](#9-数据缓存机制)
10. [V2 代码改进详解](#10-v2-代码改进详解)
11. [测试体系](#11-测试体系)
12. [如何分开测试前后端](#12-如何分开测试前后端)
13. [数据库可视化与管理](#13-数据库可视化与管理)
14. [常见问题与排查](#14-常见问题与排查)
15. [后续开发方向](#15-后续开发方向)

---

## 1. 项目概览

QuantPlatform 是一个**量化交易回测平台**，核心功能是：

- 管理多个量化策略（目前是 XLK 动量策略的 3 种变体）
- 从 Yahoo Finance 获取历史数据并缓存到数据库
- 运行回测，生成净值曲线、交易信号、绩效指标
- 前端可视化展示回测结果

### 架构总览

```
用户浏览器 (React)
    │
    │  HTTP 请求 (/api/*)
    ▼
Vite Dev Server (:5173)
    │
    │  代理转发 /api → :8000
    ▼
FastAPI 后端 (:8000)
    │
    ├── 路由层 (Routers)     → 处理 HTTP 请求/响应
    ├── 数据校验 (Schemas)   → Pydantic 模型验证（V2 使用 date 类型）
    ├── 业务逻辑 (Services)  → 策略计算、数据获取
    └── 数据层 (Models)      → SQLAlchemy ORM
            │
            ▼
    PostgreSQL (:5432)       → 数据持久化
            │
    Docker 容器              → 运行数据库
```

---

## 2. 技术栈说明

### 后端

| 技术 | 版本 | 作用 | 为什么用它 |
|------|------|------|-----------|
| **FastAPI** | 0.115.6 | Python Web 框架 | 异步支持好、自带 Swagger 文档、类型提示驱动 |
| **SQLAlchemy** | 2.0.36 | ORM（对象关系映射） | 用 Python 类操作数据库表，支持 async |
| **asyncpg** | 0.30.0 | PostgreSQL 异步驱动 | 配合 FastAPI 的异步特性，非阻塞数据库操作 |
| **Pydantic** | V2 | 数据校验 | 自动校验 API 请求/响应格式和类型（V2 新增 `date` 类型校验） |
| **Pandas** | 2.2.3 | 数据处理 | 金融数据计算（收益率、动量分数等） |
| **NumPy** | 2.2.1 | 数值计算 | 数组运算、统计计算 |
| **Requests** | 2.32.3 | HTTP 客户端 | 从 Yahoo Finance API 获取数据 |
| **Alembic** | 1.14.1 | 数据库迁移 | Schema 版本管理（已配置，开发模式用 create_all） |
| **pytest** | 9.0.2 | 测试框架 | V2 新增，核心策略逻辑单元测试 |

**FastAPI vs Flask/Django：**
- FastAPI 原生支持 async/await，适合 I/O 密集型（网络请求、数据库查询）
- 自动生成 Swagger 文档：访问 `http://localhost:8000/docs`
- 基于 Python 类型提示，代码即文档

**SQLAlchemy 做了什么：**
- 你定义一个 Python 类（如 `Strategy`），它自动对应一张数据库表
- `db.execute(select(Strategy))` 等价于 `SELECT * FROM strategies`
- 支持 async 会话（`AsyncSession`），在后台线程中使用独立引擎避免跨线程问题

### 前端

| 技术 | 版本 | 作用 |
|------|------|------|
| **React** | 19 | UI 框架，组件化开发 |
| **TypeScript** | ~5.9 | JavaScript 加类型，减少 bug |
| **Vite** | 7 | 构建工具，热更新快 |
| **Tailwind CSS** | 4 | CSS 工具类，快速写样式 |
| **Shadcn/ui + Radix** | - | UI 组件库（50+ 组件：按钮、卡片、表格等） |
| **Apache ECharts** | 6 | 图表库（净值曲线、回撤图、月度热力图） |
| **Axios** | - | HTTP 请求库，调后端 API（120 秒超时） |
| **Wouter** | 3 | 轻量路由（Hash 模式页面跳转） |
| **date-fns** | 4 | 日期格式化 |
| **Framer Motion** | - | 动画库（已安装，轻度使用） |

### 基础设施

| 技术 | 版本 | 作用 |
|------|------|------|
| **Docker** | - | 容器化运行 PostgreSQL |
| **PostgreSQL** | 16-alpine | 关系型数据库，存储所有数据 |

---

## 3. 目录结构与文件说明

### 项目根目录

```
quantplatform/
├── backend/                    # Python FastAPI 后端
├── frontend/                   # React TypeScript 前端
├── test_strategy/              # V2 新增：核心策略单元测试
│   ├── __init__.py
│   ├── conftest.py             #   共享测试 fixtures（合成数据生成）
│   ├── test_metrics_calculator.py  # metrics 计算测试（19 个用例）
│   ├── test_momentum_scores.py     # 动量评分测试（13 个用例）
│   └── test_backtest_strategy.py   # 回测引擎测试（15 个用例）
├── docker-compose.yml          # PostgreSQL 容器定义
├── PROJECT_GUIDE_v1_mvp.md     # V1 项目指南
├── PROJECT_GUIDE_v2.md         # V2 项目指南（本文件）
└── README.md                   # 快速启动指南
```

### 后端 (`backend/`)

```
backend/
├── app/
│   ├── main.py              # 入口：创建 FastAPI 应用，注册路由和中间件
│   ├── config.py            # 配置：数据库 URL、66 个 ticker、Yahoo API 请求头
│   ├── database.py          # 数据库连接：创建引擎和会话工厂 + 后台线程引擎
│   │
│   ├── models/              # ORM 模型（对应数据库表）
│   │   ├── __init__.py      #   Re-exports 所有模型
│   │   ├── strategy.py      #   strategies + backtest_runs 表
│   │   ├── price_data.py    #   daily_prices 价格缓存表
│   │   └── trade_signal.py  #   trade_signals 交易信号表
│   │
│   ├── schemas/             # Pydantic 模型（API 请求/响应格式定义）
│   │   ├── strategy.py      #   策略的输入/输出格式
│   │   ├── backtest.py      #   V2: start_date/end_date 使用 date 类型
│   │   └── market_data.py   #   行情数据的输出格式
│   │
│   ├── routers/             # API 路由（HTTP 端点）
│   │   ├── strategies.py    #   /api/strategies - 策略增删查
│   │   ├── backtest.py      #   V2: 批量插入信号, timezone-aware datetime
│   │   └── market_data.py   #   /api/market-data - 行情数据查询
│   │
│   └── services/            # 业务逻辑（核心计算）
│       ├── data_fetcher.py      # V2: 非阻塞 asyncio.sleep + raise_for_status
│       ├── momentum_strategy.py # V2: 无风险利率传递 + 移除弃用 API
│       ├── metrics_calculator.py # V2: 几何年化收益 + Sharpe 含无风险利率
│       └── regime_classifier.py  # 市场状态分类（VIX 等，预留）
│
├── seed.py                  # 种子数据脚本：创建 3 个默认策略
├── requirements.txt         # Python 依赖列表
├── alembic.ini              # Alembic 迁移配置
├── alembic/                 # 迁移文件目录
└── .env                     # 环境变量（数据库密码等）
```

### 前端 (`frontend/`)

```
frontend/src/
├── main.tsx                 # React 入口，挂载到 DOM
├── App.tsx                  # 路由配置（Hash 模式）
├── index.css                # 全局样式（Tailwind + CSS 变量主题）
│
├── api/                     # 与后端通信
│   ├── client.ts            #   Axios 实例（baseURL=/api, timeout=120s）
│   ├── strategies.ts        #   策略 API 调用函数
│   └── backtest.ts          #   回测 API + 轮询机制
│
├── types/index.ts           # TypeScript 类型定义
│
├── pages/                   # 页面组件
│   ├── Dashboard.tsx        #   策略列表页（首页）
│   ├── StrategyDetail.tsx   #   V2: 动态日期、移除假趋势、修复 useEffect
│   └── Home.tsx             #   落地页
│
├── components/
│   ├── StrategyCard.tsx     # 策略卡片组件
│   ├── ErrorBoundary.tsx    # 错误边界
│   ├── layout/              # 布局组件（DashboardLayout, Sidebar, Header）
│   ├── charts/              # 图表组件（ECharts）
│   │   ├── EquityChart.tsx      # 净值曲线（策略 vs 基准）
│   │   ├── DrawdownChart.tsx    # 回撤图
│   │   └── HeatmapChart.tsx     # 月度收益热力图
│   └── ui/                  # 50+ 基础 UI 组件（Shadcn/Radix）
│
├── contexts/ThemeContext.tsx # 主题（深色/浅色，默认深色）
├── hooks/use-mobile.ts      # 移动端检测
└── lib/
    ├── utils.ts             # cn() 工具函数
    └── mockData.ts          # 模拟数据（已弃用）
```

---

## 4. 数据库详解

### 4 张表概览

```
strategies (策略配置)
    │
    │ 1 : N
    ▼
backtest_runs (回测记录)
    │
    │ 1 : N
    ▼
trade_signals (交易信号)

daily_prices (独立的价格缓存表)
```

### 表 1: strategies（策略配置）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID | 主键 |
| name | VARCHAR(200) | 策略名称 |
| description | TEXT | 策略描述 |
| tags | ARRAY(String) | 标签，如 ['momentum', 'XLK'] |
| method | VARCHAR(30) | 动量方法：simple / risk_adjusted / volume_weighted |
| lookback_months | INTEGER | 回看周期：3 / 6 / 12 |
| decile | FLOAT | 选股比例：0.2 = 前 20% |
| tc_bps | INTEGER | 交易成本：5 = 5 个基点 |
| winsor_q | FLOAT | 极端值处理：0.01 = 裁剪前后 1% |
| status | VARCHAR(20) | 状态：active / paused |
| created_at / updated_at | TIMESTAMPTZ | 时间戳 |

### 表 2: backtest_runs（回测记录）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID | 回测运行 ID |
| strategy_id | UUID | 关联策略 ID（外键，CASCADE DELETE） |
| status | VARCHAR(20) | running / completed / failed |
| param_method | VARCHAR | 本次使用的方法（快照） |
| param_lookback | INTEGER | 本次回看周期 |
| param_start_date / param_end_date | DATE | 回测日期范围 |
| total_return | FLOAT | 总收益率（V2: 几何累积） |
| annual_return | FLOAT | **V2: 几何年化收益率** |
| sharpe_ratio | FLOAT | **V2: 含无风险利率的 Sharpe** |
| annual_volatility / max_drawdown / win_rate / profit_factor | FLOAT | 其他绩效指标 |
| total_months | INTEGER | 回测月数 |
| monthly_returns | JSONB | 每月收益率列表 |
| equity_curve | JSONB | 净值曲线数据 |
| error_message | TEXT | 失败时的错误信息 |
| started_at / finished_at | TIMESTAMPTZ | 时间戳 |

### 表 3: daily_prices（价格缓存）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | BIGINT | 自增主键 |
| ticker | VARCHAR(20) | 股票代码 |
| trade_date | DATE | 交易日期 |
| adj_close | FLOAT | 调整后收盘价 |
| volume | BIGINT | 成交量 |
| source | VARCHAR(20) | 数据来源（yahoo） |
| fetched_at | TIMESTAMPTZ | 获取时间 |

**唯一约束**: `(ticker, trade_date)` — 同一只股票同一天只存一条。

### 表 4: trade_signals（交易信号）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | BIGINT | 自增主键 |
| backtest_run_id | UUID | 关联回测 ID（CASCADE DELETE） |
| signal_date | DATE | 信号日期（月末） |
| ticker | VARCHAR(20) | 股票代码 |
| direction | VARCHAR(10) | BUY（做多，当前为 long-only 策略） |
| weight | FLOAT | 组合权重 |
| score | FLOAT | 动量分数 |
| price | FLOAT | 信号日价格 |
| pnl | FLOAT | 盈亏（预留） |

---

## 5. 后端架构

### API 端点一览

#### 策略管理 (`/api/strategies`)

| 方法 | 路径 | 作用 |
|------|------|------|
| GET | `/api/strategies` | 获取策略列表（支持 search, sort_by, order 参数） |
| GET | `/api/strategies/{id}` | 获取单个策略详情（含最新回测指标） |
| POST | `/api/strategies` | 创建新策略 |

#### 回测执行 (`/api/backtest`)

| 方法 | 路径 | 作用 |
|------|------|------|
| POST | `/api/backtest/{strategy_id}/run` | **启动回测**（后台执行，立即返回） |
| GET | `/api/backtest/runs/{run_id}/status` | **轮询回测状态** |
| GET | `/api/backtest/{strategy_id}/latest` | 获取最新完成的回测结果 |
| GET | `/api/backtest/{strategy_id}/equity` | 获取净值曲线数据 |
| GET | `/api/backtest/{strategy_id}/monthly-returns` | 获取月度收益率列表 |
| GET | `/api/backtest/{strategy_id}/trades` | 获取交易信号（分页：page, size） |

#### 行情数据 (`/api/market-data`)

| 方法 | 路径 | 作用 |
|------|------|------|
| GET | `/api/market-data/{ticker}` | 获取某只股票的缓存价格 |
| POST | `/api/market-data/fetch` | 手动触发数据获取 |

#### 系统

| 方法 | 路径 | 作用 |
|------|------|------|
| GET | `/api/health` | 健康检查 |

### 核心业务逻辑

#### data_fetcher.py — 数据获取流程

```
fetch_and_cache_prices(tickers, start_date, end_date)
    │
    ├── 1. 查数据库缓存 → _load_from_db()
    │      如果某个 ticker 在 daily_prices 表有 >10 行数据，直接用缓存
    │
    ├── 2. 缺失的 ticker → _fetch_yahoo_single()
    │      发 HTTP 请求到 Yahoo Finance API
    │      V2: 添加 response.raise_for_status() 检查状态码
    │      V2: await asyncio.sleep(0.3) 替代阻塞的 time.sleep
    │
    ├── 3. 新数据写入数据库 → _save_to_db()
    │      使用 PostgreSQL UPSERT (ON CONFLICT DO NOTHING)
    │
    └── 4. 返回 (adj_close_df, volume_df) 两个 DataFrame
```

#### momentum_strategy.py — 策略计算

```
run_full_backtest()
    │
    ├── backtest_strategy()  ← 核心回测循环
    │   │
    │   ├── 对每个月末：
    │   │   ├── calculate_momentum_scores()  ← 计算动量分数
    │   │   │   ├── simple: 价格涨幅 = (P_t / P_t-k) - 1
    │   │   │   ├── risk_adjusted: 涨幅 / 年化波动率
    │   │   │   └── volume_weighted: 涨幅 × (近期成交量 / 基线成交量)
    │   │   │
    │   │   ├── 排名，选前 decile% 做多（long-only）
    │   │   ├── 记录交易信号（ticker, direction=BUY, weight, price）
    │   │   └── 计算下月收益 = 多头收益 - 交易成本（基于换手率）
    │   │
    │   └── 返回 (月度收益率序列, 交易信号列表)
    │
    ├── V2: 从 ^IRX 提取平均无风险利率
    │
    ├── calculate_metrics(returns, rf_rate)  ← 计算绩效指标
    │   V2: 几何年化收益 + Sharpe 含无风险利率
    │
    └── build_equity_curve() ← 构建净值曲线
        从 $10,000 开始，策略 vs XLK 基准
```

---

## 6. 前端架构

### 页面路由（Hash 模式）

| 路径 | 页面 | 功能 |
|------|------|------|
| `/#/` | Dashboard | 策略卡片列表，搜索/排序 |
| `/#/strategies` | Dashboard | 同上 |
| `/#/strategies/:id` | StrategyDetail | 策略详情、运行回测、查看图表 |
| `/#/compare` | （预留） | 策略对比 — Phase 2 |

### 组件层次

```
App.tsx (路由)
├── DashboardLayout (布局壳)
│   ├── Header (顶栏 + 搜索 + 主题切换)
│   ├── Sidebar (导航菜单)
│   └── 内容区域
│       │
│       ├── Dashboard.tsx
│       │   └── StrategyCard.tsx × N
│       │
│       └── StrategyDetail.tsx
│           ├── MetricCard × 4 (收益率、夏普、回撤、胜率)
│           ├── EquityChart (净值曲线 — ECharts)
│           ├── DrawdownChart (回撤图 — ECharts)
│           ├── HeatmapChart (月度热力图 — ECharts)
│           └── Trades Dialog (交易信号弹窗 — 按月分页)
```

### API 调用层

```
api/client.ts
  └── Axios 实例，baseURL="/api"，timeout=120s

api/strategies.ts
  ├── getStrategies()     → GET /api/strategies
  ├── getStrategy(id)     → GET /api/strategies/{id}
  └── createStrategy()    → POST /api/strategies

api/backtest.ts
  ├── runBacktest()            → POST /api/backtest/{id}/run
  ├── getBacktestStatus()      → GET /api/backtest/runs/{runId}/status
  ├── runBacktestWithPolling() → 启动 + 每 3 秒轮询直到完成
  ├── getLatestBacktest()      → GET /api/backtest/{id}/latest
  ├── getEquityCurve()         → GET /api/backtest/{id}/equity
  ├── getMonthlyReturns()      → GET /api/backtest/{id}/monthly-returns
  └── getTrades()              → GET /api/backtest/{id}/trades
```

---

## 7. 前后端如何连接

### 开发环境

```
浏览器 → localhost:5173 (Vite)
              │
              │ /api/* 请求被 Vite proxy 转发
              ▼
         localhost:8000 (FastAPI)
              │
              ▼
         localhost:5432 (PostgreSQL in Docker)
```

**关键配置** — `vite.config.ts`:
```typescript
server: {
  proxy: {
    "/api": {
      target: "http://localhost:8000",
      changeOrigin: true,
    },
  },
},
```

**CORS 配置** — `main.py`:
```python
allow_origins=["http://localhost:5173", "http://localhost:3000"]
```

---

## 8. 一次回测的完整数据流

```
[前端 StrategyDetail.tsx]
  │
  │ 1. 调用 runBacktestWithPolling(strategyId, {})
  │    V2: 不再硬编码日期，使用后端默认值 (2020-01-01 ~ 2025-12-31)
  ▼
[前端 api/backtest.ts]
  │
  │ 2. POST /api/backtest/{strategyId}/run
  │    V2: Pydantic 自动校验 date 类型，非法日期返回 422
  │    后端立即返回 {id, status: "running"}
  │
  │ 3. 每 3 秒发送: GET /api/backtest/runs/{runId}/status
  ▼
[后端 routers/backtest.py]
  │
  │ 4. 创建 backtest_runs 记录 (status="running")
  │    V2: 使用 datetime.now(timezone.utc) 记录时间
  │ 5. 启动后台线程（独立事件循环 + 独立数据库引擎）
  │ 6. 立即返回 HTTP 200
  ▼
[后台线程]
  │
  │ 7. data_fetcher: 检查 daily_prices 缓存
  │    V2: await asyncio.sleep(0.3) — 非阻塞延迟
  │    V2: response.raise_for_status() — 明确的错误处理
  │
  │ 8. momentum_strategy: 动量计算 + 月度再平衡
  │    V2: adj_close.pct_change() — 无弃用参数
  │
  │ 9. metrics_calculator: 计算绩效指标
  │    V2: 几何年化收益 = (1 + total_return)^(12/n) - 1
  │    V2: Sharpe = (annual_return - rf) / annual_vol
  │    V2: rf 来自 ^IRX 平均值 / 100
  │
  │ 10. 写入数据库:
  │     V2: db.add_all(trade_signals) — 批量插入 ~1400 行
  ▼
[前端收到 status="completed"]
  │
  │ 11. 并行加载: 策略 + 净值曲线 + 月度收益 + 交易信号
  │ 12. 渲染图表
  ▼
[用户看到回测结果]
```

---

## 9. 数据缓存机制

### 价格数据缓存

- **首次运行**: 从 Yahoo Finance 下载 69 个 ticker（66 XLK 成分股 + XLK + ^VIX + ^IRX）
- **后续运行**: 直接从 `daily_prices` 表读取，秒级完成
- **判断标准**: 如果某 ticker 缓存行数 > 10，认为已有足够数据

### 回测结果缓存

- 每次回测在 `backtest_runs` 表新增一行
- 参数快照（method, lookback）存在 `backtest_runs` 表中，即使修改策略参数也不影响历史结果
- 前端默认展示最新一次 `status=completed` 的回测

---

## 10. V2 代码改进详解

### 10.1 代码质量修复

| 问题 | 修复前 | 修复后 | 文件 |
|------|--------|--------|------|
| 阻塞事件循环 | `time.sleep(0.3)` | `await asyncio.sleep(0.3)` | data_fetcher.py |
| HTTP 错误静默 | 未检查状态码 | `response.raise_for_status()` | data_fetcher.py |
| 弃用 API | `datetime.utcnow()` | `datetime.now(timezone.utc)` | backtest.py (router) |
| 弱类型校验 | `start_date: str` | `start_date: date` (Pydantic) | schemas/backtest.py |
| 弃用 pandas 参数 | `pct_change(fill_method=None)` | `pct_change()` | momentum_strategy.py |

### 10.2 性能优化

| 问题 | 修复前 | 修复后 |
|------|--------|--------|
| 逐条写入信号 | `for sig: db.add(sig)` | `db.add_all(signal_list)` 批量插入 |

### 10.3 策略逻辑修复

#### 年化收益率：算术平均 → 几何平均

```python
# V1（算术平均 — 高估收益）:
annual_return = (1 + monthly_mean) ** 12 - 1

# V2（几何平均 — 行业标准）:
total_return = (1 + monthly_returns).cumprod().iloc[-1] - 1
annual_return = (1 + total_return) ** (12 / n_months) - 1
```

**为什么重要**: 假设某月 +50%、下月 -50%：
- 算术平均月收益 = 0%，年化 = 0%（看起来没亏）
- 实际: 1.0 × 1.5 × 0.5 = 0.75，亏了 25%
- 几何年化正确反映亏损

#### Sharpe Ratio：加入无风险利率

```python
# V1:
sharpe = annual_return / annual_vol

# V2:
# ^IRX = 13 周 T-Bill 利率，例如 4.5 表示 4.5% 年化
annual_rf = adj_close["^IRX"].mean() / 100.0  # → 0.045
sharpe = (annual_return - annual_rf) / annual_vol
```

### 10.4 前端修复

| 问题 | 修复 |
|------|------|
| 回测日期硬编码 `"2026-01-30"` | 使用 `{}` 空对象，后端填充默认值 |
| MetricCard 假趋势值 (12.5, 0.4 等) | 移除 `trend` prop，不显示虚假数据 |
| useEffect 缺少 `loadData` 依赖 | `useCallback` 包裹 + 正确依赖列表 |

---

## 11. 测试体系

### 测试结构

```
test_strategy/
├── conftest.py                     # 共享 fixtures
│   ├── date_index                  # ~2 年交易日 DatetimeIndex
│   ├── simple_adj_close            # 15 只合成股票 + XLK + ^VIX + ^IRX
│   ├── simple_volume               # 对应的成交量数据
│   └── known_monthly_returns       # 手工构造的 12 个月收益率
│
├── test_metrics_calculator.py      # 19 个用例
│   ├── TestCalculateMetrics (13)
│   │   ├── test_empty_returns
│   │   ├── test_total_return           — 验证累积复利计算
│   │   ├── test_annual_return_geometric — 验证几何年化公式
│   │   ├── test_annual_return_is_not_arithmetic — 确认不是旧公式
│   │   ├── test_sharpe_with_risk_free_rate
│   │   ├── test_sharpe_without_risk_free_defaults_to_zero
│   │   ├── test_sharpe_higher_rf_means_lower_sharpe
│   │   ├── test_max_drawdown          — 已知序列 [+10%, -20%, +5%]
│   │   ├── test_win_rate
│   │   ├── test_profit_factor
│   │   ├── test_profit_factor_no_losses
│   │   ├── test_total_months
│   │   └── test_zero_volatility       — 零波动不触发除零错误
│   │
│   └── TestBuildEquityCurve (6)
│       ├── test_empty_returns
│       ├── test_start_value
│       ├── test_final_value_matches_total_return
│       ├── test_date_format
│       ├── test_benchmark_normalization
│       └── test_no_benchmark
│
├── test_momentum_scores.py         # 13 个用例
│   ├── TestSimpleMethod (4)         — 排名、正分数、类型、数量
│   ├── TestRiskAdjustedMethod (2)   — 非空、与 simple 不同
│   ├── TestVolumeWeightedMethod (1) — 非空
│   └── TestEdgeCases (6)
│       ├── test_insufficient_data
│       ├── test_fewer_than_10_stocks_returns_empty
│       ├── test_invalid_method_raises
│       ├── test_excludes_benchmark_tickers
│       ├── test_timezone_aware_index
│       └── test_different_lookback_periods
│
└── test_backtest_strategy.py       # 15 个用例
    ├── TestBacktestStrategy (9)
    │   ├── test_returns_series_not_empty
    │   ├── test_returns_are_floats
    │   ├── test_trade_signals_structure
    │   ├── test_signals_direction_is_buy
    │   ├── test_weights_sum_to_one_per_month
    │   ├── test_transaction_cost_reduces_returns
    │   ├── test_different_methods_produce_different_signals
    │   ├── test_decile_affects_portfolio_size
    │   └── test_winsorization
    │
    └── TestRunFullBacktest (6)
        ├── test_output_structure
        ├── test_metrics_keys
        ├── test_equity_curve_format
        ├── test_monthly_returns_format
        ├── test_risk_free_rate_from_irx
        └── test_empty_data
```

### 运行测试

```bash
# 激活 conda 环境
conda activate py311-tradingagents-cn

# 在 quantplatform 目录下运行
cd quantplatform
python -m pytest test_strategy/ -v

# 运行单个测试文件
python -m pytest test_strategy/test_metrics_calculator.py -v

# 运行某个测试类
python -m pytest test_strategy/test_backtest_strategy.py::TestRunFullBacktest -v
```

### 测试设计原则

1. **纯函数测试**: 所有测试不依赖数据库或网络，使用内存中的合成数据
2. **确定性**: 固定 `np.random.seed` 确保结果可复现
3. **手算验证**: `known_monthly_returns` fixture 用已知数值，可以手动验算正确性
4. **边界覆盖**: 空输入、零波动、不足数据量、无效方法名等

---

## 12. 如何分开测试前后端

### 测试后端（无需启动前端）

#### 方式 1: Swagger 文档（推荐）

1. 启动后端: `uvicorn app.main:app --reload`
2. 打开浏览器: http://localhost:8000/docs
3. 点击 "Try it out" 直接测试任意端点

#### 方式 2: curl 命令

```bash
# 健康检查
curl http://localhost:8000/api/health

# 获取策略列表
curl http://localhost:8000/api/strategies

# 启动回测（使用默认日期范围）
curl -X POST http://localhost:8000/api/backtest/{strategy_id}/run \
  -H "Content-Type: application/json" \
  -d '{}'

# 查询回测状态
curl http://localhost:8000/api/backtest/runs/{run_id}/status
```

### 测试前端（需后端运行）

1. 启动后端 + 数据库
2. `cd frontend && npm run dev`
3. 访问 `http://localhost:5173`

---

## 13. 数据库可视化与管理

### 推荐工具

| 工具 | 特点 |
|------|------|
| pgAdmin | 官方 GUI，功能全面 |
| DBeaver | 通用客户端，支持 ER 图 |
| VS Code 插件 | "Database Client" 或 "PostgreSQL" |
| psql | 命令行，通过 Docker 连接 |

### 连接信息

```
Host: localhost
Port: 5432
Database: quantplatform
Username: quant
Password: quantpass
```

### 常用 SQL

```sql
-- 查看缓存数据统计
SELECT ticker, COUNT(*) as days,
       MIN(trade_date) as first_date,
       MAX(trade_date) as last_date
FROM daily_prices GROUP BY ticker ORDER BY ticker;

-- 最近 5 次回测
SELECT id, status, sharpe_ratio, annual_return, total_return,
       max_drawdown, started_at
FROM backtest_runs ORDER BY started_at DESC LIMIT 5;

-- 某次回测的交易信号
SELECT signal_date, ticker, direction, weight, score, price
FROM trade_signals
WHERE backtest_run_id = '你的回测ID'
ORDER BY signal_date DESC, ticker LIMIT 30;
```

---

## 14. 常见问题与排查

### Q: 后端启动报错 "module 'app' has no attribute..."

确保在 `backend/` 目录下运行 `uvicorn app.main:app --reload`。

### Q: Yahoo Finance 数据获取失败

1. 检查网络连接
2. V2: 如果收到 429 (限流)，会通过 `raise_for_status()` 明确报错
3. 查看后端日志中 `[BG]` 开头的信息

### Q: 回测一直显示 "Running..."

1. 查看后端终端的 `[BG]` 日志
2. 查询 `backtest_runs` 表的 `error_message`
3. 确认 Docker 容器正在运行: `docker ps`

### Q: 前端请求返回 422 Validation Error

V2 中 `start_date` / `end_date` 是 `date` 类型。确保前端发送 ISO 格式日期字符串（`"2020-01-01"`）或不发送（使用后端默认值）。

### Q: 数据库连接失败

```bash
docker ps                    # 确认容器运行
docker-compose up -d         # 启动容器
docker-compose logs postgres # 查看日志
```

---

## 15. 后续开发方向

### 已完成（V2）

- [x] 代码质量修复（阻塞调用、弃用 API、弱类型校验）
- [x] 策略逻辑优化（几何年化、Sharpe 含无风险利率）
- [x] 性能优化（批量写入交易信号）
- [x] 前端修复（动态日期、移除假数据、useEffect 依赖）
- [x] 核心策略单元测试（47 个用例全部通过）

### 短期优化

- [ ] **API 集成测试** — TestClient + SQLite 内存数据库
- [ ] **前端策略创建页面** — 目前只有 seed.py 可以创建策略
- [ ] **回测参数可配置** — 前端添加日期选择器和参数表单
- [ ] **策略对比页面** — 多策略净值曲线叠加对比
- [ ] **数据更新机制** — 检测缓存数据过期，自动补充最新数据

### 中期功能

- [ ] **更多策略类型** — 均值回归、配对交易、因子模型
- [ ] **市场状态分类** — 集成 `regime_classifier.py`
- [ ] **风险管理模块** — 最大仓位限制、止损逻辑
- [ ] **数据源扩展** — Alpha Vantage、Polygon 等
- [ ] **用户认证** — 登录注册，每个用户独立策略

### 长期目标

- [ ] **实盘信号推送** — 邮件 / Telegram
- [ ] **策略组合优化** — Kelly 准则
- [ ] **实时数据** — WebSocket
- [ ] **部署上线** — Docker Compose 全栈 / 云端
