# QuantPlatform MVP - 项目全面指南

> 帮助你从零掌握整个项目的架构、技术栈、数据流和后续开发方向。

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
10. [如何分开测试前后端](#10-如何分开测试前后端)
11. [数据库可视化与管理](#11-数据库可视化与管理)
12. [常见问题与排查](#12-常见问题与排查)
13. [后续开发方向](#13-后续开发方向)

---

## 1. 项目概览

QuantPlatform 是一个**量化交易回测平台 MVP**，核心功能是：

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
    ├── 数据校验 (Schemas)   → Pydantic 模型验证
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

| 技术 | 作用 | 为什么用它 |
|------|------|-----------|
| **FastAPI** | Python Web 框架 | 异步支持好、自带 API 文档（Swagger）、类型提示 |
| **SQLAlchemy** | ORM（对象关系映射） | 用 Python 类操作数据库表，不用手写 SQL |
| **asyncpg** | PostgreSQL 异步驱动 | 配合 FastAPI 的异步特性，非阻塞数据库操作 |
| **Pydantic** | 数据校验 | 自动校验 API 请求/响应的数据格式和类型 |
| **Pandas / NumPy** | 数据处理 | 金融数据计算（收益率、动量分数等） |
| **Requests** | HTTP 客户端 | 从 Yahoo Finance API 获取数据 |

**FastAPI vs Flask/Django:**
- FastAPI 原生支持 async/await，适合 I/O 密集型（网络请求、数据库查询）
- 自动生成 Swagger 文档：访问 `http://localhost:8000/docs`
- 基于 Python 类型提示，代码即文档

**SQLAlchemy 做了什么：**
- 你定义一个 Python 类（如 `Strategy`），它自动对应一张数据库表
- `db.execute(select(Strategy))` 等价于 `SELECT * FROM strategies`
- 你不需要手写 SQL，但也可以在需要时直接写

### 前端

| 技术 | 作用 |
|------|------|
| **React 19** | UI 框架，组件化开发 |
| **TypeScript** | JavaScript 加类型，减少 bug |
| **Vite** | 构建工具，热更新快 |
| **Tailwind CSS** | CSS 工具类，快速写样式 |
| **Shadcn/ui + Radix** | UI 组件库（按钮、卡片、表格等） |
| **ECharts** | 图表库（净值曲线、热力图等） |
| **Axios** | HTTP 请求库，调后端 API |
| **Wouter** | 轻量路由（页面跳转） |

### 基础设施

| 技术 | 作用 |
|------|------|
| **Docker** | 容器化运行 PostgreSQL |
| **PostgreSQL** | 关系型数据库，存储所有数据 |

---

## 3. 目录结构与文件说明

### 后端 (`backend/`)

```
backend/
├── app/
│   ├── main.py              # 入口：创建 FastAPI 应用，注册路由和中间件
│   ├── config.py            # 配置：数据库 URL、66 个 ticker、Yahoo API 请求头
│   ├── database.py          # 数据库连接：创建引擎和会话工厂
│   │
│   ├── models/              # ORM 模型（对应数据库表）
│   │   ├── strategy.py      #   策略表 + 回测记录表
│   │   ├── price_data.py    #   日线价格缓存表
│   │   └── trade_signal.py  #   交易信号表
│   │
│   ├── schemas/             # Pydantic 模型（API 请求/响应格式定义）
│   │   ├── strategy.py      #   策略的输入/输出格式
│   │   ├── backtest.py      #   回测的输入/输出格式
│   │   └── market_data.py   #   行情数据的输出格式
│   │
│   ├── routers/             # API 路由（HTTP 端点）
│   │   ├── strategies.py    #   /api/strategies - 策略增删查
│   │   ├── backtest.py      #   /api/backtest - 回测执行和结果查询
│   │   └── market_data.py   #   /api/market-data - 行情数据查询
│   │
│   └── services/            # 业务逻辑（核心计算）
│       ├── data_fetcher.py      # Yahoo Finance 数据获取 + 数据库缓存
│       ├── momentum_strategy.py # 动量策略：打分、选股、回测
│       ├── metrics_calculator.py # 绩效指标计算 + 净值曲线
│       └── regime_classifier.py  # 市场状态分类（VIX 等，预留）
│
├── seed.py                  # 种子数据脚本：创建 3 个默认策略
├── requirements.txt         # Python 依赖列表
└── .env                     # 环境变量（数据库密码等）
```

### 前端 (`frontend/`)

```
frontend/src/
├── main.tsx                 # React 入口，挂载到 DOM
├── App.tsx                  # 路由配置（/ → Dashboard, /strategies/:id → Detail）
├── index.css                # 全局样式（Tailwind）
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
│   ├── StrategyDetail.tsx   #   策略详情 + 回测运行 + 图表
│   └── Home.tsx             #   落地页
│
├── components/
│   ├── StrategyCard.tsx     # 策略卡片组件
│   ├── ErrorBoundary.tsx    # 错误边界
│   ├── layout/              # 布局组件
│   │   ├── DashboardLayout.tsx
│   │   ├── Sidebar.tsx
│   │   └── Header.tsx
│   ├── charts/              # 图表组件
│   │   ├── EquityChart.tsx      # 净值曲线（策略 vs 基准）
│   │   ├── DrawdownChart.tsx    # 回撤图
│   │   └── HeatmapChart.tsx     # 月度收益热力图
│   └── ui/                  # 40+ 基础 UI 组件（Shadcn/Radix）
│
├── contexts/ThemeContext.tsx # 主题（深色/浅色）
├── hooks/use-mobile.ts      # 移动端检测
└── lib/
    ├── utils.ts             # 工具函数
    └── mockData.ts          # 模拟数据（已弃用，改用真实 API）
```

### 配置文件

| 文件 | 作用 |
|------|------|
| `docker-compose.yml` | 定义 PostgreSQL 容器（用户名 quant，密码 quantpass，数据库名 quantplatform） |
| `vite.config.ts` | Vite 构建配置 + `/api` 代理到后端 8000 端口 |
| `.env` | 后端环境变量（数据库连接字符串） |

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
| name | VARCHAR(200) | 策略名称，如 "XLK Momentum - Risk Adjusted 6M" |
| description | TEXT | 策略描述 |
| tags | ARRAY(String) | 标签，如 ['momentum', 'XLK', 'risk_adjusted'] |
| method | VARCHAR(30) | 动量计算方法：simple / risk_adjusted / volume_weighted |
| lookback_months | INTEGER | 回看周期：3 / 6 / 12 个月 |
| decile | FLOAT | 选股比例：0.2 = 前/后 20% |
| tc_bps | INTEGER | 交易成本：5 = 5 个基点（0.05%） |
| winsor_q | FLOAT | 极端值处理：0.01 = 裁剪前后 1% |
| status | VARCHAR(20) | 状态：active / paused / archived |
| created_at | TIMESTAMP | 创建时间 |
| updated_at | TIMESTAMP | 更新时间 |

### 表 2: backtest_runs（回测记录）

每次点击 "Run Backtest" 都会新增一行。

| 字段 | 类型 | 说明 |
|------|------|------|
| id | UUID | 回测运行 ID |
| strategy_id | UUID | 关联的策略 ID（外键） |
| status | VARCHAR(20) | running / completed / failed |
| started_at | TIMESTAMP | 开始时间 |
| finished_at | TIMESTAMP | 完成时间 |
| param_method | VARCHAR | 本次使用的方法（快照） |
| param_lookback | INTEGER | 本次使用的回看周期 |
| param_start_date | DATE | 回测开始日期 |
| param_end_date | DATE | 回测结束日期 |
| total_return | FLOAT | 总收益率，如 1.044 = 104.4% |
| annual_return | FLOAT | 年化收益率 |
| annual_volatility | FLOAT | 年化波动率 |
| sharpe_ratio | FLOAT | 夏普比率 |
| max_drawdown | FLOAT | 最大回撤，如 -0.191 = -19.1% |
| win_rate | FLOAT | 胜率 |
| profit_factor | FLOAT | 盈亏比 |
| total_months | INTEGER | 回测总月数 |
| monthly_returns | JSONB | 每月收益率列表 |
| equity_curve | JSONB | 净值曲线数据（前端图表直接用） |
| error_message | TEXT | 失败时的错误信息 |

### 表 3: daily_prices（价格缓存）

从 Yahoo Finance 获取的每日收盘价，避免重复下载。

| 字段 | 类型 | 说明 |
|------|------|------|
| id | BIGINT | 自增主键 |
| ticker | VARCHAR(20) | 股票代码，如 AAPL |
| trade_date | DATE | 交易日期 |
| adj_close | FLOAT | 调整后收盘价 |
| volume | BIGINT | 成交量 |
| source | VARCHAR(20) | 数据来源（yahoo） |
| fetched_at | TIMESTAMP | 获取时间 |

**唯一约束**: (ticker, trade_date) — 同一只股票同一天只存一条。

### 表 4: trade_signals（交易信号）

每次回测产生的买入/卖出信号。

| 字段 | 类型 | 说明 |
|------|------|------|
| id | BIGINT | 自增主键 |
| backtest_run_id | UUID | 关联的回测运行 ID |
| signal_date | DATE | 信号日期（月末） |
| ticker | VARCHAR(20) | 股票代码 |
| direction | VARCHAR(10) | BUY（做多）或 SELL（做空） |
| weight | FLOAT | 组合权重，如 0.083 = 8.3% |
| score | FLOAT | 动量分数 |
| price | FLOAT | 信号日的价格 |
| pnl | FLOAT | 盈亏（预留） |

### 一次回测保存了什么数据？

1. **backtest_runs 表**: 1 行 — 绩效指标 + JSONB 净值曲线 + JSONB 月度收益率
2. **trade_signals 表**: 约 1400+ 行 — 每月 ~24 条信号（12 BUY + 12 SELL）× ~60 个月
3. **daily_prices 表**: 约 66×1500 = 99,000 行 — 66 只股票 × ~1500 个交易日（2020-2026）

---

## 5. 后端架构

### API 端点一览

#### 策略管理 (`/api/strategies`)

| 方法 | 路径 | 作用 |
|------|------|------|
| GET | `/api/strategies` | 获取策略列表（含最新回测指标） |
| GET | `/api/strategies/{id}` | 获取单个策略详情 |
| POST | `/api/strategies` | 创建新策略 |

#### 回测执行 (`/api/backtest`)

| 方法 | 路径 | 作用 |
|------|------|------|
| POST | `/api/backtest/{strategy_id}/run` | **启动回测**（后台执行，立即返回） |
| GET | `/api/backtest/runs/{run_id}/status` | **轮询回测状态** |
| GET | `/api/backtest/{strategy_id}/latest` | 获取最新完成的回测结果 |
| GET | `/api/backtest/{strategy_id}/equity` | 获取净值曲线数据 |
| GET | `/api/backtest/{strategy_id}/trades` | 获取交易信号（分页） |

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
    │      每个 ticker 间隔 0.3 秒（防止被限流）
    │
    ├── 3. 新数据写入数据库 → _save_to_db()
    │      使用 PostgreSQL UPSERT（已存在则跳过）
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
    │   │   ├── 排名，选前 20% 做多，后 20% 做空
    │   │   ├── 记录交易信号（ticker, direction, weight, price）
    │   │   └── 计算下月收益 = 多头收益 - 空头收益 - 交易成本
    │   │
    │   └── 返回 (月度收益率序列, 交易信号列表)
    │
    ├── calculate_metrics()  ← 计算绩效指标
    │   夏普比率, 最大回撤, 胜率, 盈亏比 等
    │
    └── build_equity_curve() ← 构建净值曲线
        从 $10,000 开始，策略 vs XLK 基准
```

---

## 6. 前端架构

### 页面路由

| 路径 | 页面 | 功能 |
|------|------|------|
| `/` | Dashboard | 策略卡片列表，搜索/排序 |
| `/strategies/:id` | StrategyDetail | 策略详情、运行回测、查看图表 |
| `/compare` | （预留） | 策略对比 |

### 组件层次

```
App.tsx (路由)
├── DashboardLayout (布局壳)
│   ├── Header (顶栏 + 主题切换)
│   ├── Sidebar (导航菜单)
│   └── 内容区域
│       │
│       ├── Dashboard.tsx
│       │   └── StrategyCard.tsx × N
│       │
│       └── StrategyDetail.tsx
│           ├── MetricCard × 4 (收益率、夏普、回撤、胜率)
│           ├── EquityChart (净值曲线)
│           ├── DrawdownChart (回撤图)
│           ├── HeatmapChart (月度热力图)
│           └── Trades Table (交易信号列表)
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
  ├── runBacktestWithPolling() → 启动 + 每3秒轮询直到完成
  ├── getEquityCurve()         → GET /api/backtest/{id}/equity
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
      target: "http://localhost:8000",  // 后端地址
      changeOrigin: true,
    },
  },
},
```

前端所有 API 请求以 `/api` 开头，Vite 开发服务器自动将它们转发给 FastAPI 后端。前端不需要知道后端的具体地址。

### 生产环境（未来）

需要 Nginx 或类似的反向代理来替代 Vite proxy。

---

## 8. 一次回测的完整数据流

用户点击 "Run Backtest" 后发生了什么：

```
[前端 StrategyDetail.tsx]
  │
  │ 1. 调用 runBacktestWithPolling(strategyId, {start_date, end_date})
  │    按钮显示: "Submitting backtest..."
  ▼
[前端 api/backtest.ts]
  │
  │ 2. POST /api/backtest/{strategyId}/run  →  后端立即返回 {id, status: "running"}
  │    按钮显示: "Fetching data & running backtest..."
  │
  │ 3. 每 3 秒发送: GET /api/backtest/runs/{runId}/status
  │    直到 status 变为 "completed" 或 "failed"
  ▼
[后端 routers/backtest.py]
  │
  │ 4. 创建 backtest_runs 表记录 (status="running")
  │ 5. 启动后台线程（独立事件循环 + 独立数据库连接）
  │ 6. 立即返回 HTTP 200 给前端
  ▼
[后台线程]
  │
  │ 7. data_fetcher: 检查 daily_prices 缓存
  │    ├── 缓存命中: 直接读数据库（毫秒级）
  │    └── 缓存未命中: 调 Yahoo API（每个 ticker 0.3 秒）
  │        第一次运行: 66 + 3 = 69 个 ticker × 0.3s ≈ 21 秒 + API 响应时间
  │        后续运行: 大部分从缓存读取，几秒内完成
  │
  │ 8. momentum_strategy: 计算动量分数、月度再平衡
  │    ├── 对 2020-07 到 2025-12 每个月末循环
  │    ├── 每月选出做多/做空股票
  │    └── 计算月度收益率
  │
  │ 9. metrics_calculator: 计算夏普、回撤等指标
  │
  │ 10. 写入数据库:
  │     ├── 更新 backtest_runs: status="completed" + 指标 + JSONB数据
  │     └── 插入 trade_signals: ~1400+ 行交易信号
  ▼
[前端收到 status="completed"]
  │
  │ 11. 并行加载: 策略数据 + 净值曲线 + 交易信号
  │ 12. 渲染图表和数据
  ▼
[用户看到回测结果]
```

---

## 9. 数据缓存机制

### 价格数据缓存

**首次运行** 某只股票的回测时：
1. 检查 `daily_prices` 表是否已有该 ticker 的数据
2. 没有 → 从 Yahoo Finance API 下载 → 存入数据库
3. 下次再用 → 直接从数据库读取，不再请求 API

**这意味着：**
- 第一次运行回测较慢（需要下载 69 个 ticker 的数据）
- **之后再运行任何策略都会很快**，因为数据已经缓存
- 即使换一个策略（比如从 simple 换到 risk_adjusted），数据不需要重新下载
- 只有当你需要更新到最新日期的数据时，才会有新的 API 请求

### 回测结果缓存

- 每次运行回测都会在 `backtest_runs` 表新增一条记录
- 旧的回测结果不会被删除，可以对比不同时间/参数的结果
- 前端默认展示最新一次完成的回测

---

## 10. 如何分开测试前后端

### 测试后端（无需启动前端）

#### 方式 1: Swagger 文档（推荐）

1. 启动后端: `uvicorn app.main:app --reload`
2. 打开浏览器: http://localhost:8000/docs
3. 可以看到所有 API 端点，点击 "Try it out" 直接测试

#### 方式 2: curl 命令

```bash
# 健康检查
curl http://localhost:8000/api/health

# 获取策略列表
curl http://localhost:8000/api/strategies

# 启动回测（替换 {strategy_id} 为实际 UUID）
curl -X POST http://localhost:8000/api/backtest/{strategy_id}/run \
  -H "Content-Type: application/json" \
  -d '{"start_date": "2020-01-01", "end_date": "2026-01-30"}'

# 查询回测状态
curl http://localhost:8000/api/backtest/runs/{run_id}/status

# 获取净值曲线
curl http://localhost:8000/api/backtest/{strategy_id}/equity
```

#### 方式 3: Python 脚本

```python
import requests

BASE = "http://localhost:8000/api"

# 获取策略
strategies = requests.get(f"{BASE}/strategies").json()
print(strategies)

# 启动回测
sid = strategies[0]["id"]
run = requests.post(f"{BASE}/backtest/{sid}/run",
    json={"start_date": "2020-01-01", "end_date": "2026-01-30"}
).json()
print(f"Run ID: {run['id']}, Status: {run['status']}")
```

### 测试前端（可以不启动后端）

前端可以使用 `lib/mockData.ts` 中的模拟数据进行开发和测试 UI 组件。但测试完整功能需要后端运行。

### 测试数据库连接

```python
# 在 backend 目录下运行
python -c "
from app.config import settings
print(f'Database URL: {settings.DATABASE_URL}')
print(f'Tickers count: {len(settings.XLK_TICKERS)}')
"
```

---

## 11. 数据库可视化与管理

### 方式 1: pgAdmin（图形界面，推荐）

1. 下载安装 [pgAdmin](https://www.pgadmin.org/download/)
2. 添加服务器连接:
   - Host: `localhost`
   - Port: `5432`
   - Database: `quantplatform`
   - Username: `quant`
   - Password: `quantpass`
3. 可以浏览表结构、查看数据、执行 SQL

### 方式 2: DBeaver（通用数据库客户端）

1. 下载 [DBeaver](https://dbeaver.io/download/)
2. 新建 PostgreSQL 连接，填入上述信息
3. 功能更丰富，支持 ER 图可视化

### 方式 3: VS Code 插件

安装 "Database Client" 或 "PostgreSQL" 插件，直接在 VS Code 中查看。

### 方式 4: 命令行 psql

```bash
# 通过 Docker 连接
docker exec -it quantplatform-postgres-1 psql -U quant -d quantplatform

# 常用查询
\dt                              -- 列出所有表
SELECT * FROM strategies;        -- 查看策略
SELECT count(*) FROM daily_prices; -- 统计缓存的价格数据量
SELECT id, status, sharpe_ratio, total_return
  FROM backtest_runs
  ORDER BY started_at DESC
  LIMIT 5;                       -- 最近 5 次回测
SELECT ticker, direction, weight, signal_date
  FROM trade_signals
  WHERE backtest_run_id = '...'
  ORDER BY signal_date DESC
  LIMIT 20;                      -- 某次回测的交易信号
```

### 常用查询示例

```sql
-- 查看数据库缓存了多少数据
SELECT ticker, COUNT(*) as days,
       MIN(trade_date) as first_date,
       MAX(trade_date) as last_date
FROM daily_prices
GROUP BY ticker
ORDER BY ticker;

-- 查看某个策略的所有回测历史
SELECT id, status, sharpe_ratio, total_return, max_drawdown,
       started_at, finished_at
FROM backtest_runs
WHERE strategy_id = '你的策略ID'
ORDER BY started_at DESC;

-- 查看某次回测的最新交易信号
SELECT signal_date, ticker, direction, weight, score, price
FROM trade_signals
WHERE backtest_run_id = '你的回测ID'
ORDER BY signal_date DESC, direction
LIMIT 30;
```

---

## 12. 常见问题与排查

### Q: 后端启动报错 "module 'app' has no attribute..."

原因：Python 模块命名冲突。确保 `main.py` 中的 import 使用 `from app import models` 而非 `import app.models`。

### Q: Yahoo Finance 数据获取失败

1. 检查网络连接
2. Yahoo API 可能暂时限流，等几分钟再试
3. 查看后端日志中的具体错误信息
4. 某些 ticker 可能已退市或改名

### Q: 回测一直显示 "Running..."

1. 查看后端终端是否有 `[BG]` 开头的日志
2. 如果有错误，check `backtest_runs` 表中对应记录的 `error_message`
3. 数据库连接是否正常（Docker 容器是否运行）

### Q: 前端请求返回 404 或 CORS 错误

1. 确保后端正在运行（`http://localhost:8000/api/health` 返回 OK）
2. 确保 `vite.config.ts` 中的 proxy 配置正确
3. CORS 白名单在 `main.py` 中配置

### Q: 数据库连接失败

```bash
# 检查 Docker 容器是否运行
docker ps

# 如果没有运行
docker-compose up -d

# 检查日志
docker-compose logs postgres
```

---

## 13. 后续开发方向

### 短期优化（完善 MVP）

- [ ] **前端策略创建页面** — 目前只有 seed.py 可以创建策略
- [ ] **回测参数可配置** — 前端目前硬编码了日期范围，应该让用户输入
- [ ] **策略对比页面** — 多个策略净值曲线叠加对比
- [ ] **数据更新机制** — 检测缓存数据是否过期，自动补充最新数据
- [ ] **错误提示优化** — 回测失败时给用户更友好的提示

### 中期功能

- [ ] **添加更多策略** — 均值回归、配对交易、因子模型等
- [ ] **市场状态分类** — 集成 `regime_classifier.py`，根据 VIX 等指标调整策略
- [ ] **风险管理模块** — 最大仓位限制、止损逻辑
- [ ] **数据源扩展** — 支持更多 API（Alpha Vantage、Polygon 等）
- [ ] **用户认证** — 登录注册，每个用户有自己的策略

### 长期目标

- [ ] **实盘信号推送** — 每日/每月推送交易信号（邮件/Telegram）
- [ ] **策略组合优化** — 多策略资金分配（Kelly 准则等）
- [ ] **实时数据** — WebSocket 推送实时价格
- [ ] **部署上线** — Docker Compose 全栈部署 / 云端部署
