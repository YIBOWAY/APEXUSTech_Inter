# QuantPlatform MVP

基于 XLK 动量策略的量化交易平台 MVP。

## 技术栈

- **后端**: FastAPI + SQLAlchemy (async) + PostgreSQL
- **前端**: React 19 + Vite + Shadcn/ui + ECharts
- **策略**: XLK 动量策略（3种方法 × 3个回看期）

## 快速启动

### 1. 启动 PostgreSQL

```bash
cd quantplatform
docker-compose up -d
```

等待数据库启动（约10秒）。

### 2. 启动后端

```bash
cd backend

# 创建虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt

# 运行数据库种子
python seed.py

# 启动服务
uvicorn app.main:app --reload
```

后端运行在 http://localhost:8000，Swagger 文档在 http://localhost:8000/docs

### 3. 启动前端

```bash
cd frontend

# 安装依赖
pnpm install

# 启动开发服务器
pnpm dev
```

前端运行在 http://localhost:5173

## 使用说明

1. 打开 http://localhost:5173
2. Dashboard 显示预置的 3 个策略
3. 点击策略卡片进入详情页
4. 点击 "Run Backtest" 执行回测
5. 回测完成后自动显示净值曲线、回撤图、月度热力图

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/strategies` | 策略列表 |
| GET | `/api/strategies/{id}` | 策略详情 |
| POST | `/api/backtest/{id}/run` | 触发回测 |
| GET | `/api/backtest/{id}/equity` | 净值曲线 |
| GET | `/api/backtest/{id}/trades` | 交易信号 |

## 目录结构

```
quantplatform/
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI 入口
│   │   ├── models/           # ORM 模型
│   │   ├── schemas/          # Pydantic 模型
│   │   ├── routers/          # API 路由
│   │   └── services/         # 业务逻辑
│   ├── seed.py               # 数据种子
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── api/              # API 调用层
│   │   ├── pages/            # 页面组件
│   │   └── components/       # UI 组件
│   └── package.json
└── docker-compose.yml        # PostgreSQL
```

## 注意事项

- 首次回测会从 Yahoo Finance 拉取数据（约 70 个 ticker），耗时较长
- 数据会缓存到数据库，二次回测速度更快
- 回测同步执行，请求可能需要 30-60 秒
