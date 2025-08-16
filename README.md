# APEXUSTech Internship Project - Event-Driven Futures Backtesting Framework

## 项目概述
这是一个基于事件驱动架构的期货日历价差策略回测框架，使用Python实现了完整的量化交易回测系统。

## 主要功能
- 🎯 **事件驱动架构**：模块化设计，包含市场事件、信号事件、订单事件和成交事件
- 📊 **数据处理**：支持CSV格式的历史期货数据
- 📈 **策略实现**：Z-score均值回归日历价差策略
- 💰 **投资组合管理**：实时头寸和资金管理
- 🔄 **模拟执行**：包含滑点和手续费的真实模拟
- 📊 **性能分析**：完整的回测指标和可视化

## 文件结构
```
APEXUSTech_Inter/
├── project3/
│   ├── tmp/
│   │   └── temp.ipynb           # 主要回测框架和分析
│   ├── cl_data.csv              # WTI原油历史数据
│   └── crude_oil_wti_spread_data.csv  # 生成的价差数据
└── README.md
```

## 核心组件

### 事件系统
- `Event`: 基础事件类
- `MarketEvent`: 市场数据事件
- `SignalEvent`: 交易信号事件
- `OrderEvent`: 订单事件
- `FillEvent`: 成交事件

### 数据处理
- `RealCSVDataHandler`: CSV数据读取和处理

### 策略引擎
- `RealCalendarSpreadZScoreStrategy`: Z-score日历价差策略

### 投资组合管理
- `RealBasicPortfolio`: 投资组合和风险管理

### 执行系统
- `RealSimulatedExecutionHandler`: 模拟执行引擎

## 回测结果
基于真实WTI原油期货数据(2020-2025)的回测结果：
- 数据点：1,260个交易日
- 策略：Z-score均值回归日历价差
- 包含完整的性能指标和可视化分析

## 技术栈
- Python 3.x
- pandas: 数据处理
- numpy: 数值计算
- matplotlib: 可视化
- queue: 事件队列管理

## 使用方法
1. 克隆仓库
2. 安装依赖：`pip install pandas numpy matplotlib`
3. 运行 `temp.ipynb` 中的代码

## 作者
APEXUSTech实习项目

## 许可证
MIT License
