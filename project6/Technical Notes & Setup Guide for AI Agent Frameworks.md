# Technical Notes & Setup Guide for AI Agent Frameworks

*This guide summarises the key installation steps, commands and troubleshooting tips encountered during evaluation of Qlib, FinRL, FinRobot, TradingAgents and Alpha-Agent.*

---

## Environment Preparation

- It is highly recommended to use **Conda** to manage Python environments. Many dependencies require specific versions of compilers and libraries.
- Always create a **dedicated virtual environment** for each framework (e.g., `conda create -n qlib python=3.10`), activate it, and then install packages.

**Python versions supported:**

- **Qlib:** Python 3.8–3.12  
- **FinRL:** Python 3.6 (legacy) or higher; examples use 3.8  
- **FinRobot:** Python 3.10 recommended  
- **TradingAgents:** Python 3.13 (bleeding edge)  
- **Alpha-Agent:** Not yet specified; use Python 3.10+ with Qlib  

---

## Qlib Setup

### Install via pip
```bash
pip install pyqlib
```

This installs the latest stable version. If cutting-edge features are needed, clone from source:
```bash
git clone https://github.com/microsoft/qlib.git
cd qlib
pip install .    # or pip install -e .[dev] for development
```

### Prepare data
The official Qlib dataset is temporarily disabled; use the community-contributed dataset:

```bash
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
mkdir -p ~/.qlib/qlib_data/cn_data
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=1
rm -f qlib_bin.tar.gz
```

Or download via CLI from Yahoo Finance:
```bash
python -m qlib.cli.data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

### Run a basic demo
```bash
qrun examples/benchmarks/LightGBM/workflow_config_lightgbm.yaml
```

The output includes logs, trained model and performance metrics.

**Troubleshooting:**

- Missing header files during installation → switch to Conda or install gcc/g++.  
- On Mac M1 → `brew install libomp` before building LightGBM.  

---

## FinRL Setup

### Installation
```bash
pip install finrl
```
Or:
```bash
git clone https://github.com/AI4Finance-Foundation/FinRL.git
cd FinRL
pip install -e .
```

### Running examples
- Launch Jupyter and open `Stock_NeurIPS2018.ipynb`.  
- Typical workflow: `train.py` → train RL agent, `test.py` → evaluate, `trade.py` → simulate trading.  

### Data sources
- Built-in downloaders: Akshare, Alpaca, Baostock, Binance, etc.  
- Configure API keys in `.env` or config files.  

**Troubleshooting:**  

- API rate limits → reduce frequency or cache data.  
- IBKR data → install `ib_insync` and run IB Gateway.  

---

## FinRobot Setup

### Install
```bash
conda create --name finrobot python=3.10
conda activate finrobot
git clone https://github.com/AI4Finance-Foundation/FinRobot.git
cd FinRobot
pip install -e .
```

### Configure API keys
- Rename `OAI_CONFIG_LIST_sample` → `OAI_CONFIG_LIST` and remove comments.  
- Add OpenAI API key, Finnhub, etc.  

### Run tutorials
Open:
- `tutorials_beginner/agent_fingpt_forecaster.ipynb`  
- `tutorials_beginner/agent_annual_report.ipynb`  

**Notes:**

- Focus on research/reporting, **not** live trading.  
- Can integrate other LLMs via LLMOps layer.  

---

## TradingAgents Setup

### Install
```bash
git clone https://github.com/TauricResearch/TradingAgents.git
cd TradingAgents
conda create -n tradingagents python=3.13
conda activate tradingagents
pip install -r requirements.txt
```

### Set API keys
```bash
export FINNHUB_API_KEY=your_finnhub_api_key
export OPENAI_API_KEY=your_openai_api_key
```

### Run CLI
```bash
python -m cli.main
```

Choose ticker, date, LLM config and number of debate rounds → get trading decision.

### Python usage
```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

ta = TradingAgentsGraph(debug=True, config=DEFAULT_CONFIG.copy())
_, decision = ta.propagate("NVDA", "2024-05-10")
print(decision)
```

**Notes:**

- Uses LLMs extensively; pick smaller models to save cost.  
- Simulated exchange; backtesting data to come later.  

---

## Alpha-Agent Setup (Future)

- Repo still under construction.  
- Expected steps:  
  - Install via pip or source.  
  - Connect to **LLMQuant quant-wiki**, MarketPulse, LLMQuant Data.  
  - Use pipeline functions for: knowledge base, market features, strategies, code gen, Qlib backtesting, reports.  

---

## Broker Connectivity & Execution

- None provide **native IBKR execution**.  
- Qlib & FinRL → research/backtesting.  
- FinRobot, TradingAgents, Alpha-Agent → research/analysis only.  

To build live trading:  

- Use Qlib outputs + custom executor with IBKR API (`ib_insync`).  
- FinRL → use data streams, but manage live execution separately.  

---

## Summary

- **Qlib** → factor mining, workflow demos.  
- **FinRL** → reinforcement learning agents.  
- **FinRobot** → LLM-based financial research.  
- **TradingAgents** → multi-agent debate trading framework.  
- **Alpha-Agent** → integrates above, with automated loop.  

⚠️ Remember:  
- Store API keys securely.  
- Don’t expose sensitive data during experiments.  
