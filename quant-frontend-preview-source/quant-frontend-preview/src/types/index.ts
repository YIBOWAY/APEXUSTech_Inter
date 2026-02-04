export interface Metric {
  total_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  annual_volatility: number;
  win_rate: number;
  profit_factor: number;
}

export interface Strategy {
  id: string;
  name: string;
  description: string;
  tags: string[];
  metrics: Metric;
  created_at: string;
  updated_at: string;
  status: 'active' | 'paused' | 'archived';
}

export interface EquityPoint {
  date: string;
  value: number;
  benchmark: number;
}

export interface Trade {
  id: string;
  symbol: string;
  direction: 'BUY' | 'SELL';
  price: number;
  quantity: number;
  timestamp: string;
  commission: number;
  pnl?: number;
  status: 'filled' | 'pending' | 'cancelled';
}

export interface RollingMetric {
  date: string;
  value: number;
}

export interface ComparisonData {
  strategies: Strategy[];
  series: {
    date: string;
    [strategyId: string]: number | string; // value for each strategy
  }[];
}
