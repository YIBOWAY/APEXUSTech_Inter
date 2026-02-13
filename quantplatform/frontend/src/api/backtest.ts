import apiClient from './client';
import type { EquityPoint, Trade } from '@/types';

export interface BacktestRequest {
  start_date?: string;
  end_date?: string;
  method?: string;
  lookback_months?: number;
}

export interface BacktestRunResult {
  id: string;
  strategy_id: string;
  status: string;
  started_at?: string;
  finished_at?: string;
  total_return?: number;
  annual_return?: number;
  annual_volatility?: number;
  sharpe_ratio?: number;
  max_drawdown?: number;
  win_rate?: number;
  profit_factor?: number;
  total_months?: number;
  error_message?: string;
}

/** Start a backtest (returns immediately with status "running") */
export async function runBacktest(
  strategyId: string,
  params?: BacktestRequest
): Promise<BacktestRunResult> {
  const response = await apiClient.post(`/backtest/${strategyId}/run`, params || {});
  return response.data;
}

/** Poll backtest run status */
export async function getBacktestStatus(runId: string): Promise<BacktestRunResult> {
  const response = await apiClient.get(`/backtest/runs/${runId}/status`);
  return response.data;
}

/**
 * Start backtest and poll until completion.
 * Calls onStatus callback with each status update for UI progress.
 */
export async function runBacktestWithPolling(
  strategyId: string,
  params?: BacktestRequest,
  onStatus?: (status: BacktestRunResult) => void,
  pollIntervalMs = 3000,
): Promise<BacktestRunResult> {
  // 1. Start backtest
  const run = await runBacktest(strategyId, params);
  onStatus?.(run);

  // 2. Poll until done
  return new Promise((resolve, reject) => {
    const poll = async () => {
      try {
        const status = await getBacktestStatus(run.id);
        onStatus?.(status);

        if (status.status === 'completed') {
          resolve(status);
        } else if (status.status === 'failed') {
          reject(new Error(status.error_message || 'Backtest failed'));
        } else {
          // Still running, poll again
          setTimeout(poll, pollIntervalMs);
        }
      } catch (err) {
        reject(err);
      }
    };

    setTimeout(poll, pollIntervalMs);
  });
}

export async function getLatestBacktest(strategyId: string): Promise<BacktestRunResult> {
  const response = await apiClient.get(`/backtest/${strategyId}/latest`);
  return response.data;
}

export async function getEquityCurve(
  strategyId: string,
  runId?: string
): Promise<EquityPoint[]> {
  const params = runId ? { run_id: runId } : {};
  const response = await apiClient.get(`/backtest/${strategyId}/equity`, { params });
  return response.data;
}

export interface MonthlyReturn {
  date: string;
  return: number;
}

export async function getMonthlyReturns(strategyId: string): Promise<MonthlyReturn[]> {
  const response = await apiClient.get(`/backtest/${strategyId}/monthly-returns`);
  return response.data;
}

export async function getTrades(
  strategyId: string,
  params?: { run_id?: string; page?: number; size?: number }
): Promise<Trade[]> {
  const response = await apiClient.get(`/backtest/${strategyId}/trades`, { params });
  return response.data;
}
