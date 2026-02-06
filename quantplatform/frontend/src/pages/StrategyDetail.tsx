import { useState, useEffect } from "react";
import { useRoute } from "wouter";
import { DashboardLayout } from "@/components/layout/DashboardLayout";
import { MetricCard } from "@/components/ui/metric-card";
import { EquityChart } from "@/components/charts/EquityChart";
import { DrawdownChart } from "@/components/charts/DrawdownChart";
import { HeatmapChart } from "@/components/charts/HeatmapChart";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Play,
  Settings2,
  Download,
  Loader2,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { format } from "date-fns";
import { getStrategy } from "@/api/strategies";
import { runBacktestWithPolling, getEquityCurve, getTrades } from "@/api/backtest";
import type { BacktestRunResult } from "@/api/backtest";
import type { Strategy, EquityPoint, Trade } from "@/types";

export default function StrategyDetail() {
  const [, params] = useRoute("/strategies/:id");
  const id = params?.id;

  const [strategy, setStrategy] = useState<Strategy | null>(null);
  const [equityData, setEquityData] = useState<EquityPoint[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [runStatus, setRunStatus] = useState<string>("");

  useEffect(() => {
    if (id) {
      loadData();
    }
  }, [id]);

  const loadData = async () => {
    if (!id) return;
    try {
      setLoading(true);
      setError(null);
      const [strategyData, equityDataRes, tradesData] = await Promise.all([
        getStrategy(id),
        getEquityCurve(id).catch(() => []),
        getTrades(id).catch(() => []),
      ]);
      setStrategy(strategyData);
      setEquityData(equityDataRes);
      setTrades(tradesData);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load strategy");
      console.error("Failed to load strategy:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleRunBacktest = async () => {
    if (!id) return;
    setIsRunning(true);
    setError(null);
    setRunStatus("Submitting backtest...");
    try {
      await runBacktestWithPolling(
        id,
        {
          start_date: "2020-01-01",
          end_date: "2026-01-30",
        },
        (status: BacktestRunResult) => {
          if (status.status === "running") {
            setRunStatus("Fetching data & running backtest...");
          } else if (status.status === "completed") {
            setRunStatus("Completed! Loading results...");
          }
        },
      );
      // Reload data after backtest completes
      await loadData();
      setRunStatus("");
    } catch (err) {
      console.error("Backtest failed:", err);
      setError(err instanceof Error ? err.message : "Backtest failed");
      setRunStatus("");
    } finally {
      setIsRunning(false);
    }
  };

  if (loading) {
    return (
      <DashboardLayout title="Loading...">
        <div className="space-y-6">
          <Skeleton className="h-12 w-64" />
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[...Array(4)].map((_, i) => (
              <Skeleton key={i} className="h-24" />
            ))}
          </div>
          <Skeleton className="h-[400px]" />
        </div>
      </DashboardLayout>
    );
  }

  if (error || !strategy) {
    return (
      <DashboardLayout title="Error">
        <div className="text-center py-12">
          <p className="text-destructive mb-4">{error || "Strategy not found"}</p>
          <Button onClick={loadData} variant="outline">
            <Loader2 className="mr-2 h-4 w-4" />
            Retry
          </Button>
        </div>
      </DashboardLayout>
    );
  }

  const metrics = strategy.metrics || {
    total_return: 0,
    sharpe_ratio: 0,
    max_drawdown: 0,
    win_rate: 0,
  };

  return (
    <DashboardLayout title={strategy.name}>
      <div className="space-y-6">
        {/* Header Actions */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <h2 className="text-3xl font-display font-bold">{strategy.name}</h2>
              <Badge variant="outline" className={
                strategy.status === 'active'
                  ? "text-emerald-500 border-emerald-500/20 bg-emerald-500/10"
                  : "text-amber-500 border-amber-500/20 bg-amber-500/10"
              }>
                {strategy.status === 'active' ? 'Active' : 'Paused'}
              </Badge>
            </div>
            <p className="text-muted-foreground max-w-2xl">
              {strategy.description || `Momentum strategy using ${strategy.method || 'simple'} method with ${strategy.lookback_months || 6} month lookback`}
            </p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm">
              <Settings2 className="mr-2 h-4 w-4" /> Configure
            </Button>
            <Button variant="outline" size="sm">
              <Download className="mr-2 h-4 w-4" /> Export
            </Button>
            <Button onClick={handleRunBacktest} disabled={isRunning} className="bg-primary hover:bg-primary/90">
              {isRunning ? (
                <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> {runStatus || "Running..."}</>
              ) : (
                <><Play className="mr-2 h-4 w-4" /> Run Backtest</>
              )}
            </Button>
          </div>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricCard
            title="Total Return"
            value={metrics.total_return ? `${(metrics.total_return * 100).toFixed(1)}%` : 'N/A'}
            trend={metrics.total_return ? 12.5 : undefined}
            prefix={metrics.total_return && metrics.total_return > 0 ? "+" : ""}
          />
          <MetricCard
            title="Sharpe Ratio"
            value={metrics.sharpe_ratio ? metrics.sharpe_ratio.toFixed(2) : 'N/A'}
            trend={metrics.sharpe_ratio ? 0.4 : undefined}
          />
          <MetricCard
            title="Max Drawdown"
            value={metrics.max_drawdown ? `${(metrics.max_drawdown * 100).toFixed(1)}%` : 'N/A'}
            trend={metrics.max_drawdown ? -2.1 : undefined}
            valueClassName="text-destructive"
          />
          <MetricCard
            title="Win Rate"
            value={metrics.win_rate ? `${(metrics.win_rate * 100).toFixed(0)}%` : 'N/A'}
            suffix=""
            trend={metrics.win_rate ? 5.2 : undefined}
          />
        </div>

        {/* No data prompt */}
        {equityData.length === 0 && (
          <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
            <CardContent className="py-12 text-center">
              <p className="text-muted-foreground mb-4">
                No backtest results yet. Run a backtest to see performance data.
              </p>
              <Button onClick={handleRunBacktest} disabled={isRunning}>
                {isRunning ? (
                  <><Loader2 className="mr-2 h-4 w-4 animate-spin" /> {runStatus || "Running..."}</>
                ) : (
                  <><Play className="mr-2 h-4 w-4" /> Run Backtest</>
                )}
              </Button>
            </CardContent>
          </Card>
        )}

        {/* Main Charts Area */}
        {equityData.length > 0 && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Equity Curve (2/3) */}
            <Card className="lg:col-span-2 border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader>
                <CardTitle>Equity Curve</CardTitle>
              </CardHeader>
              <CardContent>
                <EquityChart data={equityData} height={400} />
              </CardContent>
            </Card>

            {/* Recent Trades (1/3) */}
            <Card className="border-border/50 bg-card/50 backdrop-blur-sm flex flex-col">
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle>Recent Trades</CardTitle>
                <Button variant="ghost" size="sm" className="h-8 text-xs">View All</Button>
              </CardHeader>
              <CardContent className="flex-1 p-0">
                <ScrollArea className="h-[400px]">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Symbol</TableHead>
                        <TableHead>Side</TableHead>
                        <TableHead className="text-right">Weight</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {trades.length === 0 ? (
                        <TableRow>
                          <TableCell colSpan={3} className="text-center text-muted-foreground">
                            No trades yet
                          </TableCell>
                        </TableRow>
                      ) : (
                        trades.slice(0, 20).map((trade) => (
                          <TableRow key={trade.id}>
                            <TableCell className="font-medium font-mono text-xs">
                              {trade.symbol}
                              <div className="text-[10px] text-muted-foreground">
                                {trade.timestamp ? format(new Date(trade.timestamp), 'MM-dd') : '-'}
                              </div>
                            </TableCell>
                            <TableCell>
                              <Badge
                                variant="secondary"
                                className={trade.direction === 'BUY' ? 'text-emerald-500 bg-emerald-500/10' : 'text-rose-500 bg-rose-500/10'}
                              >
                                {trade.direction}
                              </Badge>
                            </TableCell>
                            <TableCell className="text-right font-mono text-xs">
                              {(trade.quantity * 100).toFixed(1)}%
                            </TableCell>
                          </TableRow>
                        ))
                      )}
                    </TableBody>
                  </Table>
                </ScrollArea>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Secondary Charts */}
        {equityData.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader>
                <CardTitle>Underwater Drawdown</CardTitle>
              </CardHeader>
              <CardContent>
                <DrawdownChart data={equityData} height={300} />
              </CardContent>
            </Card>

            <Card className="border-border/50 bg-card/50 backdrop-blur-sm">
              <CardHeader>
                <CardTitle>Monthly Returns</CardTitle>
              </CardHeader>
              <CardContent>
                <HeatmapChart data={equityData} height={300} />
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}
