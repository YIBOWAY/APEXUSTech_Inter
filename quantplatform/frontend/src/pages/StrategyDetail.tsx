import { useState, useEffect, useMemo, useCallback } from "react";
import { useRoute } from "wouter";
import { DashboardLayout } from "@/components/layout/DashboardLayout";
import { MetricCard } from "@/components/ui/metric-card";
import { EquityChart } from "@/components/charts/EquityChart";
import { DrawdownChart } from "@/components/charts/DrawdownChart";
import { HeatmapChart } from "@/components/charts/HeatmapChart";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Calendar } from "@/components/ui/calendar";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import {
  Play,
  Settings2,
  Download,
  Loader2,
  ChevronLeft,
  ChevronRight,
  CalendarDays,
  Info,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
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
import { runBacktestWithPolling, getEquityCurve, getTrades, getMonthlyReturns } from "@/api/backtest";
import type { BacktestRunResult, MonthlyReturn } from "@/api/backtest";
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
  const [showAllTrades, setShowAllTrades] = useState(false);
  const [allTrades, setAllTrades] = useState<Trade[]>([]);
  const [allTradesLoading, setAllTradesLoading] = useState(false);
  const [tradePage, setTradePage] = useState(1);
  const [monthlyReturns, setMonthlyReturns] = useState<MonthlyReturn[]>([]);
  const [startDate, setStartDate] = useState<Date>(new Date(2020, 0, 1));
  const [endDate, setEndDate] = useState<Date>(new Date(2025, 11, 31));
  const [startOpen, setStartOpen] = useState(false);
  const [endOpen, setEndOpen] = useState(false);

  const loadData = useCallback(async () => {
    if (!id) return;
    try {
      setLoading(true);
      setError(null);
      const [strategyData, equityDataRes, tradesData, monthlyReturnsRes] = await Promise.all([
        getStrategy(id),
        getEquityCurve(id).catch(() => []),
        getTrades(id).catch(() => []),
        getMonthlyReturns(id).catch(() => []),
      ]);
      setStrategy(strategyData);
      setEquityData(equityDataRes);
      setTrades(tradesData);
      setMonthlyReturns(monthlyReturnsRes);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load strategy");
      console.error("Failed to load strategy:", err);
    } finally {
      setLoading(false);
    }
  }, [id]);

  useEffect(() => {
    if (id) {
      loadData();
    }
  }, [id, loadData]);

  const handleRunBacktest = async () => {
    if (!id) return;
    setIsRunning(true);
    setError(null);
    setRunStatus("Submitting backtest...");
    try {
      await runBacktestWithPolling(
        id,
        { start_date: format(startDate, "yyyy-MM-dd"), end_date: format(endDate, "yyyy-MM-dd") },
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

  const handleViewAllTrades = async () => {
    if (!id) return;
    setShowAllTrades(true);
    setAllTradesLoading(true);
    setTradePage(1);
    try {
      const pageSize = 500;
      let page = 1;
      const all: Trade[] = [];

      while (true) {
        const chunk = await getTrades(id, { page, size: pageSize });
        all.push(...chunk);
        if (chunk.length < pageSize) break;
        page += 1;
      }

      setAllTrades(all);
    } catch (err) {
      console.error("Failed to load all trades:", err);
    } finally {
      setAllTradesLoading(false);
    }
  };

  // Group trades by date for display
  const groupTradesByDate = (tradeList: Trade[]) => {
    const groups: Record<string, Trade[]> = {};
    for (const t of tradeList) {
      const date = t.timestamp || "unknown";
      if (!groups[date]) groups[date] = [];
      groups[date].push(t);
    }
    return Object.entries(groups).sort(([a], [b]) => b.localeCompare(a));
  };

  const groupedTrades = useMemo(() => groupTradesByDate(allTrades), [allTrades]);
  const totalPages = groupedTrades.length;
  const pagedTradeGroups = useMemo(() => {
    if (totalPages === 0) return [];
    const safePage = Math.min(Math.max(tradePage, 1), totalPages);
    return [groupedTrades[safePage - 1]];
  }, [groupedTrades, totalPages, tradePage]);

  useEffect(() => {
    if (totalPages > 0 && tradePage > totalPages) {
      setTradePage(totalPages);
    }
  }, [tradePage, totalPages]);

  const monthlyPerfMap = useMemo(() => {
    const perfMap = new Map<string, { monthlyReturn?: number; monthlyPnl?: number }>();
    const returnMap = new Map(monthlyReturns.map((r) => [r.date, r.return]));
    const sortedEquity = [...equityData].sort((a, b) => a.date.localeCompare(b.date));

    for (let i = 0; i < sortedEquity.length; i += 1) {
      const current = sortedEquity[i];
      const currentReturn = returnMap.get(current.date);
      let monthlyPnl: number | undefined;

      if (i === 0) {
        if (
          currentReturn !== undefined &&
          Number.isFinite(currentReturn) &&
          currentReturn > -0.999999
        ) {
          const prevValue = current.value / (1 + currentReturn);
          monthlyPnl = current.value - prevValue;
        }
      } else {
        monthlyPnl = current.value - sortedEquity[i - 1].value;
      }

      perfMap.set(current.date, {
        monthlyReturn: currentReturn,
        monthlyPnl,
      });
    }

    return perfMap;
  }, [monthlyReturns, equityData]);

  const formatSignedCurrency = (value?: number) => {
    if (value === undefined || Number.isNaN(value)) return "-";
    return `${value >= 0 ? "+" : "-"}$${Math.abs(value).toFixed(2)}`;
  };

  const formatSignedPercent = (value?: number) => {
    if (value === undefined || Number.isNaN(value)) return "-";
    return `${value >= 0 ? "+" : ""}${(value * 100).toFixed(2)}%`;
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
          <div className="space-y-2">
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-2">
                <Popover open={startOpen} onOpenChange={setStartOpen}>
                  <PopoverTrigger asChild>
                    <Button variant="outline" size="sm" className={cn("w-[150px] justify-start text-left font-mono text-xs h-9", !startDate && "text-muted-foreground")}>
                      <CalendarDays className="mr-2 h-3.5 w-3.5" />
                      {format(startDate, "yyyy-MM-dd")}
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-auto p-0" align="start">
                    <Calendar
                      mode="single"
                      captionLayout="dropdown"
                      selected={startDate}
                      onSelect={(date) => { if (date) { setStartDate(date); setStartOpen(false); } }}
                      disabled={(date) => date > endDate || date > new Date()}
                      defaultMonth={startDate}
                      startMonth={new Date(2005, 0)}
                      endMonth={new Date(new Date().getFullYear(), 11)}
                    />
                  </PopoverContent>
                </Popover>
                <span className="text-muted-foreground text-sm">—</span>
                <Popover open={endOpen} onOpenChange={setEndOpen}>
                  <PopoverTrigger asChild>
                    <Button variant="outline" size="sm" className={cn("w-[150px] justify-start text-left font-mono text-xs h-9", !endDate && "text-muted-foreground")}>
                      <CalendarDays className="mr-2 h-3.5 w-3.5" />
                      {format(endDate, "yyyy-MM-dd")}
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-auto p-0" align="start">
                    <Calendar
                      mode="single"
                      captionLayout="dropdown"
                      selected={endDate}
                      onSelect={(date) => { if (date) { setEndDate(date); setEndOpen(false); } }}
                      disabled={(date) => date < startDate || date > new Date()}
                      defaultMonth={endDate}
                      startMonth={new Date(2005, 0)}
                      endMonth={new Date(new Date().getFullYear(), 11)}
                    />
                  </PopoverContent>
                </Popover>
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
            {strategy.lookback_months && (
              <p className="text-xs text-muted-foreground flex items-center gap-1">
                <Info className="h-3 w-3 shrink-0" />
                First {strategy.lookback_months} months are lookback warmup — trading starts from month {strategy.lookback_months + 1}.
              </p>
            )}
          </div>
        </div>

        {/* Metrics Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricCard
            title="Total Return"
            value={metrics.total_return ? `${(metrics.total_return * 100).toFixed(1)}%` : 'N/A'}
            prefix={metrics.total_return && metrics.total_return > 0 ? "+" : ""}
          />
          <MetricCard
            title="Sharpe Ratio"
            value={metrics.sharpe_ratio ? metrics.sharpe_ratio.toFixed(2) : 'N/A'}
          />
          <MetricCard
            title="Max Drawdown"
            value={metrics.max_drawdown ? `${(metrics.max_drawdown * 100).toFixed(1)}%` : 'N/A'}
            valueClassName="text-destructive"
          />
          <MetricCard
            title="Win Rate"
            value={metrics.win_rate ? `${(metrics.win_rate * 100).toFixed(0)}%` : 'N/A'}
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
                <div>
                  <CardTitle>Recent Trades</CardTitle>
                  {trades.length > 0 && trades[0].timestamp && (
                    <p className="text-xs text-muted-foreground mt-1">
                      Latest: {format(new Date(trades[0].timestamp), 'yyyy-MM-dd')}
                    </p>
                  )}
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-8 text-xs"
                  onClick={handleViewAllTrades}
                  disabled={trades.length === 0}
                >
                  View All
                </Button>
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
                                {trade.timestamp ? format(new Date(trade.timestamp), 'yyyy-MM-dd') : '-'}
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
        {/* All Trades Dialog */}
        <Dialog open={showAllTrades} onOpenChange={setShowAllTrades}>
          <DialogContent className="max-w-3xl max-h-[80vh] flex flex-col">
            <DialogHeader>
              <DialogTitle>All Trade Signals — {strategy.name}</DialogTitle>
              <p className="text-sm text-muted-foreground">
                {allTrades.length} total signals | {totalPages} rebalance months | Month {Math.min(tradePage, Math.max(totalPages, 1))}/{Math.max(totalPages, 1)}
              </p>
            </DialogHeader>
            {allTradesLoading ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : (
              <>
                <ScrollArea className="flex-1 min-h-0">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Date</TableHead>
                        <TableHead className="text-right">Monthly PnL</TableHead>
                        <TableHead>Symbol</TableHead>
                        <TableHead>Side</TableHead>
                        <TableHead className="text-right">Weight</TableHead>
                        <TableHead className="text-right">Price</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {pagedTradeGroups.length === 0 ? (
                        <TableRow>
                          <TableCell colSpan={6} className="text-center text-muted-foreground">
                            No trades yet
                          </TableCell>
                        </TableRow>
                      ) : pagedTradeGroups.map(([date, dateTrades]) => (
                        dateTrades.map((trade, idx) => (
                          <TableRow key={trade.id} className={idx === 0 ? "border-t-2 border-border/80" : ""}>
                            {idx === 0 ? (
                              <TableCell rowSpan={dateTrades.length} className="font-mono text-xs font-medium align-top whitespace-nowrap">
                                {format(new Date(date), 'yyyy-MM-dd')}
                                <div className="text-[10px] text-muted-foreground">
                                  {dateTrades.length} signals
                                </div>
                              </TableCell>
                            ) : null}
                            {idx === 0 ? (
                              <TableCell rowSpan={dateTrades.length} className="text-right align-top whitespace-nowrap">
                                <div className={`font-mono text-xs font-medium ${(monthlyPerfMap.get(date)?.monthlyPnl ?? 0) >= 0 ? "text-emerald-500" : "text-rose-500"}`}>
                                  {formatSignedCurrency(monthlyPerfMap.get(date)?.monthlyPnl)}
                                </div>
                                <div className={`text-[10px] ${(monthlyPerfMap.get(date)?.monthlyReturn ?? 0) >= 0 ? "text-emerald-400/90" : "text-rose-400/90"}`}>
                                  {formatSignedPercent(monthlyPerfMap.get(date)?.monthlyReturn)}
                                </div>
                              </TableCell>
                            ) : null}
                            <TableCell className="font-mono text-sm font-medium">
                              {trade.symbol}
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
                            <TableCell className="text-right font-mono text-xs">
                              {trade.price > 0 ? `$${trade.price.toFixed(2)}` : '-'}
                            </TableCell>
                          </TableRow>
                        ))
                      ))}
                    </TableBody>
                  </Table>
                </ScrollArea>

                {/* Pagination */}
                {totalPages > 1 && (
                  <div className="flex items-center justify-between pt-4 border-t">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setTradePage(p => Math.max(1, p - 1))}
                      disabled={tradePage <= 1}
                    >
                      <ChevronLeft className="h-4 w-4 mr-1" /> Prev
                    </Button>
                    <span className="text-sm text-muted-foreground">
                      Month {tradePage} of {totalPages}
                    </span>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setTradePage(p => Math.min(totalPages, p + 1))}
                      disabled={tradePage >= totalPages}
                    >
                      Next <ChevronRight className="h-4 w-4 ml-1" />
                    </Button>
                  </div>
                )}
              </>
            )}
          </DialogContent>
        </Dialog>
      </div>
    </DashboardLayout>
  );
}
