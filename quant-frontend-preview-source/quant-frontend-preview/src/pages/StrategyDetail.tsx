import { useMemo, useState } from "react";
import { useRoute } from "wouter";
import { DashboardLayout } from "@/components/layout/DashboardLayout";
import { generateEquityCurve, generateTrades, generateStrategies } from "@/lib/mockData";
import { MetricCard } from "@/components/ui/metric-card";
import { EquityChart } from "@/components/charts/EquityChart";
import { DrawdownChart } from "@/components/charts/DrawdownChart";
import { HeatmapChart } from "@/components/charts/HeatmapChart";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { 
  Play, 
  Settings2, 
  Download, 
  Share2, 
  Calendar,
  ArrowUpRight,
  ArrowDownRight
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

export default function StrategyDetail() {
  const [match, params] = useRoute("/strategies/:id");
  const id = params?.id;

  // Simulate fetching strategy details
  const strategy = useMemo(() => generateStrategies(1)[0], []);
  const equityData = useMemo(() => generateEquityCurve(365, 10000, 0.02, 0.0005), []);
  const trades = useMemo(() => generateTrades(20), []);

  const [isRunning, setIsRunning] = useState(false);

  const handleRunBacktest = () => {
    setIsRunning(true);
    setTimeout(() => setIsRunning(false), 2000);
  };

  return (
    <DashboardLayout title={strategy.name}>
      <div className="space-y-6">
        {/* Header Actions */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <h2 className="text-3xl font-display font-bold">{strategy.name}</h2>
              <Badge variant="outline" className="text-emerald-500 border-emerald-500/20 bg-emerald-500/10">
                Live Trading
              </Badge>
            </div>
            <p className="text-muted-foreground max-w-2xl">
              {strategy.description}
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
                <>Running...</>
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
            value={`${(strategy.metrics.total_return * 100).toFixed(1)}%`} 
            trend={12.5} 
            prefix="+"
          />
          <MetricCard 
            title="Sharpe Ratio" 
            value={strategy.metrics.sharpe_ratio.toFixed(2)} 
            trend={0.4} 
          />
          <MetricCard 
            title="Max Drawdown" 
            value={`${(strategy.metrics.max_drawdown * 100).toFixed(1)}%`} 
            trend={-2.1} 
            valueClassName="text-destructive"
          />
          <MetricCard 
            title="Win Rate" 
            value={`${(strategy.metrics.win_rate * 100).toFixed(0)}%`} 
            suffix=""
            trend={5.2}
          />
        </div>

        {/* Main Charts Area */}
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
                      <TableHead className="text-right">PnL</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {trades.map((trade) => (
                      <TableRow key={trade.id}>
                        <TableCell className="font-medium font-mono text-xs">
                          {trade.symbol}
                          <div className="text-[10px] text-muted-foreground">
                            {format(new Date(trade.timestamp), 'MM-dd HH:mm')}
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
                        <TableCell className={`text-right font-mono text-xs ${trade.pnl && trade.pnl > 0 ? 'text-emerald-500' : 'text-rose-500'}`}>
                          {trade.pnl ? (trade.pnl > 0 ? '+' : '') + trade.pnl : '-'}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </ScrollArea>
            </CardContent>
          </Card>
        </div>

        {/* Secondary Charts */}
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
      </div>
    </DashboardLayout>
  );
}
