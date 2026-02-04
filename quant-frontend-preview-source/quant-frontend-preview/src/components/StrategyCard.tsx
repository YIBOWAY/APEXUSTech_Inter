import type { Strategy } from "@/types";
import { Card, CardContent, CardFooter, CardHeader } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { MetricCard } from "@/components/ui/metric-card";
import { ArrowUpRight, ArrowDownRight, Activity } from "lucide-react";
import { Link } from "wouter";

interface StrategyCardProps {
  strategy: Strategy;
}

export function StrategyCard({ strategy }: StrategyCardProps) {
  const isPositive = strategy.metrics.total_return > 0;

  return (
    <Card className="flex flex-col overflow-hidden transition-all hover:border-primary/50 hover:shadow-lg group">
      <CardHeader className="p-5 pb-2">
        <div className="flex justify-between items-start">
          <div className="space-y-1">
            <h3 className="font-display font-bold text-xl tracking-tight group-hover:text-primary transition-colors">
              {strategy.name}
            </h3>
            <div className="flex gap-2">
              {strategy.tags.map(tag => (
                <Badge key={tag} variant="secondary" className="text-xs font-normal">
                  {tag}
                </Badge>
              ))}
            </div>
          </div>
          <Badge 
            variant={strategy.status === 'active' ? 'default' : 'outline'}
            className={strategy.status === 'active' ? 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20' : ''}
          >
            {strategy.status === 'active' ? 'Live' : 'Paused'}
          </Badge>
        </div>
      </CardHeader>
      
      <CardContent className="p-5 grid grid-cols-2 gap-4">
        <div>
          <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">Total Return</p>
          <div className={`text-2xl font-bold font-mono tracking-tighter ${isPositive ? 'text-success' : 'text-destructive'}`}>
            {isPositive ? '+' : ''}{(strategy.metrics.total_return * 100).toFixed(1)}%
          </div>
        </div>
        <div>
          <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">Sharpe</p>
          <div className="text-2xl font-bold font-mono tracking-tighter text-foreground">
            {strategy.metrics.sharpe_ratio.toFixed(2)}
          </div>
        </div>
        <div>
          <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">Drawdown</p>
          <div className="text-lg font-mono text-muted-foreground">
            {(strategy.metrics.max_drawdown * 100).toFixed(1)}%
          </div>
        </div>
        <div>
          <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">Win Rate</p>
          <div className="text-lg font-mono text-muted-foreground">
            {(strategy.metrics.win_rate * 100).toFixed(0)}%
          </div>
        </div>
      </CardContent>
      
      <CardFooter className="p-2 bg-secondary/30 mt-auto">
        <Link href={`/strategies/${strategy.id}`} className="w-full">
          <Button variant="ghost" className="w-full justify-between group-hover:bg-primary group-hover:text-primary-foreground transition-colors">
            View Analysis
            <Activity className="w-4 h-4 opacity-50" />
          </Button>
        </Link>
      </CardFooter>
    </Card>
  );
}
