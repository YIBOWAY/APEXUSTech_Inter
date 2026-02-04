import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { ArrowDownRight, ArrowUpRight } from "lucide-react";

interface MetricCardProps {
  title: string;
  value: string | number;
  trend?: number;
  trendLabel?: string;
  icon?: React.ReactNode;
  className?: string;
  valueClassName?: string;
  prefix?: string;
  suffix?: string;
}

export function MetricCard({
  title,
  value,
  trend,
  trendLabel = "vs last month",
  icon,
  className,
  valueClassName,
  prefix = "",
  suffix = ""
}: MetricCardProps) {
  const isPositive = trend && trend > 0;
  const isNeutral = trend === 0;

  return (
    <Card className={cn("overflow-hidden border-border bg-card shadow-sm hover:shadow-md transition-shadow", className)}>
      <CardContent className="p-6">
        <div className="flex items-center justify-between space-y-0 pb-2">
          <p className="text-sm font-medium text-muted-foreground tracking-wide uppercase">
            {title}
          </p>
          {icon && <div className="text-muted-foreground">{icon}</div>}
        </div>
        <div className="flex items-baseline gap-1">
          {prefix && <span className="text-xl font-medium text-muted-foreground">{prefix}</span>}
          <div className={cn("text-3xl font-bold font-display tracking-tight text-foreground", valueClassName)}>
            {value}
          </div>
          {suffix && <span className="text-lg font-medium text-muted-foreground">{suffix}</span>}
        </div>
        {trend !== undefined && (
          <div className="mt-2 flex items-center text-xs">
            <span
              className={cn(
                "flex items-center font-medium",
                isPositive ? "text-success" : isNeutral ? "text-muted-foreground" : "text-destructive"
              )}
            >
              {isPositive ? <ArrowUpRight className="mr-1 h-3 w-3" /> : <ArrowDownRight className="mr-1 h-3 w-3" />}
              {Math.abs(trend).toFixed(2)}%
            </span>
            <span className="ml-2 text-muted-foreground">{trendLabel}</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
