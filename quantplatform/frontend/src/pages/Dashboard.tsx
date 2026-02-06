import { useState, useEffect } from "react";
import { DashboardLayout } from "@/components/layout/DashboardLayout";
import { StrategyCard } from "@/components/StrategyCard";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Filter, Plus, Loader2 } from "lucide-react";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Skeleton } from "@/components/ui/skeleton";
import { getStrategies } from "@/api/strategies";
import type { Strategy } from "@/types";

export default function Dashboard() {
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  const [sort, setSort] = useState("sharpe");

  useEffect(() => {
    loadStrategies();
  }, []);

  const loadStrategies = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getStrategies();
      setStrategies(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load strategies");
      console.error("Failed to load strategies:", err);
    } finally {
      setLoading(false);
    }
  };

  const filteredStrategies = strategies
    .filter(s => s.name.toLowerCase().includes(filter.toLowerCase()))
    .sort((a, b) => {
      const aMetrics = a.metrics || { sharpe_ratio: 0, total_return: 0 };
      const bMetrics = b.metrics || { sharpe_ratio: 0, total_return: 0 };
      if (sort === "sharpe") return bMetrics.sharpe_ratio - aMetrics.sharpe_ratio;
      if (sort === "return") return bMetrics.total_return - aMetrics.total_return;
      return 0;
    });

  return (
    <DashboardLayout title="Strategy Lab">
      <div className="space-y-6">
        {/* Action Bar */}
        <div className="flex flex-col sm:flex-row gap-4 justify-between items-start sm:items-center">
          <Tabs defaultValue="grid" className="w-[200px]">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="grid">Grid</TabsTrigger>
              <TabsTrigger value="list">List</TabsTrigger>
            </TabsList>
          </Tabs>

          <div className="flex gap-2 w-full sm:w-auto">
            <div className="relative flex-1 sm:w-64">
              <Input
                placeholder="Search strategies..."
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
                className="pl-9"
              />
              <Filter className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
            </div>

            <Select value={sort} onValueChange={setSort}>
              <SelectTrigger className="w-[160px]">
                <SelectValue placeholder="Sort by" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="sharpe">Highest Sharpe</SelectItem>
                <SelectItem value="return">Total Return</SelectItem>
                <SelectItem value="drawdown">Low Drawdown</SelectItem>
              </SelectContent>
            </Select>

            <Button className="gap-2">
              <Plus className="h-4 w-4" />
              New Strategy
            </Button>
          </div>
        </div>

        {/* Loading State */}
        {loading && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="space-y-3">
                <Skeleton className="h-[200px] w-full rounded-lg" />
              </div>
            ))}
          </div>
        )}

        {/* Error State */}
        {error && !loading && (
          <div className="text-center py-12">
            <p className="text-destructive mb-4">{error}</p>
            <Button onClick={loadStrategies} variant="outline">
              <Loader2 className="mr-2 h-4 w-4" />
              Retry
            </Button>
          </div>
        )}

        {/* Empty State */}
        {!loading && !error && strategies.length === 0 && (
          <div className="text-center py-12">
            <p className="text-muted-foreground mb-4">No strategies found. Create your first strategy!</p>
            <Button className="gap-2">
              <Plus className="h-4 w-4" />
              New Strategy
            </Button>
          </div>
        )}

        {/* Grid */}
        {!loading && !error && filteredStrategies.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {filteredStrategies.map(strategy => (
              <StrategyCard key={strategy.id} strategy={strategy} />
            ))}
          </div>
        )}
      </div>
    </DashboardLayout>
  );
}
