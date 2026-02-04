import { useState, useMemo } from "react";
import { DashboardLayout } from "@/components/layout/DashboardLayout";
import { generateStrategies } from "@/lib/mockData";
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
import { Filter, Plus } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function Dashboard() {
  const strategies = useMemo(() => generateStrategies(8), []);
  const [filter, setFilter] = useState("");
  const [sort, setSort] = useState("sharpe");

  const filteredStrategies = strategies
    .filter(s => s.name.toLowerCase().includes(filter.toLowerCase()))
    .sort((a, b) => {
      if (sort === "sharpe") return b.metrics.sharpe_ratio - a.metrics.sharpe_ratio;
      if (sort === "return") return b.metrics.total_return - a.metrics.total_return;
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

        {/* Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {filteredStrategies.map(strategy => (
            <StrategyCard key={strategy.id} strategy={strategy} />
          ))}
        </div>
      </div>
    </DashboardLayout>
  );
}
