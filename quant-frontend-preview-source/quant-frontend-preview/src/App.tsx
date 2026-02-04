import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Router, Route, Switch } from "wouter";
import { useHashLocation } from "wouter/use-hash-location";
import ErrorBoundary from "@/components/ErrorBoundary";
import { ThemeProvider } from "@/contexts/ThemeContext";
import Dashboard from "@/pages/Dashboard";
import StrategyDetail from "@/pages/StrategyDetail";
import { Button } from "@/components/ui/button";
import { Link } from "wouter";

function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-background text-foreground">
      <h1 className="text-4xl font-display font-bold mb-4">404</h1>
      <p className="text-muted-foreground mb-8">Page not found</p>
      <Link href="/">
        <Button>Return Home</Button>
      </Link>
    </div>
  );
}

function AppRouter() {
  return (
    <Router hook={useHashLocation}>
      <Switch>
        <Route path="/" component={Dashboard} />
        <Route path="/strategies" component={Dashboard} />
        <Route path="/strategies/:id" component={StrategyDetail} />
        <Route path="/compare">
          <div className="flex flex-col items-center justify-center min-h-screen bg-background text-foreground">
            <h1 className="text-2xl font-display font-bold mb-2">Comparison Module</h1>
            <p className="text-muted-foreground">Coming soon in Phase 2</p>
            <Link href="/" className="mt-4"><Button variant="outline">Back to Dashboard</Button></Link>
          </div>
        </Route>
        <Route component={NotFound} />
      </Switch>
    </Router>
  );
}

function App() {
  return (
    <ErrorBoundary>
      <ThemeProvider defaultTheme="dark">
        <TooltipProvider>
          <Toaster />
          <AppRouter />
        </TooltipProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;
