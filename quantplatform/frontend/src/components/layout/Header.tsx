import { useTheme } from "@/contexts/ThemeContext";
import { Button } from "@/components/ui/button";
import { Moon, Sun, Bell, Search } from "lucide-react";
import { Input } from "@/components/ui/input";

export function Header({ title }: { title?: string }) {
  const { theme, toggleTheme } = useTheme();

  return (
    <header className="flex h-14 items-center gap-4 border-b border-border bg-background/50 px-6 backdrop-blur-sm sticky top-0 z-10">
      <div className="flex-1">
        <h1 className="text-lg font-semibold font-display tracking-tight text-foreground">
          {title || "Dashboard"}
        </h1>
      </div>
      
      <div className="flex items-center gap-2 md:gap-4">
        <div className="relative hidden sm:block w-64">
          <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            type="search"
            placeholder="Search strategies..."
            className="h-9 w-full rounded-md border border-input bg-background pl-9 focus-visible:ring-1"
          />
        </div>
        
        <Button variant="ghost" size="icon" className="h-9 w-9">
          <Bell className="h-4 w-4" />
        </Button>
        
        <Button 
          variant="ghost" 
          size="icon" 
          className="h-9 w-9" 
          onClick={toggleTheme}
        >
          {theme === "dark" ? (
            <Sun className="h-4 w-4" />
          ) : (
            <Moon className="h-4 w-4" />
          )}
          <span className="sr-only">Toggle theme</span>
        </Button>
      </div>
    </header>
  );
}
