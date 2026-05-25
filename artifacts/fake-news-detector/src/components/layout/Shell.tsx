import { Link, useLocation } from "wouter";
import { Search, History, FlaskConical, BarChart2 } from "lucide-react";
import { cn } from "@/lib/utils";

const navItems = [
  { href: "/",            label: "Classifier",  icon: Search       },
  { href: "/history",     label: "History",     icon: History      },
  { href: "/performance", label: "Performance", icon: BarChart2    },
  { href: "/features",    label: "Features",    icon: FlaskConical },
];

export function Shell({ children }: { children: React.ReactNode }) {
  const [location] = useLocation();

  return (
    <div className="min-h-screen flex w-full flex-col bg-background">
      <header className="sticky top-0 z-40 w-full border-b bg-card/80 backdrop-blur-md">
        <div className="container mx-auto flex h-14 items-center gap-4 px-4 sm:px-6">
          <div className="flex items-center gap-2 mr-6">
            <div className="h-6 w-6 rounded bg-primary text-primary-foreground flex items-center justify-center font-mono text-xs font-bold">
              FN
            </div>
            <span className="font-semibold text-sm tracking-tight">Fake News Detector</span>
            <span className="ml-2 rounded bg-muted px-1.5 py-0.5 text-[10px] font-mono text-muted-foreground">
              PROTOTYPE
            </span>
          </div>

          <nav className="flex items-center gap-1 overflow-x-auto">
            {navItems.map((item) => {
              const isActive =
                location === item.href ||
                (item.href !== "/" && location.startsWith(item.href));
              const Icon = item.icon;
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  data-testid={`nav-${item.label.toLowerCase()}`}
                  className={cn(
                    "flex items-center gap-2 px-3 py-1.5 text-sm font-medium transition-colors rounded-md hover:bg-muted whitespace-nowrap",
                    isActive
                      ? "text-foreground bg-muted"
                      : "text-muted-foreground",
                  )}
                >
                  <Icon className="h-4 w-4" />
                  {item.label}
                </Link>
              );
            })}
          </nav>
        </div>
      </header>

      <main className="flex-1 container mx-auto px-4 py-8 sm:px-6 max-w-6xl">
        {children}
      </main>
    </div>
  );
}
