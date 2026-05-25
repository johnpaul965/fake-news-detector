import { Switch, Route, Router as WouterRouter } from "wouter";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import NotFound from "@/pages/not-found";

import { Shell } from "@/components/layout/Shell";
import Home from "@/pages/home";
import Results from "@/pages/results";
import History from "@/pages/history";
import Features from "@/pages/features";
import Performance from "@/pages/performance";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

function Router() {
  return (
    <Shell>
      <Switch>
        <Route path="/"              component={Home}        />
        <Route path="/results/:id"   component={Results}     />
        <Route path="/history"       component={History}     />
        <Route path="/performance"   component={Performance} />
        <Route path="/features"      component={Features}    />
        <Route                       component={NotFound}    />
      </Switch>
    </Shell>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <WouterRouter base={import.meta.env.BASE_URL?.replace(/\/$/, "") || ""}>
          <Router />
        </WouterRouter>
        <Toaster />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
