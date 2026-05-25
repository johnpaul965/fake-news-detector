import { useState } from "react";
import { Link } from "wouter";
import { History as HistoryIcon, ChevronLeft, ChevronRight, Search } from "lucide-react";
import { format } from "date-fns";

import { useGetHistory } from "@workspace/api-client-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Progress } from "@/components/ui/progress";

export default function History() {
  const [page, setPage] = useState(1);
  const limit = 10;
  
  const { data, isLoading } = useGetHistory({ limit, offset: (page - 1) * limit });

  const totalPages = data ? Math.ceil(data.total / limit) : 0;

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div>
        <h1 className="text-3xl font-bold tracking-tight mb-2 flex items-center gap-2">
          <HistoryIcon className="h-7 w-7 text-primary" />
          Classification History
        </h1>
        <p className="text-muted-foreground text-lg">
          Review past evaluations and model predictions.
        </p>
      </div>

      <Card className="shadow-sm bg-card/50">
        <CardContent className="p-0">
          <div className="rounded-md overflow-hidden">
            {isLoading ? (
              <div className="divide-y divide-border/50">
                {[1, 2, 3, 4, 5].map((i) => (
                  <div key={i} className="p-4 sm:p-6 flex items-center gap-4">
                    <Skeleton className="h-12 w-12 rounded-full" />
                    <div className="space-y-2 flex-1">
                      <Skeleton className="h-5 w-3/4" />
                      <Skeleton className="h-4 w-1/4" />
                    </div>
                  </div>
                ))}
              </div>
            ) : data?.items.length === 0 ? (
              <div className="text-center py-12 px-4">
                <Search className="h-12 w-12 text-muted mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-foreground">No history yet</h3>
                <p className="text-muted-foreground mt-1 mb-4">You haven't classified any articles yet.</p>
                <Button asChild>
                  <Link href="/">Classify an Article</Link>
                </Button>
              </div>
            ) : (
              <div className="divide-y divide-border/50">
                {data?.items.map((item) => {
                  const isFake = item.prediction === "fake";
                  return (
                    <Link key={item.id} href={`/results/${item.id}`}>
                      <div className="p-4 sm:p-6 flex flex-col sm:flex-row sm:items-center gap-4 hover:bg-muted/40 transition-colors cursor-pointer group">
                        
                        <div className="flex-1 min-w-0">
                          <h3 className="text-base font-semibold text-foreground truncate group-hover:text-primary transition-colors">
                            {item.title}
                          </h3>
                          <div className="text-xs text-muted-foreground mt-1.5 flex items-center gap-3">
                            <span className="font-mono">ID: {item.id.toString().padStart(5, "0")}</span>
                            <span>•</span>
                            <span>{format(new Date(item.createdAt), "MMM d, yyyy HH:mm")}</span>
                          </div>
                        </div>

                        <div className="flex items-center gap-4 sm:min-w-[200px] justify-between sm:justify-end">
                          <div className="flex flex-col items-end gap-1.5 w-full max-w-[120px]">
                            <div className="flex justify-between w-full text-xs font-mono">
                              <span className="text-muted-foreground">Confidence</span>
                              <span className="font-bold">{(item.confidence * 100).toFixed(0)}%</span>
                            </div>
                            <Progress 
                              value={item.confidence * 100} 
                              className={`h-1.5 w-full ${isFake ? "[&>div]:bg-destructive" : "[&>div]:bg-chart-1"}`} 
                            />
                          </div>
                          
                          <Badge 
                            variant={isFake ? "destructive" : "outline"} 
                            className={`w-16 justify-center ${!isFake ? "bg-chart-1/10 text-chart-1 border-chart-1/20" : ""}`}
                          >
                            {item.prediction.toUpperCase()}
                          </Badge>
                        </div>

                      </div>
                    </Link>
                  );
                })}
              </div>
            )}
          </div>

          {totalPages > 1 && (
            <div className="flex items-center justify-between px-6 py-4 border-t border-border/50 bg-muted/20">
              <div className="text-sm text-muted-foreground">
                Page <span className="font-medium text-foreground">{page}</span> of <span className="font-medium text-foreground">{totalPages}</span>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={page === 1}
                >
                  <ChevronLeft className="h-4 w-4 mr-1" />
                  Previous
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                  disabled={page === totalPages}
                >
                  Next
                  <ChevronRight className="h-4 w-4 ml-1" />
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}