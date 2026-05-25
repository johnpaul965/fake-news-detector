import { useParams, Link } from "wouter";
import { ArrowLeft, AlertTriangle, ShieldCheck, Activity, BrainCircuit, MessageSquareQuote, ExternalLink, Search } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

import { useGetHistoryItem } from "@workspace/api-client-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";

interface SearchLink {
  label: string;
  url: string;
  snippet: string;
  publisher: string;
  type: "credible" | "factcheck";
}

function buildSearchLinks(title: string, prediction: "fake" | "real"): SearchLink[] {
  const keywords = title
    .split(/\s+/)
    .slice(0, 7)
    .join(" ")
    .replace(/[^a-zA-Z0-9 ]/g, "")
    .trim();
  const q = encodeURIComponent(keywords);

  if (prediction === "real") {
    return [
      {
        label: `Search "${keywords}" on Inquirer`,
        url: `https://newsinfo.inquirer.net/?s=${q}`,
        snippet: "Philippine Daily Inquirer — one of the Philippines' most trusted broadsheets",
        publisher: "inquirer.net",
        type: "credible",
      },
      {
        label: `Search "${keywords}" on Rappler`,
        url: `https://www.rappler.com/search/#stq=${q}`,
        snippet: "Rappler — award-winning Philippine digital news outlet",
        publisher: "rappler.com",
        type: "credible",
      },
      {
        label: `Search "${keywords}" on GMA News`,
        url: `https://www.gmanetwork.com/news/search/?q=${q}`,
        snippet: "GMA Network News — major Philippine broadcast network",
        publisher: "gmanetwork.com",
        type: "credible",
      },
      {
        label: `Search "${keywords}" on Philippine News Agency`,
        url: `https://www.pna.gov.ph/?s=${q}`,
        snippet: "PNA — official Philippine government news agency",
        publisher: "pna.gov.ph",
        type: "credible",
      },
      {
        label: `Search "${keywords}" on Manila Bulletin`,
        url: `https://mb.com.ph/?s=${q}`,
        snippet: "Manila Bulletin — one of the Philippines' oldest newspapers",
        publisher: "mb.com.ph",
        type: "credible",
      },
    ];
  } else {
    return [
      {
        label: `Search "${keywords}" on VERA Files`,
        url: `https://verafiles.org/?s=${q}`,
        snippet: "VERA Files — Philippine fact-checking organization accredited by IFCN",
        publisher: "verafiles.org",
        type: "factcheck",
      },
      {
        label: `Search "${keywords}" on AFP Fact Check PH`,
        url: `https://factcheck.afp.com/?s=${q}`,
        snippet: "AFP Fact Check — global newswire fact-checking team covering the Philippines",
        publisher: "factcheck.afp.com",
        type: "factcheck",
      },
      {
        label: `Search "${keywords}" on Rappler Fact Check`,
        url: `https://www.rappler.com/fact-check/?s=${q}`,
        snippet: "Rappler Fact Check — Philippine digital news outlet's dedicated fact-checking section",
        publisher: "rappler.com",
        type: "factcheck",
      },
      {
        label: `Search "${keywords}" on TSEK.PH`,
        url: `https://tsek.ph/?s=${q}`,
        snippet: "TSEK.PH — collaborative Philippine fact-checking platform",
        publisher: "tsek.ph",
        type: "factcheck",
      },
      {
        label: `Search "${keywords}" on PolitiFact`,
        url: `https://www.politifact.com/search/?q=${q}`,
        snippet: "PolitiFact — Pulitzer Prize-winning international fact-checking organization",
        publisher: "politifact.com",
        type: "factcheck",
      },
    ];
  }
}

export default function Results() {
  const { id } = useParams<{ id: string }>();
  const { data: result, isLoading, isError } = useGetHistoryItem(Number(id));

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-4">
          <Skeleton className="h-10 w-24" />
          <Skeleton className="h-10 w-64" />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Skeleton className="h-48 md:col-span-1" />
          <Skeleton className="h-48 md:col-span-2" />
        </div>
        <Skeleton className="h-96 w-full" />
      </div>
    );
  }

  if (isError || !result) {
    return (
      <div className="text-center py-12">
        <AlertTriangle className="h-12 w-12 text-destructive mx-auto mb-4" />
        <h2 className="text-xl font-bold mb-2">Result not found</h2>
        <p className="text-muted-foreground mb-6">Could not load the classification result.</p>
        <Button asChild variant="outline">
          <Link href="/">Return to Classifier</Link>
        </Button>
      </div>
    );
  }

  const isFake = result.prediction === "fake";
  const confidencePercent = (result.confidence * 100).toFixed(1);

  const sortedFeatures = [...result.features].sort((a, b) => b.importance - a.importance);
  const topFeatures = sortedFeatures.slice(0, 15);

  const linguisticFeatures = result.features.filter((f) => f.category === "linguistic");
  const structuralFeatures = result.features.filter((f) => f.category === "structural");

  const searchLinks = buildSearchLinks(result.title, result.prediction as "fake" | "real");

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500 pb-20">
      <div className="flex items-center gap-4">
        <Button variant="ghost" size="icon" asChild className="rounded-full">
          <Link href="/">
            <ArrowLeft className="h-4 w-4" />
          </Link>
        </Button>
        <h1 className="text-2xl font-bold tracking-tight truncate flex-1" title={result.title}>
          {result.title}
        </h1>
        <div className="text-sm text-muted-foreground font-mono">
          ID: {result.id.toString().padStart(5, "0")}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Verdict Panel */}
        <Card className={`lg:col-span-1 border-t-4 ${isFake ? "border-t-destructive" : "border-t-chart-1"} shadow-md bg-card/50`}>
          <CardHeader className="pb-2">
            <CardDescription className="font-mono text-xs uppercase tracking-wider">
              Random Forest Verdict
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col items-center justify-center py-6 text-center">
            {isFake ? (
              <AlertTriangle className="h-16 w-16 text-destructive mb-4" />
            ) : (
              <ShieldCheck className="h-16 w-16 text-chart-1 mb-4" />
            )}
            <div className={`text-5xl font-black tracking-tighter mb-2 ${isFake ? "text-destructive" : "text-chart-1"}`}>
              {result.prediction.toUpperCase()}
            </div>
            <div className="text-3xl font-mono font-bold text-foreground">
              {confidencePercent}%
            </div>
            <div className="text-sm text-muted-foreground mt-1">Confidence Score</div>
          </CardContent>
        </Card>

        {/* Experiment Comparison */}
        <Card className="lg:col-span-2 shadow-sm bg-card/50">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Activity className="h-5 w-5 text-primary" />
              Configuration Comparison
            </CardTitle>
            <CardDescription>
              Performance across different feature sets for this document
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-3 gap-4 h-full">
              {result.experiments.map((exp) => (
                <div key={exp.configuration} className="flex flex-col items-center justify-center p-4 rounded-lg bg-muted/50 border border-border/50 text-center">
                  <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3">
                    {exp.configuration.replace("-", " ")}
                  </div>
                  <Badge
                    variant={exp.prediction === "fake" ? "destructive" : "outline"}
                    className={exp.prediction === "real" ? "bg-chart-1/10 text-chart-1 border-chart-1/20" : ""}
                  >
                    {exp.prediction.toUpperCase()}
                  </Badge>
                  <div className="mt-4 text-2xl font-mono font-bold">
                    {(exp.confidence * 100).toFixed(1)}%
                  </div>
                  <div className="text-[10px] text-muted-foreground mt-2">
                    {exp.featureCount} features
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Explanation Panel */}
      <Card className={`shadow-sm border-l-4 ${isFake ? "border-l-destructive" : "border-l-chart-1"} bg-card/50`}>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center gap-2">
            <MessageSquareQuote className="h-5 w-5 text-primary" />
            Why this verdict?
          </CardTitle>
          <CardDescription>
            Plain-language explanation of the key signals that drove this classification
          </CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-sm leading-relaxed text-foreground/90">
            {result.explanation}
          </p>
        </CardContent>
      </Card>

      {/* Evidence Panel */}
      <Card className={`shadow-sm border-l-4 ${isFake ? "border-l-destructive/60" : "border-l-chart-1/60"} bg-card/50`}>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center gap-2">
            <Search className="h-5 w-5 text-primary" />
            {isFake ? "Fact-Check Sources" : "Corroborating Sources"}
          </CardTitle>
          <CardDescription>
            {isFake
              ? "Search these Philippine and international fact-checkers to verify this claim"
              : "Search these credible Philippine outlets to corroborate this article"}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {searchLinks.map((link, i) => (
              <div key={i} className={`rounded-md border p-4 space-y-1.5 ${
                link.type === "factcheck"
                  ? "border-destructive/20 bg-destructive/5"
                  : "border-chart-1/20 bg-chart-1/5"
              }`}>
                <div className="flex items-center gap-2 flex-wrap">
                  <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded uppercase tracking-wider ${
                    link.type === "factcheck"
                      ? "bg-destructive/10 text-destructive"
                      : "bg-chart-1/10 text-chart-1"
                  }`}>
                    {link.type === "factcheck" ? "Fact-Checker" : "Credible Source"}
                  </span>
                  <span className="text-xs font-semibold text-muted-foreground">{link.publisher}</span>
                </div>
                <p className="text-sm font-medium text-foreground leading-snug">{link.label}</p>
                <p className="text-xs text-muted-foreground leading-relaxed">{link.snippet}</p>
                <a
                  href={link.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 text-xs text-primary hover:underline"
                >
                  Search this outlet <ExternalLink className="h-3 w-3" />
                </a>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Feature Importance Chart */}
      <Card className="shadow-sm bg-card/50">
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <BrainCircuit className="h-5 w-5 text-primary" />
            Top 15 Feature Importances
          </CardTitle>
          <CardDescription>
            Features contributing most to the classification decision
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[350px] w-full mt-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={topFeatures} layout="vertical" margin={{ top: 5, right: 30, left: 100, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="hsl(var(--border))" />
                <XAxis type="number" hide />
                <YAxis
                  dataKey="name"
                  type="category"
                  axisLine={false}
                  tickLine={false}
                  tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                  width={100}
                />
                <Tooltip
                  cursor={{ fill: "hsl(var(--muted)/0.4)" }}
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="bg-popover border border-border p-3 rounded-md shadow-md text-sm">
                          <div className="font-bold text-foreground mb-1">{data.label} ({data.name})</div>
                          <div className="text-muted-foreground flex justify-between gap-4 mb-1">
                            <span>Category:</span>
                            <span className="capitalize">{data.category}</span>
                          </div>
                          <div className="text-muted-foreground flex justify-between gap-4 mb-1">
                            <span>Value:</span>
                            <span className="font-mono">{data.value.toFixed(4)}</span>
                          </div>
                          <div className="text-muted-foreground flex justify-between gap-4">
                            <span>Importance:</span>
                            <span className="font-mono">{(data.importance * 100).toFixed(2)}%</span>
                          </div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                  {topFeatures.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.category === "linguistic" ? "hsl(var(--chart-1))" : "hsl(var(--chart-2))"}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="flex justify-center gap-6 mt-4 pt-4 border-t border-border/50 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-sm bg-chart-1"></div>
              <span className="text-muted-foreground">Linguistic</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-sm bg-chart-2"></div>
              <span className="text-muted-foreground">Structural</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Feature Details Tabs */}
      <Tabs defaultValue="linguistic" className="w-full">
        <TabsList className="grid w-full grid-cols-2 max-w-md">
          <TabsTrigger value="linguistic">Linguistic Features</TabsTrigger>
          <TabsTrigger value="structural">Structural Features</TabsTrigger>
        </TabsList>

        <TabsContent value="linguistic" className="mt-4">
          <Card className="shadow-sm bg-card/50">
            <CardContent className="p-0">
              <div className="rounded-md border border-border/50 overflow-hidden">
                <table className="w-full text-sm text-left">
                  <thead className="bg-muted/50 text-muted-foreground font-medium border-b border-border/50">
                    <tr>
                      <th className="px-4 py-3">Feature</th>
                      <th className="px-4 py-3">Value</th>
                      <th className="px-4 py-3 text-right">Importance Weight</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border/50">
                    {linguisticFeatures.map((f) => (
                      <tr key={f.name} className="hover:bg-muted/30 transition-colors">
                        <td className="px-4 py-3">
                          <div className="font-medium text-foreground">{f.label}</div>
                          <div className="text-xs text-muted-foreground font-mono mt-0.5">{f.name}</div>
                        </td>
                        <td className="px-4 py-3 font-mono">{f.value.toFixed(4)}</td>
                        <td className="px-4 py-3 font-mono text-right">{(f.importance * 100).toFixed(2)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="structural" className="mt-4">
          <Card className="shadow-sm bg-card/50">
            <CardContent className="p-0">
              <div className="rounded-md border border-border/50 overflow-hidden">
                <table className="w-full text-sm text-left">
                  <thead className="bg-muted/50 text-muted-foreground font-medium border-b border-border/50">
                    <tr>
                      <th className="px-4 py-3">Feature</th>
                      <th className="px-4 py-3">Value</th>
                      <th className="px-4 py-3 text-right">Importance Weight</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border/50">
                    {structuralFeatures.map((f) => (
                      <tr key={f.name} className="hover:bg-muted/30 transition-colors">
                        <td className="px-4 py-3">
                          <div className="font-medium text-foreground">{f.label}</div>
                          <div className="text-xs text-muted-foreground font-mono mt-0.5">{f.name}</div>
                        </td>
                        <td className="px-4 py-3 font-mono">{f.value.toFixed(4)}</td>
                        <td className="px-4 py-3 font-mono text-right">{(f.importance * 100).toFixed(2)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
