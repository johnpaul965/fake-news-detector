import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { FlaskConical, Target, ShieldCheck, AlertTriangle } from "lucide-react";
import { useGetModelPerformance } from "@workspace/api-client-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";

const CONFIG_LABELS: Record<string, string> = {
  "linguistic-only": "Linguistic Only",
  "structural-only": "Structural Only",
  combined:          "Combined",
};

const METRIC_KEYS = [
  { key: "accuracy",  label: "Accuracy"  },
  { key: "precision", label: "Precision" },
  { key: "recall",    label: "Recall"    },
  { key: "f1Score",   label: "F1 Score"  },
  { key: "rocAuc",    label: "ROC-AUC"   },
] as const;

function MetricBadge({ value }: { value: number }) {
  const pct = (value * 100).toFixed(2);
  const color = value >= 0.85 ? "text-chart-1" : value >= 0.75 ? "text-yellow-500" : "text-destructive";
  return <span className={`font-mono font-bold ${color}`}>{pct}%</span>;
}

function ConfusionMatrix({ matrix }: { matrix: [[number, number], [number, number]] }) {
  // matrix = [[TN, FP], [FN, TP]]
  const [[tn, fp], [fn, tp]] = matrix;
  return (
    <div className="space-y-1">
      <div className="text-[10px] text-muted-foreground uppercase tracking-wider mb-2">
        Confusion Matrix
      </div>
      <div className="grid grid-cols-[auto_1fr_1fr] gap-1 text-xs">
        <div />
        <div className="text-center text-muted-foreground font-medium pb-1">Pred Fake</div>
        <div className="text-center text-muted-foreground font-medium pb-1">Pred Real</div>
        <div className="text-muted-foreground font-medium pr-2 flex items-center">Actual Fake</div>
        <div
          data-testid="matrix-tp"
          className="text-center p-2 rounded bg-chart-1/15 border border-chart-1/30 font-mono font-bold text-chart-1"
        >
          {tp}<div className="text-[9px] font-normal text-muted-foreground">TP</div>
        </div>
        <div
          data-testid="matrix-fn"
          className="text-center p-2 rounded bg-destructive/10 border border-destructive/20 font-mono text-destructive/80"
        >
          {fn}<div className="text-[9px] font-normal text-muted-foreground">FN</div>
        </div>
        <div className="text-muted-foreground font-medium pr-2 flex items-center">Actual Real</div>
        <div
          data-testid="matrix-fp"
          className="text-center p-2 rounded bg-destructive/10 border border-destructive/20 font-mono text-destructive/80"
        >
          {fp}<div className="text-[9px] font-normal text-muted-foreground">FP</div>
        </div>
        <div
          data-testid="matrix-tn"
          className="text-center p-2 rounded bg-chart-1/15 border border-chart-1/30 font-mono font-bold text-chart-1"
        >
          {tn}<div className="text-[9px] font-normal text-muted-foreground">TN</div>
        </div>
      </div>
    </div>
  );
}

export default function Performance() {
  const { data: perf, isLoading } = useGetModelPerformance();

  if (isLoading) {
    return (
      <div className="space-y-6">
        <Skeleton className="h-10 w-64" />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[0, 1, 2].map((i) => <Skeleton key={i} className="h-72" />)}
        </div>
        <Skeleton className="h-80" />
      </div>
    );
  }

  if (!perf || !perf.experiments?.length) return null;

  const combined = perf.experiments.find((e) => e.configuration === "combined");

  // Derive class counts from combined confusion matrix: [[TN, FP], [FN, TP]]
  const cm = combined?.confusionMatrix;
  const fakeCount = cm ? cm[0][0] + cm[0][1] : null;
  const realCount = cm ? cm[1][0] + cm[1][1] : null;
  const classLabel = fakeCount != null && realCount != null
    ? `${fakeCount} fake, ${realCount} real`
    : "50% fake, 50% real";

  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500 pb-10">
      <div>
        <h1 className="text-3xl font-bold tracking-tight mb-2 flex items-center gap-2">
          <FlaskConical className="h-7 w-7 text-primary" />
          Model Evaluation
        </h1>
        <p className="text-muted-foreground text-base">
          Stratified {perf.folds}-fold cross-validation results across three experimental configurations.
          Trained on {perf.trainingSamples} samples ({classLabel}).
        </p>
      </div>

      {/* Combined summary stats */}
      {combined && (
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
          {METRIC_KEYS.map(({ key, label }) => (
            <Card key={key} className="bg-card/50 shadow-sm text-center">
              <CardContent className="pt-4 pb-3">
                <div className="text-xs text-muted-foreground uppercase tracking-wider mb-1">{label}</div>
                <div className="text-2xl"><MetricBadge value={(combined as any)[key]} /></div>
                <div className="text-[10px] text-muted-foreground mt-0.5">Combined</div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Per-configuration experiment results */}
      <div>
        <h2 className="text-lg font-semibold mb-3">Configuration Comparison (Ablation Study)</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {perf.experiments.map((exp) => (
            <Card
              key={exp.configuration}
              data-testid={`card-experiment-${exp.configuration}`}
              className={`bg-card/50 shadow-sm border-t-4 ${
                exp.configuration === "combined"
                  ? "border-t-primary"
                  : exp.configuration === "linguistic-only"
                  ? "border-t-chart-1"
                  : "border-t-chart-2"
              }`}
            >
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center justify-between">
                  {CONFIG_LABELS[exp.configuration]}
                  <Badge variant="outline" className="font-mono text-[10px]">
                    {exp.featureCount} features
                  </Badge>
                </CardTitle>
                <CardDescription className="font-mono text-xs uppercase tracking-wider">
                  {exp.configuration}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <table className="w-full text-sm">
                  <tbody className="divide-y divide-border/40">
                    {METRIC_KEYS.map(({ key, label }) => (
                      <tr key={key} className="flex justify-between py-1">
                        <td className="text-muted-foreground">{label}</td>
                        <td><MetricBadge value={(exp as any)[key]} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <ConfusionMatrix matrix={exp.confusionMatrix as [[number, number], [number, number]]} />
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      {/* Confusion matrix legend */}
      <Card className="bg-muted/30 border-border/40">
        <CardContent className="py-4">
          <div className="flex flex-wrap gap-6 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-chart-1/50 border border-chart-1/50" />
              <span><strong>TP</strong> — True Positive: correctly identified as Fake</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-chart-1/50 border border-chart-1/50" />
              <span><strong>TN</strong> — True Negative: correctly identified as Real</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-destructive/30 border border-destructive/30" />
              <span><strong>FP</strong> — False Positive: Real classified as Fake</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-destructive/30 border border-destructive/30" />
              <span><strong>FN</strong> — False Negative: Fake classified as Real</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Feature Importances */}
      <Card className="bg-card/50 shadow-sm">
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Target className="h-5 w-5 text-primary" />
            Feature Importances
          </CardTitle>
          <CardDescription>
            Mean decrease in impurity across all Random Forest trees, calibrated from literature
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-[500px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={perf.featureImportances}
                layout="vertical"
                margin={{ top: 5, right: 40, left: 130, bottom: 5 }}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  horizontal={true}
                  vertical={false}
                  stroke="hsl(var(--border))"
                />
                <XAxis
                  type="number"
                  tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                  tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                />
                <YAxis
                  dataKey="label"
                  type="category"
                  axisLine={false}
                  tickLine={false}
                  tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
                  width={130}
                />
                <Tooltip
                  cursor={{ fill: "hsl(var(--muted)/0.3)" }}
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const d = payload[0].payload;
                      return (
                        <div className="bg-popover border border-border p-3 rounded-md shadow-md text-sm">
                          <div className="font-bold mb-1">{d.label}</div>
                          <div className="text-muted-foreground flex justify-between gap-4">
                            <span>Category:</span>
                            <span className="capitalize">{d.category}</span>
                          </div>
                          <div className="text-muted-foreground flex justify-between gap-4">
                            <span>Importance:</span>
                            <span className="font-mono">{(d.importance * 100).toFixed(2)}%</span>
                          </div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                  {perf.featureImportances.map((entry, i) => (
                    <Cell
                      key={`cell-${i}`}
                      fill={
                        entry.category === "linguistic"
                          ? "hsl(var(--chart-1))"
                          : "hsl(var(--chart-2))"
                      }
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="flex justify-center gap-6 mt-4 pt-4 border-t border-border/50 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-sm bg-chart-1" />
              <span className="text-muted-foreground">Linguistic</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-sm bg-chart-2" />
              <span className="text-muted-foreground">Structural</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
