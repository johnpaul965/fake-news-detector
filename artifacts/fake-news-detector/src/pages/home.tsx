import { useState } from "react";
import { useLocation } from "wouter";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { useQueryClient } from "@tanstack/react-query";
import { Search, Loader2, Link, FileText, AlertCircle } from "lucide-react";

import { useClassifyArticle, getGetHistoryQueryKey, getGetModelPerformanceQueryKey } from "@workspace/api-client-react";

import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useToast } from "@/hooks/use-toast";

const MIN_BODY_CHARS = 150;

const formSchema = z.object({
  title: z.string().min(1, "Headline is required").max(500),
  body: z.string().min(10, "Article body must be at least 10 characters"),
});

const urlSchema = z.object({
  url: z.string().url("Please enter a valid URL (e.g. https://inquirer.net/...)"),
});

export default function Home() {
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [isFetching, setIsFetching] = useState(false);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState("url");

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: { title: "", body: "" },
  });

  const urlForm = useForm<z.infer<typeof urlSchema>>({
    resolver: zodResolver(urlSchema),
    defaultValues: { url: "" },
  });

  const classifyMutation = useClassifyArticle();

  function onSubmit(values: z.infer<typeof formSchema>) {
    classifyMutation.mutate(
      { data: values },
      {
        onSuccess: (result) => {
          queryClient.invalidateQueries({ queryKey: getGetHistoryQueryKey() });
          queryClient.invalidateQueries({ queryKey: getGetModelPerformanceQueryKey() });
          setLocation(`/results/${result.id}`);
        },
        onError: () => {
          toast({
            variant: "destructive",
            title: "Classification failed",
            description: "There was an error processing the article.",
          });
        },
      }
    );
  }

  async function onFetchUrl(values: z.infer<typeof urlSchema>) {
    setIsFetching(true);
    setFetchError(null);
    try {
      const resp = await fetch(
        `/api/classify/fetch-url?url=${encodeURIComponent(values.url)}`
      );
      const data = await resp.json();

      if (!resp.ok) {
        setFetchError(data.error ?? "Failed to fetch article.");
        return;
      }

      form.setValue("title", data.title ?? "");
      form.setValue("body", data.body ?? "");
      setActiveTab("paste");
      toast({
        title: "Article fetched",
        description: "Headline and body text have been extracted. Review and click Run Classification.",
      });
    } catch {
      setFetchError("Could not reach the server. Make sure the API is running.");
    } finally {
      setIsFetching(false);
    }
  }

  return (
    <div className="max-w-3xl mx-auto space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div>
        <h1 className="text-3xl font-bold tracking-tight mb-2">News Classifier</h1>
        <p className="text-muted-foreground text-lg">
          Analyze news articles using Random Forest classification against linguistic and structural features.
        </p>
      </div>

      <Card className="border-border/50 shadow-sm bg-card/50 backdrop-blur">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Search className="h-5 w-5 text-primary" />
            Article Analysis
          </CardTitle>
          <CardDescription>
            Paste an article directly or fetch it from a URL to extract features and predict veracity.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="paste" className="flex items-center gap-2">
                <FileText className="h-4 w-4" />
                Paste Text
              </TabsTrigger>
              <TabsTrigger value="url" className="flex items-center gap-2">
                <Link className="h-4 w-4" />
                Fetch from URL
              </TabsTrigger>
            </TabsList>

            {/* ── Tab: Paste Text ── */}
            <TabsContent value="paste">
              <Form {...form}>
                <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
                  <FormField
                    control={form.control}
                    name="title"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel className="text-sm font-semibold">Headline</FormLabel>
                        <FormControl>
                          <Input
                            placeholder="Enter the article headline..."
                            className="font-mono text-sm bg-background/50"
                            {...field}
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                  <FormField
                    control={form.control}
                    name="body"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel className="text-sm font-semibold flex items-center justify-between">
                          <span>Article Body</span>
                          <span className="text-xs font-normal text-muted-foreground">
                            {field.value.length} characters
                          </span>
                        </FormLabel>
                        <FormControl>
                          <Textarea
                            placeholder="Paste the full article text here..."
                            className="min-h-[250px] font-mono text-sm resize-y bg-background/50 leading-relaxed"
                            {...field}
                          />
                        </FormControl>
                        {field.value.length > 0 && field.value.length < MIN_BODY_CHARS ? (
                          <p className="text-sm text-amber-600 dark:text-amber-400">
                            Text is too short for reliable analysis. Paste the full article body — at least {MIN_BODY_CHARS} characters recommended.
                          </p>
                        ) : (
                          <FormDescription>
                            Paste the complete article text for accurate feature extraction.
                          </FormDescription>
                        )}
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                  <Button
                    type="submit"
                    className="w-full h-12 text-base font-semibold tracking-wide"
                    disabled={classifyMutation.isPending}
                  >
                    {classifyMutation.isPending ? (
                      <>
                        <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                        Extracting Features & Classifying...
                      </>
                    ) : (
                      "Run Classification"
                    )}
                  </Button>
                </form>
              </Form>
            </TabsContent>

            {/* ── Tab: Fetch from URL ── */}
            <TabsContent value="url">
              <Form {...urlForm}>
                <form onSubmit={urlForm.handleSubmit(onFetchUrl)} className="space-y-4">
                  <FormField
                    control={urlForm.control}
                    name="url"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel className="text-sm font-semibold">Article URL</FormLabel>
                        <FormControl>
                          <Input
                            placeholder="https://inquirer.net/..."
                            className="font-mono text-sm bg-background/50"
                            type="url"
                            {...field}
                          />
                        </FormControl>
                        <FormDescription>
                          Enter the full URL of the news article. The headline and body text will be extracted automatically.
                        </FormDescription>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  {fetchError && (
                    <div className="flex items-start gap-2 rounded-md border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive">
                      <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
                      <span>{fetchError}</span>
                    </div>
                  )}

                  <div className="rounded-md bg-muted/50 border border-border/40 p-3 text-xs text-muted-foreground space-y-1">
                    <p className="font-medium text-foreground/70">Works best with:</p>
                    <p>Inquirer, Manila Bulletin, Manila Times, Rappler, GMA News, PNA</p>
                    <p className="mt-1">Some sites block automated fetching — use Paste Text tab as fallback.</p>
                  </div>

                  <Button
                    type="submit"
                    className="w-full h-12 text-base font-semibold tracking-wide"
                    disabled={isFetching}
                  >
                    {isFetching ? (
                      <>
                        <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                        Fetching Article...
                      </>
                    ) : (
                      <>
                        <Link className="mr-2 h-5 w-5" />
                        Fetch & Analyze
                      </>
                    )}
                  </Button>
                </form>
              </Form>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
}
