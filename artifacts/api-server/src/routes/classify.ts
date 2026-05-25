import { Router } from "express";
import { db } from "@workspace/db";
import { classificationsTable } from "@workspace/db";
import { ClassifyArticleBody } from "@workspace/api-zod";
import { extractFeatures, FEATURE_META } from "../lib/features.js";
import { classifyArticle } from "../lib/classifier.js";
import { getModelStore } from "../lib/model-store.js";
import { generateExplanation, featureRecordsToExtracted } from "../lib/explainer.js";
import { queryFactCheck } from "../lib/factcheck.js";
import { findCorroboratingSources } from "../lib/websearch.js";
import { eq } from "drizzle-orm";

const router = Router();

// ── POST /api/classify ────────────────────────────────────────────────────────
router.post("/classify", async (req, res) => {
  const parsed = ClassifyArticleBody.safeParse(req.body);
  if (!parsed.success) {
    res.status(400).json({ error: "Invalid input", details: parsed.error.issues });
    return;
  }

  const { title, body } = parsed.data;
  const { linguistic, structural, combined } = getModelStore();

  const features = extractFeatures(title, body);
  const result   = classifyArticle(features, linguistic, structural, combined);

  const explanation = generateExplanation(features, result.prediction);

  const [saved] = await db
    .insert(classificationsTable)
    .values({
      title,
      bodyPreview: body.slice(0, 300),
      prediction:  result.prediction,
      confidence:  result.confidence,
      featuresJson:    result.features,
      experimentsJson: result.experiments,
    })
    .returning();

  res.json({
    id:          saved.id,
    title,
    prediction:  result.prediction,
    confidence:  result.confidence,
    features:    result.features,
    experiments: result.experiments,
    explanation,
    createdAt:   saved.createdAt.toISOString(),
  });
});

// ── GET /api/classify/fetch-url ───────────────────────────────────────────────
router.get("/classify/fetch-url", async (req, res) => {
  const url = String(req.query.url ?? "").trim();

  if (!url || (!url.startsWith("http://") && !url.startsWith("https://"))) {
    res.status(400).json({ error: "A valid http/https URL is required" });
    return;
  }

  try {
    const response = await fetch(url, {
      headers: {
        "User-Agent": "Mozilla/5.0 (compatible; FakeNewsDetector/1.0; research prototype)",
        "Accept": "text/html,application/xhtml+xml",
      },
      signal: AbortSignal.timeout(12000),
    });

    if (!response.ok) {
      res.status(400).json({ error: `Could not fetch URL (HTTP ${response.status})` });
      return;
    }

    const html = await response.text();

    const titleMatch = html.match(/<title[^>]*>([^<]{1,300})<\/title>/i);
    const rawTitle = titleMatch
      ? titleMatch[1].replace(/\s+/g, " ").trim()
      : "";

    const ogTitleMatch = html.match(/<meta[^>]+property=["']og:title["'][^>]+content=["']([^"']{1,300})["']/i)
      || html.match(/<meta[^>]+content=["']([^"']{1,300})["'][^>]+property=["']og:title["']/i);
    const headline = ogTitleMatch ? ogTitleMatch[1].trim() : rawTitle;

    const linkMatches = html.match(/<a\s[^>]*href=["']https?:\/\/[^"']+["']/gi) || [];
    const linkCount = linkMatches.length;

    const stripped = html
      .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, " ")
      .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, " ")
      .replace(/<nav[^>]*>[\s\S]*?<\/nav>/gi, " ")
      .replace(/<header[^>]*>[\s\S]*?<\/header>/gi, " ")
      .replace(/<footer[^>]*>[\s\S]*?<\/footer>/gi, " ")
      .replace(/<aside[^>]*>[\s\S]*?<\/aside>/gi, " ")
      .replace(/<form[^>]*>[\s\S]*?<\/form>/gi, " ")
      .replace(/<!--[\s\S]*?-->/g, " ")
      .replace(/<[^>]+>/g, " ")
      .replace(/&nbsp;/g, " ")
      .replace(/&amp;/g, "&")
      .replace(/&lt;/g, "<")
      .replace(/&gt;/g, ">")
      .replace(/&quot;/g, '"')
      .replace(/&#\d+;/g, " ")
      .replace(/\s{2,}/g, " ")
      .trim();

    const candidates = stripped
      .split(/(?<=[.!?])\s+/)
      .map((s) => s.trim())
      .filter((s) => s.length > 40 && s.split(/\s+/).length > 6);

    const body = candidates.slice(0, 80).join(" ").trim();

    if (body.length < 100) {
      res.status(422).json({
        error: "Could not extract enough article text from this URL. Try copying and pasting the text manually.",
      });
      return;
    }

    res.json({ title: headline, body, url, linkCount });
  } catch (err: any) {
    if (err?.name === "TimeoutError" || err?.name === "AbortError") {
      res.status(504).json({ error: "Request timed out. The website took too long to respond." });
    } else {
      res.status(500).json({ error: "Failed to fetch or parse the URL. The site may block automated requests." });
    }
  }
});

// ── GET /api/corroborate ──────────────────────────────────────────────────────
router.get("/corroborate", async (req, res) => {
  const q = String(req.query.q ?? "").trim();
  const prediction = String(req.query.prediction ?? "").trim() as "fake" | "real";
  if (!q || !["fake", "real"].includes(prediction)) {
    res.status(400).json({ error: "Parameters 'q' and 'prediction' (fake|real) are required" });
    return;
  }
  const results = await findCorroboratingSources(q, prediction);
  res.json({ results });
});

// ── GET /api/fact-check ───────────────────────────────────────────────────────
router.get("/fact-check", async (req, res) => {
  const q = String(req.query.q ?? "").trim();
  if (!q) {
    res.status(400).json({ error: "Query parameter 'q' is required" });
    return;
  }
  const results = await queryFactCheck(q);
  res.json({ results });
});

// ── GET /api/history ──────────────────────────────────────────────────────────
router.get("/history", async (req, res) => {
  const limit  = Math.min(100, parseInt(String(req.query.limit  ?? "20"), 10) || 20);
  const offset =               parseInt(String(req.query.offset ?? "0"),  10) || 0;

  const items = await db
    .select()
    .from(classificationsTable)
    .orderBy(classificationsTable.createdAt)
    .limit(limit)
    .offset(offset);

  const total = await db.$count(classificationsTable);

  res.json({
    items: items.map((r) => ({
      id:         r.id,
      title:      r.title,
      prediction: r.prediction,
      confidence: r.confidence,
      createdAt:  r.createdAt.toISOString(),
    })),
    total,
  });
});

// ── GET /api/history/:id ──────────────────────────────────────────────────────
router.get("/history/:id", async (req, res) => {
  const id = parseInt(req.params.id, 10);
  if (isNaN(id)) { res.status(400).json({ error: "Invalid id" }); return; }

  const [item] = await db
    .select()
    .from(classificationsTable)
    .where(eq(classificationsTable.id, id))
    .limit(1);

  if (!item) { res.status(404).json({ error: "Not found" }); return; }

  const storedFeatures = item.featuresJson as Array<{ name: string; value: number }>;
  const extractedFeatures = featureRecordsToExtracted(storedFeatures);
  const explanation = generateExplanation(extractedFeatures, item.prediction as "fake" | "real");

  res.json({
    id:          item.id,
    title:       item.title,
    prediction:  item.prediction,
    confidence:  item.confidence,
    features:    item.featuresJson,
    experiments: item.experimentsJson,
    explanation,
    createdAt:   item.createdAt.toISOString(),
  });
});

// ── GET /api/model/features ───────────────────────────────────────────────────
router.get("/model/features", (_req, res) => {
  res.json(FEATURE_META);
});

// ── GET /api/model/performance ────────────────────────────────────────────────
router.get("/model/performance", (_req, res) => {
  const { performance } = getModelStore();
  res.json(performance);
});

export default router;
