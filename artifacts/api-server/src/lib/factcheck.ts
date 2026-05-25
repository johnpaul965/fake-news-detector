export interface FactCheckResult {
  claimant: string;
  claimText: string;
  claimDate: string | null;
  rating: string;
  publisher: string;
  publisherSite: string;
  url: string;
  reviewDate: string | null;
}

export async function queryFactCheck(title: string): Promise<FactCheckResult[]> {
  const apiKey = process.env.GOOGLE_FACT_CHECK_API_KEY;
  if (!apiKey) return [];

  const keywords = title
    .split(/\s+/)
    .slice(0, 10)
    .join(" ")
    .trim();

  if (!keywords) return [];

  try {
    const url =
      `https://factchecktools.googleapis.com/v1alpha1/claims:search` +
      `?query=${encodeURIComponent(keywords)}&key=${encodeURIComponent(apiKey)}&languageCode=en`;

    const resp = await fetch(url, {
      signal: AbortSignal.timeout(6000),
    });

    if (!resp.ok) return [];

    const data = await resp.json() as {
      claims?: Array<{
        text?: string;
        claimant?: string;
        claimDate?: string;
        claimReview?: Array<{
          publisher?: { name?: string; site?: string };
          url?: string;
          title?: string;
          reviewDate?: string;
          textualRating?: string;
        }>;
      }>;
    };

    if (!data.claims) return [];

    const results: FactCheckResult[] = [];
    for (const claim of data.claims.slice(0, 5)) {
      for (const review of (claim.claimReview ?? []).slice(0, 1)) {
        results.push({
          claimant:      claim.claimant                ?? "Unknown",
          claimText:     claim.text                    ?? "",
          claimDate:     claim.claimDate               ?? null,
          rating:        review.textualRating          ?? "Unknown",
          publisher:     review.publisher?.name        ?? "Unknown",
          publisherSite: review.publisher?.site        ?? "",
          url:           review.url                    ?? "",
          reviewDate:    review.reviewDate             ?? null,
        });
      }
    }

    return results;
  } catch {
    return [];
  }
}
