import { ExtractedFeatures } from "./features.js";

interface Signal {
  strength: number;
  fakeSentence: string;
  realSentence: string;
}

export function generateExplanation(
  features: ExtractedFeatures,
  prediction: "fake" | "real",
): string {
  const f = features;
  const isFake = prediction === "fake";
  const c = (v: number) => Math.max(0, Math.min(1, v));

  const signals: Signal[] = [
    {
      strength: isFake ? c(f.titleBodyRatio / 0.20) : c(1 - f.titleBodyRatio / 0.20),
      fakeSentence: "its title is disproportionately long relative to the article body — a known clickbait signal",
      realSentence: "its title length is proportional to the article body, consistent with professional journalism",
    },
    {
      strength: isFake ? c(1 - f.avgSentenceLength / 0.45) : c(f.avgSentenceLength / 0.45),
      fakeSentence: "it uses unusually short sentences, typical of sensational or low-quality writing",
      realSentence: "it uses well-structured, longer sentences consistent with professional news reporting",
    },
    {
      strength: isFake ? c(f.exclamationDensity / 0.25) : c(1 - f.exclamationDensity / 0.25),
      fakeSentence: "it contains an elevated density of exclamation marks — a hallmark of emotionally amplified misinformation",
      realSentence: "it avoids excessive exclamation marks, consistent with objective factual reporting",
    },
    {
      strength: isFake ? c(f.capsWordRatio / 0.12) : c(1 - f.capsWordRatio / 0.12),
      fakeSentence: "an unusually high proportion of words are in ALL CAPS, a common sensationalism indicator",
      realSentence: "it avoids excessive capitalization, consistent with measured professional writing",
    },
    {
      strength: isFake ? c(1 - f.quotationFrequency / 0.15) : c(f.quotationFrequency / 0.15),
      fakeSentence: "it contains few or no quoted sources — credible journalism typically cites direct quotes from named persons",
      realSentence: "it contains multiple quoted sources, consistent with professional journalism standards",
    },
    {
      strength: isFake ? c(f.negativeAffectRatio / 0.20) : c(1 - f.negativeAffectRatio / 0.20),
      fakeSentence: "it is heavily loaded with negative emotional language (fear, threat, danger), characteristic of alarmist misinformation",
      realSentence: "it maintains measured emotional language consistent with objective reporting",
    },
    {
      strength: isFake ? c(f.certaintyRatio / 0.15) : c(1 - f.certaintyRatio / 0.15),
      fakeSentence: "it overuses certainty language ('always', 'never', 'proven', 'secret'), typically used to assert credibility without evidence",
      realSentence: "it avoids overconfident absolutes, using measured language typical of verified reporting",
    },
    {
      strength: isFake ? c(1 - f.wordCount / 0.40) : c(f.wordCount / 0.40),
      fakeSentence: "the article is unusually short — misinformation is often thin on verifiable details",
      realSentence: "the article is substantive in length, consistent with thorough factual reporting",
    },
    {
      strength: isFake ? c(1 - f.lexicalDiversity / 0.80) : c(f.lexicalDiversity / 0.80),
      fakeSentence: "it uses a limited and repetitive vocabulary, associated with low-quality or formulaic writing",
      realSentence: "it uses a rich and varied vocabulary, consistent with professional writing standards",
    },
    {
      strength: isFake ? c(f.firstPersonRatio / 0.12) : c(1 - f.firstPersonRatio / 0.12),
      fakeSentence: "it makes heavy use of first-person pronouns, indicating opinion-heavy writing rather than objective journalism",
      realSentence: "it avoids first-person pronouns, maintaining the objective perspective of factual reporting",
    },
    {
      strength: isFake ? c(f.sentimentPolarity / 0.30) : c(1 - f.sentimentPolarity / 0.30),
      fakeSentence: "it exhibits extreme emotional tone rather than the neutral, measured language of factual reporting",
      realSentence: "it maintains a neutral emotional tone consistent with objective news reporting",
    },
    {
      strength: isFake ? c(1 - f.paragraphCount / 0.50) : c(f.paragraphCount / 0.50),
      fakeSentence: "it has very few paragraphs, suggesting thin or rushed content",
      realSentence: "it is well-structured with multiple paragraphs, consistent with thorough journalism",
    },
    {
      strength: isFake ? c(f.adverbRatio / 0.25) : c(1 - f.adverbRatio / 0.25),
      fakeSentence: "it relies heavily on adverbs ('definitely', 'absolutely', 'incredibly'), indicating exaggerated or persuasive writing",
      realSentence: "it uses adverbs sparingly, consistent with factual and measured reporting",
    },
  ];

  const top3 = signals
    .filter((s) => s.strength > 0.15)
    .sort((a, b) => b.strength - a.strength)
    .slice(0, 3);

  if (top3.length === 0) {
    return `This article was classified as ${prediction} based on a combination of subtle linguistic and structural patterns identified by the Random Forest model. No single feature strongly dominates the classification decision.`;
  }

  const reasons = top3.map((s) => (isFake ? s.fakeSentence : s.realSentence));
  const verdict = isFake ? "likely fake" : "likely real";

  let explanation: string;
  if (reasons.length === 1) {
    explanation = `This article was classified as ${verdict} primarily because ${reasons[0]}.`;
  } else if (reasons.length === 2) {
    explanation = `This article was classified as ${verdict} because ${reasons[0]}, and ${reasons[1]}.`;
  } else {
    explanation = `This article was classified as ${verdict} because ${reasons[0]}, ${reasons[1]}, and ${reasons[2]}.`;
  }

  return (
    explanation +
    " This analysis is based on linguistic and structural patterns identified by the Random Forest model trained on 22,933 Philippine news articles."
  );
}

export function featureRecordsToExtracted(
  featureRecords: Array<{ name: string; value: number }>,
): ExtractedFeatures {
  const result: Record<string, number> = {};
  for (const f of featureRecords) {
    result[f.name] = f.value;
  }
  return result as unknown as ExtractedFeatures;
}
