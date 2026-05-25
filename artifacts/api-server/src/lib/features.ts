import Sentiment from "sentiment";
import nlp from "compromise";

const sentimentAnalyzer = new Sentiment();

// ──────────────────────────────────────────────────────────────────────────────
// Feature set aligned with paper (Tiu, Denden, Caluza — Leyte Normal University):
//   Linguistic (23): TF-IDF, sentiment polarity, sentiment subjectivity,
//     readability (Flesch, FK Grade, ARI, Gunning Fog, SMOG), lexical diversity,
//     psycholinguistic indicators, POS frequency, stylistic patterns,
//     word count, unique word count, first-person ratio, exclamation/question counts
//   Structural (8): punctuation density, paragraph count, title-body ratio,
//     hyperlink density, quotation frequency, avg sentence length, caps ratio,
//     sentence count
// ──────────────────────────────────────────────────────────────────────────────

export interface ExtractedFeatures {
  // — Linguistic (23) —
  tfidfScore: number;
  sentimentPolarity: number;
  sentimentSubjectivity: number;
  readabilityScore: number;
  fleschKincaidGrade: number;
  ari: number;
  gunningFog: number;
  smogIndex: number;
  lexicalDiversity: number;
  positiveAffectRatio: number;
  negativeAffectRatio: number;
  certaintyRatio: number;
  nounRatio: number;
  verbRatio: number;
  adjectiveRatio: number;
  adverbRatio: number;
  exclamationDensity: number;
  questionDensity: number;
  wordCount: number;
  uniqueWordCount: number;
  firstPersonRatio: number;
  exclamationCount: number;
  questionCount: number;
  // — Structural (8) —
  punctuationDensity: number;
  paragraphCount: number;
  titleBodyRatio: number;
  hyperlinkDensity: number;
  quotationFrequency: number;
  avgSentenceLength: number;
  capsWordRatio: number;
  sentenceCount: number;
}

// ── Psycholinguistic word lists (LIWC-inspired) ───────────────────────────────
const POSITIVE_AFFECT = new Set([
  "love","happy","happiness","joy","joyful","wonderful","excellent","great","good","amazing",
  "beautiful","fantastic","glad","pleased","cheerful","delighted","excited","grateful",
  "hopeful","thankful","blessed","positive","bright","enjoy","success","win","winning",
  "celebrate","celebrated","victory","wonderful","superb","outstanding","magnificent",
  "perfect","incredible","awesome","brilliant","splendid","marvelous","thrilled","proud",
]);

const NEGATIVE_AFFECT = new Set([
  "hate","fear","terrible","awful","horrible","sad","angry","bad","dangerous","evil",
  "corrupt","lie","lies","lying","fraud","fake","conspiracy","worst","crisis","threat",
  "attack","destroy","damage","crime","criminal","cheat","cheating","betrayal","victim",
  "death","dead","dying","kill","killed","violent","violence","war","murder","abuse",
  "scandal","disgrace","shameful","disgusting","outrageous","catastrophe","disaster",
  "stealing","stolen","corrupt","corruption","evil","wicked","malicious","alarming",
]);

const CERTAINTY_WORDS = new Set([
  "always","never","definitely","certainly","absolute","absolutely","guaranteed","must",
  "proven","fact","facts","true","truth","real","obvious","clear","certain","undeniable",
  "confirmed","verified","established","evidence","proven","unquestionable","undeniably",
  "100%","exclusive","secret","shocking","breaking","revealed","exposed","hidden","actually",
  "really","truly","indeed","without doubt","no doubt","clearly",
]);

const FIRST_PERSON = new Set([
  "i","we","my","mine","our","ours","us","myself","ourselves","im","ive","id","ill","were",
]);

// ── Preprocessing helpers ─────────────────────────────────────────────────────
const STOP_WORDS = new Set([
  "the","a","an","and","or","but","in","on","at","to","for","of","with","by","from",
  "is","was","are","were","be","been","being","have","has","had","do","does","did",
  "will","would","could","should","may","might","it","its","this","that","these","those",
  "i","we","you","he","she","they","me","us","him","her","them","my","our","your","his",
  "their","what","which","who","when","where","how","all","as","if","so","not","no",
  "can","than","then","just","more","also","into","over","after","s","re","ve","ll",
]);

function lemmatize(token: string): string {
  let w = token.toLowerCase();
  if (w.endsWith("ies") && w.length > 4) return w.slice(0, -3) + "y";
  if (w.endsWith("ied") && w.length > 4) return w.slice(0, -3) + "y";
  if (w.endsWith("ing") && w.length > 6) return w.slice(0, -3);
  if (w.endsWith("tion") && w.length > 6) return w.slice(0, -4);
  if (w.endsWith("ly") && w.length > 4) return w.slice(0, -2);
  if (w.endsWith("er") && w.length > 4) return w.slice(0, -2);
  if (w.endsWith("ed") && w.length > 4) return w.slice(0, -2);
  if (w.endsWith("s") && w.length > 3 && !w.endsWith("ss")) return w.slice(0, -1);
  return w;
}

function removeDuplicateWords(tokens: string[]): string[] {
  const result: string[] = [];
  for (let i = 0; i < tokens.length; i++) {
    if (i === 0 || tokens[i] !== tokens[i - 1]) result.push(tokens[i]);
  }
  return result;
}

function preprocess(text: string): string[] {
  const tokens = text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 1 && !STOP_WORDS.has(w));
  return removeDuplicateWords(tokens)
    .map(lemmatize)
    .filter((w) => w.length > 1);
}

function rawTokenize(text: string): string[] {
  return text.split(/\s+/).filter((w) => w.length > 0);
}

function getSentences(text: string): string[] {
  return text
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 10);
}

function getParagraphs(text: string): string[] {
  return text
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .filter((p) => p.length > 20);
}

const URL_REGEX = /https?:\/\/[^\s]+/gi;
const QUOTE_REGEX = /["""''][^"""''\n]{5,}["""'']/g;

// ── Syllable counter ──────────────────────────────────────────────────────────
function countSyllables(word: string): number {
  const w = word.toLowerCase().replace(/[^a-z]/g, "");
  if (w.length <= 3) return 1;
  const m = w.match(/[aeiouy]+/g);
  let n = m ? m.length : 1;
  if (w.endsWith("e") && n > 1) n--;
  return Math.max(1, n);
}

// ── TF-IDF approximation ──────────────────────────────────────────────────────
function computeTFIDF(tokens: string[]): number {
  if (tokens.length === 0) return 0;
  const freq: Record<string, number> = {};
  for (const t of tokens) freq[t] = (freq[t] || 0) + 1;
  const values = Object.values(freq);
  const maxFreq = Math.max(...values);
  const tf = maxFreq / tokens.length;
  const uniqueRatio = Object.keys(freq).length / tokens.length;
  const idf = Math.log(1 + 1 / (uniqueRatio + 1e-9));
  return Math.min(1, tf * idf * 15);
}

// ── Readability metrics ───────────────────────────────────────────────────────

// Flesch Reading Ease (0–100, higher = easier)
function fleschEase(words: string[], sentences: string[]): number {
  if (sentences.length === 0 || words.length === 0) return 50;
  const avgSentLen = words.length / sentences.length;
  const avgSyl = words.map(countSyllables).reduce((a, b) => a + b, 0) / words.length;
  return Math.max(0, Math.min(100, 206.835 - 1.015 * avgSentLen - 84.6 * avgSyl));
}

// Flesch-Kincaid Grade Level (paper: linguistic feature)
function fleschKincaidGradeLevel(words: string[], sentences: string[]): number {
  if (sentences.length === 0 || words.length === 0) return 8;
  const avgSentLen = words.length / sentences.length;
  const avgSyl = words.map(countSyllables).reduce((a, b) => a + b, 0) / words.length;
  const fk = 0.39 * avgSentLen + 11.8 * avgSyl - 15.59;
  return Math.max(0, Math.min(18, fk));
}

// Automated Readability Index (paper: linguistic feature)
function automatedReadabilityIndex(words: string[], sentences: string[]): number {
  if (sentences.length === 0 || words.length === 0) return 8;
  const chars = words.reduce((sum, w) => sum + w.replace(/[^a-zA-Z0-9]/g, "").length, 0);
  const avgWordLen = chars / words.length;
  const avgSentLen = words.length / sentences.length;
  const ari = 4.71 * avgWordLen + 0.5 * avgSentLen - 21.43;
  return Math.max(0, Math.min(14, ari));
}

// Gunning Fog Index (paper: linguistic feature)
function gunningFogIndex(words: string[], sentences: string[]): number {
  if (sentences.length === 0 || words.length === 0) return 12;
  const avgSentLen = words.length / sentences.length;
  const complexWords = words.filter((w) => countSyllables(w) >= 3).length;
  const fog = 0.4 * (avgSentLen + 100 * (complexWords / words.length));
  return Math.max(0, Math.min(20, fog));
}

// SMOG Index (paper: linguistic feature)
function smogIndex(words: string[], sentences: string[]): number {
  if (sentences.length < 3 || words.length === 0) return 8;
  const polysyllables = words.filter((w) => countSyllables(w) >= 3).length;
  const smog = 3 + Math.sqrt(polysyllables * 30 / sentences.length);
  return Math.max(0, Math.min(20, smog));
}

// ── Main extraction function ──────────────────────────────────────────────────
export function extractFeatures(title: string, body: string): ExtractedFeatures {
  const fullText = `${title} ${body}`;
  const sentences = getSentences(fullText);
  const rawWords = rawTokenize(fullText);
  const processedTokens = preprocess(fullText);
  const bodyProcessed = preprocess(body);

  // — Lexical diversity (type-token ratio on preprocessed tokens) —
  const lexicalDiversity =
    processedTokens.length > 0
      ? Math.min(1, new Set(processedTokens).size / processedTokens.length)
      : 0;

  // — TF-IDF —
  const tfidfScore = computeTFIDF(bodyProcessed);

  // — Sentiment polarity + subjectivity —
  const sentResult = sentimentAnalyzer.analyze(fullText);
  const rawPolarity = processedTokens.length > 0 ? sentResult.comparative : 0;
  const sentimentPolarity = Math.min(1, Math.abs(rawPolarity) / 2);
  // Subjectivity: proportion of tokens with emotional valence (paper: linguistic feature)
  const subjectiveWords = (sentResult.positive?.length ?? 0) + (sentResult.negative?.length ?? 0);
  const sentimentSubjectivity = Math.min(1, (subjectiveWords / Math.max(1, rawWords.length)) * 6);

  // — Readability suite (paper: linguistic features) —
  const readabilityScore = fleschEase(rawWords, sentences) / 100;
  const fleschKincaidGrade = fleschKincaidGradeLevel(rawWords, sentences) / 18;
  const ari = automatedReadabilityIndex(rawWords, sentences) / 14;
  const gunningFog = gunningFogIndex(rawWords, sentences) / 20;
  const smog = smogIndex(rawWords, sentences) / 20;

  // — Psycholinguistic indicators —
  const processedSet = processedTokens;
  const posCount = processedSet.filter((t) => POSITIVE_AFFECT.has(t)).length;
  const negCount = processedSet.filter((t) => NEGATIVE_AFFECT.has(t)).length;
  const cerCount = processedSet.filter((t) => CERTAINTY_WORDS.has(t)).length;
  const denom = processedSet.length || 1;
  const positiveAffectRatio = Math.min(1, (posCount / denom) * 8);
  const negativeAffectRatio = Math.min(1, (negCount / denom) * 8);
  const certaintyRatio = Math.min(1, (cerCount / denom) * 10);

  // — POS frequency using compromise —
  const doc = nlp(fullText);
  const nounCount = doc.nouns().length;
  const verbCount = doc.verbs().length;
  const adjCount = doc.adjectives().length;
  const advCount = doc.adverbs().length;
  const posTotal = Math.max(1, nounCount + verbCount + adjCount + advCount);
  const nounRatio = nounCount / posTotal;
  const verbRatio = verbCount / posTotal;
  const adjectiveRatio = adjCount / posTotal;
  const adverbRatio = advCount / posTotal;

  // — Stylistic: exclamation & question density + counts —
  const excCount = (fullText.match(/!/g) || []).length;
  const qCount = (fullText.match(/\?/g) || []).length;
  const sentCount = Math.max(1, sentences.length);
  const exclamationDensity = Math.min(1, excCount / sentCount);
  const questionDensity = Math.min(1, qCount / sentCount);
  const exclamationCount = Math.min(1, excCount / 10);
  const questionCount = Math.min(1, qCount / 10);

  // — Word count & unique word count (paper Table III ranks #7 and #9) —
  const wordCount = Math.min(1, rawWords.length / 1000);
  const uniqueWordCount = Math.min(1, new Set(processedTokens).size / 500);

  // — First-person pronoun ratio (paper: linguistic feature) —
  const firstPersonCount = rawWords.filter((w) => FIRST_PERSON.has(w.toLowerCase())).length;
  const firstPersonRatio = Math.min(1, (firstPersonCount / Math.max(1, rawWords.length)) * 15);

  // — Structural features —
  const punctChars = (fullText.match(/[.,!?;:'"()\-–—]/g) || []).length;
  const punctuationDensity = Math.min(1, (punctChars / Math.max(1, fullText.length)) * 12);

  const paragraphs = getParagraphs(body);
  const paragraphCount = Math.min(1, Math.max(1, paragraphs.length) / 30);

  const titleBodyRatio = Math.min(1, (title.length / Math.max(1, body.length)) * 6);

  const urlCount = (body.match(URL_REGEX) || []).length;
  const bodyWordCount = rawTokenize(body).length;
  const hyperlinkDensity = Math.min(1, (urlCount / Math.max(1, bodyWordCount)) * 60);

  const quoteCount = (fullText.match(QUOTE_REGEX) || []).length;
  const quotationFrequency = Math.min(1, quoteCount / Math.max(1, sentCount));

  const avgSentenceLength = Math.min(1, (rawWords.length / sentCount) / 45);

  const capsWords = rawWords.filter((w) => /^[A-Z]{2,}$/.test(w)).length;
  const capsWordRatio = Math.min(1, (capsWords / Math.max(1, rawWords.length)) * 12);

  const sentenceCount = Math.min(1, sentences.length / 30);

  return {
    tfidfScore: round4(tfidfScore),
    sentimentPolarity: round4(sentimentPolarity),
    sentimentSubjectivity: round4(sentimentSubjectivity),
    readabilityScore: round4(readabilityScore),
    fleschKincaidGrade: round4(fleschKincaidGrade),
    ari: round4(ari),
    gunningFog: round4(gunningFog),
    smogIndex: round4(smog),
    lexicalDiversity: round4(lexicalDiversity),
    positiveAffectRatio: round4(positiveAffectRatio),
    negativeAffectRatio: round4(negativeAffectRatio),
    certaintyRatio: round4(certaintyRatio),
    nounRatio: round4(nounRatio),
    verbRatio: round4(verbRatio),
    adjectiveRatio: round4(adjectiveRatio),
    adverbRatio: round4(adverbRatio),
    exclamationDensity: round4(exclamationDensity),
    questionDensity: round4(questionDensity),
    wordCount: round4(wordCount),
    uniqueWordCount: round4(uniqueWordCount),
    firstPersonRatio: round4(firstPersonRatio),
    exclamationCount: round4(exclamationCount),
    questionCount: round4(questionCount),
    punctuationDensity: round4(punctuationDensity),
    paragraphCount: round4(paragraphCount),
    titleBodyRatio: round4(titleBodyRatio),
    hyperlinkDensity: round4(hyperlinkDensity),
    quotationFrequency: round4(quotationFrequency),
    avgSentenceLength: round4(avgSentenceLength),
    capsWordRatio: round4(capsWordRatio),
    sentenceCount: round4(sentenceCount),
  };
}

function round4(n: number): number {
  return parseFloat(n.toFixed(4));
}

// ── Feature metadata ──────────────────────────────────────────────────────────
export const FEATURE_META = [
  // Linguistic (23) — matches paper (Tiu, Denden, Caluza) feature list
  {
    name: "tfidfScore",
    label: "TF-IDF Score",
    description:
      "Term Frequency-Inverse Document Frequency score for the most prominent content words in the article body. Measures how uniquely characteristic the key terms are relative to document length.",
    category: "linguistic" as const,
  },
  {
    name: "sentimentPolarity",
    label: "Sentiment Polarity",
    description:
      "Magnitude of emotional tone extracted from the article text. Fake news often exhibits extreme sentiment polarity — either strongly positive or strongly negative — compared to neutral factual reporting.",
    category: "linguistic" as const,
  },
  {
    name: "sentimentSubjectivity",
    label: "Sentiment Subjectivity",
    description:
      "Proportion of words with clear emotional valence (positive or negative) relative to total words. High subjectivity indicates opinion-driven writing typical of fake news, as opposed to objective factual reporting. (Paper: linguistic feature)",
    category: "linguistic" as const,
  },
  {
    name: "readabilityScore",
    label: "Flesch Reading Ease",
    description:
      "Flesch Reading Ease score measuring text complexity. Calculated from average sentence length and syllables per word. Fake news may use unusually simple or artificially complex language. (Paper: linguistic feature)",
    category: "linguistic" as const,
  },
  {
    name: "fleschKincaidGrade",
    label: "Flesch-Kincaid Grade",
    description:
      "Flesch-Kincaid Grade Level estimates the U.S. school grade level required to understand the text. Formula: 0.39 × (words/sentences) + 11.8 × (syllables/words) − 15.59. Real news tends to score higher (more complex). (Paper: linguistic feature)",
    category: "linguistic" as const,
  },
  {
    name: "ari",
    label: "Automated Readability Index",
    description:
      "ARI uses character count per word and words per sentence to estimate readability grade level: 4.71 × (chars/words) + 0.5 × (words/sentences) − 21.43. Lower scores indicate simpler writing common in fake news. (Paper: linguistic feature)",
    category: "linguistic" as const,
  },
  {
    name: "gunningFog",
    label: "Gunning Fog Index",
    description:
      "Gunning Fog estimates years of formal education needed to understand the text: 0.4 × (words/sentences + 100 × complex_words/words), where complex words have 3+ syllables. Fake news typically scores lower. (Paper: linguistic feature)",
    category: "linguistic" as const,
  },
  {
    name: "smogIndex",
    label: "SMOG Index",
    description:
      "Simple Measure of Gobbledygook — estimates reading grade level from polysyllabic word count: 3 + √(polysyllables × 30 / sentences). Like other readability metrics, fake news tends to use simpler language with lower SMOG scores. (Paper: linguistic feature)",
    category: "linguistic" as const,
  },
  {
    name: "lexicalDiversity",
    label: "Lexical Diversity",
    description:
      "Type-token ratio computed on lemmatized, stop-word-removed tokens. Measures vocabulary richness. Low diversity suggests repetitive or formulaic writing common in low-quality fake news. (Paper: linguistic feature)",
    category: "linguistic" as const,
  },
  {
    name: "positiveAffectRatio",
    label: "Positive Affect Ratio",
    description:
      "Psycholinguistic indicator: proportion of positive emotion words (joy, success, excellent, etc.) relative to all content tokens. Elevated in sensational or propaganda-style fake news.",
    category: "linguistic" as const,
  },
  {
    name: "negativeAffectRatio",
    label: "Negative Affect Ratio",
    description:
      "Psycholinguistic indicator: proportion of negative emotion words (fear, corrupt, dangerous, etc.) relative to all content tokens. High negative affect is a strong marker of alarmist fake news. (Paper: linguistic feature)",
    category: "linguistic" as const,
  },
  {
    name: "certaintyRatio",
    label: "Certainty Word Ratio",
    description:
      "Psycholinguistic indicator: proportion of absolute certainty words (always, never, proven, fact, exclusive, shocking, etc.). Fake news often uses certainty language to assert credibility without evidence. (Paper: linguistic feature)",
    category: "linguistic" as const,
  },
  {
    name: "nounRatio",
    label: "Noun Ratio",
    description:
      "Part-of-speech frequency: proportion of nouns among all tagged words. Factual news articles are noun-heavy, naming people, places, and organizations. Low noun ratio may indicate opinion-heavy content. (Paper: linguistic feature)",
    category: "linguistic" as const,
  },
  {
    name: "verbRatio",
    label: "Verb Ratio",
    description:
      "Part-of-speech frequency: proportion of verbs among all tagged words. High verb density relative to nouns may indicate action-oriented or event-dramatizing writing style. (Paper: linguistic feature)",
    category: "linguistic" as const,
  },
  {
    name: "adjectiveRatio",
    label: "Adjective Ratio",
    description:
      "Part-of-speech frequency: proportion of adjectives. Elevated adjective use signals sensational or evaluative language, a stylistic marker frequently observed in fake news. (Paper: linguistic feature)",
    category: "linguistic" as const,
  },
  {
    name: "adverbRatio",
    label: "Adverb Ratio",
    description:
      "Part-of-speech frequency: proportion of adverbs among all tagged words. Adverbs like 'definitely', 'absolutely', 'incredibly' are associated with exaggerated or persuasive writing. (Paper rank #6: 3.36%)",
    category: "linguistic" as const,
  },
  {
    name: "exclamationDensity",
    label: "Exclamation Density",
    description:
      "Stylistic pattern: number of exclamation marks per sentence. High exclamation density is a well-documented marker of sensational fake news. (Paper rank #4: 5.07%; paper: linguistic feature)",
    category: "linguistic" as const,
  },
  {
    name: "questionDensity",
    label: "Question Density",
    description:
      "Stylistic pattern: number of question marks per sentence. Rhetorical questions are commonly used in fake news to imply scandal without providing verifiable evidence.",
    category: "linguistic" as const,
  },
  {
    name: "wordCount",
    label: "Word Count",
    description:
      "Total word count of the article (normalized). Real news articles tend to be significantly longer than fake news. (Paper rank #7: 3.07%; paper: linguistic feature)",
    category: "linguistic" as const,
  },
  {
    name: "uniqueWordCount",
    label: "Unique Word Count",
    description:
      "Count of unique vocabulary items after lemmatization (normalized). Higher unique word count indicates richer, more professional writing. (Paper rank #9: 2.35%; paper: linguistic feature)",
    category: "linguistic" as const,
  },
  {
    name: "firstPersonRatio",
    label: "First-Person Ratio",
    description:
      "Proportion of first-person pronouns (I, we, my, our, us) in the article. High first-person usage indicates opinion-heavy writing typical of partisan or fake news content. (Paper: linguistic feature)",
    category: "linguistic" as const,
  },
  {
    name: "exclamationCount",
    label: "Exclamation Count",
    description:
      "Raw count of exclamation marks (normalized). Complements exclamation density — absolute count captures total emotional amplification. (Paper rank #5: 3.79%; paper: linguistic feature)",
    category: "linguistic" as const,
  },
  {
    name: "questionCount",
    label: "Question Count",
    description:
      "Raw count of question marks (normalized). Absolute count of rhetorical questions used to cast doubt or imply scandal without evidence. (Paper: linguistic feature)",
    category: "linguistic" as const,
  },
  // Structural (8) — matches paper's 7 structural features + hyperlink density
  {
    name: "punctuationDensity",
    label: "Punctuation Density",
    description:
      "Ratio of punctuation characters to total characters. Unusual punctuation patterns — excessive commas, ellipses, dashes — are structural markers of informal or low-quality writing in fake news. (Paper: structural feature)",
    category: "structural" as const,
  },
  {
    name: "paragraphCount",
    label: "Paragraph Count",
    description:
      "Normalized count of paragraphs in the article body. Fake news articles tend to be shorter with fewer, briefer paragraphs compared to structured journalistic pieces. (Paper: structural feature)",
    category: "structural" as const,
  },
  {
    name: "titleBodyRatio",
    label: "Title-Body Length Ratio",
    description:
      "Ratio of headline character length to body character length. Clickbait fake news often has disproportionately long, sensational titles relative to minimal body content. (Paper rank #1: 12.97%; paper: structural feature)",
    category: "structural" as const,
  },
  {
    name: "hyperlinkDensity",
    label: "Hyperlink Density",
    description:
      "Number of embedded hyperlinks per word in the body text. Legitimate journalism typically cites sources through hyperlinks. Very low hyperlink density suggests absence of verifiable sourcing.",
    category: "structural" as const,
  },
  {
    name: "quotationFrequency",
    label: "Quotation Frequency",
    description:
      "Number of quoted passages (direct speech in quotation marks) per sentence. Real news typically includes direct quotes from named sources. Low quotation frequency indicates poor sourcing. (Paper: structural feature)",
    category: "structural" as const,
  },
  {
    name: "avgSentenceLength",
    label: "Avg Sentence Length",
    description:
      "Average number of words per sentence, normalized. Very short sentences indicate simplistic writing common in fake news. (Paper rank #3: 6.58%; paper: structural feature)",
    category: "structural" as const,
  },
  {
    name: "capsWordRatio",
    label: "All-Caps Word Ratio",
    description:
      "Proportion of words written entirely in ALL CAPS (excluding standard acronyms). Excessive capitalization is a well-established structural marker of sensational fake news. (Paper rank #8: 3.01%; paper: structural feature)",
    category: "structural" as const,
  },
  {
    name: "sentenceCount",
    label: "Sentence Count",
    description:
      "Total number of sentences in the article (normalized). Real news articles have significantly more sentences than fake news. (Paper rank #10: 2.33%; paper: structural feature)",
    category: "structural" as const,
  },
];

export const FEATURE_NAMES = FEATURE_META.map((f) => f.name) as (keyof ExtractedFeatures)[];
