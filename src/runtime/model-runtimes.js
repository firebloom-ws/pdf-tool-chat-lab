import { readTensorPreview } from "./safetensors.js";
import { BytePairTokenizer } from "./tokenizer-bpe.js";
import { PixtralVisionPreprocessor } from "./pixtral-preprocessor.js";
import {
  DEFAULT_QWEN_WEBGPU_MODEL,
  QwenWebGpuTextRuntime
} from "./qwen-webgpu-runtime.js";
import { DocToLoraEncoder as DocumentProfileEncoder } from "./doc-to-lora.js";

const LIGHTON_MODEL = "lightonai/LightOnOCR-2-1B-bbox-soup";

function unionBoxes(blocks) {
  return blocks.reduce((merged, block) => {
    if (!merged) {
      return { ...block.bbox };
    }
    const x = Math.min(merged.x, block.bbox.x);
    const y = Math.min(merged.y, block.bbox.y);
    const width = Math.max(merged.x + merged.width, block.bbox.x + block.bbox.width) - x;
    const height = Math.max(merged.y + merged.height, block.bbox.y + block.bbox.height) - y;
    return { x, y, width, height };
  }, null);
}

function cleanSearchQuery(prompt) {
  return prompt
    .replace(/\b(open|show|find|search|jump|go to|take me to|please|the|an?)\b/gi, " ")
    .replace(/\b(what is|what are|how does|how do|explain|tell me|describe|define|give me)\b/gi, " ")
    .replace(/\bpage\s+\d+\b/gi, " ")
    .replace(/[?!]+$/, "")
    .replace(/\s+/g, " ")
    .trim();
}

/** Remove table-of-contents noise: lines with many dots, only numbers, or too short. */
function isNoisyChunk(snippet) {
  const s = snippet.replace(/\s+/g, " ").trim();
  if (s.length < 40) return true;
  const dotRatio = (s.match(/\./g) ?? []).length / s.length;
  if (dotRatio > 0.1) return true;
  const alphaRatio = (s.match(/[a-zA-Z]/g) ?? []).length / s.length;
  if (alphaRatio < 0.35) return true;
  return false;
}

/**
 * Detect chunks that are predominantly math/formulas extracted from PDFs.
 * These are hard to read: lots of isolated single letters/numbers, equals signs, etc.
 */
function isFormulaHeavy(snippet) {
  const s = snippet.replace(/\s+/g, " ").trim();
  const words = s.split(/\s+/);
  if (!words.length) return false;
  // Single-character tokens (like `b k = n k`) are a strong formula signal
  const singleCharTokens = words.filter((w) => /^[a-zA-Z0-9]$/.test(w)).length;
  const singleCharRatio = singleCharTokens / words.length;
  // Math operator density
  const mathOps = (s.match(/[=+\-×÷<>≤≥≈∑∏∫√]/g) ?? []).length;
  const mathRatio = mathOps / Math.max(s.length, 1);
  return singleCharRatio > 0.28 || mathRatio > 0.04;
}

/**
 * Clean up common PDF extraction artifacts for display.
 * - Collapse `b k = ( n ) k` → `bk = (n)k`  (isolated math tokens)
 * - Remove duplicate page number stamps (e.g. ". 37" at end of line)
 * - Normalize whitespace
 */
function cleanSnippet(snippet) {
  return snippet
    .replace(/\s+/g, " ")                          // normalize whitespace
    .replace(/\s+\.\s+\d+\s*$/, "")                // strip trailing ". 37"
    .replace(/\b(\d+)\s*\n/g, "")                  // strip standalone page nums
    .trim();
}

/** Score a chunk for prose quality — higher = more readable. */
function readabilityScore(snippet) {
  if (isFormulaHeavy(snippet)) return 0;
  const s = cleanSnippet(snippet);
  const wordCount = s.split(/\s+/).length;
  const hasVerb = /\b(is|are|was|were|has|have|can|will|define|describe|explain|show|mean|refer|state|present|propose|use)\b/i.test(s);
  const hasSentenceEnd = /[.!?]/.test(s);
  const isDefinition = /\b(is defined as|is a |is an |refers to|we define|means that|can be described)\b/i.test(s);
  let score = Math.min(wordCount / 20, 1);
  if (hasVerb) score += 0.3;
  if (hasSentenceEnd) score += 0.2;
  if (isDefinition) score += 0.5;
  return score;
}

function dedupeByChunk(results) {
  const seen = new Set();
  return results.filter((result) => {
    const key = result.id ?? `${result.pageNumber}:${result.snippet ?? result.text ?? ""}`;
    if (seen.has(key)) {
      return false;
    }
    seen.add(key);
    return true;
  });
}

function dedupeByPage(results) {
  const seen = new Set();
  return results.filter((r) => {
    if (seen.has(r.pageNumber)) return false;
    seen.add(r.pageNumber);
    return true;
  });
}

function buildRegionReferences(results, { limit = 5 } = {}) {
  const filtered = dedupeByChunk(
    (results ?? []).filter((result) => !isNoisyChunk(result.snippet ?? result.text ?? ""))
  );
  const pool = filtered.length ? filtered : dedupeByChunk(results ?? []);

  return pool.slice(0, limit).map((result, index) => ({
    id: `r${index + 1}`,
    chunkId: result.id ?? null,
    pageNumber: result.pageNumber ?? null,
    bbox: result.bbox ?? null,
    label: result.pageNumber ? `p.${result.pageNumber} · r${index + 1}` : `r${index + 1}`,
    snippet: truncateForPrompt(result.snippet ?? result.text ?? "", 120),
    elementType: result.elementType ?? null
  }));
}

function buildReferenceMap(references) {
  return new Map(
    (references ?? [])
      .filter((reference) => reference?.chunkId)
      .map((reference) => [reference.chunkId, reference])
  );
}

function citationForResult(result, referenceMap) {
  const reference = referenceMap.get(result.id);
  return reference ? `[ref:${reference.id}]` : `[p.${result.pageNumber}]`;
}

function describeBboxForTrace(bbox) {
  if (!bbox) {
    return "none";
  }
  return [bbox.x, bbox.y, bbox.width, bbox.height]
    .map((value) => Number(value).toFixed(3))
    .join(",");
}

/**
 * Synthesize a grounded, readable answer from retrieved chunks.
 * Handles summary, list, and default single-best-passage modes.
 * Deprioritizes formula-heavy and noisy chunks.
 */
function synthesizeGroundedAnswer(question, results, references = []) {
  const referenceMap = buildReferenceMap(references);
  // Remove noise, then sort by readability (definitions/prose first, formulas last)
  const readable = results
    .filter((r) => !isNoisyChunk(r.snippet))
    .map((r) => ({ ...r, _score: readabilityScore(r.snippet) }))
    .sort((a, b) => b._score - a._score);

  // If filtering removed everything, fall back to raw (still deduped)
  const pool = dedupeByPage(readable.length >= 1 ? readable : results).slice(0, 5);

  if (!pool.length) {
    return "I couldn’t find relevant content for that. Try rephrasing or asking about a specific section.";
  }

  const q = question.toLowerCase();

  const isSummary =
    /\b(summar|overview|what is this|what(‘s| is) (this|the (paper|doc|article))|about|main (point|topic|idea|finding|contribution|argument)|tl;?dr|in a nutshell)\b/i.test(q);

  const isList =
    /\b(list|enumerate|what are (the|all)|all the|every|each|bullet)\b/i.test(q);

  const isDefinition =
    /\b(what is (a |an |the )?|define |meaning of |explain (what|how)|how does)\b/i.test(q);

  // Extract the topic being asked about for the preamble
  const topicMatch = question.match(/what is (?:a |an |the )?(.+?)[\?\.]*$/i);
  const topic = topicMatch ? topicMatch[1].trim() : null;
  const preamble = topic
    ? `From the document on "${topic}":\n\n`
    : "From the document:\n\n";

  if (isSummary) {
    const paragraphs = pool
      .map((r) => cleanSnippet(r.snippet))
      .join("\n\n");
    const sources = pool
      .map((result) => citationForResult(result, referenceMap))
      .filter((value, index, array) => array.indexOf(value) === index)
      .join(" ");
    return `${preamble}${paragraphs}${sources ? `\n\nSources: ${sources}` : ""}`;
  }

  if (isList) {
    return (
      preamble +
      pool
        .map((r) => `• ${citationForResult(r, referenceMap)} ${cleanSnippet(r.snippet).slice(0, 200)}`)
        .join("\n")
    );
  }

  // Definition / default: lead with the most readable passage
  const best = pool[0];
  const rest = pool.slice(1, 3).filter((r) => !isFormulaHeavy(r.snippet));

  let answer = (isDefinition ? preamble : "") +
    `${citationForResult(best, referenceMap)} ${cleanSnippet(best.snippet)}`;

  if (rest.length) {
    answer +=
      "\n\nRelated:\n" +
      rest
        .map((r) => `• ${citationForResult(r, referenceMap)} ${cleanSnippet(r.snippet).slice(0, 160)}`)
        .join("\n");
  }

  return answer;
}

/** Extract the last substantive user query from conversation history. */
function lastSubstantiveQuery(messages) {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.role === "user" && m.content.trim().split(/\s+/).length > 3) {
      return m.content;
    }
  }
  return null;
}

/** True when the user is expressing dissatisfaction or requesting a retry. */
function isFollowUpCorrection(prompt) {
  return /\b(that’?s? (not|wrong|incorrect|bad)|no[,.]? (that’?s?|it’?s?)|not (what|a |the )|again|retry|redo|try again|more detail|expand|elaborate)\b/i.test(
    prompt
  );
}

function isPointingFollowUp(prompt) {
  return /\b(point (it|that)? out|point me to|show me where|where is that|where is it|open that|open it|take me there|jump there)\b/i.test(
    prompt
  );
}

function isSummaryLikeQuestion(question) {
  return /\b(summar|overview|main point|main idea|what is this about|what is the paper about|whole document|entire document|big picture)\b/i.test(
    question
  );
}

function isBroadAnswerQuestion(question) {
  return /\b(summar|overview|explain|describe|compare|why|how|walk me through|tell me about|what does .* mean)\b/i.test(
    question
  );
}

const PROFILE_STOPWORDS = new Set([
  "about",
  "after",
  "before",
  "between",
  "could",
  "describe",
  "does",
  "explain",
  "from",
  "have",
  "into",
  "just",
  "main",
  "more",
  "over",
  "paper",
  "section",
  "should",
  "show",
  "some",
  "than",
  "that",
  "their",
  "there",
  "these",
  "this",
  "those",
  "what",
  "when",
  "where",
  "which",
  "with",
  "would"
]);

function truncateForPrompt(text, maxChars = 220) {
  const clean = cleanSnippet(text ?? "");
  if (clean.length <= maxChars) {
    return clean;
  }
  return `${clean.slice(0, maxChars).trimEnd()}…`;
}

function looksLikeTruncatedAnswer(text) {
  const trimmed = String(text ?? "").trim();
  if (!trimmed) {
    return false;
  }
  if (/[.!?]["'”’\])}]*$/.test(trimmed)) {
    return false;
  }
  if (/[:;,(\[{/-]$/.test(trimmed)) {
    return true;
  }
  return /[A-Za-z0-9\])\]}]$/.test(trimmed);
}

function mergeContinuationText(existing, addition) {
  const left = String(existing ?? "");
  const right = String(addition ?? "");
  if (!left.trim()) {
    return right.trimStart();
  }
  if (!right.trim()) {
    return left;
  }

  const rightTrimmed = right.replace(/^\s+/, "");
  const maxOverlap = Math.min(80, left.length, rightTrimmed.length);
  let overlap = 0;
  for (let size = maxOverlap; size >= 12; size -= 1) {
    const suffix = left.slice(-size).toLowerCase();
    const prefix = rightTrimmed.slice(0, size).toLowerCase();
    if (suffix === prefix) {
      overlap = size;
      break;
    }
  }

  const remainder = overlap ? rightTrimmed.slice(overlap).replace(/^\s+/, "") : rightTrimmed;
  if (!remainder) {
    return left;
  }
  if (/^[,.;:!?)/\]}]/.test(remainder)) {
    return `${left}${remainder}`;
  }
  if (/\s$/.test(left) || /^\s/.test(right)) {
    return `${left}${remainder}`;
  }
  return `${left} ${remainder}`;
}

function buildContinuationMessages(baseMessages, partialAnswer) {
  return [
    ...baseMessages,
    {
      role: "assistant",
      content: partialAnswer
    },
    {
      role: "user",
      content:
        "Continue exactly where you left off. Do not restart, do not repeat previous sentences, and finish the same answer with the same grounding style."
    }
  ];
}

function extractProfileQueryTerms(question) {
  const tokens = String(question ?? "").toLowerCase().match(/[a-z0-9]{3,}/g) ?? [];
  return [...new Set(tokens.filter((token) => !PROFILE_STOPWORDS.has(token)))];
}

function parseDocumentProfile(memoryPrompt) {
  const lines = String(memoryPrompt ?? "")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);

  const [titleLine = "", ...entryLines] = lines;
  const entries = entryLines
    .map((line) => {
      const match = line.match(/^\[(p\.\d+(?:–\d+)?)\]\s*(.*)$/i);
      const pageLabel = match?.[1] ?? null;
      const text = match?.[2] ?? line;
      const firstPage = pageLabel?.match(/p\.(\d+)/i)?.[1] ?? null;
      return {
        pageLabel,
        pageNumber: firstPage ? Number(firstPage) : null,
        text,
        raw: line
      };
    })
    .filter((entry) => entry.text);

  return {
    titleLine,
    entries
  };
}

function scoreDocumentProfileEntry(entry, queryTerms, retrievedPages) {
  const haystack = `${entry.pageLabel ?? ""} ${entry.text}`.toLowerCase();
  let score = 0;

  for (const term of queryTerms) {
    if (haystack.includes(term)) {
      score += Math.min(term.length, 6);
    }
  }

  if (entry.pageNumber && retrievedPages.has(entry.pageNumber)) {
    score -= 0.75;
  }

  if (isNoisyChunk(entry.text)) {
    score -= 1.5;
  }
  if (isFormulaHeavy(entry.text)) {
    score -= 1.25;
  }

  score += Math.min(entry.text.length / 180, 1);
  return score;
}

function buildDocumentProfileContext(question, results, documentProfile) {
  if (!documentProfile?.memoryPrompt) {
    return {
      prefix: "",
      applied: false,
      reason: "unavailable",
      sections: 0,
      linesUsed: 0
    };
  }

  const parsed = parseDocumentProfile(documentProfile.memoryPrompt);
  const queryTerms = extractProfileQueryTerms(question);
  const retrievedPages = new Set(
    results.map((result) => result.pageNumber).filter((pageNumber) => Number.isFinite(pageNumber))
  );
  const broadQuestion = isSummaryLikeQuestion(question) || results.length === 0;
  const maxEntries = broadQuestion ? 4 : results.length > 0 ? 2 : 3;

  const rankedEntries = parsed.entries
    .map((entry) => ({
      entry,
      score: scoreDocumentProfileEntry(entry, queryTerms, retrievedPages)
    }))
    .sort((left, right) => {
      if (right.score !== left.score) {
        return right.score - left.score;
      }
      return (left.entry.pageNumber ?? Number.MAX_SAFE_INTEGER) - (right.entry.pageNumber ?? Number.MAX_SAFE_INTEGER);
    });

  let selected = rankedEntries.filter((item) => item.score > 0).slice(0, maxEntries);
  let reason = queryTerms.length && selected.length ? "query-match" : broadQuestion ? "broad-context" : "global-context";

  if (!selected.length) {
    selected = rankedEntries.slice(0, Math.min(maxEntries, rankedEntries.length));
    reason = broadQuestion ? "broad-context" : "fallback";
  }

  const lines = [];
  if (parsed.titleLine) {
    lines.push(truncateForPrompt(parsed.titleLine, 90));
  }
  for (const { entry } of selected) {
    const prefix = entry.pageLabel ? `[${entry.pageLabel}] ` : "";
    lines.push(`${prefix}${truncateForPrompt(entry.text, broadQuestion ? 160 : 130)}`);
  }

  if (!lines.length) {
    return {
      prefix: "",
      applied: false,
      reason: "empty",
      sections: parsed.entries.length,
      linesUsed: 0
    };
  }

  return {
    prefix: `Document profile:\n${lines.join("\n")}\n`,
    applied: true,
    reason,
    sections: parsed.entries.length,
    linesUsed: lines.length
  };
}

function buildGroundedChatRequest(messages, question, results, documentProfile = null) {
  const recentTurns = messages.slice(-3, -1);
  const history = recentTurns.length
    ? recentTurns
        .map((message) => `${message.role.toUpperCase()}: ${message.content}`)
        .join("\n")
    : "";

  const filteredResults = results.filter((result) => !isNoisyChunk(result.snippet));
  const contextResults = filteredResults.length ? filteredResults : results;
  const references = buildRegionReferences(contextResults, {
    limit: Math.min(5, Math.max(3, contextResults.length))
  });
  const referenceMap = buildReferenceMap(references);
  const context = contextResults.length
    ? contextResults
        .slice(0, 5)
        .map((result) => {
          const reference = referenceMap.get(result.id);
          const regionToken = reference ? `[ref:${reference.id}] ` : "";
          return `${regionToken}[p.${result.pageNumber}] ${truncateForPrompt(result.snippet, 220)}`;
        })
        .join("\n\n")
    : "No strong matches were found in the current PDF.";

  const documentProfileContext = buildDocumentProfileContext(question, results, documentProfile);

  return {
    documentProfileContext,
    messages: [
      {
        role: "system",
        content:
          "You answer questions about an uploaded PDF. Use only the provided document context. Cite specific regions with tokens like [ref:r1] when the context includes them. You may also mention pages like [p.12]. If the snippets do not support the answer, say that directly."
      },
      {
        role: "user",
        content: [
          documentProfileContext.prefix,
          history ? `Conversation so far:\n${history}\n` : "",
          `Question: ${question}`,
          "",
          "Document context:",
          context,
          "",
          "Write a concise answer grounded in the document."
        ]
          .filter(Boolean)
          .join("\n")
      }
    ],
    references
  };
}

function choosePreviewTensor(manifest) {
  return (
    manifest.tensors.find((tensor) => ["F32", "F16", "BF16"].includes(tensor.dtype)) ??
    manifest.tensors[0] ??
    null
  );
}

function summarizeConfigArchitecture(config, { supportedFeatures = [] } = {}) {
  if (!config) {
    return { name: "unknown", unsupportedFeatures: [] };
  }

  const architectures = config.architectures ?? [];
  const layerTypes =
    config.text_config?.layer_types ??
    config.layer_types ??
    config.model_type ??
    [];
  const unsupportedFeatures = [];
  const supportSet = new Set(supportedFeatures);

  const normalizedLayers = Array.isArray(layerTypes) ? layerTypes : [layerTypes];
  if (normalizedLayers.includes("linear_attention") && !supportSet.has("linear_attention")) {
    unsupportedFeatures.push("linear_attention");
  }
  if (
    (config.architectures ?? []).some((name) => String(name).includes("ConditionalGeneration")) &&
    !supportSet.has("multimodal-conditional-generation")
  ) {
    unsupportedFeatures.push("multimodal-conditional-generation");
  }

  return {
    name: architectures[0] ?? config.model_type ?? "unknown",
    unsupportedFeatures
  };
}

class BrowserModelRuntime {
  constructor({ hubClient, webgpuRuntime, repo, keyFiles, createExecutor = null }) {
    this.hubClient = hubClient;
    this.webgpuRuntime = webgpuRuntime;
    this.repo = repo;
    this.keyFiles = keyFiles;
    this.createExecutor = createExecutor;
    this.state = null;
    this.preparePromise = null;
  }

  async prepare() {
    if (this.state) {
      return this.state;
    }
    if (!this.preparePromise) {
      this.preparePromise = this.#prepareInternal().finally(() => {
        this.preparePromise = null;
      });
    }
    this.state = await this.preparePromise;
    return this.state;
  }

  async #prepareInternal() {
    const [model, tokenizerBundle, config, manifest, gpuInfo] = await Promise.all([
      this.hubClient.describeModel(this.repo, this.keyFiles),
      this.hubClient.loadTokenizerBundle(this.repo),
      this.hubClient.fetchJson(this.repo, "config.json"),
      this.hubClient.inspectSafetensors(this.repo),
      this.webgpuRuntime.probe()
    ]);

    const previewTensor = choosePreviewTensor(manifest);
    let previewValues = [];
    if (previewTensor) {
      previewValues = Array.from(
        await readTensorPreview(this.hubClient, this.repo, previewTensor, 16).catch(() => [])
      ).slice(0, 8);
    }

    let tokenizer = null;
    try {
      tokenizer = BytePairTokenizer.fromModelBundle(tokenizerBundle);
    } catch {
      tokenizer = null;
    }

    let gpuSmokeTest = null;
    if (gpuInfo.available) {
      gpuSmokeTest = await this.webgpuRuntime.smokeTest().catch(() => null);
    }

    let executor = null;
    let supportedFeatures = [];
    let executorInfo = {
      status: "unavailable",
      reason: "executor-not-created"
    };
    if (this.createExecutor) {
      try {
        executor = await this.createExecutor({
          hubClient: this.hubClient,
          webgpuRuntime: this.webgpuRuntime,
          repo: this.repo,
          model,
          config,
          tokenizerBundle,
          tokenizer,
          manifest
        });
        executorInfo = executor
          ? {
              status: "ready",
              reason: "executor-created"
            }
          : {
              status: "skipped",
              reason: "executor-conditions-not-met"
            };
      } catch (error) {
        console.error(`Failed to create executor for ${this.repo}`, error);
        executorInfo = {
          status: "error",
          reason: error instanceof Error ? error.message : String(error)
        };
      }
      supportedFeatures = executor?.supportedFeatures ?? [];
    }

    return {
      model,
      config,
      tokenizerBundle,
      tokenizer,
      manifest,
      previewTensor,
      previewValues,
      gpuInfo,
      gpuSmokeTest,
      executor,
      executorInfo,
      architecture: summarizeConfigArchitecture(config, { supportedFeatures })
    };
  }
}

export class LightOnOCRRuntime {
  constructor(hubClient, webgpuRuntime) {
    this.runtime = new BrowserModelRuntime({
      hubClient,
      webgpuRuntime,
      repo: LIGHTON_MODEL,
      createExecutor: async ({ config }) => {
        if (!config?.vision_config) {
          return null;
        }

        return {
          supportedFeatures: [],
          visionPreprocessor: new PixtralVisionPreprocessor(config)
        };
      },
      keyFiles: [
        "config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer.json"
      ]
    });
  }

  async probe() {
    return this.runtime.prepare();
  }

  async analyzePage(pageRecord, options = {}) {
    const state = this.runtime.state;
    const visionPreprocessor = state?.executor?.visionPreprocessor ?? null;
    let visionPlan = null;

    if (visionPreprocessor && options.renderPageImage) {
      try {
        visionPlan = visionPreprocessor.prepare(await options.renderPageImage());
      } catch (error) {
        console.error("Failed to prepare LightOn/Pixtral vision input", error);
      }
    }

    if (pageRecord.textItems.length === 0) {
      return {
        source:
          state && visionPlan
            ? "hf-assets-loaded/pixtral-vision-preprocessed"
            : state
              ? "hf-assets-loaded/pdf.js-fallback"
              : "pdf.js-fallback",
        blocks: [
          {
            text: visionPlan
              ? `[Image-only page prepared for LightOnOCR vision decoding: ${visionPlan.imageTokenCount} image tokens across a ${visionPlan.patchGrid.rows}x${visionPlan.patchGrid.cols} patch grid]`
              : "[Image-only page pending LightOnOCR WebGPU runtime]",
            bbox: { x: 0.08, y: 0.08, width: 0.84, height: 0.84 }
          }
        ],
        documentBox: { x: 0.08, y: 0.08, width: 0.84, height: 0.84 }
      };
    }

    return {
      source:
        state && visionPlan
          ? "hf-assets-loaded/pixtral-vision-preprocessed/pdf.js-text-layer"
          : state
            ? "hf-assets-loaded/pdf.js-text-layer"
            : "pdf.js-text-layer",
      blocks: pageRecord.textItems,
      documentBox: unionBoxes(pageRecord.textItems)
    };
  }
}

export class QwenToolRuntime {
  constructor(hubClient, webgpuRuntime) {
    this.hubClient = hubClient;
    this.webgpuRuntime = webgpuRuntime;
    this.textRuntime = new QwenWebGpuTextRuntime({
      webgpuRuntime,
      modelId: DEFAULT_QWEN_WEBGPU_MODEL
    });
    this.toolRegistry = null;

    /** Document-profile encoder used by the shipped WebGPU worker path. */
    this.profileEncoder = new DocumentProfileEncoder();
    /** The most-recently compiled adapter (null until encode() completes). */
    this._documentProfile = null;
    /** Listener cleanup for profileEncoder subscription. */
    this._profileUnsub = this.profileEncoder.subscribe(state => {
      this._documentProfile = state.adapter ?? null;
    });
  }

  setToolRegistry(toolRegistry) {
    this.toolRegistry = toolRegistry;
  }

  /**
   * Subscribe to document-profile state changes.
   * Listener receives { state, detail, adapter }.
   * Returns an unsubscribe function.
   */
  subscribeDocumentProfile(listener) {
    return this.profileEncoder.subscribe(listener);
  }

  subscribeLoRA(listener) {
    return this.subscribeDocumentProfile(listener);
  }

  resetDocumentProfile(detail = "") {
    this._documentProfile = null;
    this.profileEncoder.clear(detail);
  }

  /**
   * Encode a document index into a bounded document profile for the active
   * WebGPU worker runtime.
   *
   * Should be called after the document is indexed (index available) and
   * model config is known (i.e. after at least one probe() call so modelId
   * is set).  Safe to call again when a new document is loaded.
   *
   * @param {Object} index      — from buildDocumentIndex
   * @param {Object} options
   * @param {string} options.title       — document title
   * @param {Function} options.onProgress — (fraction, message) callback
   * @returns {Promise<import('./doc-to-lora.js').DocToLoraAdapter>}
   */
  async encodeDocument(index, { title = "document", onProgress } = {}) {
    const adapter = await this.profileEncoder.encode(index, this.textRuntime.modelId, {
      title,
      onProgress,
      // The live WebGPU worker runtime cannot patch model weights, so the
      // shipped path always compiles a bounded document profile instead.
      compileWeights: false
    });
    this._documentProfile = adapter;
    return adapter;
  }

  /** The active document adapter/profile (null if not yet encoded). */
  get documentProfile() {
    return this._documentProfile;
  }

  get loraAdapter() {
    return this._documentProfile;
  }

  /** Human-readable document-profile state string. */
  get documentProfileStatus() {
    const s = this.profileEncoder.state;
    const d = this.profileEncoder.detail;
    if (s === "ready") return `Document profile ready · ${d}`;
    if (s === "encoding") return `Profiling… ${d}`;
    if (s === "error") return `Document profile error: ${d}`;
    return null;
  }

  get loraStatus() {
    return this.documentProfileStatus;
  }

  getModelOptions() {
    return this.textRuntime.getModelOptions();
  }

  getModelState() {
    return this.textRuntime.getState();
  }

  subscribe(listener) {
    return this.textRuntime.subscribe(listener);
  }

  setModel(modelId) {
    this.textRuntime.setModel(modelId);
  }

  async probe() {
    return this.textRuntime.probe();
  }

  async run(messages, options = {}) {
    if (!this.toolRegistry) {
      return {
        mode: "fallback",
        text: "Upload a PDF first, then I can answer questions about it.",
        trace: []
      };
    }

    const prompt = messages.at(-1)?.content ?? "";
    const trace = [];
    let results = [];

    // ── Resolve the effective search query ────────────────────────────
    // If the user is expressing dissatisfaction ("that's not a summary"),
    // re-use the last substantive question instead of searching "summary".
    const effectivePrompt =
      (isFollowUpCorrection(prompt) || isPointingFollowUp(prompt)) && messages.length > 2
        ? (lastSubstantiveQuery(messages.slice(0, -1)) ?? prompt)
        : prompt;

    const explicitPage = prompt.match(/\bpage\s+(\d+)\b/i);

    // ── Tokenize for trace (best-effort) ──────────────────────────────
    trace.push(`tokenize(promptTokens: ${effectivePrompt.split(/\s+/).filter(Boolean).length})`);

    // ── Tool: openPage for explicit page numbers ───────────────────────
    if (explicitPage) {
      const pageNumber = Number(explicitPage[1]);
      await this.toolRegistry.openPage({ pageNumber });
      trace.push(`openPage(pageNumber: ${pageNumber})`);
      const page = this.toolRegistry.peekPage(pageNumber);
      results = page
        ? [{ pageNumber, snippet: page.summary, text: page.summary, bbox: null }]
        : [];
    }

    // ── Tool: search via vector DB ────────────────────────────────────
    const searchQuery = cleanSearchQuery(effectivePrompt);
    if (searchQuery && !explicitPage) {
      // Fetch more candidates so noise + formula filters have room to work
      results = await this.toolRegistry.search(searchQuery, { limit: 12 });
      trace.push(`search(query: "${searchQuery}")`);
    }

    // ── Tool: openPage for strongest result when navigating ───────────
    const shouldNavigate =
      (/\b(open|show|jump|take me|go to|navigate)\b/i.test(prompt) ||
        isPointingFollowUp(prompt)) &&
      results[0];
    if (shouldNavigate) {
      const clean = results.find((r) => !isNoisyChunk(r.snippet)) ?? results[0];
      if (clean?.id) {
        await this.toolRegistry.openChunk({ chunkId: clean.id });
        trace.push(
          `openChunk(chunkId: ${clean.id}, pageNumber: ${clean.pageNumber}, bbox: ${describeBboxForTrace(clean.bbox)})`
        );
      } else {
        await this.toolRegistry.openPage({ pageNumber: clean.pageNumber, bbox: clean.bbox });
        trace.push(
          `openPage(pageNumber: ${clean.pageNumber}, bbox: ${describeBboxForTrace(clean.bbox)})`
        );
      }
    }

    const modelState = this.textRuntime.getState();
    const responseReferences = buildRegionReferences(results, {
      limit: Math.min(5, Math.max(3, results.length || 0))
    });
    if (this._documentProfile) {
      trace.push(
        `docProfile(available: ${this._documentProfile.numGroups} sections, chars: ${this._documentProfile.memoryPrompt.length})`
      );
    } else if (this.profileEncoder.state === "encoding") {
      trace.push(`docProfile(status: encoding…)`);
    }

    if (!modelState.ready) {
      const reason =
        modelState.status === "loading"
          ? "model-loading"
          : modelState.error ?? modelState.status;
      if (this._documentProfile) {
        trace.push("docProfile(skipped: model-not-ready)");
      }
      trace.push(`qwen.generate(skipped: ${reason})`);
      return {
        mode: "grounded",
        text:
          (modelState.status === "loading"
            ? `Qwen ${modelState.label} is still loading on WebGPU, so this answer uses retrieval only for now.\n\n`
            : modelState.status === "error"
              ? `Qwen failed to load (${modelState.error}), so this answer uses retrieval only.\n\n`
              : "Qwen is not loaded yet, so this answer uses retrieval only.\n\n") +
          synthesizeGroundedAnswer(effectivePrompt, results, responseReferences),
        trace,
        references: responseReferences
      };
    }

    try {
      const groundedRequest = buildGroundedChatRequest(
        messages,
        effectivePrompt,
        results,
        this._documentProfile
      );
      const profileContext = groundedRequest.documentProfileContext;
      if (this._documentProfile) {
        trace.push(
          profileContext.applied
            ? `docProfile(applied: ${profileContext.linesUsed} lines, reason: ${profileContext.reason})`
            : `docProfile(skipped: ${profileContext.reason})`
        );
      }
      const baseMaxNewTokens = isBroadAnswerQuestion(effectivePrompt) ? 192 : 144;
      const continuationMaxNewTokens = 96;
      const maxPasses = 3;
      let accumulatedText = "";
      let totalOutputTokens = 0;
      let passesUsed = 0;
      let hitTokenLimit = false;
      let currentMessages = groundedRequest.messages;
      let finalGeneration = null;
      let initialPromptTokens = null;

      for (let passIndex = 0; passIndex < maxPasses; passIndex += 1) {
        const passNumber = passIndex + 1;
        const maxNewTokens = passIndex === 0 ? baseMaxNewTokens : continuationMaxNewTokens;
        const generated = await this.textRuntime.generate({
          messages: currentMessages,
          maxNewTokens,
          onPartial: (partialText) => {
            const merged = mergeContinuationText(accumulatedText, partialText);
            options.onPartial?.(merged);
          }
        });
        finalGeneration = generated;
        passesUsed = passNumber;
        if (initialPromptTokens == null) {
          initialPromptTokens = generated.promptTokens ?? null;
        }
        totalOutputTokens += Number(generated.outputTokens ?? 0);
        hitTokenLimit = Boolean(generated.hitTokenLimit);
        accumulatedText = mergeContinuationText(accumulatedText, generated.text ?? "");

        if (!(generated.text?.trim())) {
          break;
        }

        if (!(generated.hitTokenLimit && looksLikeTruncatedAnswer(accumulatedText))) {
          break;
        }

        const nextPassNumber = passNumber + 1;
        if (nextPassNumber > maxPasses) {
          trace.push("qwen.continue(skipped: pass-cap)");
          break;
        }
        currentMessages = buildContinuationMessages(groundedRequest.messages, accumulatedText);
        trace.push(`qwen.continue(pass: ${nextPassNumber}, reason: token-limit)`);
      }

      if (accumulatedText.trim()) {
        trace.push(
          `qwen.generate(model: ${finalGeneration?.modelId ?? this.textRuntime.modelId}, promptTokens: ${initialPromptTokens ?? "?"}, newTokens: ${totalOutputTokens}, passes: ${passesUsed}${hitTokenLimit && looksLikeTruncatedAnswer(accumulatedText) ? ", capped" : ""})`
        );
        return {
          mode: "webgpu",
          text: accumulatedText.trim(),
          trace,
          references: groundedRequest.references
        };
      }
    } catch (error) {
      console.error("Qwen WebGPU generation failed, using synthesized fallback", error);
      trace.push(
        `qwen.generate(failed: ${error instanceof Error ? error.message : String(error)})`
      );
    }

    // ── Grounded synthesis fallback (always available) ────────────────
    return {
      mode: "grounded",
      text: synthesizeGroundedAnswer(effectivePrompt, results, responseReferences),
      trace,
      references: responseReferences
    };
  }

  interrupt() {
    this.textRuntime.interrupt();
  }
}
