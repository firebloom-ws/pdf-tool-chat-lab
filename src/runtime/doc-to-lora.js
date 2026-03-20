/**
 * Document-profile / Doc-to-LoRA compatibility encoder.
 *
 * The active browser runtime ships a bounded document profile that is injected
 * into the prompt. A legacy custom JS Qwen path can still emit per-layer
 * LoRA-style matrices, but that path is not the one used by the WebGPU worker.
 *
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │  Architecture                                                            │
 * │                                                                          │
 * │  PDF chunks ──► selectMetaChunks ──► metaEmbeddings ──► per-layer A, B  │
 * │                       (≤8 groups)     (hash-projected)    (combined avg) │
 * │                                                                          │
 * │  Two inference paths:                                                    │
 * │  ① Qwen35TextEngine  — inject A,B into matVec at runtime (true LoRA)    │
 * │  ② WebGPU worker     — prepend compressed document profile to the prompt │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 * Key design decisions (addressing known instabilities):
 *
 * MAX_ADAPTER_CHUNKS = 8
 *   Training was done with ≤8 chunks of ≤512 tokens each (≤2048 chars).
 *   Naïve combine_lora concatenates along the rank dimension —
 *   effective rank grows linearly with chunks → garbled generation at n > 8.
 *   We enforce the limit by grouping pages into ≤8 meta-groups.
 *
 * COMBINATION: weighted average, NOT concatenation
 *   Concatenation:  combined_A = [A₁; A₂; …; Aₙ]   → rank = n × r   ← BAD
 *   Averaging:      combined_A = Σ wᵢ × Aᵢ           → rank = r      ← GOOD
 *   Weights = readability score per meta-group (prose > formula/noise).
 *
 * LORA_SCALE = (α / r) × dampFactor
 *   We use alpha=16, rank=8 but dampen by 0.04 so LoRA deltas are small
 *   relative to base-model activations.  The model's RAG context still
 *   drives answers; LoRA biases attention toward document-relevant directions.
 *
 * HASH-PROJECTED MATRICES (no stored projection matrix)
 *   The 384-dim document embeddings are projected into (dIn, dOut) spaces
 *   using FNV1a hashing — the same trick as HashLayoutEmbedder.
 *   Memory: O(r × max(dIn,dOut)) per matrix, never the full projection matrix.
 *   Deterministic: same document always produces the same adapter.
 */

// ─── Constants ────────────────────────────────────────────────────────────────

export const LORA_RANK = 8;
export const LORA_ALPHA = 16;

/** Conservative scale factor: (α/r) × 0.04 = 0.08 */
export const LORA_SCALE = (LORA_ALPHA / LORA_RANK) * 0.04;

/** Max meta-chunks — must match training distribution (≤8) */
export const MAX_ADAPTER_CHUNKS = 8;

/** Max chars per chunk before grouping (≈512 tokens) */
export const MAX_CHARS_PER_CHUNK = 2048;

/**
 * Known Qwen3.5 hidden-state dimensions for each model variant.
 * Used for LoRA matrix sizing in the Qwen35TextEngine path.
 * head_dim is derived if absent.
 */
const QWEN35_CONFIGS = {
  "onnx-community/Qwen3.5-0.8B-ONNX": {
    hidden_size: 1024,
    num_hidden_layers: 28,
    num_attention_heads: 16,
    num_key_value_heads: 8,
    head_dim: 64
  },
  "onnx-community/Qwen3.5-2B-ONNX": {
    hidden_size: 1536,
    num_hidden_layers: 28,
    num_attention_heads: 16,
    num_key_value_heads: 8,
    head_dim: 96
  },
  "onnx-community/Qwen3.5-4B-ONNX": {
    hidden_size: 2560,
    num_hidden_layers: 36,
    num_attention_heads: 32,
    num_key_value_heads: 8,
    head_dim: 80
  }
};

export function resolveModelConfig(modelId) {
  return QWEN35_CONFIGS[modelId] ?? QWEN35_CONFIGS["onnx-community/Qwen3.5-0.8B-ONNX"];
}

// ─── Math utilities ───────────────────────────────────────────────────────────

function fnv1a(str, seed = 2166136261) {
  let h = (seed | 0) >>> 0;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h;
}

function l2Normalize(vec) {
  let sum = 0;
  for (let i = 0; i < vec.length; i++) sum += vec[i] * vec[i];
  if (sum < 1e-12) return vec;
  const inv = 1 / Math.sqrt(sum);
  for (let i = 0; i < vec.length; i++) vec[i] *= inv;
  return vec;
}

/**
 * Hash-project srcVec into a tgtDim-dimensional vector.
 *
 * Each source dimension i gets mapped to a target bucket via FNV1a(i, seed).
 * Sign is determined by a second hash so contributions cancel when uncorrelated.
 * Memory: O(tgtDim) — the full projection matrix is never materialised.
 */
function hashProject(srcVec, tgtDim, seed) {
  const out = new Float32Array(tgtDim);
  for (let i = 0; i < srcVec.length; i++) {
    const bucket = fnv1a(String(i), seed) % tgtDim;
    const sign = fnv1a(String(i), seed + 1) & 1 ? 1 : -1;
    out[bucket] += sign * srcVec[i];
  }
  return l2Normalize(out);
}

// ─── Adapter matrix derivation ────────────────────────────────────────────────

/**
 * Derive A ∈ ℝ^{rank × dIn} from a document meta-embedding.
 *
 * Each row r is a dIn-dimensional direction that the document embedding
 * "suggests" should be amplified at this layer and projection.
 * Rows are L2-normalised so scale comes purely from B and the LoRA scale factor.
 */
function deriveLoraA(docVec, dIn, rank, seedBase) {
  const A = new Float32Array(rank * dIn);
  for (let r = 0; r < rank; r++) {
    const row = hashProject(docVec, dIn, fnv1a(`A${r}`, seedBase));
    A.set(row, r * dIn);
  }
  return A;
}

/**
 * Derive B ∈ ℝ^{dOut × rank} from a document meta-embedding.
 *
 * Each column r is a dOut-dimensional "output direction" for the document.
 * Initialised small (×0.1) — the LoRA scale factor does the final scaling.
 */
function deriveLoraB(docVec, dOut, rank, seedBase) {
  const B = new Float32Array(dOut * rank);
  for (let r = 0; r < rank; r++) {
    const col = hashProject(docVec, dOut, fnv1a(`B${r}`, seedBase));
    for (let i = 0; i < dOut; i++) {
      B[i * rank + r] = col[i] * 0.1;
    }
  }
  return B;
}

// ─── LoRA application ─────────────────────────────────────────────────────────

/**
 * Apply a LoRA delta in place: output += scale × B @ (A @ input)
 *
 * Returns output unchanged (same reference) if dimensions don't match —
 * ensures a bad adapter never corrupts generation.
 *
 * @param {Float32Array} output  — existing projection output [dOut]
 * @param {Float32Array} input   — layer input [dIn]
 * @param {Object} lora         — { A, B, rank, scale }
 * @returns {Float32Array}
 */
export function applyLoraToVector(output, input, lora) {
  if (!lora) return output;
  const { A, B, rank, scale } = lora;
  const dIn = input.length;
  const dOut = output.length;

  if (A.length !== rank * dIn || B.length !== dOut * rank) {
    // Dimension mismatch — silently skip rather than corrupt
    return output;
  }

  // ax = A @ input  [rank]
  const ax = new Float32Array(rank);
  for (let r = 0; r < rank; r++) {
    let s = 0;
    const off = r * dIn;
    for (let j = 0; j < dIn; j++) s += A[off + j] * input[j];
    ax[r] = s;
  }

  // output += scale × (B @ ax)  [dOut]
  for (let i = 0; i < dOut; i++) {
    let delta = 0;
    const off = i * rank;
    for (let r = 0; r < rank; r++) delta += B[off + r] * ax[r];
    output[i] += scale * delta;
  }
  return output;
}

// ─── Chunk quality scoring ────────────────────────────────────────────────────

function isFormulaHeavy(text) {
  const words = text.replace(/\s+/g, " ").trim().split(/\s+/);
  if (!words.length) return false;
  const singleChar = words.filter(w => /^[a-zA-Z0-9]$/.test(w)).length / words.length;
  const mathOps = (text.match(/[=+\-×÷<>≤≥≈∑∏∫√]/g) ?? []).length / Math.max(text.length, 1);
  return singleChar > 0.28 || mathOps > 0.04;
}

function chunkReadabilityScore(chunk) {
  const s = (chunk.text ?? "").replace(/\s+/g, " ").trim();
  if (!s || s.length < 40) return 0;
  if (isFormulaHeavy(s)) return 0;

  const words = s.split(/\s+/);
  let score = Math.min(words.length / 20, 1.0);
  if (/\b(is|are|was|were|has|have|can|will|define|present|describe)\b/i.test(s)) score += 0.3;
  if (/[.!?]/.test(s)) score += 0.2;
  if (/\b(is defined as|is a |is an |refers to|we define)\b/i.test(s)) score += 0.5;
  return score;
}

// ─── Meta-chunk selection ─────────────────────────────────────────────────────

/**
 * Group document chunks into ≤maxGroups meta-groups.
 *
 * Steps:
 * 1. Filter out pure formula / noise chunks (keep prose-heavy ones).
 * 2. If filtered pool is too small, fall back to all chunks.
 * 3. Split sorted pool into equal-size bins.
 * 4. Return at most maxGroups bins (≤8 to match training distribution).
 *
 * This ensures:
 * - We respect the ≤8 chunk training distribution.
 * - High-quality (prose) sections are prioritised.
 * - Long documents are proportionally sampled (not arbitrarily truncated).
 */
export function selectMetaChunks(chunks, maxGroups = MAX_ADAPTER_CHUNKS) {
  if (!chunks?.length) return [];

  const prose = chunks.filter(c => chunkReadabilityScore(c) > 0);
  const pool = prose.length >= Math.min(4, Math.ceil(chunks.length * 0.2)) ? prose : [...chunks];

  if (pool.length <= maxGroups) {
    return pool.map(c => [c]);
  }

  const binSize = Math.ceil(pool.length / maxGroups);
  const groups = [];
  for (let i = 0; i < pool.length && groups.length < maxGroups; i += binSize) {
    groups.push(pool.slice(i, i + binSize));
  }
  return groups;
}

/**
 * Compute a readability-weighted average embedding for a group of chunks.
 * Returns null if no chunks have valid embeddings.
 */
export function buildMetaEmbedding(group) {
  const valid = group.filter(c => c.vector?.length);
  if (!valid.length) return null;

  const dim = valid[0].vector.length;
  const weights = valid.map(c => Math.max(chunkReadabilityScore(c), 0.05));
  const total = weights.reduce((a, b) => a + b, 0);

  const meta = new Float32Array(dim);
  for (let i = 0; i < valid.length; i++) {
    const w = weights[i] / total;
    for (let j = 0; j < dim; j++) {
      meta[j] += w * valid[i].vector[j];
    }
  }
  return l2Normalize(meta);
}

// ─── Single-chunk adapter construction ───────────────────────────────────────

/**
 * Extract attention projection dimensions from a Qwen3.5-style text config.
 */
function extractAttentionDims(textConfig) {
  const h = textConfig.hidden_size;
  const headDim = textConfig.head_dim ?? Math.floor(h / textConfig.num_attention_heads);
  const nKv = textConfig.num_key_value_heads;
  const nAttn = textConfig.num_attention_heads;
  return {
    dModel: h,
    qOutDim: h * 2,           // q_proj output includes gating: [hidden*2, hidden]
    kvOutDim: nKv * headDim,  // k_proj / v_proj: [nKv * headDim, hidden]
    oInDim: nAttn * headDim,  // o_proj input:  [hidden, nAttn * headDim]
    oOutDim: h               // o_proj output: [hidden]
  };
}

/**
 * Build per-layer LoRA matrices for a single meta-embedding.
 * Returns an array of layer adapter objects indexed by layer number.
 */
function buildSingleChunkAdapter(metaVec, textConfig) {
  const numLayers = textConfig.num_hidden_layers;
  const { dModel, qOutDim, kvOutDim, oInDim, oOutDim } = extractAttentionDims(textConfig);
  const layers = [];

  for (let l = 0; l < numLayers; l++) {
    const layerSeed = fnv1a(`layer_${l}`);
    layers.push({
      q: {
        A: deriveLoraA(metaVec, dModel, LORA_RANK, fnv1a("q_A", layerSeed)),
        B: deriveLoraB(metaVec, qOutDim, LORA_RANK, fnv1a("q_B", layerSeed)),
        rank: LORA_RANK,
        scale: LORA_SCALE
      },
      k: {
        A: deriveLoraA(metaVec, dModel, LORA_RANK, fnv1a("k_A", layerSeed)),
        B: deriveLoraB(metaVec, kvOutDim, LORA_RANK, fnv1a("k_B", layerSeed)),
        rank: LORA_RANK,
        scale: LORA_SCALE
      },
      v: {
        A: deriveLoraA(metaVec, dModel, LORA_RANK, fnv1a("v_A", layerSeed)),
        B: deriveLoraB(metaVec, kvOutDim, LORA_RANK, fnv1a("v_B", layerSeed)),
        rank: LORA_RANK,
        scale: LORA_SCALE
      },
      o: {
        A: deriveLoraA(metaVec, oInDim, LORA_RANK, fnv1a("o_A", layerSeed)),
        B: deriveLoraB(metaVec, oOutDim, LORA_RANK, fnv1a("o_B", layerSeed)),
        rank: LORA_RANK,
        scale: LORA_SCALE
      }
    });
  }
  return layers;
}

// ─── Adapter combination ─────────────────────────────────────────────────────

/**
 * Combine multiple per-chunk adapters via readability-weighted averaging.
 *
 * WHY NOT CONCATENATION (the naive approach):
 *   combine_lora pads A along row axis: [A₁; A₂; …; Aₙ] → rank = n × r
 *   For n = 10 chunks, effective rank grows from 8 to 80.
 *   This is far outside the training distribution (max 8 chunks, rank 8)
 *   and causes the base model's generation to collapse or loop.
 *
 * WEIGHTED AVERAGE keeps rank fixed at LORA_RANK regardless of chunk count.
 *
 * @param {{ layers: Array, weight: number }[]} chunkedAdapters
 * @returns {Array} combined layer adapters
 */
function combineAdapters(chunkedAdapters) {
  if (!chunkedAdapters.length) return [];
  if (chunkedAdapters.length === 1) return chunkedAdapters[0].layers;

  const totalWeight = chunkedAdapters.reduce((s, c) => s + c.weight, 0) || 1;
  const numLayers = chunkedAdapters[0].layers.length;
  const combined = [];

  for (let l = 0; l < numLayers; l++) {
    const layerOut = {};
    for (const proj of ["q", "k", "v", "o"]) {
      const ref = chunkedAdapters[0].layers[l][proj];
      const cA = new Float32Array(ref.A.length);
      const cB = new Float32Array(ref.B.length);

      for (const chunk of chunkedAdapters) {
        const w = chunk.weight / totalWeight;
        const src = chunk.layers[l][proj];
        for (let i = 0; i < cA.length; i++) cA[i] += w * src.A[i];
        for (let i = 0; i < cB.length; i++) cB[i] += w * src.B[i];
      }
      layerOut[proj] = { A: cA, B: cB, rank: ref.rank, scale: ref.scale };
    }
    combined.push(layerOut);
  }
  return combined;
}

// ─── Memory prompt generation (transformers.js fallback) ─────────────────────

/**
 * Build a compact document profile for the transformers.js ONNX path.
 *
 * Since the ONNX model can't have weights patched at runtime, we inject
 * the document encoding as a compressed system-level summary. This gives
 * the model a bounded, reusable document profile without the token cost of
 * replaying the full document on every turn.
 *
 * @param {Array} metaGroups  — array of chunk arrays
 * @param {string} title      — document title
 * @returns {string}
 */
function buildMemoryPrompt(metaGroups, title) {
  const lines = [`Document: "${title}"`];

  for (let i = 0; i < metaGroups.length; i++) {
    const group = metaGroups[i];
    const best = group
      .slice()
      .sort((a, b) => chunkReadabilityScore(b) - chunkReadabilityScore(a))[0];
    if (!best) continue;

    const text = best.text?.replace(/\s+/g, " ").trim() ?? "";
    if (text.length < 30) continue;

    const pages = [...new Set(group.map(c => c.pageNumber))].sort((a, b) => a - b);
    const pageLabel = pages.length === 1
      ? `p.${pages[0]}`
      : `p.${pages[0]}–${pages[pages.length - 1]}`;

    // Truncate to ~200 chars for each section summary
    lines.push(`[${pageLabel}] ${text.slice(0, 200)}${text.length > 200 ? "…" : ""}`);
  }

  return lines.join("\n");
}

// ─── DocToLoraAdapter ─────────────────────────────────────────────────────────

/**
 * A compiled LoRA adapter derived from a loaded PDF document.
 *
 * Contains:
 * - Per-layer A, B matrices for Q/K/V/O attention projections
 *   (used by Qwen35TextEngine when injecting into custom JS forward pass)
 * - A compressed document profile prompt
 *   (used by the transformers.js path as an additional system prefix)
 */
export class DocToLoraAdapter {
  constructor({ layers, rank, scale, numGroups, title, encodingMs, memoryPrompt, mode = "document-profile" }) {
    this.layers = layers;           // Array<{ q, k, v, o: { A, B, rank, scale } }>
    this.rank = rank;
    this.scale = scale;
    this.numGroups = numGroups;
    this.title = title;
    this.encodingMs = encodingMs;
    this.memoryPrompt = memoryPrompt;
    this.mode = mode;
  }

  /** Per-layer adapter for a given layer index (null-safe). */
  getLayer(layerIndex) {
    return this.layers[layerIndex] ?? null;
  }

  get numLayers() {
    return this.layers.length;
  }

  /** Estimated in-memory footprint of all A, B matrices. */
  get memoryBytes() {
    if (!this.layers.length) return 0;
    let total = 0;
    for (const l of this.layers) {
      for (const p of ["q", "k", "v", "o"]) {
        total += l[p].A.byteLength + l[p].B.byteLength;
      }
    }
    return total;
  }

  formatMemory() {
    const b = this.memoryBytes;
    return b < 1024 * 1024
      ? `${(b / 1024).toFixed(0)} KB`
      : `${(b / (1024 * 1024)).toFixed(1)} MB`;
  }

  /** Short human-readable summary. */
  describe() {
    if (this.mode === "document-profile") {
      return (
        `document profile · ${this.numGroups} section${this.numGroups !== 1 ? "s" : ""} ` +
        `· ${this.memoryPrompt.length} chars · ${this.encodingMs}ms`
      );
    }
    return (
      `rank=${this.rank} · ${this.numGroups} section${this.numGroups !== 1 ? "s" : ""} ` +
      `· ${this.numLayers} layers · ${this.formatMemory()} · ${this.encodingMs}ms`
    );
  }
}

// ─── DocToLoraEncoder ─────────────────────────────────────────────────────────

/**
 * Encodes a document index into a DocToLoraAdapter.
 *
 * Usage:
 *   const encoder = new DocToLoraEncoder();
 *   const unsub = encoder.subscribe(state => updateUI(state));
 *   const adapter = await encoder.encode(index, "onnx-community/Qwen3.5-0.8B-ONNX", { title });
 *   engine.setLoraAdapter(adapter);
 */
export class DocToLoraEncoder {
  constructor() {
    this.state = "idle";   // idle | encoding | ready | error
    this.detail = "";
    this.lastAdapter = null;
    this._listeners = new Set();
  }

  /** Subscribe to encoding state changes. Returns unsubscribe fn. */
  subscribe(listener) {
    this._listeners.add(listener);
    listener({ state: this.state, detail: this.detail, adapter: this.lastAdapter });
    return () => this._listeners.delete(listener);
  }

  _emit() {
    for (const fn of this._listeners) {
      try {
        fn({ state: this.state, detail: this.detail, adapter: this.lastAdapter });
      } catch { /* ignore listener errors */ }
    }
  }

  _setState(state, detail = "") {
    this.state = state;
    this.detail = detail;
    this._emit();
  }

  clear(detail = "") {
    this.lastAdapter = null;
    this._setState("idle", detail);
  }

  /**
   * Encode a document index into either adapter weights or a document profile.
   *
   * @param {Object} index       — from buildDocumentIndex: { chunks, pages }
   * @param {string} modelId     — e.g. "onnx-community/Qwen3.5-0.8B-ONNX"
   * @param {Object} options
   * @param {string} [options.title]      — document title for labelling
   * @param {Function} [options.onProgress] — (fraction 0–1, message) callback
   * @param {boolean} [options.compileWeights] — generate layer weights when runtime can consume them
   * @returns {Promise<DocToLoraAdapter>}
   */
  async encode(index, modelId, { title = "document", onProgress, compileWeights = true } = {}) {
    if (this.state === "encoding") {
      throw new Error("Already encoding — await the current encode() to finish first.");
    }

    const t0 = performance.now();
    this.lastAdapter = null;
    this._setState("encoding", "Selecting document sections…");

    try {
      // ── Phase 1: select meta-groups ───────────────────────────────────────
      const metaGroups = selectMetaChunks(index.chunks, MAX_ADAPTER_CHUNKS);

      if (!metaGroups.length) {
        throw new Error("No usable chunks found — document may be image-only or empty.");
      }
      if (!metaGroups[0][0]?.vector?.length) {
        throw new Error("Chunks are missing embedding vectors. Re-index the document.");
      }

      const nGroups = metaGroups.length;
      this._setState(
        "encoding",
        compileWeights
          ? `Building ${nGroups} section adapter${nGroups !== 1 ? "s" : ""}…`
          : `Building ${nGroups} section document profile${nGroups !== 1 ? "s" : ""}…`
      );
      onProgress?.(
        0.05,
        compileWeights
          ? `Encoding ${nGroups} section${nGroups !== 1 ? "s" : ""}…`
          : `Profiling ${nGroups} section${nGroups !== 1 ? "s" : ""}…`
      );

      // ── Phase 2: meta-embeddings ──────────────────────────────────────────
      const metaEmbeddings = metaGroups.map(buildMetaEmbedding).filter(Boolean);
      if (!metaEmbeddings.length) {
        throw new Error("Meta-embedding computation returned no valid vectors.");
      }

      // ── Phase 3: resolve model config ─────────────────────────────────────
      const textConfig = resolveModelConfig(modelId);

      let combinedLayers = [];
      if (compileWeights) {
        // ── Phase 4: build per-group adapters ───────────────────────────────
        const chunkedAdapters = [];

        for (let i = 0; i < metaEmbeddings.length; i++) {
          const pct = 0.1 + (i / metaEmbeddings.length) * 0.65;
          this._setState("encoding", `Encoding section ${i + 1} / ${metaEmbeddings.length}…`);
          onProgress?.(pct, `Section ${i + 1} of ${metaEmbeddings.length}`);

          const layers = buildSingleChunkAdapter(metaEmbeddings[i], textConfig);
          const weight = metaGroups[i].reduce(
            (s, c) => s + Math.max(chunkReadabilityScore(c), 0.05),
            0
          );
          chunkedAdapters.push({ layers, weight });

          await new Promise(r => setTimeout(r, 0));
        }

        // ── Phase 5: combine via weighted average ───────────────────────────
        this._setState("encoding", "Combining section adapters (weighted average)…");
        onProgress?.(0.80, "Combining adapters…");
        await new Promise(r => setTimeout(r, 0));

        combinedLayers = combineAdapters(chunkedAdapters);
      } else {
        this._setState("encoding", "Compiling document profile…");
        onProgress?.(0.80, "Compiling document profile…");
        await new Promise(r => setTimeout(r, 0));
      }

      // ── Phase 6: build memory prompt (ONNX path fallback) ─────────────────
      const memoryPrompt = buildMemoryPrompt(metaGroups, title);

      const encodingMs = Math.round(performance.now() - t0);
      const adapter = new DocToLoraAdapter({
        layers: combinedLayers,
        rank: LORA_RANK,
        scale: LORA_SCALE,
        numGroups: nGroups,
        title,
        encodingMs,
        memoryPrompt,
        mode: compileWeights ? "weight-adapter" : "document-profile"
      });

      this.lastAdapter = adapter;
      this._setState(
        "ready",
        `${nGroups} section${nGroups !== 1 ? "s" : ""} encoded · ${adapter.describe()}`
      );
      onProgress?.(
        1.0,
        compileWeights
          ? `Doc-to-LoRA ready — ${adapter.formatMemory()}`
          : "Document profile ready"
      );

      const logLabel = compileWeights ? "DocToLoRA" : "DocumentProfile";
      console.info(
        `[${logLabel}] Encoded "${title}" → ` +
        `mode=${adapter.mode}, sections=${nGroups}, ` +
        `layers=${combinedLayers.length}, ` +
        `size=${compileWeights ? adapter.formatMemory() : `${adapter.memoryPrompt.length} chars`}, ` +
        `time=${encodingMs}ms`
      );

      return adapter;

    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this.lastAdapter = null;
      this._setState("error", msg);
      console.error("[DocumentProfile] Encoding failed:", err);
      throw err;
    }
  }
}
