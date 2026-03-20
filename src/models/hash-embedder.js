const DEFAULT_DIMENSIONS = 384;
const LAYOUT_FEATURES = 16;

function fnv1a(input, seed = 2166136261) {
  let hash = seed >>> 0;
  for (let index = 0; index < input.length; index += 1) {
    hash ^= input.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function l2Normalize(vector) {
  let norm = 0;
  for (let index = 0; index < vector.length; index += 1) {
    norm += vector[index] * vector[index];
  }
  if (norm === 0) {
    return vector;
  }
  const scale = 1 / Math.sqrt(norm);
  for (let index = 0; index < vector.length; index += 1) {
    vector[index] *= scale;
  }
  return vector;
}

function tokenize(text) {
  const lowered = text.toLowerCase();
  const words = lowered.match(/[a-z0-9][a-z0-9-_/.:]{1,}/g) ?? [];
  const trigrams = [];
  for (const word of words) {
    if (word.length < 3) {
      continue;
    }
    for (let index = 0; index <= word.length - 3; index += 1) {
      trigrams.push(word.slice(index, index + 3));
    }
  }
  return [...words, ...trigrams];
}

function addFeature(vector, offset, value) {
  if (Number.isFinite(value)) {
    vector[offset] += value;
  }
}

export class HashLayoutEmbedder {
  constructor(dimensions = DEFAULT_DIMENSIONS) {
    this.dimensions = dimensions;
    this.textDimensions = dimensions - LAYOUT_FEATURES;
  }

  encodeText(text, metadata = {}) {
    const vector = new Float32Array(this.dimensions);
    const tokens = tokenize(text);

    for (const token of tokens) {
      const primaryHash = fnv1a(token);
      const signedHash = fnv1a(token, 1469598103);
      const bucket = primaryHash % this.textDimensions;
      const sign = signedHash % 2 === 0 ? 1 : -1;
      const weight = token.length > 5 ? 1.4 : token.length > 3 ? 1 : 0.65;
      vector[bucket] += sign * weight;
    }

    const base = this.textDimensions;
    const bbox = metadata.bbox ?? { x: 0, y: 0, width: 1, height: 1 };
    const pageRatio = metadata.pageCount ? metadata.pageNumber / metadata.pageCount : 0;
    const lengthRatio = Math.min(text.length / 700, 1);
    const digitRatio = text.length
      ? (text.match(/\d/g)?.length ?? 0) / text.length
      : 0;
    const upperRatio = text.length
      ? (text.match(/[A-Z]/g)?.length ?? 0) / text.length
      : 0;

    addFeature(vector, base + 0, bbox.x);
    addFeature(vector, base + 1, bbox.y);
    addFeature(vector, base + 2, bbox.width);
    addFeature(vector, base + 3, bbox.height);
    addFeature(vector, base + 4, pageRatio);
    addFeature(vector, base + 5, lengthRatio);
    addFeature(vector, base + 6, digitRatio);
    addFeature(vector, base + 7, upperRatio);
    addFeature(vector, base + 8, metadata.blockCount ? Math.min(metadata.blockCount / 18, 1) : 0);
    addFeature(vector, base + 9, metadata.lineCount ? Math.min(metadata.lineCount / 32, 1) : 0);
    addFeature(vector, base + 10, metadata.isFallback ? 1 : 0);
    addFeature(vector, base + 11, text.includes("table") ? 0.7 : 0);
    addFeature(vector, base + 12, text.includes("figure") ? 0.6 : 0);
    addFeature(vector, base + 13, text.includes("%") ? 0.3 : 0);
    addFeature(vector, base + 14, text.includes("$") ? 0.3 : 0);
    addFeature(vector, base + 15, tokens.length ? Math.min(tokens.length / 64, 1) : 0);

    return l2Normalize(vector);
  }

  encodeQuery(text) {
    return this.encodeText(text, { bbox: { x: 0.5, y: 0.5, width: 1, height: 1 } });
  }

  getQueryTerms(text) {
    return [...new Set(tokenize(text).filter((token) => token.length > 2))];
  }

  packSignature(vector) {
    const packed = new Uint32Array(Math.ceil(vector.length / 32));
    for (let index = 0; index < vector.length; index += 1) {
      if (vector[index] >= 0) {
        const wordIndex = Math.floor(index / 32);
        const bitIndex = index % 32;
        packed[wordIndex] |= 1 << bitIndex;
      }
    }
    return packed;
  }

  keywordOverlap(queryTerms, text) {
    if (!queryTerms.length) {
      return 0;
    }
    const lowered = text.toLowerCase();
    let hits = 0;
    for (const term of queryTerms) {
      if (lowered.includes(term)) {
        hits += 1;
      }
    }
    return hits / queryTerms.length;
  }
}

export const EMBEDDING_DIMENSIONS = DEFAULT_DIMENSIONS;
