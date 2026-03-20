const HAMMING_WASM_BASE64 =
  "AGFzbQEAAAABDAJgAX8AYAN/f38BfwILAQNlbnYDbG9nAAADAgEBBQMBAAEHHQIGbWVtb3J5AgAQaGFtbWluZ19kaXN0YW5jZQABCoYBAYMBAQF/QQAhAwJAA0AgAkUEQAwCCyAAEAAgAyAA/QAEACAB/QAEAP1R/RsAaSAA/QAEACAB/QAEAP1R/RsBaSAA/QAEACAB/QAEAP1R/RsCaSAA/QAEACAB/QAEAP1R/RsDaWpqamohAyAAQRBqIQAgAUEQaiEBIAJBEGshAgwACwsgAwsAPQRuYW1lARgCAANsb2cBEGhhbW1pbmdfZGlzdGFuY2UCHAIAAAEEAARwdHJBAQRwdHJCAgNsZW4DBGRpc3Q=";

function base64ToUint8Array(base64) {
  return Uint8Array.from(atob(base64), (char) => char.charCodeAt(0));
}

function popcount32(value) {
  let count = 0;
  let current = value >>> 0;
  while (current) {
    current &= current - 1;
    count += 1;
  }
  return count;
}

function signatureDistance(left, right) {
  let distance = 0;
  for (let index = 0; index < left.length; index += 1) {
    distance += popcount32((left[index] ^ right[index]) >>> 0);
  }
  return distance;
}

class WasmHammingKernel {
  static async create() {
    try {
      const memory = new WebAssembly.Memory({ initial: 1 });
      const { instance } = await WebAssembly.instantiate(base64ToUint8Array(HAMMING_WASM_BASE64), {
        env: {
          memory,
          log() {}
        }
      });
      return new WasmHammingKernel(instance, memory);
    } catch {
      return null;
    }
  }

  constructor(instance, memory) {
    this.instance = instance;
    this.memory = instance.exports.memory ?? memory;
  }

  ensureCapacity(byteLength) {
    const needed = byteLength * 2;
    if (needed > this.memory.buffer.byteLength) {
      const extraPages = Math.ceil((needed - this.memory.buffer.byteLength) / 65536);
      this.memory.grow(extraPages);
    }
  }

  distance(left, right) {
    const leftBytes = new Uint8Array(left.buffer, left.byteOffset, left.byteLength);
    const rightBytes = new Uint8Array(right.buffer, right.byteOffset, right.byteLength);
    this.ensureCapacity(leftBytes.byteLength);
    const memoryView = new Uint8Array(this.memory.buffer);
    memoryView.set(leftBytes, 0);
    memoryView.set(rightBytes, leftBytes.byteLength);
    return this.instance.exports.hamming_distance(0, leftBytes.byteLength, leftBytes.byteLength);
  }
}

export class VectorDatabase {
  constructor(dimensions) {
    this.dimensions = dimensions;
    this.entries = [];
    this.hammingKernelPromise = WasmHammingKernel.create();
  }

  async hydrate(entries) {
    this.entries = entries.map((entry) => ({
      ...entry,
      vector: entry.vector instanceof Float32Array ? entry.vector : new Float32Array(entry.vector),
      signature:
        entry.signature instanceof Uint32Array
          ? entry.signature
          : new Uint32Array(entry.signature)
    }));
  }

  async search(queryVector, { limit = 6, queryTerms = [] } = {}) {
    if (!this.entries.length) {
      return [];
    }

    const querySignature = new Uint32Array(Math.ceil(queryVector.length / 32));
    for (let index = 0; index < queryVector.length; index += 1) {
      if (queryVector[index] >= 0) {
        const wordIndex = Math.floor(index / 32);
        const bitIndex = index % 32;
        querySignature[wordIndex] |= 1 << bitIndex;
      }
    }

    const hammingKernel = await this.hammingKernelPromise;
    const approximate = this.entries
      .map((entry, index) => {
        const hamming = hammingKernel
          ? hammingKernel.distance(querySignature, entry.signature)
          : signatureDistance(querySignature, entry.signature);
        return {
          index,
          hamming,
          keywordBoost: queryTerms.length
            ? queryTerms.filter((term) => entry.text.toLowerCase().includes(term)).length /
              queryTerms.length
            : 0
        };
      })
      .sort((left, right) => {
        const leftScore = left.hamming - left.keywordBoost * 12;
        const rightScore = right.hamming - right.keywordBoost * 12;
        return leftScore - rightScore;
      })
      .slice(0, Math.min(this.entries.length, limit * 8));

    const candidateMatrix = new Float32Array(approximate.length * this.dimensions);
    approximate.forEach((candidate, rowIndex) => {
      candidateMatrix.set(this.entries[candidate.index].vector, rowIndex * this.dimensions);
    });

    const scores = this.scoreInJavaScript(candidateMatrix, queryVector, approximate.length);

    return approximate
      .map((candidate, rowIndex) => ({
        ...this.entries[candidate.index],
        similarity: scores[rowIndex] + candidate.keywordBoost * 0.18
      }))
      .sort((left, right) => right.similarity - left.similarity)
      .slice(0, limit);
  }

  scoreInJavaScript(matrix, queryVector, rows) {
    const scores = new Float32Array(rows);
    for (let row = 0; row < rows; row += 1) {
      let score = 0;
      const rowOffset = row * this.dimensions;
      for (let col = 0; col < this.dimensions; col += 1) {
        score += matrix[rowOffset + col] * queryVector[col];
      }
      scores[row] = score;
    }
    return scores;
  }
}
