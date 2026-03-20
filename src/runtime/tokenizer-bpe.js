const GPT2_PATTERN =
  /'(?:[sdmt]|ll|ve|re)| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

function bytesToUnicode() {
  const bs = [];
  for (let code = 33; code <= 126; code += 1) bs.push(code);
  for (let code = 161; code <= 172; code += 1) bs.push(code);
  for (let code = 174; code <= 255; code += 1) bs.push(code);

  const cs = [...bs];
  let offset = 0;
  for (let byte = 0; byte < 256; byte += 1) {
    if (!bs.includes(byte)) {
      bs.push(byte);
      cs.push(256 + offset);
      offset += 1;
    }
  }

  return Object.fromEntries(bs.map((byte, index) => [byte, String.fromCharCode(cs[index])]));
}

const BYTE_ENCODER = bytesToUnicode();
const BYTE_DECODER = Object.fromEntries(
  Object.entries(BYTE_ENCODER).map(([key, value]) => [value, Number(key)])
);

function getPairs(word) {
  const pairs = new Set();
  for (let index = 0; index < word.length - 1; index += 1) {
    pairs.add(`${word[index]} ${word[index + 1]}`);
  }
  return pairs;
}

function normalizeMerges(merges = []) {
  return merges.map((merge) => (Array.isArray(merge) ? merge.join(" ") : merge));
}

export class BytePairTokenizer {
  static fromModelBundle(bundle) {
    if (bundle?.tokenizerJson?.model?.vocab) {
      return new BytePairTokenizer(bundle.tokenizerJson, bundle.tokenizerConfig);
    }

    if (bundle?.vocabJson && bundle?.mergesText) {
      return new BytePairTokenizer(
        {
          model: {
            vocab: bundle.vocabJson,
            merges: bundle.mergesText
              .split("\n")
              .map((line) => line.trim())
              .filter((line) => line && !line.startsWith("#"))
          },
          added_tokens: []
        },
        bundle.tokenizerConfig
      );
    }

    throw new Error("No supported tokenizer assets were found.");
  }

  constructor(tokenizerJson, tokenizerConfig = {}) {
    const vocab = tokenizerJson.model?.vocab ?? {};
    const merges = normalizeMerges(tokenizerJson.model?.merges ?? []);
    const addedTokens = tokenizerJson.added_tokens ?? [];

    this.encoder = new Map(Object.entries(vocab));
    this.decoder = new Map(Object.entries(vocab).map(([token, id]) => [id, token]));
    this.bpeRanks = new Map(merges.map((merge, index) => [merge, index]));
    this.specialTokens = new Map();
    this.specialTokenRegex = null;
    this.cache = new Map();
    this.pattern = GPT2_PATTERN;
    this.tokenizerConfig = tokenizerConfig;

    for (const token of addedTokens) {
      if (typeof token.id === "number") {
        this.specialTokens.set(token.content, token.id);
        this.decoder.set(token.id, token.content);
      }
    }

    const configSpecials = [
      tokenizerConfig.bos_token,
      tokenizerConfig.eos_token,
      tokenizerConfig.pad_token,
      ...(tokenizerConfig.additional_special_tokens ?? [])
    ].filter(Boolean);

    for (const token of configSpecials) {
      const id = this.encoder.get(token);
      if (id !== undefined) {
        this.specialTokens.set(token, id);
      }
    }

    if (this.specialTokens.size) {
      this.specialTokenRegex = new RegExp(
        [...this.specialTokens.keys()]
          .sort((left, right) => right.length - left.length)
          .map((token) => token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
          .join("|"),
        "g"
      );
    }
  }

  byteEncode(text) {
    const bytes = new TextEncoder().encode(text);
    return Array.from(bytes, (byte) => BYTE_ENCODER[byte]).join("");
  }

  bpe(token) {
    if (this.cache.has(token)) {
      return this.cache.get(token);
    }

    let word = [...token];
    let pairs = getPairs(word);

    while (pairs.size > 0) {
      let bestPair = null;
      let bestRank = Number.POSITIVE_INFINITY;

      for (const pair of pairs) {
        const rank = this.bpeRanks.get(pair);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestPair = pair;
        }
      }

      if (!bestPair) {
        break;
      }

      const [first, second] = bestPair.split(" ");
      const merged = [];
      let index = 0;

      while (index < word.length) {
        const foundAt = word.indexOf(first, index);
        if (foundAt === -1) {
          merged.push(...word.slice(index));
          break;
        }

        merged.push(...word.slice(index, foundAt));
        if (word[foundAt + 1] === second) {
          merged.push(first + second);
          index = foundAt + 2;
        } else {
          merged.push(word[foundAt]);
          index = foundAt + 1;
        }
      }

      word = merged;
      if (word.length === 1) {
        break;
      }
      pairs = getPairs(word);
    }

    const result = word.join(" ");
    this.cache.set(token, result);
    return result;
  }

  encode(text) {
    if (!text) {
      return [];
    }

    const tokens = [];
    const segments = this.specialTokenRegex
      ? text.split(this.specialTokenRegex).flatMap((segment, index, source) => {
          const special = text.match(this.specialTokenRegex) ?? [];
          const items = [segment];
          if (index < special.length) {
            items.push(special[index]);
          }
          return items;
        })
      : [text];

    for (const segment of segments) {
      if (!segment) {
        continue;
      }
      const specialId = this.specialTokens.get(segment);
      if (specialId !== undefined) {
        tokens.push(specialId);
        continue;
      }

      const pieces = segment.match(this.pattern) ?? [];
      for (const piece of pieces) {
        const encoded = this.byteEncode(piece);
        const bpeTokens = this.bpe(encoded).split(" ");
        for (const token of bpeTokens) {
          const id = this.encoder.get(token);
          if (id === undefined) {
            throw new Error(`Unknown BPE token: ${token}`);
          }
          tokens.push(id);
        }
      }
    }

    return tokens;
  }

  countTokens(text) {
    return this.encode(text).length;
  }

  decode(tokenIds) {
    const encodedText = tokenIds
      .map((id) => this.decoder.get(id) ?? "")
      .join("");

    const bytes = [];
    for (const char of encodedText) {
      const byte = BYTE_DECODER[char];
      if (byte !== undefined) {
        bytes.push(byte);
      }
    }
    return new TextDecoder().decode(new Uint8Array(bytes));
  }
}
