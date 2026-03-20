function halfToFloat32(value) {
  const sign = (value & 0x8000) << 16;
  const exponent = (value & 0x7c00) >> 10;
  const fraction = value & 0x03ff;

  let bits;
  if (exponent === 0) {
    if (fraction === 0) {
      bits = sign;
    } else {
      let mantissa = fraction;
      let shift = 0;
      while ((mantissa & 0x0400) === 0) {
        mantissa <<= 1;
        shift += 1;
      }
      mantissa &= 0x03ff;
      bits = sign | ((127 - 15 - shift) << 23) | (mantissa << 13);
    }
  } else if (exponent === 0x1f) {
    bits = sign | 0x7f800000 | (fraction << 13);
  } else {
    bits = sign | ((exponent + 112) << 23) | (fraction << 13);
  }

  return new Float32Array(new Uint32Array([bits]).buffer)[0];
}

function bfloat16ToFloat32(value) {
  const bits = value << 16;
  return new Float32Array(new Uint32Array([bits]).buffer)[0];
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-value));
}

function softplus(value) {
  if (value > 20) {
    return value;
  }
  if (value < -20) {
    return Math.exp(value);
  }
  return Math.log1p(Math.exp(value));
}

function silu(value) {
  return value * sigmoid(value);
}

function clampTokens(tokens, maxContextTokens) {
  if (tokens.length <= maxContextTokens) {
    return tokens;
  }
  return tokens.slice(tokens.length - maxContextTokens);
}

function argmax(values) {
  let bestIndex = 0;
  let bestValue = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (value > bestValue) {
      bestValue = value;
      bestIndex = index;
    }
  }
  return bestIndex;
}

function inferMatrixOrientation(shape, inputLength) {
  if (shape.length !== 2) {
    throw new Error(`Expected a matrix tensor, got shape ${shape.join("x")}`);
  }
  if (shape[1] === inputLength) {
    return { rows: shape[0], cols: shape[1], transposed: false };
  }
  if (shape[0] === inputLength) {
    return { rows: shape[1], cols: shape[0], transposed: true };
  }
  throw new Error(
    `Could not align matrix shape ${shape.join("x")} with input length ${inputLength}`
  );
}

function sanitizeGeneratedText(text) {
  return text
    .replace(/<\|[^|>]+?\|>/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function formatBytes(byteCount) {
  if (!Number.isFinite(byteCount) || byteCount <= 0) {
    return "0 B";
  }
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = byteCount;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(value >= 100 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

class RawTensor {
  constructor(descriptor, bytes) {
    this.name = descriptor.name;
    this.shape = descriptor.shape;
    this.dtype = descriptor.dtype;
    this.byteLength = descriptor.byteLength;

    switch (descriptor.dtype) {
      case "F32":
        this.data = new Float32Array(bytes);
        break;
      case "BF16":
      case "F16":
        this.data = new Uint16Array(bytes);
        break;
      case "I32":
        this.data = new Int32Array(bytes);
        break;
      case "U32":
        this.data = new Uint32Array(bytes);
        break;
      default:
        throw new Error(`Unsupported tensor dtype: ${descriptor.dtype}`);
    }
  }

  valueAt(index) {
    switch (this.dtype) {
      case "F32":
        return this.data[index];
      case "BF16":
        return bfloat16ToFloat32(this.data[index]);
      case "F16":
        return halfToFloat32(this.data[index]);
      case "I32":
      case "U32":
        return this.data[index];
      default:
        throw new Error(`Unsupported tensor dtype: ${this.dtype}`);
    }
  }

  vectorAt(rowIndex) {
    if (this.shape.length === 1) {
      return this.toFloat32Array();
    }
    const width = this.shape[this.shape.length - 1];
    const start = rowIndex * width;
    const output = new Float32Array(width);
    for (let index = 0; index < width; index += 1) {
      output[index] = this.valueAt(start + index);
    }
    return output;
  }

  toFloat32Array() {
    const output = new Float32Array(this.data.length);
    for (let index = 0; index < this.data.length; index += 1) {
      output[index] = this.valueAt(index);
    }
    return output;
  }
}

class HubTensorStore {
  constructor({ hubClient, repo, manifest }) {
    this.hubClient = hubClient;
    this.repo = repo;
    this.tensorIndex = new Map(manifest.tensors.map((tensor) => [tensor.name, tensor]));
    this.cache = new Map();
  }

  listNames() {
    return [...this.tensorIndex.keys()];
  }

  has(name) {
    return this.tensorIndex.has(name);
  }

  findBySuffix(suffix) {
    const match = this.listNames().find((name) => name.endsWith(suffix));
    return match ?? null;
  }

  async get(name) {
    if (this.cache.has(name)) {
      return this.cache.get(name);
    }

    const descriptor = this.tensorIndex.get(name);
    if (!descriptor) {
      throw new Error(`Unknown tensor: ${name}`);
    }

    const bytes = await this.hubClient.fetchRange(
      this.repo,
      descriptor.filePath,
      descriptor.byteStart,
      descriptor.byteEnd
    );
    if (!bytes) {
      throw new Error(`Could not read tensor bytes for ${name}`);
    }

    const tensor = new RawTensor(descriptor, bytes);
    this.cache.set(name, tensor);
    return tensor;
  }
}

class RotaryEmbedding {
  constructor({ dims, base = 10000, ropeScaling = null }) {
    this.dims = dims;
    this.base = base;
    this.scale = ropeScaling?.type === "linear" ? 1 / ropeScaling.factor : 1;
    this.invFrequencies = new Float32Array(dims / 2);
    for (let index = 0; index < dims / 2; index += 1) {
      this.invFrequencies[index] = 1 / base ** ((2 * index) / dims);
    }
  }

  apply(head, offset) {
    const rotated = head.slice();
    for (let index = 0; index < this.dims; index += 2) {
      const inv = this.invFrequencies[index / 2];
      const angle = offset * inv * this.scale;
      const cos = Math.cos(angle);
      const sin = Math.sin(angle);
      const even = head[index];
      const odd = head[index + 1];
      rotated[index] = even * cos - odd * sin;
      rotated[index + 1] = even * sin + odd * cos;
    }
    return rotated;
  }
}

class JsKVCache {
  constructor() {
    this.keys = [];
    this.values = [];
    this.offset = 0;
  }

  append(keys, values) {
    this.keys.push(keys.map((key) => key.slice()));
    this.values.push(values.map((value) => value.slice()));
    this.offset += 1;
  }
}

class JsLinearCache {
  constructor({ convKernelSize, convDim, numValueHeads, valueHeadDim, keyHeadDim }) {
    this.convKernelSize = convKernelSize;
    this.convDim = convDim;
    this.history = [];
    this.state = new Float32Array(numValueHeads * valueHeadDim * keyHeadDim);
  }

  pushHistory(vector) {
    this.history.push(vector.slice());
    const keep = this.convKernelSize - 1;
    if (this.history.length > keep) {
      this.history.shift();
    }
  }
}

function addInPlace(left, right) {
  const output = new Float32Array(left.length);
  for (let index = 0; index < left.length; index += 1) {
    output[index] = left[index] + right[index];
  }
  return output;
}

function rmsNorm(input, weight, epsilon) {
  let sumSquares = 0;
  for (let index = 0; index < input.length; index += 1) {
    sumSquares += input[index] * input[index];
  }
  const scale = 1 / Math.sqrt(sumSquares / input.length + epsilon);
  const output = new Float32Array(input.length);
  for (let index = 0; index < input.length; index += 1) {
    output[index] = input[index] * scale * weight[index];
  }
  return output;
}

function unitRmsNorm(input, epsilon) {
  let sumSquares = 0;
  for (let index = 0; index < input.length; index += 1) {
    sumSquares += input[index] * input[index];
  }
  const scale = 1 / Math.sqrt(sumSquares / input.length + epsilon);
  const output = new Float32Array(input.length);
  for (let index = 0; index < input.length; index += 1) {
    output[index] = input[index] * scale;
  }
  return output;
}

function matVec(weightTensor, input) {
  const { rows, cols, transposed } = inferMatrixOrientation(weightTensor.shape, input.length);
  const output = new Float32Array(rows);

  if (!transposed) {
    for (let row = 0; row < rows; row += 1) {
      let total = 0;
      const rowOffset = row * cols;
      for (let col = 0; col < cols; col += 1) {
        total += weightTensor.valueAt(rowOffset + col) * input[col];
      }
      output[row] = total;
    }
    return output;
  }

  for (let row = 0; row < rows; row += 1) {
    let total = 0;
    for (let col = 0; col < cols; col += 1) {
      total += weightTensor.valueAt(col * rows + row) * input[col];
    }
    output[row] = total;
  }
  return output;
}

function swiglu(gate, value) {
  const output = new Float32Array(gate.length);
  for (let index = 0; index < gate.length; index += 1) {
    output[index] = silu(gate[index]) * value[index];
  }
  return output;
}

function reshapeHeads(vector, headCount, headDim) {
  const heads = new Array(headCount);
  for (let head = 0; head < headCount; head += 1) {
    heads[head] = vector.slice(head * headDim, (head + 1) * headDim);
  }
  return heads;
}

function flattenHeads(heads) {
  const headDim = heads[0]?.length ?? 0;
  const output = new Float32Array(heads.length * headDim);
  for (let head = 0; head < heads.length; head += 1) {
    output.set(heads[head], head * headDim);
  }
  return output;
}

function softmax(scores) {
  let maxScore = Number.NEGATIVE_INFINITY;
  for (const score of scores) {
    if (score > maxScore) {
      maxScore = score;
    }
  }

  const weights = new Float32Array(scores.length);
  let total = 0;
  for (let index = 0; index < scores.length; index += 1) {
    const value = Math.exp(scores[index] - maxScore);
    weights[index] = value;
    total += value;
  }

  if (!total) {
    return weights;
  }

  for (let index = 0; index < weights.length; index += 1) {
    weights[index] /= total;
  }
  return weights;
}

function dot(left, right) {
  let total = 0;
  for (let index = 0; index < left.length; index += 1) {
    total += left[index] * right[index];
  }
  return total;
}

function convKernelAt(weightTensor, channel, kernelIndex) {
  const shape = weightTensor.shape;
  if (shape.length === 2) {
    return weightTensor.valueAt(channel * shape[1] + kernelIndex);
  }
  if (shape.length !== 3) {
    throw new Error(`Unsupported conv tensor shape ${shape.join("x")}`);
  }

  if (shape[1] === 1) {
    return weightTensor.valueAt(channel * shape[2] + kernelIndex);
  }
  if (shape[2] === 1) {
    return weightTensor.valueAt(channel * shape[1] + kernelIndex);
  }
  return weightTensor.valueAt(channel * shape[1] * shape[2] + kernelIndex);
}

function detectTextPrefix(store) {
  const layer0 = store.findBySuffix("layers.0.input_layernorm.weight");
  if (!layer0) {
    throw new Error("Could not detect the language-model tensor prefix.");
  }
  return layer0.slice(0, -"layers.0.input_layernorm.weight".length);
}

function normalizeTokenList(value) {
  if (Array.isArray(value)) {
    return value;
  }
  if (typeof value === "number") {
    return [value];
  }
  return [];
}

function buildStopTokens(config, tokenizerConfig = {}) {
  const stop = new Set();
  for (const token of normalizeTokenList(config?.eos_token_id)) {
    if (typeof token === "number") {
      stop.add(token);
    }
  }
  for (const token of normalizeTokenList(config?.text_config?.eos_token_id)) {
    if (typeof token === "number") {
      stop.add(token);
    }
  }
  for (const token of normalizeTokenList(tokenizerConfig.eos_token_id)) {
    if (typeof token === "number") {
      stop.add(token);
    }
  }
  return stop;
}

export class Qwen35TextEngine {
  constructor({ hubClient, repo, config, manifest, tokenizer, tokenizerConfig }) {
    this.hubClient = hubClient;
    this.repo = repo;
    this.config = config;
    this.textConfig = config.text_config ?? config;
    this.tokenizer = tokenizer;
    this.tokenizerConfig = tokenizerConfig ?? {};
    this.tensorStore = new HubTensorStore({ hubClient, repo, manifest });
    this.vectorCache = new Map();
    /** @type {import('./doc-to-lora.js').DocToLoraAdapter|null} */
    this.loraAdapter = null;
    this.textPrefix = detectTextPrefix(this.tensorStore);
    this.embedName = `${this.textPrefix}embed_tokens.weight`;
    this.finalNormName = `${this.textPrefix}norm.weight`;
    this.lmHeadName =
      this.tensorStore.findBySuffix("lm_head.weight") ??
      (this.textConfig.tie_word_embeddings ? this.embedName : null);
    this.stopTokens = buildStopTokens(config, tokenizerConfig);
    this.maxContextTokens = 320;
    this.maxNewTokens = 48;
    this.shouldShiftNormWeights = this.#detectNormShift();
    this.layerPlans = this.#buildLayerPlans();
    this.headDim =
      this.textConfig.head_dim ??
      Math.floor(this.textConfig.hidden_size / this.textConfig.num_attention_heads);
    this.partialRotaryFactor =
      this.textConfig.partial_rotary_factor ??
      this.textConfig.rope_parameters?.partial_rotary_factor ??
      0.25;
    this.rotary = new RotaryEmbedding({
      dims: Math.max(
        2,
        2 * Math.floor((this.headDim * this.partialRotaryFactor) / 2)
      ),
      base:
        this.textConfig.rope_theta ??
        this.textConfig.rope_parameters?.rope_theta ??
        100000,
      ropeScaling: this.textConfig.rope_scaling ?? this.textConfig.rope_parameters ?? null
    });
    this.requiredTensorNames = this.#collectRequiredTensorNames();
    this.coldStartBytes = this.#estimateRequiredBytes();
  }

  #buildLayerPlans() {
    if ((this.textConfig.num_experts ?? 0) > 0) {
      throw new Error("MoE Qwen3.5 variants are not supported by this runtime yet.");
    }

    const plans = [];
    for (let index = 0; index < this.textConfig.num_hidden_layers; index += 1) {
      const prefix = `${this.textPrefix}layers.${index}.`;
      const isLinear = (index + 1) % this.textConfig.full_attention_interval !== 0;
      plans.push({
        index,
        isLinear,
        inputNorm: `${prefix}input_layernorm.weight`,
        postNorm: `${prefix}post_attention_layernorm.weight`,
        gateProj: `${prefix}mlp.gate_proj.weight`,
        upProj: `${prefix}mlp.up_proj.weight`,
        downProj: `${prefix}mlp.down_proj.weight`,
        attention: isLinear
          ? {
              inProjQkv: `${prefix}linear_attn.in_proj_qkv.weight`,
              inProjZ: `${prefix}linear_attn.in_proj_z.weight`,
              inProjB: `${prefix}linear_attn.in_proj_b.weight`,
              inProjA: `${prefix}linear_attn.in_proj_a.weight`,
              conv1d: `${prefix}linear_attn.conv1d.weight`,
              dtBias: `${prefix}linear_attn.dt_bias`,
              aLog: `${prefix}linear_attn.A_log`,
              norm: `${prefix}linear_attn.norm.weight`,
              outProj: `${prefix}linear_attn.out_proj.weight`
            }
          : {
              qProj: `${prefix}self_attn.q_proj.weight`,
              kProj: `${prefix}self_attn.k_proj.weight`,
              vProj: `${prefix}self_attn.v_proj.weight`,
              oProj: `${prefix}self_attn.o_proj.weight`,
              qNorm: `${prefix}self_attn.q_norm.weight`,
              kNorm: `${prefix}self_attn.k_norm.weight`
            }
      });
    }
    return plans;
  }

  #detectNormShift() {
    for (const name of this.tensorStore.listNames()) {
      if (name.includes("mtp.")) {
        return true;
      }
      if (name.endsWith("conv1d.weight")) {
        const descriptor = this.tensorStore.tensorIndex.get(name);
        if (descriptor?.shape?.length === 3 && descriptor.shape.at(-1) !== 1) {
          return true;
        }
      }
    }
    return false;
  }

  #collectRequiredTensorNames() {
    const names = new Set([this.embedName, this.finalNormName]);
    if (this.lmHeadName) {
      names.add(this.lmHeadName);
    }

    for (const layerPlan of this.layerPlans) {
      names.add(layerPlan.inputNorm);
      names.add(layerPlan.postNorm);
      names.add(layerPlan.gateProj);
      names.add(layerPlan.upProj);
      names.add(layerPlan.downProj);
      for (const name of Object.values(layerPlan.attention)) {
        names.add(name);
      }
    }

    return [...names];
  }

  #estimateRequiredBytes() {
    let total = 0;
    for (const name of this.requiredTensorNames) {
      const descriptor = this.tensorStore.tensorIndex.get(name);
      if (descriptor?.byteLength) {
        total += descriptor.byteLength;
      }
    }
    return total;
  }

  getRuntimeStatus({ maxInteractiveBytes = 256 * 1024 * 1024 } = {}) {
    if (!this.lmHeadName) {
      return {
        interactive: false,
        reason: "lm-head-unavailable",
        coldStartBytes: this.coldStartBytes,
        coldStartLabel: formatBytes(this.coldStartBytes)
      };
    }

    const missing = this.requiredTensorNames.filter(
      (name) => !this.tensorStore.tensorIndex.has(name)
    );
    if (missing.length) {
      return {
        interactive: false,
        reason: `missing-tensors:${missing[0]}`,
        coldStartBytes: this.coldStartBytes,
        coldStartLabel: formatBytes(this.coldStartBytes)
      };
    }

    if (this.coldStartBytes > maxInteractiveBytes) {
      return {
        interactive: false,
        reason: `cold-start-too-large:${formatBytes(this.coldStartBytes)}`,
        coldStartBytes: this.coldStartBytes,
        coldStartLabel: formatBytes(this.coldStartBytes)
      };
    }

    return {
      interactive: true,
      reason: "interactive-ready",
      coldStartBytes: this.coldStartBytes,
      coldStartLabel: formatBytes(this.coldStartBytes)
    };
  }

  async #getWeightVector(name) {
    if (this.vectorCache.has(name)) {
      return this.vectorCache.get(name);
    }
    const tensor = await this.tensorStore.get(name);
    const vector = tensor.toFloat32Array();
    if (
      this.shouldShiftNormWeights &&
      (
        name.endsWith(".input_layernorm.weight") ||
        name.endsWith(".post_attention_layernorm.weight") ||
        name.endsWith(".q_norm.weight") ||
        name.endsWith(".k_norm.weight") ||
        name.endsWith(".linear_attn.norm.weight") ||
        name.endsWith(".norm.weight")
      )
    ) {
      for (let index = 0; index < vector.length; index += 1) {
        vector[index] += 1;
      }
    }
    this.vectorCache.set(name, vector);
    return vector;
  }

  /**
   * Attach a Doc-to-LoRA adapter.  The adapter will be applied to every
   * subsequent forwardToken() call until cleared (pass null to remove).
   *
   * @param {import('./doc-to-lora.js').DocToLoraAdapter|null} adapter
   */
  setLoraAdapter(adapter) {
    this.loraAdapter = adapter ?? null;
    if (adapter) {
      console.info(
        `[Qwen35Engine] LoRA adapter attached — ` +
        `${adapter.numLayers} layers, rank=${adapter.rank}, ` +
        `size=${adapter.formatMemory()}`
      );
    }
  }

  /**
   * Apply a LoRA delta to a projection output vector.
   *
   * output += scale × B @ (A @ input)
   *
   * No-ops gracefully if loraAdapter is null, layerIndex is out of range,
   * or projection dimensions don't match — never corrupts generation.
   */
  #applyLoraDelta(output, input, layerIndex, projName) {
    const layer = this.loraAdapter?.getLayer(layerIndex);
    if (!layer) return output;
    const lora = layer[projName];
    if (!lora) return output;

    const { A, B, rank, scale } = lora;
    const dIn = input.length;
    const dOut = output.length;

    if (A.length !== rank * dIn || B.length !== dOut * rank) {
      return output; // dimension mismatch — skip silently
    }

    // ax = A @ input  [rank]
    const ax = new Float32Array(rank);
    for (let r = 0; r < rank; r++) {
      let s = 0;
      const off = r * dIn;
      for (let j = 0; j < dIn; j++) s += A[off + j] * input[j];
      ax[r] = s;
    }

    // output += scale × (B @ ax)
    for (let i = 0; i < dOut; i++) {
      let delta = 0;
      const off = i * rank;
      for (let r = 0; r < rank; r++) delta += B[off + r] * ax[r];
      output[i] += scale * delta;
    }
    return output;
  }

  makeCaches() {
    return this.layerPlans.map((layer) =>
      layer.isLinear
        ? new JsLinearCache({
            convKernelSize: this.textConfig.linear_conv_kernel_dim,
            convDim:
              this.textConfig.linear_num_key_heads * this.textConfig.linear_key_head_dim * 2 +
              this.textConfig.linear_num_value_heads * this.textConfig.linear_value_head_dim,
            numValueHeads: this.textConfig.linear_num_value_heads,
            valueHeadDim: this.textConfig.linear_value_head_dim,
            keyHeadDim: this.textConfig.linear_key_head_dim
          })
        : new JsKVCache()
    );
  }

  async embedToken(tokenId) {
    const tensor = await this.tensorStore.get(this.embedName);
    return tensor.vectorAt(tokenId);
  }

  async #runMlp(input, layerPlan) {
    const [gateProj, upProj, downProj] = await Promise.all([
      this.tensorStore.get(layerPlan.gateProj),
      this.tensorStore.get(layerPlan.upProj),
      this.tensorStore.get(layerPlan.downProj)
    ]);

    const gate = matVec(gateProj, input);
    const up = matVec(upProj, input);
    return matVec(downProj, swiglu(gate, up));
  }

  async #runAttention(input, layerPlan, cache) {
    const [qProj, kProj, vProj, oProj] = await Promise.all([
      this.tensorStore.get(layerPlan.attention.qProj),
      this.tensorStore.get(layerPlan.attention.kProj),
      this.tensorStore.get(layerPlan.attention.vProj),
      this.tensorStore.get(layerPlan.attention.oProj)
    ]);
    const [qNormWeight, kNormWeight] = await Promise.all([
      this.#getWeightVector(layerPlan.attention.qNorm),
      this.#getWeightVector(layerPlan.attention.kNorm)
    ]);

    // Base projections + optional LoRA deltas
    // delta = (α/r) × B @ (A @ input), applied in-place to the matVec result
    const l = layerPlan.index;
    const qRaw = this.#applyLoraDelta(matVec(qProj, input), input, l, "q");
    const gate = qRaw.slice(qRaw.length / 2);
    const queries = reshapeHeads(
      qRaw.slice(0, qRaw.length / 2),
      this.textConfig.num_attention_heads,
      this.headDim
    ).map((head) => this.rotary.apply(rmsNorm(head, qNormWeight, this.textConfig.rms_norm_eps), cache.offset));

    const kRaw = this.#applyLoraDelta(matVec(kProj, input), input, l, "k");
    const keys = reshapeHeads(
      kRaw,
      this.textConfig.num_key_value_heads,
      this.headDim
    ).map((head) => this.rotary.apply(rmsNorm(head, kNormWeight, this.textConfig.rms_norm_eps), cache.offset));

    const vRaw = this.#applyLoraDelta(matVec(vProj, input), input, l, "v");
    const values = reshapeHeads(
      vRaw,
      this.textConfig.num_key_value_heads,
      this.headDim
    );

    cache.append(keys, values);

    const groupSize = this.textConfig.num_attention_heads / this.textConfig.num_key_value_heads;
    const scale = this.headDim ** -0.5;
    const outputs = [];

    for (let headIndex = 0; headIndex < this.textConfig.num_attention_heads; headIndex += 1) {
      const kvHead = Math.floor(headIndex / groupSize);
      const scores = [];
      for (let position = 0; position < cache.keys.length; position += 1) {
        scores.push(dot(queries[headIndex], cache.keys[position][kvHead]) * scale);
      }

      const weights = softmax(scores);
      const mixed = new Float32Array(this.headDim);
      for (let position = 0; position < cache.values.length; position += 1) {
        const value = cache.values[position][kvHead];
        const weight = weights[position];
        for (let dim = 0; dim < value.length; dim += 1) {
          mixed[dim] += value[dim] * weight;
        }
      }
      outputs.push(mixed);
    }

    const flattened = flattenHeads(outputs);
    for (let index = 0; index < flattened.length; index += 1) {
      flattened[index] *= sigmoid(gate[index]);
    }
    // Apply LoRA to output projection (o_proj)
    return this.#applyLoraDelta(matVec(oProj, flattened), flattened, l, "o");
  }

  async #runLinearAttention(input, layerPlan, cache) {
    const tensors = await Promise.all([
      this.tensorStore.get(layerPlan.attention.inProjQkv),
      this.tensorStore.get(layerPlan.attention.inProjZ),
      this.tensorStore.get(layerPlan.attention.inProjB),
      this.tensorStore.get(layerPlan.attention.inProjA),
      this.tensorStore.get(layerPlan.attention.conv1d),
      this.tensorStore.get(layerPlan.attention.dtBias),
      this.tensorStore.get(layerPlan.attention.aLog),
      this.tensorStore.get(layerPlan.attention.outProj)
    ]);
    const [inProjQkv, inProjZ, inProjB, inProjA, conv1d, dtBias, aLog, outProj] = tensors;
    const normWeight = await this.#getWeightVector(layerPlan.attention.norm);

    const nk = this.textConfig.linear_num_key_heads;
    const nv = this.textConfig.linear_num_value_heads;
    const dk = this.textConfig.linear_key_head_dim;
    const dv = this.textConfig.linear_value_head_dim;
    const headsPerKey = nv / nk;

    const q = new Array(nk);
    const k = new Array(nk);
    const v = new Array(nv);
    const z = new Array(nv);
    const qkv = matVec(inProjQkv, input);
    const zFlat = matVec(inProjZ, input);
    const b = matVec(inProjB, input);
    const a = matVec(inProjA, input);

    let cursorQkv = 0;
    for (let head = 0; head < nk; head += 1) {
      q[head] = qkv.slice(cursorQkv, cursorQkv + dk);
      cursorQkv += dk;
    }
    for (let head = 0; head < nk; head += 1) {
      k[head] = qkv.slice(cursorQkv, cursorQkv + dk);
      cursorQkv += dk;
    }
    for (let head = 0; head < nv; head += 1) {
      v[head] = qkv.slice(cursorQkv, cursorQkv + dv);
      cursorQkv += dv;
      z[head] = zFlat.slice(head * dv, (head + 1) * dv);
    }

    const mixedQkv = new Float32Array(cache.convDim);
    let cursor = 0;
    for (const head of q) {
      mixedQkv.set(head, cursor);
      cursor += head.length;
    }
    for (const head of k) {
      mixedQkv.set(head, cursor);
      cursor += head.length;
    }
    for (const head of v) {
      mixedQkv.set(head, cursor);
      cursor += head.length;
    }

    const convWindow = [];
    const missing = this.textConfig.linear_conv_kernel_dim - 1 - cache.history.length;
    for (let index = 0; index < missing; index += 1) {
      convWindow.push(new Float32Array(cache.convDim));
    }
    convWindow.push(...cache.history);
    convWindow.push(mixedQkv);

    const convOut = new Float32Array(cache.convDim);
    for (let channel = 0; channel < cache.convDim; channel += 1) {
      let total = 0;
      for (let step = 0; step < convWindow.length; step += 1) {
        total += convWindow[step][channel] * convKernelAt(conv1d, channel, step);
      }
      convOut[channel] = silu(total);
    }
    cache.pushHistory(mixedQkv);

    cursor = 0;
    const qConv = new Array(nk);
    const kConv = new Array(nk);
    const vConv = new Array(nv);
    for (let head = 0; head < nk; head += 1) {
      qConv[head] = unitRmsNorm(convOut.slice(cursor, cursor + dk), 1e-6);
      cursor += dk;
    }
    for (let head = 0; head < nk; head += 1) {
      kConv[head] = unitRmsNorm(convOut.slice(cursor, cursor + dk), 1e-6);
      cursor += dk;
    }
    for (let head = 0; head < nv; head += 1) {
      vConv[head] = convOut.slice(cursor, cursor + dv);
      cursor += dv;
    }

    const invScale = dk ** -0.5;
    for (let head = 0; head < nk; head += 1) {
      for (let dim = 0; dim < dk; dim += 1) {
        qConv[head][dim] *= invScale * invScale;
        kConv[head][dim] *= invScale;
      }
    }
    const outputHeads = new Array(nv);
    for (let valueHead = 0; valueHead < nv; valueHead += 1) {
      const keyHead = Math.floor(valueHead / headsPerKey);
      const qHead = qConv[keyHead];
      const kHead = kConv[keyHead];
      const g = Math.exp(-Math.exp(aLog.valueAt(valueHead)) * softplus(a[valueHead] + dtBias.valueAt(valueHead)));
      const beta = sigmoid(b[valueHead]);
      const output = new Float32Array(dv);

      for (let dValue = 0; dValue < dv; dValue += 1) {
        let kvMem = 0;
        for (let dKey = 0; dKey < dk; dKey += 1) {
          const stateIndex = ((valueHead * dv + dValue) * dk) + dKey;
          cache.state[stateIndex] *= g;
          kvMem += cache.state[stateIndex] * kHead[dKey];
        }

        const delta = (vConv[valueHead][dValue] - kvMem) * beta;
        let headOut = 0;
        for (let dKey = 0; dKey < dk; dKey += 1) {
          const stateIndex = ((valueHead * dv + dValue) * dk) + dKey;
          cache.state[stateIndex] += kHead[dKey] * delta;
          headOut += cache.state[stateIndex] * qHead[dKey];
        }
        output[dValue] = headOut;
      }

      const normalized = rmsNorm(output, normWeight, this.textConfig.rms_norm_eps);
      for (let dim = 0; dim < normalized.length; dim += 1) {
        normalized[dim] *= silu(z[valueHead][dim]);
      }
      outputHeads[valueHead] = normalized;
    }

    return matVec(outProj, flattenHeads(outputHeads));
  }

  async forwardToken(tokenId, caches) {
    let hidden = await this.embedToken(tokenId);

    for (const [index, layerPlan] of this.layerPlans.entries()) {
      const [inputNormWeight, postNormWeight] = await Promise.all([
        this.#getWeightVector(layerPlan.inputNorm),
        this.#getWeightVector(layerPlan.postNorm)
      ]);

      const attentionInput = rmsNorm(
        hidden,
        inputNormWeight,
        this.textConfig.rms_norm_eps
      );
      const residual = layerPlan.isLinear
        ? await this.#runLinearAttention(attentionInput, layerPlan, caches[index])
        : await this.#runAttention(attentionInput, layerPlan, caches[index]);
      hidden = addInPlace(hidden, residual);

      const mlpInput = rmsNorm(
        hidden,
        postNormWeight,
        this.textConfig.rms_norm_eps
      );
      hidden = addInPlace(hidden, await this.#runMlp(mlpInput, layerPlan));
    }

    const finalNorm = await this.#getWeightVector(this.finalNormName);
    const normalized = rmsNorm(hidden, finalNorm, this.textConfig.rms_norm_eps);
    const lmHead = await this.tensorStore.get(this.lmHeadName);
    return matVec(lmHead, normalized);
  }

  async generate(prompt, { maxNewTokens = this.maxNewTokens } = {}) {
    if (!this.tokenizer) {
      throw new Error("The Qwen tokenizer is not available.");
    }

    const inputIds = clampTokens(this.tokenizer.encode(prompt), this.maxContextTokens);
    if (!inputIds.length) {
      return { text: "", tokenIds: [], promptTokenCount: 0 };
    }

    const caches = this.makeCaches();
    let logits = null;
    for (const tokenId of inputIds) {
      logits = await this.forwardToken(tokenId, caches);
    }

    const generated = [];
    for (let step = 0; step < maxNewTokens; step += 1) {
      const nextToken = argmax(logits);
      if (this.stopTokens.has(nextToken)) {
        break;
      }
      generated.push(nextToken);
      logits = await this.forwardToken(nextToken, caches);
    }

    return {
      text: sanitizeGeneratedText(this.tokenizer.decode(generated)),
      tokenIds: generated,
      promptTokenCount: inputIds.length
    };
  }
}
