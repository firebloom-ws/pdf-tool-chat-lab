import { OpfsCache } from "./opfs-cache.js";
import { decodeTensorBytesToFloat32 } from "./safetensors.js";

const DTYPE_BYTE_SIZE = {
  BOOL: 1,
  U8: 1,
  I8: 1,
  U16: 2,
  I16: 2,
  F16: 2,
  BF16: 2,
  U32: 4,
  I32: 4,
  F32: 4
};

const LM_HEAD_CHUNK_ROWS = 8192;
const LM_HEAD_GROUP_SIZE = 128;

const PAPERTRAIL_GPTQ_QZEROS_MATVEC_WGSL = /* wgsl */ `
struct Params {
  input_size: u32,
  output_size: u32,
  group_size: u32,
  packed_output_size: u32,
  zero_point_bias: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> qweight: array<u32>;
@group(0) @binding(2) var<storage, read> qzeros: array<u32>;
@group(0) @binding(3) var<storage, read> scales: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

fn unpack_nibble(word: u32, nibble_index: u32) -> f32 {
  return f32((word >> (nibble_index * 4u)) & 0xFu);
}

fn unpack_scale(linear_index: u32) -> f32 {
  let packed = scales[linear_index / 2u];
  let pair = unpack2x16float(packed);
  if ((linear_index & 1u) == 0u) {
    return pair.x;
  }
  return pair.y;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let output_col = gid.x;
  if (output_col >= params.output_size) {
    return;
  }

  let packed_output_col = output_col / 8u;
  let output_nibble = output_col & 7u;
  let group_count = params.input_size / params.group_size;
  var acc = 0.0;

  for (var group = 0u; group < group_count; group += 1u) {
    let scale = unpack_scale(group * params.output_size + output_col);
    let zero_word = qzeros[group * params.packed_output_size + packed_output_col];
    let zero = unpack_nibble(zero_word, output_nibble) + f32(params.zero_point_bias);
    let start = group * params.group_size;
    let end = min(start + params.group_size, params.input_size);

    for (var input_index = start; input_index < end; input_index += 1u) {
      let packed_input_row = input_index / 8u;
      let weight_word = qweight[packed_input_row * params.output_size + output_col];
      let weight = unpack_nibble(weight_word, input_index & 7u);
      acc += (weight - zero) * scale * input[input_index];
    }
  }

  output[output_col] = acc;
}
`;

const PAPERTRAIL_VECTOR_ADD_WGSL = /* wgsl */ `
struct Params {
  length: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let index = gid.x;
  if (index >= params.length) {
    return;
  }
  output[index] = left[index] + right[index];
}
`;

const PAPERTRAIL_SWIGLU_WGSL = /* wgsl */ `
struct Params {
  length: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

fn silu(x: f32) -> f32 {
  return x / (1.0 + exp(-x));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let index = gid.x;
  if (index >= params.length) {
    return;
  }
  output[index] = silu(gate[index]) * up[index];
}
`;

const PAPERTRAIL_RMSNORM_WGSL = /* wgsl */ `
struct Params {
  length: u32,
  epsilon: f32,
  _pad0: u32,
  _pad1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> partialSums: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3u) {
  let tid = lid.x;
  var sum = 0.0;
  var index = tid;
  loop {
    if (index >= params.length) {
      break;
    }
    let value = input[index];
    sum += value * value;
    index += 256u;
  }

  partialSums[tid] = sum;
  workgroupBarrier();

  var stride = 128u;
  loop {
    if (stride == 0u) {
      break;
    }
    if (tid < stride) {
      partialSums[tid] += partialSums[tid + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  let inv = inverseSqrt(partialSums[0] / f32(params.length) + params.epsilon);
  index = tid;
  loop {
    if (index >= params.length) {
      break;
    }
    output[index] = input[index] * inv * weight[index];
    index += 256u;
  }
}
`;

function modelSlug(modelId) {
  return String(modelId).replace(/[^a-z0-9]+/gi, "-").replace(/^-+|-+$/g, "").toLowerCase();
}

function toArrayBuffer(bytes) {
  const view = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  return view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength);
}

function alignTo4(value) {
  return Math.max(4, Math.ceil(value / 4) * 4);
}

function createUintParams(values) {
  const params = new Uint32Array(4);
  params.set(values.slice(0, 4));
  return params;
}

function createRmsNormParams(length, epsilon) {
  const buffer = new ArrayBuffer(16);
  const view = new DataView(buffer);
  view.setUint32(0, length, true);
  view.setFloat32(4, epsilon, true);
  view.setUint32(8, 0, true);
  view.setUint32(12, 0, true);
  return new Uint8Array(buffer);
}

async function readBufferBytes(device, buffer, byteLength) {
  const staging = device.createBuffer({
    size: alignTo4(byteLength),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, byteLength);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const bytes = new Uint8Array(staging.getMappedRange()).slice(0, byteLength);
  staging.unmap();
  staging.destroy();
  return bytes;
}

async function readFloat32Buffer(device, buffer, length) {
  const bytes = await readBufferBytes(device, buffer, length * 4);
  return new Float32Array(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + length * 4)
  );
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

function sanitizeGeneratedText(text) {
  return String(text ?? "")
    .replace(/<\|[^|>]+?\|>/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function shouldReportStep(stepIndex, totalSteps, buckets = 12) {
  if (!Number.isFinite(totalSteps) || totalSteps <= 0) {
    return false;
  }
  if (stepIndex <= 0 || stepIndex + 1 >= totalSteps) {
    return true;
  }
  const interval = Math.max(1, Math.floor(totalSteps / Math.max(1, buckets)));
  return (stepIndex + 1) % interval === 0;
}

function clampTokens(tokens, maxContextTokens) {
  if (tokens.length <= maxContextTokens) {
    return tokens;
  }
  return tokens.slice(tokens.length - maxContextTokens);
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

function resolveTokenizerTokenId(tokenizer, token) {
  if (!tokenizer || typeof token !== "string") {
    return null;
  }
  if (tokenizer.specialTokens?.has(token)) {
    return tokenizer.specialTokens.get(token);
  }
  if (tokenizer.encoder?.has(token)) {
    return tokenizer.encoder.get(token);
  }
  return null;
}

function buildStopTokens(config, tokenizerConfig = {}, tokenizer = null) {
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
  for (const token of normalizeTokenList(tokenizerConfig?.eos_token_id)) {
    if (typeof token === "number") {
      stop.add(token);
    }
  }
  for (const token of [
    tokenizerConfig?.eos_token,
    config?.eos_token,
    config?.text_config?.eos_token
  ]) {
    const tokenId = resolveTokenizerTokenId(tokenizer, token);
    if (typeof tokenId === "number") {
      stop.add(tokenId);
    }
  }
  return stop;
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

function splitGatedQueryProjection(vector, headCount, headDim) {
  const queries = new Array(headCount);
  const gate = new Float32Array(headCount * headDim);
  const fusedHeadDim = headDim * 2;

  for (let head = 0; head < headCount; head += 1) {
    const headStart = head * fusedHeadDim;
    const queryHead = vector.slice(headStart, headStart + headDim);
    const gateHead = vector.slice(headStart + headDim, headStart + fusedHeadDim);
    queries[head] = queryHead;
    gate.set(gateHead, head * headDim);
  }

  return { queries, gate };
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

function convKernelAt(weight, channel, kernelIndex) {
  const shape = weight.shape;
  if (shape.length === 2) {
    return weight.values[channel * shape[1] + kernelIndex];
  }
  if (shape.length !== 3) {
    throw new Error(`Unsupported conv tensor shape ${shape.join("x")}`);
  }
  if (shape[1] === 1) {
    return weight.values[channel * shape[2] + kernelIndex];
  }
  if (shape[2] === 1) {
    return weight.values[channel * shape[1] + kernelIndex];
  }
  return weight.values[channel * shape[1] * shape[2] + kernelIndex];
}

function detectTextPrefix(tensorNames) {
  const suffix = "layers.0.input_layernorm.weight";
  const match = tensorNames.find((name) => name.endsWith(suffix));
  if (!match) {
    throw new Error("Could not detect the language-model tensor prefix.");
  }
  return match.slice(0, -suffix.length);
}

function projectionKind(baseName, tensorNames) {
  if (tensorNames.has(`${baseName}.qweight`)) {
    return "gptq";
  }
  if (tensorNames.has(`${baseName}.weight`)) {
    return "dense";
  }
  return "missing";
}

function buildLayerPlans(textConfig, textPrefix) {
  const plans = [];
  for (let index = 0; index < textConfig.num_hidden_layers; index += 1) {
    const prefix = `${textPrefix}layers.${index}.`;
    const isLinear = (index + 1) % textConfig.full_attention_interval !== 0;
    plans.push({
      index,
      isLinear,
      inputNorm: `${prefix}input_layernorm.weight`,
      postNorm: `${prefix}post_attention_layernorm.weight`,
      gateProj: `${prefix}mlp.gate_proj`,
      upProj: `${prefix}mlp.up_proj`,
      downProj: `${prefix}mlp.down_proj`,
      attention: isLinear
        ? {
            inProjQkv: `${prefix}linear_attn.in_proj_qkv`,
            inProjZ: `${prefix}linear_attn.in_proj_z`,
            inProjB: `${prefix}linear_attn.in_proj_b`,
            inProjA: `${prefix}linear_attn.in_proj_a`,
            conv1d: `${prefix}linear_attn.conv1d.weight`,
            dtBias: `${prefix}linear_attn.dt_bias`,
            aLog: `${prefix}linear_attn.A_log`,
            norm: `${prefix}linear_attn.norm.weight`,
            outProj: `${prefix}linear_attn.out_proj`
          }
        : {
            qProj: `${prefix}self_attn.q_proj`,
            kProj: `${prefix}self_attn.k_proj`,
            vProj: `${prefix}self_attn.v_proj`,
            oProj: `${prefix}self_attn.o_proj`,
            qNorm: `${prefix}self_attn.q_norm.weight`,
            kNorm: `${prefix}self_attn.k_norm.weight`
          }
    });
  }
  return plans;
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
    const half = this.dims / 2;
    for (let index = 0; index < half; index += 1) {
      const inv = this.invFrequencies[index];
      const angle = offset * inv * this.scale;
      const cos = Math.cos(angle);
      const sin = Math.sin(angle);
      const low = head[index];
      const high = head[index + half];
      rotated[index] = low * cos - high * sin;
      rotated[index + half] = high * cos + low * sin;
    }
    return rotated;
  }
}

class FullAttentionCache {
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

class LinearAttentionCache {
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

class RemoteTensorStore {
  constructor({ modelId, revision, manifest, fetchRangeBytes, statusCallback = null }) {
    this.modelId = modelId;
    this.revision = revision;
    this.fetchRangeBytes = fetchRangeBytes;
    this.statusCallback = statusCallback ?? (() => {});
    this.descriptors = new Map((manifest?.tensors ?? []).map((tensor) => [tensor.name, tensor]));
    this.bytesCache = new Map();
    this.vectorCache = new Map();
    this.rowCache = new Map();
    this.cache = new OpfsCache("papertrail-qwen-custom-v2");
    this.slug = modelSlug(modelId);
    this.reportedFetches = new Set();
    this.transferProgress = new Map();
  }

  listNames() {
    return [...this.descriptors.keys()];
  }

  has(name) {
    return this.descriptors.has(name);
  }

  findBySuffix(suffix) {
    return this.listNames().find((name) => name.endsWith(suffix)) ?? null;
  }

  getDescriptor(name) {
    const descriptor = this.descriptors.get(name);
    if (!descriptor) {
      throw new Error(`Unknown tensor: ${name}`);
    }
    return descriptor;
  }

  #tensorCacheKey(name) {
    return `tensor:${this.slug}:${this.revision}:${name}`;
  }

  #rowCacheKey(name, rowIndex) {
    return `row:${this.slug}:${this.revision}:${name}:${rowIndex}`;
  }

  #rowsCacheKey(name, startRow, rowCount) {
    return `rows:${this.slug}:${this.revision}:${name}:${startRow}:${rowCount}`;
  }

  #reportTransferProgress(action, name, loaded, total) {
    if (!Number.isFinite(total) || total <= 0) {
      return;
    }
    const key = `${action}:${name}`;
    const percent = Math.max(0, Math.min(100, Math.floor((loaded / total) * 100)));
    const lastPercent = this.transferProgress.get(key) ?? -1;
    const shouldEmit = percent >= 100 || lastPercent === -1 || percent - lastPercent >= 2;
    if (!shouldEmit) {
      return;
    }
    this.transferProgress.set(key, percent);
    this.statusCallback({
      phase: action,
      detail: `${action === "warming-weights" ? "Caching" : "Loading"} ${name} (${percent}%)`
    });
  }

  #clearTransferProgress(action, name) {
    this.transferProgress.delete(`${action}:${name}`);
  }

  async #readCachedTensorRange(name, start, end) {
    const cacheKey = this.#tensorCacheKey(name);
    if (this.bytesCache.has(name)) {
      return this.bytesCache.get(name).slice(start, end);
    }
    return this.cache.readBytesRange(cacheKey, start, end);
  }

  async getTensorBytes(name) {
    if (this.bytesCache.has(name)) {
      return this.bytesCache.get(name);
    }

    const cacheKey = this.#tensorCacheKey(name);
    const cached = await this.cache.readBytes(cacheKey);
    if (cached) {
      this.bytesCache.set(name, cached);
      return cached;
    }

    const descriptor = this.getDescriptor(name);
    if (!this.reportedFetches.has(name)) {
      this.reportedFetches.add(name);
      this.statusCallback({
        phase: "loading-weights",
        detail: `Loading tensor ${name}…`
      });
    }
    const bytes = await this.fetchRangeBytes(
      this.modelId,
      descriptor.filePath,
      this.revision,
      descriptor.byteStart,
      descriptor.byteEnd,
      {
        onProgress: ({ loaded, total }) =>
          this.#reportTransferProgress("loading-weights", name, loaded, total)
      }
    );
    this.#clearTransferProgress("loading-weights", name);
    await this.cache.writeBytes(cacheKey, bytes);
    this.bytesCache.set(name, bytes);
    return bytes;
  }

  async ensureTensorCached(name) {
    if (this.bytesCache.has(name)) {
      return;
    }

    const cacheKey = this.#tensorCacheKey(name);
    const cached = await this.cache.readBytes(cacheKey);
    if (cached) {
      return;
    }

    const descriptor = this.getDescriptor(name);
    this.statusCallback({
      phase: "warming-weights",
      detail: `Caching tensor ${name}…`
    });
    const bytes = await this.fetchRangeBytes(
      this.modelId,
      descriptor.filePath,
      this.revision,
      descriptor.byteStart,
      descriptor.byteEnd,
      {
        onProgress: ({ loaded, total }) =>
          this.#reportTransferProgress("warming-weights", name, loaded, total)
      }
    );
    this.#clearTransferProgress("warming-weights", name);
    await this.cache.writeBytes(cacheKey, bytes);
  }

  async getVector(name, { shiftNorm = false } = {}) {
    const cacheKey = `${name}:${shiftNorm ? "shift" : "plain"}`;
    if (this.vectorCache.has(cacheKey)) {
      return this.vectorCache.get(cacheKey);
    }
    const bytes = await this.getTensorBytes(name);
    const values = decodeTensorBytesToFloat32(
      this.getDescriptor(name).dtype,
      toArrayBuffer(bytes)
    );
    if (shiftNorm) {
      for (let index = 0; index < values.length; index += 1) {
        values[index] += 1;
      }
    }
    this.vectorCache.set(cacheKey, values);
    return values;
  }

  async getStructuredVector(name, options = {}) {
    const values = await this.getVector(name, options);
    return {
      name,
      shape: this.getDescriptor(name).shape,
      values
    };
  }

  async getRow(name, rowIndex, { status = true } = {}) {
    const rowKey = `${name}:${rowIndex}`;
    if (this.rowCache.has(rowKey)) {
      return this.rowCache.get(rowKey);
    }

    const cacheKey = this.#rowCacheKey(name, rowIndex);
    const cached = await this.cache.readBytes(cacheKey);
    if (cached) {
      const row = decodeTensorBytesToFloat32(
        this.getDescriptor(name).dtype,
        toArrayBuffer(cached)
      );
      this.rowCache.set(rowKey, row);
      return row;
    }

    const descriptor = this.getDescriptor(name);
    if (descriptor.shape.length !== 2) {
      throw new Error(`Tensor ${name} is not row-addressable.`);
    }
    const rowWidth = descriptor.shape[1];
    const rowByteLength = rowWidth * (DTYPE_BYTE_SIZE[descriptor.dtype] ?? 0);
    const localStart = rowIndex * rowByteLength;
    const localEnd = localStart + rowByteLength;
    const cachedTensorSlice = await this.#readCachedTensorRange(name, localStart, localEnd);
    if (cachedTensorSlice?.byteLength === rowByteLength) {
      const row = decodeTensorBytesToFloat32(descriptor.dtype, toArrayBuffer(cachedTensorSlice));
      this.rowCache.set(rowKey, row);
      return row;
    }

    const start = descriptor.byteStart + localStart;
    const end = descriptor.byteStart + localEnd;
    if (status) {
      this.statusCallback({
        phase: "loading-weights",
        detail: `Loading ${name} row ${rowIndex + 1}…`
      });
    }
    const bytes = await this.fetchRangeBytes(
      this.modelId,
      descriptor.filePath,
      this.revision,
      start,
      end,
      {
        onProgress: ({ loaded, total }) =>
          this.#reportTransferProgress("loading-weights", `${name} row ${rowIndex + 1}`, loaded, total)
      }
    );
    this.#clearTransferProgress("loading-weights", `${name} row ${rowIndex + 1}`);
    await this.cache.writeBytes(cacheKey, bytes);
    const row = decodeTensorBytesToFloat32(descriptor.dtype, toArrayBuffer(bytes));
    this.rowCache.set(rowKey, row);
    return row;
  }

  async prefetchRows(name, rowIndices, { concurrency = 12, onProgress = null } = {}) {
    const uniqueRows = [...new Set(rowIndices.filter((value) => Number.isInteger(value)))];
    if (!uniqueRows.length) {
      return;
    }

    let completed = 0;
    const report = () => {
      onProgress?.({
        completed,
        total: uniqueRows.length,
        progress: uniqueRows.length ? completed / uniqueRows.length : 1,
        detail: `Caching prompt embeddings ${completed}/${uniqueRows.length}…`
      });
    };
    report();

    let cursor = 0;
    const workers = new Array(Math.min(concurrency, uniqueRows.length)).fill(null).map(async () => {
      while (cursor < uniqueRows.length) {
        const rowIndex = uniqueRows[cursor];
        cursor += 1;
        await this.getRow(name, rowIndex, { status: false });
        completed += 1;
        report();
      }
    });

    await Promise.all(workers);
  }

  async getRowsBytes(name, startRow, rowCount) {
    const cacheKey = this.#rowsCacheKey(name, startRow, rowCount);
    const cached = await this.cache.readBytes(cacheKey);
    if (cached) {
      return cached;
    }

    const descriptor = this.getDescriptor(name);
    const rowWidth = descriptor.shape[1];
    const rowByteLength = rowWidth * (DTYPE_BYTE_SIZE[descriptor.dtype] ?? 0);
    const localStart = startRow * rowByteLength;
    const localEnd = localStart + rowCount * rowByteLength;
    const cachedTensorSlice = await this.#readCachedTensorRange(name, localStart, localEnd);
    if (cachedTensorSlice?.byteLength === rowCount * rowByteLength) {
      return cachedTensorSlice;
    }

    const start = descriptor.byteStart + localStart;
    const end = descriptor.byteStart + localEnd;
    this.statusCallback({
      phase: "loading-weights",
      detail: `Loading ${name} rows ${startRow + 1}-${startRow + rowCount}…`
    });
    const bytes = await this.fetchRangeBytes(
      this.modelId,
      descriptor.filePath,
      this.revision,
      start,
      end,
      {
        onProgress: ({ loaded, total }) =>
          this.#reportTransferProgress(
            "loading-weights",
            `${name} rows ${startRow + 1}-${startRow + rowCount}`,
            loaded,
            total
          )
      }
    );
    this.#clearTransferProgress(
      "loading-weights",
      `${name} rows ${startRow + 1}-${startRow + rowCount}`
    );
    await this.cache.writeBytes(cacheKey, bytes);
    return bytes;
  }
}

class GpuProjectionRunner {
  constructor({ gpu, shaders, tensorStore, modelId, revision, quantConfig, statusCallback }) {
    this.gpu = gpu;
    this.shaders = shaders ?? {};
    this.tensorStore = tensorStore;
    this.modelId = modelId;
    this.revision = revision;
    this.quantConfig = quantConfig ?? {};
    this.statusCallback = statusCallback ?? (() => {});
    this.hasF16 = Boolean(gpu?.hasF16);
    this.projectionCache = new Map();
    this.quantizedChunkCache = new Map();
    this.quantCache = new OpfsCache("papertrail-qwen-int4-v2");
    this.slug = modelSlug(modelId);
  }

  #chunkMetaKey(name, startRow, rowCount) {
    return `qchunk-meta:${this.slug}:${this.revision}:${name}:${startRow}:${rowCount}`;
  }

  #chunkQweightKey(name, startRow, rowCount) {
    return `qchunk-qweight:${this.slug}:${this.revision}:${name}:${startRow}:${rowCount}`;
  }

  #chunkScalesKey(name, startRow, rowCount) {
    return `qchunk-scales:${this.slug}:${this.revision}:${name}:${startRow}:${rowCount}`;
  }

  async #readBufferBytes(buffer, byteLength) {
    return readBufferBytes(this.gpu.device, buffer, byteLength);
  }

  async readFloatBuffer(buffer, length) {
    return readFloat32Buffer(this.gpu.device, buffer, length);
  }

  async #denseMatrix(name) {
    if (this.projectionCache.has(name)) {
      return this.projectionCache.get(name);
    }
    const descriptor = this.tensorStore.getDescriptor(name);
    const bytes = await this.tensorStore.getTensorBytes(name);
    const entry = {
      kind: "dense",
      name,
      inputSize: descriptor.shape[1],
      outputSize: descriptor.shape[0],
      weightBuffer: this.gpu.createBufferFromData(
        `pt-dense:${name}`,
        bytes
      )
    };
    this.tensorStore.bytesCache.delete(name);
    this.projectionCache.set(name, entry);
    return entry;
  }

  async #projection(baseName) {
    if (this.projectionCache.has(baseName)) {
      return this.projectionCache.get(baseName);
    }

    const tensorNames = new Set(this.tensorStore.listNames());
    const kind = projectionKind(baseName, tensorNames);
    if (kind === "missing") {
      throw new Error(`Missing projection weights for ${baseName}`);
    }

    if (kind === "dense") {
      const matrix = await this.#denseMatrix(`${baseName}.weight`);
      this.projectionCache.set(baseName, matrix);
      return matrix;
    }

    const qweightName = `${baseName}.qweight`;
    const qzerosName = `${baseName}.qzeros`;
    const scalesName = `${baseName}.scales`;
    const qweightDescriptor = this.tensorStore.getDescriptor(qweightName);
    const scalesDescriptor = this.tensorStore.getDescriptor(scalesName);
    const qzerosDescriptor = this.tensorStore.has(qzerosName)
      ? this.tensorStore.getDescriptor(qzerosName)
      : null;
    const byteFetches = [
      this.tensorStore.getTensorBytes(qweightName),
      this.tensorStore.getTensorBytes(scalesName)
    ];
    if (qzerosDescriptor) {
      byteFetches.push(this.tensorStore.getTensorBytes(qzerosName));
    }
    const [qweightBytes, scalesBytes, qzerosBytes = null] = await Promise.all(byteFetches);

    const entry = {
      kind: "gptq",
      name: baseName,
      inputSize: qweightDescriptor.shape[0] * 8,
      outputSize: qweightDescriptor.shape[1],
      packedOutputSize:
        qzerosDescriptor?.shape?.[1] ??
        Math.ceil(qweightDescriptor.shape[1] / 8),
      groupSize:
        this.quantConfig.group_size ??
        this.quantConfig.groupSize ??
        Math.max(8, (qweightDescriptor.shape[0] * 8) / scalesDescriptor.shape[0]),
      symmetric: Boolean(this.quantConfig.sym),
      qweightBuffer: this.gpu.createBufferFromData(
        `pt-qweight:${baseName}`,
        qweightBytes
      ),
      qzerosBuffer: qzerosBytes
        ? this.gpu.createBufferFromData(
            `pt-qzeros:${baseName}`,
            qzerosBytes
          )
        : null,
      scalesBuffer: this.gpu.createBufferFromData(
        `pt-scales:${baseName}`,
        scalesBytes
      )
    };
    this.tensorStore.bytesCache.delete(qweightName);
    this.tensorStore.bytesCache.delete(qzerosName);
    this.tensorStore.bytesCache.delete(scalesName);
    this.projectionCache.set(baseName, entry);
    return entry;
  }

  async #runDense(weightBuffer, input, inputSize, outputSize) {
    const inputBuffer = this.gpu.createBufferFromData("pt-input", input);
    try {
      const outputBuffer = this.#runDenseToBuffer(
        weightBuffer,
        inputBuffer,
        inputSize,
        outputSize
      );
      try {
        return await this.readFloatBuffer(outputBuffer, outputSize);
      } finally {
        outputBuffer.destroy();
      }
    } finally {
      inputBuffer.destroy();
    }
  }

  #runDenseToBuffer(weightBuffer, inputBuffer, inputSize, outputSize) {
    const shader = this.shaders?.bf16_matvec;
    if (!shader) {
      throw new Error("Missing bf16_matvec shader.");
    }
    const outputBuffer = this.gpu.createBuffer(
      "pt-output",
      outputSize * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    );
    const params = this.gpu.createBufferFromData(
      "pt-dense-params",
      createUintParams([inputSize, outputSize, 0, 0]),
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );
    const pipeline = this.gpu.getOrCreatePipeline("pt-bf16-matvec", shader);
    const bindGroup = this.gpu.createBindGroup(pipeline, 0, [
      inputBuffer,
      weightBuffer,
      outputBuffer,
      params
    ]);
    this.gpu.dispatch(pipeline, [bindGroup], Math.ceil(outputSize / 256));
    params.destroy();
    return outputBuffer;
  }

  async #runGptq(qweightBuffer, scalesBuffer, input, inputSize, outputSize, groupSize) {
    const inputBuffer = this.gpu.createBufferFromData("pt-input", input);
    try {
      const outputBuffer = this.#runGptqToBuffer(
        qweightBuffer,
        scalesBuffer,
        inputBuffer,
        inputSize,
        outputSize,
        groupSize
      );
      try {
        return await this.readFloatBuffer(outputBuffer, outputSize);
      } finally {
        outputBuffer.destroy();
      }
    } finally {
      inputBuffer.destroy();
    }
  }

  #runGptqToBuffer(qweightBuffer, scalesBuffer, inputBuffer, inputSize, outputSize, groupSize) {
    const numGroups = inputSize / groupSize;
    const use4t = numGroups % 4 === 0;
    const shaderKey = use4t
      ? this.hasF16
        ? "gptq_matvec_4t_f16"
        : "gptq_matvec_4t"
      : this.hasF16
        ? "gptq_matvec_f16"
        : "gptq_matvec";
    const shader = this.shaders?.[shaderKey];
    if (!shader) {
      throw new Error(`Missing ${shaderKey} shader.`);
    }

    const outputBuffer = this.gpu.createBuffer(
      "pt-output",
      outputSize * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    );
    const params = this.gpu.createBufferFromData(
      "pt-gptq-params",
      createUintParams([inputSize, outputSize, groupSize, 0]),
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );
    const pipeline = this.gpu.getOrCreatePipeline(`pt-${shaderKey}`, shader);
    const bindGroup = this.gpu.createBindGroup(pipeline, 0, [
      inputBuffer,
      qweightBuffer,
      scalesBuffer,
      outputBuffer,
      params
    ]);
    this.gpu.dispatch(
      pipeline,
      [bindGroup],
      use4t ? Math.ceil(outputSize / 8) : Math.ceil(outputSize / 32)
    );
    params.destroy();
    return outputBuffer;
  }

  async #runGptqWithQzeros(
    qweightBuffer,
    qzerosBuffer,
    scalesBuffer,
    input,
    inputSize,
    outputSize,
    groupSize,
    packedOutputSize,
    symmetric
  ) {
    if (!qzerosBuffer) {
      throw new Error("GPTQ zero-point buffer is unavailable.");
    }
    const inputBuffer = this.gpu.createBufferFromData("pt-input", input);
    try {
      const outputBuffer = this.#runGptqWithQzerosToBuffer(
        qweightBuffer,
        qzerosBuffer,
        scalesBuffer,
        inputBuffer,
        inputSize,
        outputSize,
        groupSize,
        packedOutputSize,
        symmetric
      );
      try {
        return await this.readFloatBuffer(outputBuffer, outputSize);
      } finally {
        outputBuffer.destroy();
      }
    } finally {
      inputBuffer.destroy();
    }
  }

  #runGptqWithQzerosToBuffer(
    qweightBuffer,
    qzerosBuffer,
    scalesBuffer,
    inputBuffer,
    inputSize,
    outputSize,
    groupSize,
    packedOutputSize,
    symmetric
  ) {
    const outputBuffer = this.gpu.createBuffer(
      "pt-output",
      outputSize * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    );
    const params = this.gpu.createBufferFromData(
      "pt-gptq-qzeros-params",
      new Uint32Array([
        inputSize,
        outputSize,
        groupSize,
        packedOutputSize,
        symmetric ? 1 : 0,
        0,
        0,
        0
      ]),
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );
    const pipeline = this.gpu.getOrCreatePipeline(
      "pt-gptq-qzeros-matvec",
      PAPERTRAIL_GPTQ_QZEROS_MATVEC_WGSL
    );
    const bindGroup = this.gpu.createBindGroup(pipeline, 0, [
      inputBuffer,
      qweightBuffer,
      qzerosBuffer,
      scalesBuffer,
      outputBuffer,
      params
    ]);
    this.gpu.dispatch(pipeline, [bindGroup], Math.ceil(outputSize / 64));
    params.destroy();
    return outputBuffer;
  }

  async project(baseName, input) {
    const projection = await this.#projection(baseName);
    if (input.length !== projection.inputSize) {
      throw new Error(
        `Projection ${baseName} expected ${projection.inputSize} inputs, got ${input.length}`
      );
    }
    const inputBuffer = this.gpu.createBufferFromData("pt-input", input);
    try {
      const outputBuffer = await this.projectToBuffer(baseName, inputBuffer, projection.inputSize);
      try {
        return await this.readFloatBuffer(outputBuffer, projection.outputSize);
      } finally {
        outputBuffer.destroy();
      }
    } finally {
      inputBuffer.destroy();
    }
  }

  async projectToBuffer(baseName, inputBuffer, inputSize) {
    const projection = await this.#projection(baseName);
    if (inputSize !== projection.inputSize) {
      throw new Error(
        `Projection ${baseName} expected ${projection.inputSize} inputs, got ${inputSize}`
      );
    }
    return projection.kind === "dense"
      ? this.#runDenseToBuffer(
          projection.weightBuffer,
          inputBuffer,
          projection.inputSize,
          projection.outputSize
        )
      : projection.qzerosBuffer
        ? this.#runGptqWithQzerosToBuffer(
            projection.qweightBuffer,
            projection.qzerosBuffer,
            projection.scalesBuffer,
            inputBuffer,
            projection.inputSize,
            projection.outputSize,
            projection.groupSize,
            projection.packedOutputSize,
            projection.symmetric
          )
        : this.#runGptqToBuffer(
            projection.qweightBuffer,
            projection.scalesBuffer,
            inputBuffer,
            projection.inputSize,
            projection.outputSize,
            projection.groupSize
          );
  }

  async projectDenseByName(name, input) {
    const matrix = await this.#denseMatrix(name);
    if (input.length !== matrix.inputSize) {
      throw new Error(`Dense matrix ${name} expected ${matrix.inputSize} inputs, got ${input.length}`);
    }
    return this.#runDense(matrix.weightBuffer, input, matrix.inputSize, matrix.outputSize);
  }

  async #argmaxBuffer(logitsBuffer, length) {
    const shader = this.shaders?.argmax;
    if (!shader) {
      throw new Error("Missing argmax shader.");
    }
    const resultBuffer = this.gpu.createBuffer(
      "pt-argmax-result",
      8,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    );
    const params = this.gpu.createBufferFromData(
      "pt-argmax-params",
      new Uint32Array([length]),
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );
    try {
      const pipeline = this.gpu.getOrCreatePipeline("pt-argmax", shader);
      const bindGroup = this.gpu.createBindGroup(pipeline, 0, [
        logitsBuffer,
        resultBuffer,
        params
      ]);
      this.gpu.dispatch(pipeline, [bindGroup], 1);
      const bytes = await this.#readBufferBytes(resultBuffer, 8);
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      return {
        index: view.getUint32(0, true),
        value: view.getFloat32(4, true)
      };
    } finally {
      resultBuffer.destroy();
      params.destroy();
    }
  }

  async #quantizeChunk(denseName, startRow, rowCount, inputSize) {
    const key = `${denseName}:${startRow}:${rowCount}`;
    if (this.quantizedChunkCache.has(key)) {
      return this.quantizedChunkCache.get(key);
    }

    const metaKey = this.#chunkMetaKey(denseName, startRow, rowCount);
    const qweightKey = this.#chunkQweightKey(denseName, startRow, rowCount);
    const scalesKey = this.#chunkScalesKey(denseName, startRow, rowCount);
    const cachedMeta = await this.quantCache.readJson(metaKey);
    const [cachedQweight, cachedScales] = await Promise.all([
      this.quantCache.readBytes(qweightKey),
      this.quantCache.readBytes(scalesKey)
    ]);

    if (cachedMeta && cachedQweight && cachedScales) {
      const entry = {
        inputSize,
        rowCount,
        startRow,
        qweightBuffer: this.gpu.createBufferFromData(
          `pt-lmhead-qweight:${startRow}`,
          cachedQweight
        ),
        scalesBuffer: this.gpu.createBufferFromData(
          `pt-lmhead-scales:${startRow}`,
          cachedScales
        )
      };
      this.quantizedChunkCache.set(key, entry);
      return entry;
    }

    this.statusCallback({
      phase: "quantizing-lm-head",
      detail: `Quantizing lm_head rows ${startRow + 1}-${startRow + rowCount}…`
    });

    const bf16Bytes = await this.tensorStore.getRowsBytes(denseName, startRow, rowCount);
    const bf16Buffer = this.gpu.createBufferFromData(
      `pt-lmhead-bf16:${startRow}`,
      bf16Bytes
    );
    const qweightByteLength = (inputSize / 8) * rowCount * 4;
    const numScales = (inputSize / LM_HEAD_GROUP_SIZE) * rowCount;
    const packedScaleCount = Math.ceil(numScales / 2);
    const qweightBuffer = this.gpu.createBuffer(
      `pt-lmhead-qweight:${startRow}`,
      qweightByteLength,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    );
    const scalesF32Buffer = this.gpu.createBuffer(
      `pt-lmhead-scales-f32:${startRow}`,
      numScales * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    );
    const packedScalesBuffer = this.gpu.createBuffer(
      `pt-lmhead-scales-packed:${startRow}`,
      packedScaleCount * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    );
    const quantParams = this.gpu.createBufferFromData(
      `pt-lmhead-quant-params:${startRow}`,
      new Uint32Array([inputSize, rowCount, LM_HEAD_GROUP_SIZE]),
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );
    const packParams = this.gpu.createBufferFromData(
      `pt-lmhead-pack-params:${startRow}`,
      new Uint32Array([packedScaleCount, 65535]),
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );

    try {
      const quantPipeline = this.gpu.getOrCreatePipeline(
        "pt-quantize-bf16-to-int4",
        this.shaders.quantize_bf16_to_int4
      );
      const quantBindGroup = this.gpu.createBindGroup(quantPipeline, 0, [
        bf16Buffer,
        qweightBuffer,
        scalesF32Buffer,
        quantParams
      ]);
      this.gpu.dispatch(
        quantPipeline,
        [quantBindGroup],
        Math.min(rowCount, 65535),
        Math.ceil(rowCount / 65535)
      );

      const packPipeline = this.gpu.getOrCreatePipeline(
        "pt-pack-scales-f16",
        this.shaders.pack_f32_to_f16_pairs
      );
      const packBindGroup = this.gpu.createBindGroup(packPipeline, 0, [
        scalesF32Buffer,
        packedScalesBuffer,
        packParams
      ]);
      this.gpu.dispatch(
        packPipeline,
        [packBindGroup],
        Math.min(packedScaleCount, 65535),
        Math.ceil(packedScaleCount / 65535)
      );

      const [qweightBytes, scalesBytes] = await Promise.all([
        this.#readBufferBytes(qweightBuffer, qweightByteLength),
        this.#readBufferBytes(packedScalesBuffer, packedScaleCount * 4)
      ]);

      await Promise.all([
        this.quantCache.writeJson(metaKey, {
          modelId: this.modelId,
          revision: this.revision,
          denseName,
          startRow,
          rowCount,
          inputSize,
          groupSize: LM_HEAD_GROUP_SIZE
        }),
        this.quantCache.writeBytes(qweightKey, qweightBytes),
        this.quantCache.writeBytes(scalesKey, scalesBytes)
      ]);
    } finally {
      bf16Buffer.destroy();
      scalesF32Buffer.destroy();
      quantParams.destroy();
      packParams.destroy();
    }

    const entry = {
      inputSize,
      rowCount,
      startRow,
      qweightBuffer,
      scalesBuffer: packedScalesBuffer
    };
    this.quantizedChunkCache.set(key, entry);
    return entry;
  }

  async prequantizeLmHead(denseName, { onProgress = null } = {}) {
    const descriptor = this.tensorStore.getDescriptor(denseName);
    const vocabSize = descriptor.shape[0];
    const inputSize = descriptor.shape[1];
    const totalChunks = Math.ceil(vocabSize / LM_HEAD_CHUNK_ROWS);
    for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex += 1) {
      const startRow = chunkIndex * LM_HEAD_CHUNK_ROWS;
      const rowCount = Math.min(LM_HEAD_CHUNK_ROWS, vocabSize - startRow);
      onProgress?.({
        chunkIndex: chunkIndex + 1,
        totalChunks,
        progress: (chunkIndex + 1) / totalChunks,
        detail: `Prepacking lm_head chunk ${chunkIndex + 1}/${totalChunks}…`
      });
      await this.#quantizeChunk(denseName, startRow, rowCount, inputSize);
    }
  }

  async prefetchProjection(baseName, { uploadToGpu = false } = {}) {
    if (uploadToGpu || this.projectionCache.has(baseName)) {
      await this.#projection(baseName);
      return;
    }

    const tensorNames = new Set(this.tensorStore.listNames());
    const kind = projectionKind(baseName, tensorNames);
    if (kind === "missing") {
      throw new Error(`Missing projection weights for ${baseName}`);
    }
    if (kind === "dense") {
      await this.tensorStore.ensureTensorCached(`${baseName}.weight`);
      return;
    }
    await Promise.all([
      this.tensorStore.ensureTensorCached(`${baseName}.qweight`),
      this.tensorStore.ensureTensorCached(`${baseName}.scales`),
      this.tensorStore.has(`${baseName}.qzeros`)
        ? this.tensorStore.ensureTensorCached(`${baseName}.qzeros`)
        : Promise.resolve()
    ]);
  }

  async nextTokenFromLmHead(input, denseName) {
    const descriptor = this.tensorStore.getDescriptor(denseName);
    const inputSize = descriptor.shape[1];
    if (input.length !== inputSize) {
      throw new Error(`lm_head expected ${inputSize} inputs, got ${input.length}`);
    }
    const inputBuffer = this.gpu.createBufferFromData("pt-lmhead-input", input);
    try {
      return await this.nextTokenFromLmHeadBuffer(inputBuffer, inputSize, denseName);
    } finally {
      inputBuffer.destroy();
    }
  }

  async nextTokenFromLmHeadBuffer(inputBuffer, inputSize, denseName) {
    const descriptor = this.tensorStore.getDescriptor(denseName);
    const vocabSize = descriptor.shape[0];
    if (inputSize !== descriptor.shape[1]) {
      throw new Error(`lm_head expected ${descriptor.shape[1]} inputs, got ${inputSize}`);
    }

    let bestToken = 0;
    let bestValue = Number.NEGATIVE_INFINITY;
    const totalChunks = Math.ceil(vocabSize / LM_HEAD_CHUNK_ROWS);

    const numGroups = inputSize / LM_HEAD_GROUP_SIZE;
    const use4t = numGroups % 4 === 0;
    const shaderKey = use4t
      ? this.hasF16
        ? "gptq_matvec_4t_f16"
        : "gptq_matvec_4t"
      : this.hasF16
        ? "gptq_matvec_f16"
        : "gptq_matvec";
    const shader = this.shaders?.[shaderKey];
    if (!shader) {
      throw new Error(`Missing ${shaderKey} shader.`);
    }
    const pipeline = this.gpu.getOrCreatePipeline(`pt-lmhead-${shaderKey}`, shader);

    for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex += 1) {
      const startRow = chunkIndex * LM_HEAD_CHUNK_ROWS;
      const rowCount = Math.min(LM_HEAD_CHUNK_ROWS, vocabSize - startRow);
      const quantizedChunk = await this.#quantizeChunk(denseName, startRow, rowCount, inputSize);
      const logitsBuffer = this.gpu.createBuffer(
        `pt-lmhead-logits:${chunkIndex}`,
        rowCount * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
      );
      const params = this.gpu.createBufferFromData(
        `pt-lmhead-params:${chunkIndex}`,
        createUintParams([inputSize, rowCount, LM_HEAD_GROUP_SIZE, 0]),
        GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      );
      try {
        const bindGroup = this.gpu.createBindGroup(pipeline, 0, [
          inputBuffer,
          quantizedChunk.qweightBuffer,
          quantizedChunk.scalesBuffer,
          logitsBuffer,
          params
        ]);
        this.gpu.dispatch(
          pipeline,
          [bindGroup],
          use4t ? Math.ceil(rowCount / 8) : Math.ceil(rowCount / 32)
        );
        const localBest = await this.#argmaxBuffer(logitsBuffer, rowCount);
        if (localBest.value > bestValue) {
          bestValue = localBest.value;
          bestToken = startRow + localBest.index;
        }
      } finally {
        logitsBuffer.destroy();
        params.destroy();
      }
    }

    return bestToken;
  }
}

function shouldShiftNormWeights(tensorNames, descriptors) {
  for (const name of tensorNames) {
    if (name.includes("mtp.")) {
      return true;
    }
    if (name.endsWith("conv1d.weight")) {
      const descriptor = descriptors.get(name);
      if (descriptor?.shape?.length === 3 && descriptor.shape.at(-1) !== 1) {
        return true;
      }
    }
  }
  return false;
}

export class PapertrailQwenCustomEngine {
  constructor({
    modelId,
    revision,
    config,
    manifest,
    tokenizer,
    tokenizerConfig,
    fetchRangeBytes,
    gpu,
    shaders,
    quantConfig,
    contextLength = 1024,
    statusCallback = null
  }) {
    this.modelId = modelId;
    this.revision = revision;
    this.config = config;
    this.textConfig = config?.text_config ?? config ?? {};
    this.tokenizer = tokenizer;
    this.tokenizerConfig = tokenizerConfig ?? {};
    this.contextLength = contextLength;
    this.statusCallback = statusCallback ?? (() => {});
    this.tensorStore = new RemoteTensorStore({
      modelId,
      revision,
      manifest,
      fetchRangeBytes,
      statusCallback
    });
    this.gpu = gpu;
    this.shaders = shaders ?? {};
    this.tensorNames = this.tensorStore.listNames();
    this.textPrefix = detectTextPrefix(this.tensorNames);
    this.layerPlans = buildLayerPlans(this.textConfig, this.textPrefix);
    this.embedName = `${this.textPrefix}embed_tokens.weight`;
    this.finalNormName = `${this.textPrefix}norm.weight`;
    this.lmHeadName =
      this.tensorStore.findBySuffix("lm_head.weight") ??
      (this.textConfig.tie_word_embeddings ? this.embedName : null);
    this.stopTokens = buildStopTokens(config, tokenizerConfig, tokenizer);
    this.shouldShiftNormWeights = shouldShiftNormWeights(
      this.tensorNames,
      this.tensorStore.descriptors
    );
    this.headDim =
      this.textConfig.head_dim ??
      Math.floor(this.textConfig.hidden_size / this.textConfig.num_attention_heads);
    this.partialRotaryFactor =
      this.textConfig.partial_rotary_factor ??
      this.textConfig.rope_parameters?.partial_rotary_factor ??
      0.25;
    this.rotary = new RotaryEmbedding({
      dims: Math.max(2, 2 * Math.floor((this.headDim * this.partialRotaryFactor) / 2)),
      base:
        this.textConfig.rope_theta ??
        this.textConfig.rope_parameters?.rope_theta ??
        100000,
      ropeScaling: this.textConfig.rope_scaling ?? this.textConfig.rope_parameters ?? null
    });
    this.projector = new GpuProjectionRunner({
      gpu,
      shaders,
      tensorStore: this.tensorStore,
      modelId,
      revision,
      quantConfig,
      statusCallback
    });
    this.gpuWeightBufferCache = new Map();
  }

  async prepareForInference({ onProgress = null, warmupMode = "essential" } = {}) {
    const vectorNames = new Set([this.finalNormName]);
    const projectionNames = new Set();

    for (const layerPlan of this.layerPlans) {
      vectorNames.add(layerPlan.inputNorm);
      vectorNames.add(layerPlan.postNorm);
      projectionNames.add(layerPlan.gateProj);
      projectionNames.add(layerPlan.upProj);
      projectionNames.add(layerPlan.downProj);
      if (layerPlan.isLinear) {
        vectorNames.add(layerPlan.attention.norm);
        vectorNames.add(layerPlan.attention.dtBias);
        vectorNames.add(layerPlan.attention.aLog);
        vectorNames.add(layerPlan.attention.conv1d);
        projectionNames.add(layerPlan.attention.inProjQkv);
        projectionNames.add(layerPlan.attention.inProjZ);
        projectionNames.add(layerPlan.attention.inProjB);
        projectionNames.add(layerPlan.attention.inProjA);
        projectionNames.add(layerPlan.attention.outProj);
      } else {
        vectorNames.add(layerPlan.attention.qNorm);
        vectorNames.add(layerPlan.attention.kNorm);
        projectionNames.add(layerPlan.attention.qProj);
        projectionNames.add(layerPlan.attention.kProj);
        projectionNames.add(layerPlan.attention.vProj);
        projectionNames.add(layerPlan.attention.oProj);
      }
    }

    const vectorList = [...vectorNames];
    for (let index = 0; index < vectorList.length; index += 1) {
      const name = vectorList[index];
      onProgress?.({
        phase: "warmup-vectors",
        progress: vectorList.length ? (index + 1) / vectorList.length : 1,
        detail: `Caching vector ${index + 1}/${vectorList.length}: ${name}`
      });
      if (name.endsWith("conv1d.weight")) {
        await this.tensorStore.getStructuredVector(name);
      } else {
        await this.#weightVector(name);
      }
      await this.#gpuWeightBuffer(name, {
        shiftNorm:
          this.shouldShiftNormWeights &&
          (
            name.endsWith(".input_layernorm.weight") ||
            name.endsWith(".post_attention_layernorm.weight") ||
            name.endsWith(".q_norm.weight") ||
            name.endsWith(".k_norm.weight") ||
            name === this.finalNormName
          )
      });
    }

    if (warmupMode !== "full") {
      onProgress?.({
        phase: "warmup-essential-ready",
        progress: 1,
        detail: "Essential decoder weights are ready. Large tensors will stream on first use."
      });
      return;
    }

    onProgress?.({
      phase: "warmup-embeddings",
      progress: 0,
      detail: `Caching full embedding tensor: ${this.embedName}`
    });
    await this.tensorStore.ensureTensorCached(this.embedName);

    const projectionList = [...projectionNames];
    for (let index = 0; index < projectionList.length; index += 1) {
      const name = projectionList[index];
      onProgress?.({
        phase: "warmup-projections",
        progress: projectionList.length ? (index + 1) / projectionList.length : 1,
        detail: `Caching projection ${index + 1}/${projectionList.length}: ${name}`
      });
      await this.projector.prefetchProjection(name, { uploadToGpu: false });
    }

    if (this.lmHeadName && this.lmHeadName !== this.embedName) {
      await this.projector.prequantizeLmHead(this.lmHeadName, {
        onProgress: ({ progress, detail }) => {
          onProgress?.({
            phase: "warmup-lm-head",
            progress,
            detail
          });
        }
      });
    } else if (this.lmHeadName) {
      await this.projector.prequantizeLmHead(this.lmHeadName, {
        onProgress: ({ progress, detail }) => {
          onProgress?.({
            phase: "warmup-lm-head",
            progress,
            detail
          });
        }
      });
    }
  }

  makeCaches() {
    return this.layerPlans.map((layer) =>
      layer.isLinear
        ? new LinearAttentionCache({
            convKernelSize: this.textConfig.linear_conv_kernel_dim,
            convDim:
              this.textConfig.linear_num_key_heads * this.textConfig.linear_key_head_dim * 2 +
              this.textConfig.linear_num_value_heads * this.textConfig.linear_value_head_dim,
            numValueHeads: this.textConfig.linear_num_value_heads,
            valueHeadDim: this.textConfig.linear_value_head_dim,
            keyHeadDim: this.textConfig.linear_key_head_dim
          })
        : new FullAttentionCache()
    );
  }

  async #weightVector(name) {
    return this.tensorStore.getVector(name, {
      shiftNorm:
        this.shouldShiftNormWeights &&
        (
          name.endsWith(".input_layernorm.weight") ||
          name.endsWith(".post_attention_layernorm.weight") ||
          name.endsWith(".q_norm.weight") ||
          name.endsWith(".k_norm.weight") ||
          name === this.finalNormName
        )
    });
  }

  async #gpuWeightBuffer(name, { shiftNorm = false } = {}) {
    const cacheKey = `${name}:${shiftNorm ? "shift" : "plain"}`;
    if (this.gpuWeightBufferCache.has(cacheKey)) {
      return this.gpuWeightBufferCache.get(cacheKey);
    }
    const values = await this.tensorStore.getVector(name, { shiftNorm });
    const buffer = this.gpu.createBufferFromData(`pt-weight:${cacheKey}`, values);
    this.gpuWeightBufferCache.set(cacheKey, buffer);
    return buffer;
  }

  async #makeHiddenBuffer(tokenId) {
    const row = await this.embedToken(tokenId);
    return this.gpu.createBufferFromData(`pt-hidden:${tokenId}`, row);
  }

  #destroyBuffer(buffer) {
    buffer?.destroy?.();
  }

  async #rmsNormToBuffer(inputBuffer, weightBuffer, length) {
    const outputBuffer = this.gpu.createBuffer(
      "pt-rmsnorm-output",
      length * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    );
    const params = this.gpu.createBufferFromData(
      "pt-rmsnorm-params",
      createRmsNormParams(length, this.textConfig.rms_norm_eps),
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );
    const pipeline = this.gpu.getOrCreatePipeline(
      "pt-rmsnorm-custom",
      PAPERTRAIL_RMSNORM_WGSL
    );
    const bindGroup = this.gpu.createBindGroup(pipeline, 0, [
      inputBuffer,
      weightBuffer,
      outputBuffer,
      params
    ]);
    this.gpu.dispatch(pipeline, [bindGroup], 1);
    params.destroy();
    return outputBuffer;
  }

  async #addBuffers(leftBuffer, rightBuffer, length) {
    const outputBuffer = this.gpu.createBuffer(
      "pt-add-output",
      length * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    );
    const params = this.gpu.createBufferFromData(
      "pt-add-params",
      createUintParams([length, 0, 0, 0]),
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );
    const pipeline = this.gpu.getOrCreatePipeline(
      "pt-add-custom",
      PAPERTRAIL_VECTOR_ADD_WGSL
    );
    const bindGroup = this.gpu.createBindGroup(pipeline, 0, [
      leftBuffer,
      rightBuffer,
      outputBuffer,
      params
    ]);
    this.gpu.dispatch(pipeline, [bindGroup], Math.ceil(length / 256));
    params.destroy();
    return outputBuffer;
  }

  async #swigluToBuffer(gateBuffer, upBuffer, length) {
    const outputBuffer = this.gpu.createBuffer(
      "pt-swiglu-output",
      length * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
    );
    const params = this.gpu.createBufferFromData(
      "pt-swiglu-params",
      createUintParams([length, 0, 0, 0]),
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );
    const pipeline = this.gpu.getOrCreatePipeline(
      "pt-swiglu-custom",
      PAPERTRAIL_SWIGLU_WGSL
    );
    const bindGroup = this.gpu.createBindGroup(pipeline, 0, [
      gateBuffer,
      upBuffer,
      outputBuffer,
      params
    ]);
    this.gpu.dispatch(pipeline, [bindGroup], Math.ceil(length / 256));
    params.destroy();
    return outputBuffer;
  }

  async embedToken(tokenId) {
    return this.tensorStore.getRow(this.embedName, tokenId);
  }

  async #runMlp(inputBuffer, layerPlan) {
    const hiddenSize = this.textConfig.hidden_size;
    const intermediateSize = this.textConfig.intermediate_size;
    const [gateBuffer, upBuffer] = await Promise.all([
      this.projector.projectToBuffer(layerPlan.gateProj, inputBuffer, hiddenSize),
      this.projector.projectToBuffer(layerPlan.upProj, inputBuffer, hiddenSize)
    ]);
    const swigluBuffer = await this.#swigluToBuffer(
      gateBuffer,
      upBuffer,
      intermediateSize
    );
    this.#destroyBuffer(gateBuffer);
    this.#destroyBuffer(upBuffer);
    try {
      return await this.projector.projectToBuffer(
        layerPlan.downProj,
        swigluBuffer,
        intermediateSize
      );
    } finally {
      this.#destroyBuffer(swigluBuffer);
    }
  }

  async #runAttention(inputBuffer, layerPlan, cache) {
    const [qNormWeight, kNormWeight] = await Promise.all([
      this.#weightVector(layerPlan.attention.qNorm),
      this.#weightVector(layerPlan.attention.kNorm)
    ]);
    const hiddenSize = this.textConfig.hidden_size;
    const [qBuffer, kBuffer, vBuffer] = await Promise.all([
      this.projector.projectToBuffer(layerPlan.attention.qProj, inputBuffer, hiddenSize),
      this.projector.projectToBuffer(layerPlan.attention.kProj, inputBuffer, hiddenSize),
      this.projector.projectToBuffer(layerPlan.attention.vProj, inputBuffer, hiddenSize)
    ]);
    const [qRaw, kRaw, vRaw] = await Promise.all([
      this.projector.readFloatBuffer(
        qBuffer,
        this.textConfig.num_attention_heads * this.headDim * 2
      ),
      this.projector.readFloatBuffer(
        kBuffer,
        this.textConfig.num_key_value_heads * this.headDim
      ),
      this.projector.readFloatBuffer(
        vBuffer,
        this.textConfig.num_key_value_heads * this.headDim
      )
    ]);
    this.#destroyBuffer(qBuffer);
    this.#destroyBuffer(kBuffer);
    this.#destroyBuffer(vBuffer);

    const { queries: fusedQueries, gate } = splitGatedQueryProjection(
      qRaw,
      this.textConfig.num_attention_heads,
      this.headDim
    );

    const queries = fusedQueries.map((head) =>
      this.rotary.apply(
        rmsNorm(head, qNormWeight, this.textConfig.rms_norm_eps),
        cache.offset
      )
    );

    const keys = reshapeHeads(
      kRaw,
      this.textConfig.num_key_value_heads,
      this.headDim
    ).map((head) =>
      this.rotary.apply(
        rmsNorm(head, kNormWeight, this.textConfig.rms_norm_eps),
        cache.offset
      )
    );
    const values = reshapeHeads(
      vRaw,
      this.textConfig.num_key_value_heads,
      this.headDim
    );

    cache.append(keys, values);

    const groupSize =
      this.textConfig.num_attention_heads / this.textConfig.num_key_value_heads;
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
    const flattenedBuffer = this.gpu.createBufferFromData("pt-attn-flat", flattened);
    try {
      return await this.projector.projectToBuffer(
        layerPlan.attention.oProj,
        flattenedBuffer,
        flattened.length
      );
    } finally {
      this.#destroyBuffer(flattenedBuffer);
    }
  }

  async #runLinearAttention(inputBuffer, layerPlan, cache) {
    const nk = this.textConfig.linear_num_key_heads;
    const nv = this.textConfig.linear_num_value_heads;
    const dk = this.textConfig.linear_key_head_dim;
    const dv = this.textConfig.linear_value_head_dim;
    const [
      normWeight,
      convWeight,
      dtBias,
      aLog,
      qkvBuffer,
      zBuffer,
      bBuffer,
      aBuffer
    ] = await Promise.all([
      this.#weightVector(layerPlan.attention.norm),
      this.tensorStore.getStructuredVector(layerPlan.attention.conv1d),
      this.#weightVector(layerPlan.attention.dtBias),
      this.#weightVector(layerPlan.attention.aLog),
      this.projector.projectToBuffer(
        layerPlan.attention.inProjQkv,
        inputBuffer,
        this.textConfig.hidden_size
      ),
      this.projector.projectToBuffer(
        layerPlan.attention.inProjZ,
        inputBuffer,
        this.textConfig.hidden_size
      ),
      this.projector.projectToBuffer(
        layerPlan.attention.inProjB,
        inputBuffer,
        this.textConfig.hidden_size
      ),
      this.projector.projectToBuffer(
        layerPlan.attention.inProjA,
        inputBuffer,
        this.textConfig.hidden_size
      )
    ]);
    const [qkv, zFlat, b, a] = await Promise.all([
      this.projector.readFloatBuffer(
        qkvBuffer,
        nk * dk * 2 + nv * dv
      ),
      this.projector.readFloatBuffer(zBuffer, nv * dv),
      this.projector.readFloatBuffer(bBuffer, nv),
      this.projector.readFloatBuffer(aBuffer, nv)
    ]);
    this.#destroyBuffer(qkvBuffer);
    this.#destroyBuffer(zBuffer);
    this.#destroyBuffer(bBuffer);
    this.#destroyBuffer(aBuffer);

    const headsPerKey = nv / nk;

    const q = new Array(nk);
    const k = new Array(nk);
    const v = new Array(nv);
    const z = new Array(nv);

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
        total += convWindow[step][channel] * convKernelAt(convWeight, channel, step);
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
      const g = Math.exp(
        -Math.exp(aLog[valueHead]) * softplus(a[valueHead] + dtBias[valueHead])
      );
      const beta = sigmoid(b[valueHead]);
      const output = new Float32Array(dv);

      for (let dValue = 0; dValue < dv; dValue += 1) {
        let kvMem = 0;
        for (let dKey = 0; dKey < dk; dKey += 1) {
          const stateIndex = valueHead * dv * dk + dValue * dk + dKey;
          cache.state[stateIndex] *= g;
          kvMem += cache.state[stateIndex] * kHead[dKey];
        }

        const delta = (vConv[valueHead][dValue] - kvMem) * beta;
        let headOut = 0;
        for (let dKey = 0; dKey < dk; dKey += 1) {
          const stateIndex = valueHead * dv * dk + dValue * dk + dKey;
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

    const flattened = flattenHeads(outputHeads);
    const flattenedBuffer = this.gpu.createBufferFromData("pt-linear-flat", flattened);
    try {
      return await this.projector.projectToBuffer(
        layerPlan.attention.outProj,
        flattenedBuffer,
        flattened.length
      );
    } finally {
      this.#destroyBuffer(flattenedBuffer);
    }
  }

  async #nextTokenFromHidden(hiddenBuffer) {
    const finalNormBuffer = await this.#gpuWeightBuffer(this.finalNormName, {
      shiftNorm: this.shouldShiftNormWeights
    });
    const normalizedBuffer = await this.#rmsNormToBuffer(
      hiddenBuffer,
      finalNormBuffer,
      this.textConfig.hidden_size
    );
    if (!this.lmHeadName) {
      this.#destroyBuffer(normalizedBuffer);
      throw new Error("lm_head is unavailable for this checkpoint.");
    }
    try {
      return await this.projector.nextTokenFromLmHeadBuffer(
        normalizedBuffer,
        this.textConfig.hidden_size,
        this.lmHeadName
      );
    } finally {
      this.#destroyBuffer(normalizedBuffer);
    }
  }

  async forwardHidden(tokenId, caches) {
    let hiddenBuffer = await this.#makeHiddenBuffer(tokenId);

    for (const [index, layerPlan] of this.layerPlans.entries()) {
      const [inputNormWeight, postNormWeight] = await Promise.all([
        this.#gpuWeightBuffer(layerPlan.inputNorm, {
          shiftNorm:
            this.shouldShiftNormWeights &&
            layerPlan.inputNorm.endsWith(".input_layernorm.weight")
        }),
        this.#gpuWeightBuffer(layerPlan.postNorm, {
          shiftNorm:
            this.shouldShiftNormWeights &&
            layerPlan.postNorm.endsWith(".post_attention_layernorm.weight")
        })
      ]);

      const attentionInputBuffer = await this.#rmsNormToBuffer(
        hiddenBuffer,
        inputNormWeight,
        this.textConfig.hidden_size
      );
      const residual = layerPlan.isLinear
        ? await this.#runLinearAttention(attentionInputBuffer, layerPlan, caches[index])
        : await this.#runAttention(attentionInputBuffer, layerPlan, caches[index]);
      this.#destroyBuffer(attentionInputBuffer);
      const hiddenAfterAttention = await this.#addBuffers(
        hiddenBuffer,
        residual,
        this.textConfig.hidden_size
      );
      this.#destroyBuffer(hiddenBuffer);
      this.#destroyBuffer(residual);

      const mlpInputBuffer = await this.#rmsNormToBuffer(
        hiddenAfterAttention,
        postNormWeight,
        this.textConfig.hidden_size
      );
      const mlpResidualBuffer = await this.#runMlp(mlpInputBuffer, layerPlan);
      this.#destroyBuffer(mlpInputBuffer);
      hiddenBuffer = await this.#addBuffers(
        hiddenAfterAttention,
        mlpResidualBuffer,
        this.textConfig.hidden_size
      );
      this.#destroyBuffer(hiddenAfterAttention);
      this.#destroyBuffer(mlpResidualBuffer);
    }

    return hiddenBuffer;
  }

  async forwardToken(tokenId, caches) {
    const hiddenBuffer = await this.forwardHidden(tokenId, caches);
    try {
      return await this.#nextTokenFromHidden(hiddenBuffer);
    } finally {
      this.#destroyBuffer(hiddenBuffer);
    }
  }

  async generate(prompt, { maxNewTokens = 160, onPartial = null, shouldInterrupt = null } = {}) {
    if (!this.tokenizer) {
      throw new Error("The Qwen tokenizer is not available.");
    }

    const inputIds = clampTokens(this.tokenizer.encode(prompt), this.contextLength);
    if (!inputIds.length) {
      return { text: "", tokenIds: [], promptTokenCount: 0, hitTokenLimit: false };
    }
    this.statusCallback({
      phase: "prompt-tokenized",
      detail: `Prompt tokenized to ${inputIds.length} tokens.`
    });
    await this.tensorStore.prefetchRows(this.embedName, inputIds, {
      onProgress: ({ detail }) => {
        this.statusCallback({
          phase: "warming-embeddings",
          detail
        });
      }
    });

    const caches = this.makeCaches();
    let nextToken = 0;
    const lastPromptIndex = inputIds.length - 1;
    for (let tokenIndex = 0; tokenIndex < inputIds.length; tokenIndex += 1) {
      if (shouldInterrupt?.()) {
        throw new Error("generation-interrupted");
      }
      if (shouldReportStep(tokenIndex, inputIds.length)) {
        this.statusCallback({
          phase: "prefill",
          detail: `Prefilling prompt ${tokenIndex + 1}/${inputIds.length}…`
        });
      }
      const tokenId = inputIds[tokenIndex];
      if (tokenIndex === lastPromptIndex) {
        nextToken = await this.forwardToken(tokenId, caches);
      } else {
        await this.forwardHidden(tokenId, caches);
      }
    }

    this.statusCallback({
      phase: "decode",
      detail: "Prompt ready. Decoding response…"
    });
    const generated = [];
    let stopTokenId = null;
    for (let step = 0; step < maxNewTokens; step += 1) {
      if (shouldInterrupt?.()) {
        throw new Error("generation-interrupted");
      }
      if (this.stopTokens.has(nextToken)) {
        stopTokenId = nextToken;
        break;
      }
      generated.push(nextToken);
      onPartial?.(sanitizeGeneratedText(this.tokenizer.decode(generated)));
      if (shouldReportStep(step, maxNewTokens, 8)) {
        this.statusCallback({
          phase: "decode",
          detail: `Decoding response ${step + 1}/${maxNewTokens}…`
        });
      }
      nextToken = await this.forwardToken(nextToken, caches);
    }

    this.statusCallback({
      phase: "decode-complete",
      detail: `Decoded ${generated.length} tokens.`
    });
    const rawText = this.tokenizer.decode(generated);
    const sanitizedText = sanitizeGeneratedText(rawText);
    return {
      text: sanitizedText,
      rawText,
      tokenIds: generated,
      promptTokenCount: inputIds.length,
      hitTokenLimit: generated.length >= maxNewTokens,
      stopTokenId
    };
  }
}
