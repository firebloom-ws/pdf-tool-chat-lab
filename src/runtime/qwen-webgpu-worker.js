import { BytePairTokenizer } from "./tokenizer-bpe.js";
import { TensorbendPackedCache } from "./tensorbend-packed-cache.js";

const MODEL_ID = "Intel/Qwen3.5-4B-int4-AutoRound";
const DEFAULT_REVISION = "main";
const DEFAULT_DTYPE = "int4-auto-round";
const CACHE_NAMESPACE = "papertrail-tensorbend";

let tensorbendModulesPromise = null;

const decoder = new TextDecoder();
const packedCache = new TensorbendPackedCache(CACHE_NAMESPACE);
const MAX_PENDING_CACHE_WRITES = 2;

const runtime = {
  modelId: MODEL_ID,
  revision: DEFAULT_REVISION,
  tokenizer: null,
  model: null,
  gpu: null,
  config: null,
  quantConfig: null,
  contextLength: 0,
  interruptRequested: false,
  activeRequestId: null,
  cacheManifest: null
};

function post(type, data = {}) {
  self.postMessage({ type, ...data });
}

function encodeHubPath(path) {
  return String(path)
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
}

function hubFileUrl(repo, path, revision = DEFAULT_REVISION) {
  return `https://huggingface.co/${repo}/resolve/${revision}/${encodeHubPath(path)}`;
}

async function fetchText(repo, path, revision = DEFAULT_REVISION, { optional = false } = {}) {
  const response = await fetch(hubFileUrl(repo, path, revision), {
    cache: "no-store"
  });
  if (!response.ok) {
    if (optional && response.status === 404) {
      return null;
    }
    throw new Error(`Failed to fetch ${path}: ${response.status}`);
  }
  return response.text();
}

async function fetchJson(repo, path, revision = DEFAULT_REVISION, options = {}) {
  const text = await fetchText(repo, path, revision, options);
  return text ? JSON.parse(text) : null;
}

function installTensorbendDomShims() {
  if (!globalThis.document) {
    globalThis.document = {
      createElement() {
        return {
          relList: {
            supports() {
              return true;
            }
          }
        };
      },
      querySelectorAll() {
        return [];
      }
    };
  }

  if (!globalThis.MutationObserver) {
    globalThis.MutationObserver = class MutationObserver {
      constructor() {}
      disconnect() {}
      observe() {}
      takeRecords() {
        return [];
      }
    };
  }
}

async function loadTensorbendModules() {
  if (!tensorbendModulesPromise) {
    installTensorbendDomShims();
    tensorbendModulesPromise = Promise.all([
      import(new URL("./tensorbend/gpu-ops-CgR4iK87.js", import.meta.url).href),
      import(new URL("./tensorbend/qwen35-model-DHin-Xw8.js", import.meta.url).href)
    ]).then(([gpuOpsModule, qwenModule]) => ({
      GpuOps: gpuOpsModule.G,
      Qwen35Model: qwenModule.Qwen35Model
    }));
  }
  return tensorbendModulesPromise;
}

function chooseContextLength(gpuInfo) {
  const maxBufferSize = Number(gpuInfo?.limits?.maxBufferSize ?? 0);
  const largeGpu = maxBufferSize >= 3_500_000_000;
  const mediumGpu = maxBufferSize >= 2_500_000_000;
  if (largeGpu) return 2048;
  if (mediumGpu) return 1536;
  return 1024;
}

function assignSamplingDefaults(model) {
  model.temperature = 0;
  model.topK = 1;
  model.topP = 1;
  model.repetitionPenalty = 1;
  model.presencePenalty = 0;
  model.frequencyPenalty = 0;
}

async function loadTokenizer(repo, revision = DEFAULT_REVISION) {
  const [tokenizerConfig, tokenizerJson, vocabJson, mergesText] = await Promise.all([
    fetchJson(repo, "tokenizer_config.json", revision),
    fetchJson(repo, "tokenizer.json", revision, { optional: true }),
    fetchJson(repo, "vocab.json", revision, { optional: true }),
    fetchText(repo, "merges.txt", revision, { optional: true })
  ]);

  return BytePairTokenizer.fromModelBundle({
    tokenizerConfig,
    tokenizerJson,
    vocabJson,
    mergesText
  });
}

async function loadQuantConfig(repo, revision, config = null) {
  const candidates = [
    "quantization_config.json",
    "quantize_config.json",
    "gptq_config.json"
  ];

  for (const path of candidates) {
    const direct = await fetchJson(repo, path, revision, { optional: true });
    if (direct) {
      return direct;
    }
  }

  return config?.quantization_config ?? null;
}

async function listSafetensorFiles(repo, revision = DEFAULT_REVISION) {
  const indexJson = await fetchJson(repo, "model.safetensors.index.json", revision, {
    optional: true
  });

  if (indexJson?.weight_map) {
    return [...new Set(Object.values(indexJson.weight_map))];
  }

  return ["model.safetensors"];
}

async function fetchRangeBytes(repo, path, revision, start, end, { onProgress } = {}) {
  const response = await fetch(hubFileUrl(repo, path, revision), {
    cache: "no-store",
    headers: {
      Range: `bytes=${start}-${Math.max(start, end - 1)}`
    }
  });

  if (!response.ok && response.status !== 206) {
    throw new Error(`Failed to fetch range for ${path}: ${response.status}`);
  }

  if (!response.body) {
    const bytes = new Uint8Array(await response.arrayBuffer());
    onProgress?.({ loaded: bytes.byteLength, total: end - start });
    return bytes;
  }

  const reader = response.body.getReader();
  const total = end - start;
  const bytes = new Uint8Array(total);
  let loaded = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    bytes.set(value, loaded);
    loaded += value.byteLength;
    onProgress?.({ loaded, total });
  }

  return loaded === total ? bytes : bytes.slice(0, loaded);
}

async function readSafetensorsHeader(repo, path, revision = DEFAULT_REVISION) {
  const prefix = await fetchRangeBytes(repo, path, revision, 0, 8);
  const view = new DataView(prefix.buffer, prefix.byteOffset, prefix.byteLength);
  const low = view.getUint32(0, true);
  const high = view.getUint32(4, true);
  const headerLength = low + high * 2 ** 32;
  const headerBytes = await fetchRangeBytes(repo, path, revision, 8, 8 + headerLength);
  const header = JSON.parse(decoder.decode(headerBytes));
  return {
    header,
    dataOffset: 8 + headerLength
  };
}

function collectTensorEntries(header, dataOffset) {
  const entries = [];
  for (const [name, spec] of Object.entries(header ?? {})) {
    if (name === "__metadata__") {
      continue;
    }
    const [start, end] = spec.data_offsets ?? [0, 0];
    entries.push({
      name,
      dtype: spec.dtype,
      shape: Array.isArray(spec.shape) ? spec.shape.map((value) => Number(value)) : [],
      absStart: dataOffset + start,
      absEnd: dataOffset + end,
      byteLength: end - start
    });
  }
  return entries.sort((left, right) => left.absStart - right.absStart);
}

function buildDownloadGroups(entries, maxGroupBytes = 48 * 1024 * 1024, maxGapBytes = 1 * 1024 * 1024) {
  const groups = [];
  let current = null;

  for (const entry of entries) {
    if (!current) {
      current = {
        start: entry.absStart,
        end: entry.absEnd,
        tensors: [entry]
      };
      continue;
    }

    const proposedEnd = entry.absEnd;
    const proposedSize = proposedEnd - current.start;
    const gap = entry.absStart - current.end;

    if (
      (proposedSize > maxGroupBytes && current.tensors.length > 0) ||
      gap > maxGapBytes
    ) {
      groups.push(current);
      current = {
        start: entry.absStart,
        end: entry.absEnd,
        tensors: [entry]
      };
      continue;
    }

    current.end = proposedEnd;
    current.tensors.push(entry);
  }

  if (current) {
    groups.push(current);
  }

  return groups.map((group, index) => ({
    ...group,
    groupIndex: index,
    byteLength: group.end - group.start
  }));
}

function buildShardFromGroup(bytes, group) {
  const shard = {};
  for (const tensor of group.tensors) {
    const offset = tensor.absStart - group.start;
    shard[tensor.name] = {
      dtype: tensor.dtype,
      shape: tensor.shape,
      data: bytes.subarray(offset, offset + tensor.byteLength)
    };
  }
  return shard;
}

function truncatePromptTokens(tokenIds, contextLength, maxNewTokens) {
  const promptBudget = Math.max(32, contextLength - Math.max(1, maxNewTokens));
  if (tokenIds.length <= promptBudget) {
    return {
      tokenIds,
      truncatedPromptTokens: 0
    };
  }

  if (promptBudget <= 1) {
    return {
      tokenIds: tokenIds.slice(-1),
      truncatedPromptTokens: tokenIds.length - 1
    };
  }

  return {
    tokenIds: [tokenIds[0], ...tokenIds.slice(-(promptBudget - 1))],
    truncatedPromptTokens: tokenIds.length - promptBudget
  };
}

async function hydrateFromPackedCache(model, modelId, revision, manifest) {
  const shards = [...(manifest?.shards ?? [])].sort(
    (left, right) => Number(left.shardIndex) - Number(right.shardIndex)
  );

  for (let index = 0; index < shards.length; index += 1) {
    const shardInfo = shards[index];
    const shard = await packedCache.readShard(modelId, revision, shardInfo.shardIndex);
    if (!shard) {
      throw new Error(`packed-cache-miss:${shardInfo.shardIndex}`);
    }

    model.uploadTensors(shard);
    post("load-progress", {
      phase: "cache",
      info: {
        status: "restoring_quantized_cache",
        file: shardInfo.path ?? `cached-shard-${index + 1}`,
        loaded: index + 1,
        total: shards.length,
        progress: (index + 1) / Math.max(1, shards.length),
        cached: true
      }
    });
  }
}

async function streamAndCachePackedWeights(model, modelId, revision) {
  const files = await listSafetensorFiles(modelId, revision);
  const filePlans = [];
  let totalBytes = 0;
  let totalGroups = 0;

  for (const path of files) {
    const { header, dataOffset } = await readSafetensorsHeader(modelId, path, revision);
    const entries = collectTensorEntries(header, dataOffset);
    const groups = buildDownloadGroups(entries);
    filePlans.push({ path, groups });
    totalBytes += groups.reduce((sum, group) => sum + group.byteLength, 0);
    totalGroups += groups.length;
  }

  let loadedBytes = 0;
  const shards = [];
  let globalShardIndex = 0;
  let completedGroups = 0;
  const pendingWrites = [];

  const flushPendingWrites = async (limit = 0) => {
    while (pendingWrites.length > limit) {
      const next = pendingWrites.shift();
      const result = await next;
      shards[result.shardIndex] = result.shardInfo;
    }
  };

  for (const filePlan of filePlans) {
    for (const group of filePlan.groups) {
      const bytes = await fetchRangeBytes(
        modelId,
        filePlan.path,
        revision,
        group.start,
        group.end,
        {
          onProgress: ({ loaded, total }) => {
            post("load-progress", {
              phase: "weights",
              info: {
                status: "streaming_quantized_weights",
                file: filePlan.path,
                loaded: loadedBytes + loaded,
                total: totalBytes,
                progress: totalBytes > 0 ? (loadedBytes + loaded) / totalBytes : null,
                group: completedGroups + 1,
                totalGroups
              }
            });
          }
        }
      );

      const shard = buildShardFromGroup(bytes, group);
      model.uploadTensors(shard);

      const shardIndex = globalShardIndex;
      pendingWrites.push(
        packedCache
          .writeShard(modelId, revision, shardIndex, shard, {
            path: filePlan.path,
            rangeStart: group.start,
            rangeEnd: group.end
          })
          .then((packed) => ({
            shardIndex,
            shardInfo: {
              shardIndex,
              path: filePlan.path,
              rangeStart: group.start,
              rangeEnd: group.end,
              tensorCount: packed.tensorCount,
              byteLength: packed.byteLength
            }
          }))
      );
      await flushPendingWrites(MAX_PENDING_CACHE_WRITES - 1);

      globalShardIndex += 1;
      loadedBytes += group.byteLength;
      completedGroups += 1;

      post("load-progress", {
        phase: "cache",
        info: {
          status: "packing_quantized_cache",
          file: filePlan.path,
          loaded: loadedBytes,
          total: totalBytes,
          progress: totalBytes > 0 ? loadedBytes / totalBytes : null,
          group: completedGroups,
          totalGroups
        }
      });
    }
  }

  await flushPendingWrites();

  const manifest = {
    source: "hf-stream",
    shards: shards.filter(Boolean),
    totalBytes
  };

  await packedCache.writeManifest(modelId, revision, manifest);
  return manifest;
}

async function handleLoad({ modelId = MODEL_ID, gpuInfo, revision = DEFAULT_REVISION }) {
  const resolvedModelId = MODEL_ID;

  if (runtime.model && runtime.modelId === resolvedModelId && runtime.tokenizer) {
    post("load-ready", {
      modelId: resolvedModelId,
      dtype: DEFAULT_DTYPE,
      backend: "papertrail/tensorbend-webgpu",
      contextLength: runtime.contextLength,
      cacheMode: runtime.cacheManifest ? "packed-opfs" : "memory"
    });
    return;
  }

  runtime.interruptRequested = false;
  runtime.activeRequestId = null;

  post("load-start", {
    modelId: resolvedModelId,
    dtype: DEFAULT_DTYPE,
    backend: "papertrail/tensorbend-webgpu"
  });

  const { GpuOps, Qwen35Model } = await loadTensorbendModules();

  post("load-progress", {
    phase: "config",
    info: { status: "fetching_config", progress: 0.03 }
  });

  const config = await fetchJson(resolvedModelId, "config.json", revision);
  const [quantConfig, tokenizer, manifest] = await Promise.all([
    loadQuantConfig(resolvedModelId, revision, config),
    loadTokenizer(resolvedModelId, revision),
    packedCache.readManifest(resolvedModelId, revision)
  ]);

  post("load-progress", {
    phase: "config",
    info: {
      status: manifest?.shards?.length ? "restoring_cached_model_plan" : "planning_weight_stream",
      progress: 0.06
    }
  });

  post("load-progress", {
    phase: "runtime",
    info: { status: "initializing_webgpu", progress: 0.08 }
  });

  const gpu = new GpuOps();
  await gpu.init();

  post("load-progress", {
    phase: "runtime",
    info: { status: "compiling_tensorbend_pipelines", progress: 0.1 }
  });

  const model = new Qwen35Model(gpu, config, quantConfig);
  assignSamplingDefaults(model);
  model.compilePipelines();

  let cacheManifest = manifest;

  if (cacheManifest?.shards?.length) {
    post("load-progress", {
      phase: "cache",
      info: { status: "restoring_quantized_cache", progress: 0.14 }
    });
    try {
      await hydrateFromPackedCache(model, resolvedModelId, revision, cacheManifest);
    } catch (error) {
      console.warn("Packed cache restore failed, rebuilding from network", error);
      await packedCache.clearModel(resolvedModelId, revision, cacheManifest);
      cacheManifest = null;
    }
  }

  if (!cacheManifest) {
    post("load-progress", {
      phase: "weights",
      info: { status: "streaming_quantized_weights", progress: 0.12 }
    });
    cacheManifest = await streamAndCachePackedWeights(model, resolvedModelId, revision);
  }

  post("load-progress", {
    phase: "weights",
    info: {
      status: "post_processing_weights",
      progress: 0.97
    }
  });

  await model.postProcessWeights();

  post("load-progress", {
    phase: "runtime",
    info: {
      status: "allocating_runtime_buffers",
      progress: 0.99
    }
  });

  const contextLength = chooseContextLength(gpuInfo);
  model.initBuffers(contextLength);

  runtime.modelId = resolvedModelId;
  runtime.revision = revision;
  runtime.tokenizer = tokenizer;
  runtime.model = model;
  runtime.gpu = gpu;
  runtime.config = config;
  runtime.quantConfig = quantConfig ?? {
    bits: 4,
    group_size: 128,
    sym: true,
    quant_method: "auto-round"
  };
  runtime.contextLength = contextLength;
  runtime.cacheManifest = cacheManifest;
  runtime.interruptRequested = false;

  post("load-ready", {
    modelId: resolvedModelId,
    dtype: DEFAULT_DTYPE,
    backend: "papertrail/tensorbend-webgpu",
    contextLength,
    cacheMode: "packed-opfs"
  });
}

async function handleGenerate({ requestId, prompt, maxNewTokens = 160 }) {
  if (!runtime.model || !runtime.tokenizer) {
    throw new Error("model-not-ready");
  }

  runtime.interruptRequested = false;
  runtime.activeRequestId = requestId;

  post("generate-start", {
    requestId,
    modelId: runtime.modelId
  });

  const encodedPrompt = runtime.tokenizer.encode(prompt ?? "");
  const {
    tokenIds: promptTokenIds,
    truncatedPromptTokens
  } = truncatePromptTokens(encodedPrompt, runtime.contextLength, maxNewTokens);

  const streamedTokenIds = [];
  let interrupted = false;

  const sequence = await runtime.model.generate(
    promptTokenIds,
    maxNewTokens,
    (tokenId) => {
      if (runtime.interruptRequested) {
        interrupted = true;
        return true;
      }

      streamedTokenIds.push(tokenId);
      post("generate-chunk", {
        requestId,
        text: runtime.tokenizer.decode(streamedTokenIds)
      });
      return false;
    }
  );

  const generatedTokenIds =
    Array.isArray(sequence) && sequence.length >= promptTokenIds.length
      ? sequence.slice(promptTokenIds.length)
      : streamedTokenIds;

  const finalText = runtime.tokenizer.decode(generatedTokenIds);
  const basePayload = {
    requestId,
    modelId: runtime.modelId,
    contextLength: runtime.contextLength,
    promptTokens: promptTokenIds.length,
    outputTokens: generatedTokenIds.length,
    maxNewTokens,
    truncatedPromptTokens
  };

  runtime.activeRequestId = null;

  if (interrupted || runtime.interruptRequested) {
    runtime.interruptRequested = false;
    post("generate-error", {
      ...basePayload,
      message: "generation-interrupted"
    });
    post("interrupt-complete", {
      requestId,
      contextLength: runtime.contextLength
    });
    return;
  }

  post("generate-complete", {
    ...basePayload,
    text: finalText,
    hitTokenLimit: generatedTokenIds.length >= maxNewTokens
  });
}

function handleInterrupt() {
  runtime.interruptRequested = true;
  if (!runtime.activeRequestId) {
    post("interrupt-complete", {
      contextLength: runtime.contextLength
    });
  }
}

self.addEventListener("message", async (event) => {
  const { type, data } = event.data ?? {};

  try {
    switch (type) {
      case "load":
        await handleLoad(data ?? {});
        break;

      case "generate":
        await handleGenerate(data ?? {});
        break;

      case "interrupt":
        handleInterrupt();
        break;

      default:
        break;
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (type === "load") {
      post("load-error", { message });
      return;
    }
    if (type === "generate") {
      runtime.activeRequestId = null;
      runtime.interruptRequested = false;
      post("generate-error", {
        requestId: data?.requestId ?? null,
        message,
        contextLength: runtime.contextLength
      });
    }
  }
});
