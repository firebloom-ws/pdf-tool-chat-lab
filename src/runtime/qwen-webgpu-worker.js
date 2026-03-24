import { BytePairTokenizer } from "./tokenizer-bpe.js";

const DEFAULT_REVISION = "main";
const DEFAULT_DTYPE = "bf16/f16";

let tensorbendModulesPromise = null;

const runtime = {
  modelId: null,
  revision: DEFAULT_REVISION,
  tokenizer: null,
  model: null,
  gpu: null,
  config: null,
  quantConfig: null,
  contextLength: 0,
  interruptRequested: false,
  activeRequestId: null
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
  const response = await fetch(hubFileUrl(repo, path, revision));
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
      import(new URL("./tensorbend/qwen35-model-DHin-Xw8.js", import.meta.url).href),
      import(new URL("./tensorbend/safetensors-loader-CNnqzt-J.js", import.meta.url).href)
    ]).then(([gpuOpsModule, qwenModule, safetensorsModule]) => ({
      GpuOps: gpuOpsModule.G,
      Qwen35Model: qwenModule.Qwen35Model,
      loadConfig: safetensorsModule.loadConfig,
      loadModelWeights: safetensorsModule.loadModelWeights,
      loadQuantConfig: safetensorsModule.loadQuantConfig
    }));
  }
  return tensorbendModulesPromise;
}

function chooseContextLength(modelId, gpuInfo) {
  const maxBufferSize = Number(gpuInfo?.limits?.maxBufferSize ?? 0);
  const largeGpu = maxBufferSize >= 3_500_000_000;
  const mediumGpu = maxBufferSize >= 2_500_000_000;

  if (modelId.includes("0.8B")) {
    if (largeGpu) return 8192;
    if (mediumGpu) return 6144;
    return 4096;
  }

  if (modelId.includes("2B")) {
    if (largeGpu) return 4096;
    if (mediumGpu) return 3072;
    return 2048;
  }

  return mediumGpu ? 3072 : 2048;
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

function normalizeProgressInfo(info) {
  if (!info || typeof info !== "object") {
    return { progress: null };
  }

  const total = Number(info.total ?? 0);
  const loaded = Number(info.loaded ?? 0);
  const progress = total > 0 ? loaded / total : null;

  return {
    ...info,
    progress
  };
}

async function handleLoad({ modelId, gpuInfo, revision = DEFAULT_REVISION }) {
  if (runtime.model && runtime.modelId === modelId && runtime.tokenizer) {
    post("load-ready", {
      modelId,
      dtype: runtime.quantConfig?.quant_method ?? DEFAULT_DTYPE,
      backend: "tensorbend/webgpu",
      contextLength: runtime.contextLength
    });
    return;
  }

  runtime.interruptRequested = false;
  runtime.activeRequestId = null;

  post("load-start", {
    modelId,
    dtype: DEFAULT_DTYPE,
    backend: "tensorbend/webgpu"
  });

  const {
    GpuOps,
    Qwen35Model,
    loadConfig,
    loadModelWeights,
    loadQuantConfig
  } = await loadTensorbendModules();

  post("load-progress", {
    phase: "config",
    info: { status: "fetching_config", progress: 0.05 }
  });

  const [config, quantConfig, tokenizer] = await Promise.all([
    loadConfig(modelId, revision),
    loadQuantConfig(modelId, revision).catch(() => null),
    loadTokenizer(modelId, revision)
  ]);

  post("load-progress", {
    phase: "runtime",
    info: { status: "initializing_webgpu", progress: 0.1 }
  });

  const gpu = new GpuOps();
  await gpu.init();

  const model = new Qwen35Model(gpu, config, quantConfig);
  assignSamplingDefaults(model);
  model.compilePipelines();

  post("load-progress", {
    phase: "weights",
    info: { status: "streaming_weights", progress: 0.12 }
  });

  await loadModelWeights(
    modelId,
    (info) => {
      post("load-progress", {
        phase: "weights",
        info: normalizeProgressInfo(info)
      });
    },
    {
      revision,
      onShard: async (shard) => {
        model.uploadTensors(shard);
      }
    }
  );

  post("load-progress", {
    phase: "weights",
    info: { status: "post_processing", progress: 0.96 }
  });

  await model.postProcessWeights();

  const contextLength = chooseContextLength(modelId, gpuInfo);
  model.initBuffers(contextLength);

  runtime.modelId = modelId;
  runtime.revision = revision;
  runtime.tokenizer = tokenizer;
  runtime.model = model;
  runtime.gpu = gpu;
  runtime.config = config;
  runtime.quantConfig = quantConfig;
  runtime.contextLength = contextLength;
  runtime.interruptRequested = false;

  post("load-ready", {
    modelId,
    dtype: quantConfig?.quant_method ?? DEFAULT_DTYPE,
    backend: "tensorbend/webgpu",
    contextLength
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
