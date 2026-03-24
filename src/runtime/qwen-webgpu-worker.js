import { BytePairTokenizer } from "./tokenizer-bpe.js";
import {
  analyzeCustomQwenBackend,
  CUSTOM_QWEN_BACKEND,
  formatCustomBackendDetail,
  PapertrailQwenKernelHarness
} from "./papertrail-qwen-custom-backend.js";
import { PapertrailQwenCustomEngine } from "./papertrail-qwen-custom-engine.js";

const MODEL_ID = "Intel/Qwen3.5-2B-int4-AutoRound";
const DEFAULT_REVISION = "main";
const DEFAULT_DTYPE = "int4-auto-round";

let kernelModulesPromise = null;

const decoder = new TextDecoder();

const runtime = {
  modelId: MODEL_ID,
  revision: DEFAULT_REVISION,
  tokenizer: null,
  tokenizerConfig: null,
  gpu: null,
  kernelHarness: null,
  analysis: null,
  config: null,
  quantConfig: null,
  contextLength: 0,
  interruptRequested: false,
  activeRequestId: null,
  engine: null,
  generateStatusDetail: null
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

function installKernelBundleDomShims() {
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

async function loadKernelModules() {
  if (!kernelModulesPromise) {
    installKernelBundleDomShims();
    kernelModulesPromise = import(
      new URL("./tensorbend/gpu-ops-CgR4iK87.js", import.meta.url).href
    ).then((gpuOpsModule) => ({
      GpuOps: gpuOpsModule.G,
      shaders: gpuOpsModule.S ?? {}
    }));
  }
  return kernelModulesPromise;
}

function chooseContextLength(gpuInfo) {
  const maxBufferSize = Number(gpuInfo?.limits?.maxBufferSize ?? 0);
  const largeGpu = maxBufferSize >= 3_500_000_000;
  const mediumGpu = maxBufferSize >= 2_500_000_000;
  if (largeGpu) return 2048;
  if (mediumGpu) return 1536;
  return 1024;
}

async function inspectManifest(repo, revision = DEFAULT_REVISION) {
  const files = await listSafetensorFiles(repo, revision);
  const tensors = [];
  const shards = [];

  for (const path of files) {
    const { header, dataOffset } = await readSafetensorsHeader(repo, path, revision);
    const entries = collectTensorEntries(header, dataOffset).map((entry) => ({
      ...entry,
      filePath: path,
      byteStart: entry.absStart,
      byteEnd: entry.absEnd
    }));
    shards.push({ path, tensorCount: entries.length });
    tensors.push(...entries);
  }

  return {
    shardCount: shards.length,
    shards,
    tensors
  };
}

async function loadTokenizer(repo, revision = DEFAULT_REVISION) {
  const [tokenizerConfig, tokenizerJson, vocabJson, mergesText] = await Promise.all([
    fetchJson(repo, "tokenizer_config.json", revision),
    fetchJson(repo, "tokenizer.json", revision, { optional: true }),
    fetchJson(repo, "vocab.json", revision, { optional: true }),
    fetchText(repo, "merges.txt", revision, { optional: true })
  ]);

  return {
    tokenizerConfig,
    tokenizer: BytePairTokenizer.fromModelBundle({
      tokenizerConfig,
      tokenizerJson,
      vocabJson,
      mergesText
    })
  };
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

async function handleLoad({ modelId = MODEL_ID, gpuInfo, revision = DEFAULT_REVISION }) {
  const resolvedModelId = MODEL_ID;

  if (runtime.engine && runtime.modelId === resolvedModelId) {
    post("load-ready", {
      modelId: resolvedModelId,
      dtype: DEFAULT_DTYPE,
      backend: CUSTOM_QWEN_BACKEND,
      contextLength: runtime.contextLength,
      detail: formatCustomBackendDetail(runtime.analysis, {
        selfTestPassed: runtime.analysis?.selfTest?.ok,
        decoderReady: true
      })
    });
    return;
  }

  runtime.interruptRequested = false;
  runtime.activeRequestId = null;

  post("load-start", {
    modelId: resolvedModelId,
    dtype: DEFAULT_DTYPE,
    backend: CUSTOM_QWEN_BACKEND
  });

  post("load-progress", {
    phase: "runtime",
    info: { status: "bootstrapping_custom_kernel_modules", progress: 0.04 }
  });

  const { GpuOps, shaders } = await loadKernelModules();

  post("load-progress", {
    phase: "runtime",
    info: { status: "custom_kernels_ready", progress: 0.08 }
  });

  post("load-progress", {
    phase: "config",
    info: { status: "fetching_config", progress: 0.14 }
  });

  const config = await fetchJson(resolvedModelId, "config.json", revision);
  const [quantConfig, tokenizerBundle] = await Promise.all([
    loadQuantConfig(resolvedModelId, revision, config),
    loadTokenizer(resolvedModelId, revision)
  ]);

  post("load-progress", {
    phase: "config",
    info: { status: "inspecting_checkpoint_layout", progress: 0.22 }
  });

  const manifest = await inspectManifest(resolvedModelId, revision);
  post("load-progress", {
    phase: "runtime",
    info: { status: "initializing_webgpu", progress: 0.34 }
  });

  const kernelHarness = new PapertrailQwenKernelHarness(GpuOps, shaders);
  await kernelHarness.init();

  post("load-progress", {
    phase: "runtime",
    info: { status: "running_kernel_self_test", progress: 0.52 }
  });

  const selfTest = await kernelHarness.runGptqSelfTest();
  post("load-progress", {
    phase: "runtime",
    info: {
      status: selfTest.ok ? "kernel_self_test_passed" : "kernel_self_test_failed",
      progress: 0.7
    }
  });

  const analysis = analyzeCustomQwenBackend({
    config,
    manifest,
    shaderKeys: kernelHarness.availableKernelKeys()
  });
  analysis.selfTest = selfTest;

  post("load-progress", {
    phase: "runtime",
    info: { status: "building_custom_decoder_plan", progress: 0.88 }
  });

  const contextLength = chooseContextLength(gpuInfo);

  runtime.modelId = resolvedModelId;
  runtime.revision = revision;
  runtime.tokenizer = tokenizerBundle.tokenizer;
  runtime.tokenizerConfig = tokenizerBundle.tokenizerConfig ?? {};
  runtime.gpu = kernelHarness.gpu;
  runtime.kernelHarness = kernelHarness;
  runtime.analysis = analysis;
  runtime.config = config;
  runtime.quantConfig = quantConfig ?? {
    bits: 4,
    group_size: 128,
    sym: true,
    quant_method: "auto-round"
  };
  runtime.contextLength = contextLength;
  runtime.interruptRequested = false;
  runtime.generateStatusDetail = null;
  runtime.engine = new PapertrailQwenCustomEngine({
    modelId: resolvedModelId,
    revision,
    config,
    manifest,
    tokenizer: runtime.tokenizer,
    tokenizerConfig: runtime.tokenizerConfig,
    fetchRangeBytes,
    gpu: kernelHarness.gpu,
    shaders,
    quantConfig: runtime.quantConfig,
    contextLength,
    statusCallback: ({ phase, detail }) => {
      runtime.generateStatusDetail = detail ?? phase ?? "preparing_custom_decoder";
      post("generate-status", {
        phase: phase ?? "runtime",
        detail: runtime.generateStatusDetail
      });
    }
  });

  post("load-ready", {
    modelId: resolvedModelId,
    dtype: DEFAULT_DTYPE,
    backend: CUSTOM_QWEN_BACKEND,
    contextLength,
    detail: formatCustomBackendDetail(analysis, {
      selfTestPassed: selfTest.ok,
      decoderReady: true
    }),
    analysis: {
      fullAttentionLayers: analysis.fullAttentionLayers,
      linearAttentionLayers: analysis.linearAttentionLayers,
      tensorCount: analysis.manifest.tensorCount,
      shardCount: analysis.manifest.shardCount,
      quantization: analysis.quantization,
      missingKernels: analysis.missingKernels,
      missingTensors: analysis.missingTensors.slice(0, 8),
      selfTest
    }
  });
}

async function handleGenerate({ requestId, prompt, maxNewTokens = 160 }) {
  if (!runtime.engine) {
    throw new Error("custom-backend-not-loaded");
  }

  runtime.activeRequestId = requestId;
  runtime.interruptRequested = false;
  runtime.generateStatusDetail = "Preparing custom decoder…";
  post("generate-start", { requestId, modelId: runtime.modelId });

  const heartbeat = setInterval(() => {
    post("generate-status", {
      phase: "heartbeat",
      detail: runtime.generateStatusDetail ?? "Generating…"
    });
  }, 4000);

  try {
    const generated = await runtime.engine.generate(prompt ?? "", {
      maxNewTokens,
      shouldInterrupt: () => runtime.interruptRequested,
      onPartial: (text) => {
        post("generate-chunk", {
          requestId,
          text,
          contextLength: runtime.contextLength
        });
      }
    });

    runtime.activeRequestId = null;
    runtime.interruptRequested = false;
    runtime.generateStatusDetail = null;
    post("generate-complete", {
      requestId,
      text: generated.text ?? "",
      rawTextPreview: String(generated.rawText ?? "").slice(0, 240),
      firstTokenIds: Array.isArray(generated.tokenIds) ? generated.tokenIds.slice(0, 16) : [],
      promptTokens: generated.promptTokenCount ?? 0,
      outputTokens: generated.tokenIds?.length ?? 0,
      maxNewTokens,
      hitTokenLimit: Boolean(generated.hitTokenLimit),
      stopTokenId: generated.stopTokenId ?? null,
      contextLength: runtime.contextLength
    });
  } finally {
    clearInterval(heartbeat);
  }
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
