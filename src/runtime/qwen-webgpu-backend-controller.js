import { BytePairTokenizer } from "./tokenizer-bpe.js";
import { OpfsCache } from "./opfs-cache.js";
import {
  analyzeCustomQwenBackend,
  CUSTOM_QWEN_BACKEND,
  formatCustomBackendDetail,
  PapertrailQwenKernelHarness
} from "./papertrail-qwen-custom-backend.js";
import { PapertrailQwenCustomEngine } from "./papertrail-qwen-custom-engine.js";

export const QWEN_BACKEND_MODEL_ID = "Intel/Qwen3.5-2B-int4-AutoRound";
export const QWEN_BACKEND_DEFAULT_REVISION = "main";
export const QWEN_BACKEND_DEFAULT_DTYPE = "int4-auto-round";
const MANIFEST_CACHE = new OpfsCache("papertrail-qwen-backend-v1");
const BACKEND_TEXT_CACHE = new OpfsCache("papertrail-qwen-backend-meta-v1");

function manifestCacheKey(modelId, revision) {
  return `manifest:${String(modelId)}:${String(revision)}`;
}

function textCacheKey(repo, revision, path) {
  return `text:${String(repo)}:${String(revision)}:${String(path)}`;
}

export function createQwenBackendController({ post }) {
  let kernelModulesPromise = null;
  const decoder = new TextDecoder();
  const resolvedFileUrls = new Map();

  const runtime = {
    modelId: QWEN_BACKEND_MODEL_ID,
    revision: QWEN_BACKEND_DEFAULT_REVISION,
    tokenizer: null,
    tokenizerConfig: null,
    gpu: null,
    kernelHarness: null,
    analysis: null,
    config: null,
    quantConfig: null,
    contextLength: 0,
    loading: false,
    interruptRequested: false,
    activeRequestId: null,
    engine: null,
    generateStatusDetail: null
  };

  function emit(type, data = {}) {
    post({ type, ...data });
  }

  function encodeHubPath(path) {
    return String(path)
      .split("/")
      .map((segment) => encodeURIComponent(segment))
      .join("/");
  }

  function hubFileUrl(repo, path, revision = QWEN_BACKEND_DEFAULT_REVISION) {
    return `https://huggingface.co/${repo}/resolve/${revision}/${encodeHubPath(path)}?download=1`;
  }

  function resolvedFileUrlKey(repo, path, revision) {
    return `${String(repo)}:${String(revision)}:${String(path)}`;
  }

  async function fetchText(
    repo,
    path,
    revision = QWEN_BACKEND_DEFAULT_REVISION,
    { optional = false } = {}
  ) {
    const cacheKey = textCacheKey(repo, revision, path);
    const cachedBytes = await BACKEND_TEXT_CACHE.readBytes(cacheKey);
    if (cachedBytes) {
      return decoder.decode(cachedBytes);
    }

    try {
      const response = await fetch(hubFileUrl(repo, path, revision), {
        cache: "no-store"
      });
      if (!response.ok) {
        if (optional && response.status === 404) {
          return null;
        }
        throw new Error(`Failed to fetch ${path}: ${response.status}`);
      }

      const text = await response.text();
      await BACKEND_TEXT_CACHE.writeBytes(cacheKey, new TextEncoder().encode(text));
      return text;
    } catch (error) {
      if (cachedBytes) {
        return decoder.decode(cachedBytes);
      }
      if (optional) {
        return null;
      }
      throw error;
    }
  }

  async function fetchJson(repo, path, revision = QWEN_BACKEND_DEFAULT_REVISION, options = {}) {
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

  async function inspectManifest(
    repo,
    revision = QWEN_BACKEND_DEFAULT_REVISION,
    { onProgress = null } = {}
  ) {
    const cached = await MANIFEST_CACHE.readJson(manifestCacheKey(repo, revision));
    if (cached?.tensors?.length) {
      onProgress?.({
        completed: cached.shardCount ?? cached.shards?.length ?? 0,
        total: cached.shardCount ?? cached.shards?.length ?? 0,
        detail: "Using cached checkpoint manifest."
      });
      return cached;
    }

    const files = await listSafetensorFiles(repo, revision);
    let completed = 0;
    const results = await Promise.all(
      files.map(async (path, index) => {
        onProgress?.({
          completed,
          total: files.length,
          detail: `Inspecting shard ${index + 1}/${files.length}: ${path}`
        });
        const { header, dataOffset } = await readSafetensorsHeader(repo, path, revision);
        const entries = collectTensorEntries(header, dataOffset).map((entry) => ({
          ...entry,
          filePath: path,
          byteStart: entry.absStart,
          byteEnd: entry.absEnd
        }));
        completed += 1;
        onProgress?.({
          completed,
          total: files.length,
          detail: `Shard ${completed}/${files.length} ready: ${path}`
        });
        return {
          path,
          tensorCount: entries.length,
          entries
        };
      })
    );

    const manifest = {
      shardCount: results.length,
      shards: results.map((result) => ({
        path: result.path,
        tensorCount: result.tensorCount
      })),
      tensors: results.flatMap((result) => result.entries)
    };
    await MANIFEST_CACHE.writeJson(manifestCacheKey(repo, revision), manifest);
    return manifest;
  }

  async function loadTokenizer(repo, revision = QWEN_BACKEND_DEFAULT_REVISION) {
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

  async function listSafetensorFiles(repo, revision = QWEN_BACKEND_DEFAULT_REVISION) {
    const indexJson = await fetchJson(repo, "model.safetensors.index.json", revision, {
      optional: true
    });

    if (indexJson?.weight_map) {
      return [...new Set(Object.values(indexJson.weight_map))];
    }

    return ["model.safetensors"];
  }

  async function readRequestedRange(response, start, end, { onProgress } = {}) {
    const total = Math.max(0, end - start);
    if (total === 0) {
      return new Uint8Array();
    }

    if (!response.body) {
      const bytes = new Uint8Array(await response.arrayBuffer());
      if (response.status === 206) {
        const sliced = bytes.slice(0, Math.min(total, bytes.byteLength));
        onProgress?.({ loaded: sliced.byteLength, total });
        return sliced;
      }
      const sliced = bytes.slice(start, Math.min(end, bytes.byteLength));
      onProgress?.({ loaded: sliced.byteLength, total });
      return sliced;
    }

    const reader = response.body.getReader();
    const bytes = new Uint8Array(total);
    let loaded = 0;
    let skip = response.status === 206 ? 0 : start;

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      if (!value?.byteLength) {
        continue;
      }

      let chunk = value;
      if (skip > 0) {
        if (skip >= chunk.byteLength) {
          skip -= chunk.byteLength;
          onProgress?.({ loaded, total });
          continue;
        }
        chunk = chunk.subarray(skip);
        skip = 0;
      }

      const remaining = total - loaded;
      if (remaining <= 0) {
        break;
      }
      const writeChunk = chunk.byteLength > remaining ? chunk.subarray(0, remaining) : chunk;
      bytes.set(writeChunk, loaded);
      loaded += writeChunk.byteLength;
      onProgress?.({ loaded, total });

      if (loaded >= total) {
        await reader.cancel("range-complete");
        break;
      }
    }

    return loaded === total ? bytes : bytes.slice(0, loaded);
  }

  async function fetchRangeResponse(url, start, end) {
    return fetch(url, {
      cache: "no-store",
      headers: {
        Range: `bytes=${start}-${Math.max(start, end - 1)}`
      }
    });
  }

  async function fetchRangeBytes(repo, path, revision, start, end, { onProgress } = {}) {
    const resolvedKey = resolvedFileUrlKey(repo, path, revision);
    const initialUrl = resolvedFileUrls.get(resolvedKey) ?? hubFileUrl(repo, path, revision);
    let response = await fetchRangeResponse(initialUrl, start, end);

    if (!(response.ok || response.status === 206)) {
      throw new Error(`Failed to fetch range for ${path}: ${response.status}`);
    }
    if (response.url) {
      resolvedFileUrls.set(resolvedKey, response.url);
    }

    if (response.status !== 206 && start > 0 && response.url && response.url !== initialUrl) {
      const retry = await fetchRangeResponse(response.url, start, end);
      if (retry.ok || retry.status === 206) {
        response = retry;
      }
    }

    return readRequestedRange(response, start, end, { onProgress });
  }

  async function readSafetensorsHeader(repo, path, revision = QWEN_BACKEND_DEFAULT_REVISION) {
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

  async function handleLoad({
    modelId = QWEN_BACKEND_MODEL_ID,
    gpuInfo,
    revision = QWEN_BACKEND_DEFAULT_REVISION
  }) {
    const resolvedModelId = QWEN_BACKEND_MODEL_ID;
    runtime.loading = true;
    let loadHeartbeat = null;

    try {
      if (runtime.engine && runtime.modelId === resolvedModelId) {
        emit("load-ready", {
          modelId: resolvedModelId,
          dtype: QWEN_BACKEND_DEFAULT_DTYPE,
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
      runtime.generateStatusDetail = "Bootstrapping custom backend…";

      emit("load-start", {
        modelId: resolvedModelId,
        dtype: QWEN_BACKEND_DEFAULT_DTYPE,
        backend: CUSTOM_QWEN_BACKEND
      });

      loadHeartbeat = setInterval(() => {
        emit("load-progress", {
          phase: "runtime",
          info: {
            status: "load_heartbeat",
            detail: runtime.generateStatusDetail ?? "Loading custom backend…"
          }
        });
      }, 4000);

      emit("load-progress", {
        phase: "runtime",
        info: { status: "bootstrapping_custom_kernel_modules", progress: 0.04 }
      });

      const { GpuOps, shaders } = await loadKernelModules();

      emit("load-progress", {
        phase: "runtime",
        info: { status: "custom_kernels_ready", progress: 0.08 }
      });

      emit("load-progress", {
        phase: "config",
        info: { status: "fetching_config", progress: 0.14 }
      });

      const config = await fetchJson(resolvedModelId, "config.json", revision);
      const [quantConfig, tokenizerBundle] = await Promise.all([
        loadQuantConfig(resolvedModelId, revision, config),
        loadTokenizer(resolvedModelId, revision)
      ]);

      emit("load-progress", {
        phase: "config",
        info: { status: "inspecting_checkpoint_layout", progress: 0.22 }
      });

      const manifest = await inspectManifest(resolvedModelId, revision, {
        onProgress: ({ completed = 0, total = 0, detail = "" }) => {
          const fraction = total > 0 ? completed / total : 0;
          emit("load-progress", {
            phase: "config",
            info: {
              status: "inspecting_checkpoint_layout",
              progress: 0.22 + Math.max(0, Math.min(1, fraction)) * 0.1,
              detail
            }
          });
        }
      });
      emit("load-progress", {
        phase: "runtime",
        info: { status: "initializing_webgpu", progress: 0.34 }
      });

      const kernelHarness = new PapertrailQwenKernelHarness(GpuOps, shaders);
      await kernelHarness.init();

      emit("load-progress", {
        phase: "runtime",
        info: { status: "running_kernel_self_test", progress: 0.52 }
      });

      const selfTest = await kernelHarness.runGptqSelfTest();
      emit("load-progress", {
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

      emit("load-progress", {
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
          if (runtime.activeRequestId) {
            emit("generate-status", {
              phase: phase ?? "runtime",
              detail: runtime.generateStatusDetail
            });
            return;
          }
          emit("load-progress", {
            phase: "warmup",
            info: {
              status: phase ?? "warming_decoder_caches",
              detail: runtime.generateStatusDetail
            }
          });
        }
      });

      emit("load-progress", {
        phase: "warmup",
        info: { status: "warming_decoder_caches", progress: 0.92 }
      });

      await runtime.engine.prepareForInference({
        onProgress: ({ phase, detail, progress = 0 }) => {
          emit("load-progress", {
            phase: "warmup",
            info: {
              status: phase ?? "warming_decoder_caches",
              progress: 0.92 + Math.max(0, Math.min(1, progress)) * 0.07,
              detail
            }
          });
        }
      });

      emit("load-ready", {
        modelId: resolvedModelId,
        dtype: QWEN_BACKEND_DEFAULT_DTYPE,
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
    } finally {
      runtime.loading = false;
      if (loadHeartbeat) {
        clearInterval(loadHeartbeat);
      }
    }
  }

  async function handleGenerate({ requestId, prompt, maxNewTokens = 160 }) {
    if (!runtime.engine) {
      throw new Error("custom-backend-not-loaded");
    }

    runtime.activeRequestId = requestId;
    runtime.interruptRequested = false;
    runtime.generateStatusDetail = "Preparing custom decoder…";
    emit("generate-start", { requestId, modelId: runtime.modelId });

    const heartbeat = setInterval(() => {
      emit("generate-status", {
        phase: "heartbeat",
        detail: runtime.generateStatusDetail ?? "Generating…"
      });
    }, 4000);

    try {
      const generated = await runtime.engine.generate(prompt ?? "", {
        maxNewTokens,
        shouldInterrupt: () => runtime.interruptRequested,
        onPartial: (text) => {
          emit("generate-chunk", {
            requestId,
            text,
            contextLength: runtime.contextLength
          });
        }
      });

      runtime.activeRequestId = null;
      runtime.interruptRequested = false;
      runtime.generateStatusDetail = null;
      emit("generate-complete", {
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
      emit("interrupt-complete", {
        contextLength: runtime.contextLength
      });
    }
  }

  async function handleMessage(message) {
    const { type, data } = message ?? {};

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
      const text = error instanceof Error ? error.message : String(error);
      runtime.loading = false;
      if (type === "load") {
        emit("load-error", { message: text });
        return;
      }
      if (type === "generate") {
        runtime.activeRequestId = null;
        runtime.interruptRequested = false;
        emit("generate-error", {
          requestId: data?.requestId ?? null,
          message: text,
          contextLength: runtime.contextLength
        });
      }
    }
  }

  function terminate() {
    runtime.interruptRequested = true;
    runtime.activeRequestId = null;
    runtime.generateStatusDetail = null;
  }

  return {
    handleMessage,
    terminate
  };
}
