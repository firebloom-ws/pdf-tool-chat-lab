const QWEN_MODEL_ALIASES = {
  "onnx-community/Qwen3.5-0.8B-ONNX": "Intel/Qwen3.5-2B-int4-AutoRound",
  "onnx-community/Qwen3.5-2B-ONNX": "Intel/Qwen3.5-2B-int4-AutoRound",
  "onnx-community/Qwen3.5-4B-ONNX": "Intel/Qwen3.5-2B-int4-AutoRound",
  "Qwen/Qwen3.5-0.8B": "Intel/Qwen3.5-2B-int4-AutoRound",
  "Qwen/Qwen3.5-2B": "Intel/Qwen3.5-2B-int4-AutoRound",
  "Qwen/Qwen3.5-4B": "Intel/Qwen3.5-2B-int4-AutoRound"
};

export const QWEN_WEBGPU_MODELS = [
  {
    id: "Intel/Qwen3.5-2B-int4-AutoRound",
    label: "2B AutoRound",
    description: "Custom WebGPU kernel scaffold"
  }
];

export const DEFAULT_QWEN_WEBGPU_MODEL = QWEN_WEBGPU_MODELS[0].id;

function normalizeModelId(modelId) {
  return QWEN_MODEL_ALIASES[modelId] ?? modelId ?? DEFAULT_QWEN_WEBGPU_MODEL;
}

function modelLabelFor(modelId) {
  const normalized = normalizeModelId(modelId);
  return (
    QWEN_WEBGPU_MODELS.find((model) => model.id === normalized)?.label ??
    normalized.split("/").at(-1) ??
    normalized
  );
}

function cloneableGpuInfo(gpuInfo) {
  if (!gpuInfo || typeof gpuInfo !== "object") {
    return gpuInfo ?? null;
  }
  const limits =
    gpuInfo.limits && typeof gpuInfo.limits === "object"
      ? { ...gpuInfo.limits }
      : null;
  const adapterInfo =
    gpuInfo.adapterInfo && typeof gpuInfo.adapterInfo === "object"
      ? {
          vendor: gpuInfo.adapterInfo.vendor ?? null,
          architecture: gpuInfo.adapterInfo.architecture ?? null,
          device: gpuInfo.adapterInfo.device ?? null,
          description: gpuInfo.adapterInfo.description ?? null
        }
      : null;

  return {
    available: Boolean(gpuInfo.available),
    reason: gpuInfo.reason ?? null,
    fp16Supported: Boolean(gpuInfo.fp16Supported),
    limits,
    adapterInfo
  };
}

function normalizeProgress(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return null;
  }
  return value <= 1 ? Math.round(value * 100) : Math.round(value);
}

function summarizeProgress(info, phase = "model") {
  if (!info) {
    return `Preparing ${phase}…`;
  }

  const fileName = info.file ? String(info.file).split("/").at(-1) : null;
  const groupLabel =
    Number.isFinite(info.group) && Number.isFinite(info.totalGroups)
      ? `, chunk ${info.group}/${info.totalGroups}`
      : "";
  const percent =
    normalizeProgress(info.progress) ??
    (typeof info.loaded === "number" &&
    typeof info.total === "number" &&
    info.total > 0
      ? Math.round((info.loaded / info.total) * 100)
      : null);

  if (fileName && percent !== null) {
    return `Loading ${fileName}${groupLabel} (${percent}%)`;
  }

  if (fileName) {
    return `Loading ${fileName}${groupLabel}…`;
  }

  if (typeof info.phase === "string" && info.phase.trim()) {
    return `${phase}: ${info.phase.replace(/_/g, " ")}`;
  }

  if (typeof info.status === "string" && info.status.trim()) {
    return `${phase}: ${info.status.replace(/_/g, " ")}`;
  }

  if (percent !== null) {
    return `Preparing ${phase} (${percent}%)`;
  }

  return `Preparing ${phase}…`;
}

function describeReadyDetail(message, fallbackLabel, fallbackDtype) {
  const label = modelLabelFor(message.modelId ?? fallbackLabel);
  const dtype = message.dtype ?? fallbackDtype ?? "int4";
  const contextLength = Number(message.contextLength);
  const cacheMode = message.cacheMode === "packed-opfs" ? "packed local cache" : null;
  if (Number.isFinite(contextLength) && contextLength > 0) {
    return `${label} loaded on WebGPU via the custom backend (${dtype}, ${contextLength}-token context${cacheMode ? `, ${cacheMode}` : ""}).`;
  }
  return `${label} loaded on WebGPU via the custom backend (${dtype}${cacheMode ? `, ${cacheMode}` : ""}).`;
}

export class QwenWebGpuTextRuntime {
  constructor({
    webgpuRuntime,
    hubClient,
    modelId = DEFAULT_QWEN_WEBGPU_MODEL
  } = {}) {
    this.webgpuRuntime = webgpuRuntime;
    this.hubClient = hubClient;
    this.modelId = normalizeModelId(modelId);
    this.supportsLoraWeightInjection = false;
    this.worker = null;
    this.listeners = new Set();
    this.pendingLoad = null;
    this.pendingGeneration = null;
    this.loadPromise = null;
    this.requestCounter = 0;
    this.state = {
      backend: "papertrail/qwen35-custom",
      modelId: this.modelId,
      label: modelLabelFor(this.modelId),
      status: "idle",
      ready: false,
      dtype: null,
      progress: 0,
      detail: `${modelLabelFor(this.modelId)} is not loaded yet.`,
      error: null
    };
  }

  getModelOptions() {
    return QWEN_WEBGPU_MODELS.map((model) => ({ ...model }));
  }

  getState() {
    return { ...this.state };
  }

  subscribe(listener) {
    this.listeners.add(listener);
    listener(this.getState());
    return () => {
      this.listeners.delete(listener);
    };
  }

  setModel(modelId) {
    const normalized = normalizeModelId(modelId);
    if (!normalized || normalized === this.modelId) {
      return;
    }

    this.modelId = normalized;
    this.#disposeWorker();
    this.#updateState({
      modelId: normalized,
      label: modelLabelFor(normalized),
      status: "idle",
      ready: false,
      dtype: null,
      progress: 0,
      error: null,
      detail: `${modelLabelFor(normalized)} is not loaded yet.`
    });
  }

  async probe() {
    return this.load();
  }

  async load() {
    const gpuInfo = await this.webgpuRuntime.probe();
    if (!gpuInfo.available) {
      this.#updateState({
        status: "unsupported",
        ready: false,
        error: gpuInfo.reason,
        detail: gpuInfo.reason
      });
      return {
        ...this.getState(),
        gpuInfo
      };
    }

    if (this.state.ready) {
      return {
        ...this.getState(),
        gpuInfo
      };
    }

    if (this.loadPromise) {
      return this.loadPromise;
    }

    const worker = this.#ensureWorker();
    const safeGpuInfo = cloneableGpuInfo(gpuInfo);
    this.#updateState({
      modelId: this.modelId,
      label: modelLabelFor(this.modelId),
      status: "loading",
      ready: false,
      dtype: null,
      progress: 0,
      error: null,
      detail: `Loading ${modelLabelFor(this.modelId)} with the custom backend…`
    });

    this.loadPromise = new Promise((resolve, reject) => {
      const pendingLoad = {
        gpuInfo: safeGpuInfo,
        stallTimer: null,
        resolve: (value) => {
          if (pendingLoad.stallTimer) {
            clearTimeout(pendingLoad.stallTimer);
          }
          resolve(value);
        },
        reject: (error) => {
          if (pendingLoad.stallTimer) {
            clearTimeout(pendingLoad.stallTimer);
          }
          reject(error);
        }
      };
      pendingLoad.stallTimer = setTimeout(() => {
        if (this.pendingLoad !== pendingLoad) {
          return;
        }
        const error = new Error("custom-backend-bootstrap-stalled");
        this.#updateState({
          status: "error",
          ready: false,
          error: error.message,
          detail:
            "The custom backend stalled before checkpoint analysis completed."
        });
        this.#disposeWorker();
        pendingLoad.reject(error);
      }, 45000);
      this.pendingLoad = pendingLoad;
      worker.postMessage({
        type: "load",
        data: {
          modelId: this.modelId,
          gpuInfo: safeGpuInfo
        }
      });
    }).finally(() => {
      this.loadPromise = null;
    });

    return this.loadPromise;
  }

  async generate({ messages, maxNewTokens = 160, onPartial, onStatus } = {}) {
    if (!this.state.ready) {
      throw new Error(
        this.state.status === "loading"
          ? "model-loading"
          : this.state.status === "parked"
            ? "custom-backend-parked"
          : this.state.error ?? "model-not-ready"
      );
    }

    if (this.pendingGeneration) {
      throw new Error("generation-in-progress");
    }

    if (!this.hubClient) {
      throw new Error("hub-client-unavailable");
    }

    const prompt = await this.hubClient.renderChatPrompt(this.modelId, messages, {
      add_generation_prompt: true,
      enable_thinking: false
    });

    const worker = this.#ensureWorker();
    const requestId = ++this.requestCounter;

    return new Promise((resolve, reject) => {
      this.pendingGeneration = {
        requestId,
        resolve,
        reject,
        onPartial,
        onStatus
      };
      worker.postMessage({
        type: "generate",
        data: {
          requestId,
          prompt,
          maxNewTokens
        }
      });
    });
  }

  interrupt() {
    this.worker?.postMessage({ type: "interrupt" });
  }

  #ensureWorker() {
    if (this.worker) {
      return this.worker;
    }

    const WorkerCtor = globalThis.Worker;
    if (typeof WorkerCtor !== "function") {
      throw new Error("worker-unavailable");
    }

    const workerUrl = new URL("./qwen-webgpu-worker.js", import.meta.url);
    this.worker = new WorkerCtor(workerUrl.href, {
      type: "module",
      name: "papertrail-qwen-worker"
    });

    this.worker.addEventListener("message", (event) => {
      this.#handleWorkerMessage(event.data);
    });

    this.worker.addEventListener("error", (event) => {
      const message = event.message || "qwen-worker-error";
      this.#updateState({
        status: "error",
        ready: false,
        error: message,
        detail: `Model worker failed: ${message}`
      });
      this.#rejectPending(message);
    });

    return this.worker;
  }

  #disposeWorker() {
    this.worker?.terminate();
    this.worker = null;
    this.pendingLoad = null;
    this.pendingGeneration = null;
    this.loadPromise = null;
  }

  #rejectPending(message) {
    const error = new Error(message);
    if (this.pendingLoad) {
      this.pendingLoad.reject(error);
      this.pendingLoad = null;
    }
    if (this.pendingGeneration) {
      this.pendingGeneration.reject(error);
      this.pendingGeneration = null;
    }
  }

  #handleWorkerMessage(message) {
    switch (message?.type) {
      case "load-start": {
        this.#updateState({
          status: "loading",
          ready: false,
          dtype: message.dtype ?? this.state.dtype,
          detail: `Loading ${modelLabelFor(message.modelId ?? this.modelId)} with the custom backend…`
        });
        break;
      }

      case "load-progress": {
        if (this.pendingLoad?.stallTimer) {
          const pendingLoad = this.pendingLoad;
          clearTimeout(this.pendingLoad.stallTimer);
          this.pendingLoad.stallTimer = setTimeout(() => {
            if (this.pendingLoad !== pendingLoad) {
              return;
            }
            const error = new Error("custom-backend-load-stalled");
            this.#updateState({
              status: "error",
              ready: false,
              error: error.message,
              detail:
                "The custom backend stalled while analyzing the checkpoint or kernels."
            });
            this.#disposeWorker();
            pendingLoad.reject(error);
          }, 45000);
        }
        this.#updateState({
          status: "loading",
          ready: false,
          progress:
            normalizeProgress(message.info?.progress) ??
            normalizeProgress(message.progress) ??
            this.state.progress,
          detail: summarizeProgress(message.info, message.phase ?? "model")
        });
        break;
      }

      case "load-parked": {
        this.#updateState({
          modelId: message.modelId ?? this.modelId,
          label: modelLabelFor(message.modelId ?? this.modelId),
          status: "parked",
          ready: false,
          dtype: message.dtype ?? this.state.dtype ?? "int4",
          progress: 100,
          error: null,
          detail:
            message.detail ??
            "The custom backend finished analysis but the live decoder graph is not ported yet.",
          backend: message.backend ?? this.state.backend
        });
        if (this.pendingLoad) {
          this.pendingLoad.resolve({
            ...this.getState(),
            gpuInfo: this.pendingLoad.gpuInfo
          });
          this.pendingLoad = null;
        }
        break;
      }

      case "load-ready": {
        const detail = describeReadyDetail(
          message,
          message.modelId ?? this.modelId,
          this.state.dtype
        );
        this.#updateState({
          modelId: message.modelId ?? this.modelId,
          label: modelLabelFor(message.modelId ?? this.modelId),
          status: "ready",
          ready: true,
          dtype: message.dtype ?? this.state.dtype ?? "int4",
          progress: 100,
          error: null,
          detail,
          backend: message.backend ?? this.state.backend
        });
        if (this.pendingLoad) {
          this.pendingLoad.resolve({
            ...this.getState(),
            gpuInfo: this.pendingLoad.gpuInfo
          });
          this.pendingLoad = null;
        }
        break;
      }

      case "load-error": {
        const detail = message.message || "Model load failed.";
        this.#updateState({
          status: "error",
          ready: false,
          error: detail,
          detail
        });
        if (this.pendingLoad) {
          this.pendingLoad.reject(new Error(detail));
          this.pendingLoad = null;
        }
        break;
      }

      case "generate-start": {
        this.#updateState({
          status: "generating",
          detail: `Generating with ${this.state.label}…`
        });
        break;
      }

      case "generate-chunk": {
        if (
          this.pendingGeneration &&
          message.requestId === this.pendingGeneration.requestId
        ) {
          this.pendingGeneration.onPartial?.(message.text ?? "");
        }
        break;
      }

      case "generate-status": {
        if (this.pendingGeneration) {
          this.pendingGeneration.onStatus?.(message.detail ?? message.phase ?? "working");
        }
        if (this.state.status === "generating") {
          this.#updateState({
            status: "generating",
            detail: message.detail ?? this.state.detail
          });
        }
        break;
      }

      case "generate-complete": {
        if (
          this.pendingGeneration &&
          message.requestId === this.pendingGeneration.requestId
        ) {
          this.pendingGeneration.resolve({
            text: message.text ?? "",
            rawTextPreview: message.rawTextPreview ?? "",
            firstTokenIds: message.firstTokenIds ?? [],
            promptTokens: message.promptTokens ?? null,
            outputTokens: message.outputTokens ?? null,
            maxNewTokens: message.maxNewTokens ?? null,
            hitTokenLimit: Boolean(message.hitTokenLimit),
            stopTokenId: message.stopTokenId ?? null,
            modelId: this.state.modelId,
            truncatedPromptTokens: message.truncatedPromptTokens ?? 0
          });
          this.pendingGeneration = null;
        }
        this.#updateState({
          status: "ready",
          detail: describeReadyDetail(
            {
              modelId: this.state.modelId,
              dtype: this.state.dtype,
              contextLength: message.contextLength,
              cacheMode: message.cacheMode
            },
            this.state.modelId,
            this.state.dtype
          )
        });
        break;
      }

      case "generate-error": {
        const detail = message.message || "Generation failed.";
        if (
          this.pendingGeneration &&
          message.requestId === this.pendingGeneration.requestId
        ) {
          this.pendingGeneration.reject(new Error(detail));
          this.pendingGeneration = null;
        }
        this.#updateState({
          status: "ready",
          detail: describeReadyDetail(
            {
              modelId: this.state.modelId,
              dtype: this.state.dtype,
              contextLength: message.contextLength,
              cacheMode: message.cacheMode
            },
            this.state.modelId,
            this.state.dtype
          )
        });
        break;
      }

      case "interrupt-complete": {
        this.#updateState({
          status: this.state.ready ? "ready" : this.state.status,
          detail: this.state.ready
            ? describeReadyDetail(
                {
                  modelId: this.state.modelId,
                  dtype: this.state.dtype,
                  contextLength: message.contextLength,
                  cacheMode: message.cacheMode
                },
                this.state.modelId,
                this.state.dtype
              )
            : this.state.detail
        });
        break;
      }

      default:
        break;
    }
  }

  #updateState(patch) {
    this.state = {
      ...this.state,
      ...patch
    };
    for (const listener of this.listeners) {
      listener(this.getState());
    }
  }
}
