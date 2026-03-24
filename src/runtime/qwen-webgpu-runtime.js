const QWEN_MODEL_ALIASES = {
  "onnx-community/Qwen3.5-0.8B-ONNX": "Qwen/Qwen3.5-0.8B",
  "onnx-community/Qwen3.5-2B-ONNX": "Qwen/Qwen3.5-2B"
};

export const QWEN_WEBGPU_MODELS = [
  {
    id: "Qwen/Qwen3.5-0.8B",
    label: "0.8B",
    description: "Smallest Tensorbend WebGPU model"
  },
  {
    id: "Qwen/Qwen3.5-2B",
    label: "2B",
    description: "Stronger answers, larger download"
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
  const percent =
    normalizeProgress(info.progress) ??
    (typeof info.loaded === "number" &&
    typeof info.total === "number" &&
    info.total > 0
      ? Math.round((info.loaded / info.total) * 100)
      : null);

  if (fileName && percent !== null) {
    return `Loading ${fileName} (${percent}%)`;
  }

  if (fileName) {
    return `Loading ${fileName}…`;
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
  const dtype = message.dtype ?? fallbackDtype ?? "bf16/f16";
  const contextLength = Number(message.contextLength);
  if (Number.isFinite(contextLength) && contextLength > 0) {
    return `${label} loaded on WebGPU via Tensorbend (${dtype}, ${contextLength}-token context).`;
  }
  return `${label} loaded on WebGPU via Tensorbend (${dtype}).`;
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
      backend: "tensorbend/webgpu",
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
    this.#updateState({
      modelId: this.modelId,
      label: modelLabelFor(this.modelId),
      status: "loading",
      ready: false,
      dtype: null,
      progress: 0,
      error: null,
      detail: `Loading ${modelLabelFor(this.modelId)} with Tensorbend…`
    });

    this.loadPromise = new Promise((resolve, reject) => {
      this.pendingLoad = { resolve, reject, gpuInfo };
      worker.postMessage({
        type: "load",
        data: {
          modelId: this.modelId,
          gpuInfo
        }
      });
    }).finally(() => {
      this.loadPromise = null;
    });

    return this.loadPromise;
  }

  async generate({ messages, maxNewTokens = 160, onPartial } = {}) {
    if (!this.state.ready) {
      throw new Error(
        this.state.status === "loading"
          ? "model-loading"
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
        onPartial
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

    this.worker = new Worker(new URL("./qwen-webgpu-worker.js", import.meta.url), {
      type: "module"
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
          detail: `Loading ${modelLabelFor(message.modelId ?? this.modelId)} with Tensorbend…`
        });
        break;
      }

      case "load-progress": {
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
          dtype: message.dtype ?? this.state.dtype ?? "bf16/f16",
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

      case "generate-complete": {
        if (
          this.pendingGeneration &&
          message.requestId === this.pendingGeneration.requestId
        ) {
          this.pendingGeneration.resolve({
            text: message.text ?? "",
            promptTokens: message.promptTokens ?? null,
            outputTokens: message.outputTokens ?? null,
            maxNewTokens: message.maxNewTokens ?? null,
            hitTokenLimit: Boolean(message.hitTokenLimit),
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
              contextLength: message.contextLength
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
              contextLength: message.contextLength
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
                  contextLength: message.contextLength
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
