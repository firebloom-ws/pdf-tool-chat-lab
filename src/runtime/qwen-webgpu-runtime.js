const QWEN_TRANSFORMERS_CDN =
  "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.0.0-next.8";

export const QWEN_WEBGPU_MODELS = [
  {
    id: "onnx-community/Qwen3.5-0.8B-ONNX",
    label: "0.8B",
    description: "Smallest local WebGPU model"
  },
  {
    id: "onnx-community/Qwen3.5-2B-ONNX",
    label: "2B",
    description: "Stronger answers, slower downloads"
  },
  {
    id: "onnx-community/Qwen3.5-4B-ONNX",
    label: "4B",
    description: "Largest browser option"
  }
];

export const DEFAULT_QWEN_WEBGPU_MODEL = QWEN_WEBGPU_MODELS[0].id;

function modelLabelFor(modelId) {
  return (
    QWEN_WEBGPU_MODELS.find((model) => model.id === modelId)?.label ??
    modelId.split("/").at(-1) ??
    modelId
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

  if (typeof info.status === "string" && info.status.trim()) {
    return `${phase}: ${info.status.replace(/_/g, " ")}`;
  }

  return `Preparing ${phase}…`;
}

export class QwenWebGpuTextRuntime {
  constructor({ webgpuRuntime, modelId = DEFAULT_QWEN_WEBGPU_MODEL } = {}) {
    this.webgpuRuntime = webgpuRuntime;
    this.modelId = modelId;
    this.supportsLoraWeightInjection = false;
    this.worker = null;
    this.listeners = new Set();
    this.pendingLoad = null;
    this.pendingGeneration = null;
    this.loadPromise = null;
    this.requestCounter = 0;
    this.state = {
      backend: "transformers.js/webgpu",
      cdn: QWEN_TRANSFORMERS_CDN,
      modelId,
      label: modelLabelFor(modelId),
      status: "idle",
      ready: false,
      dtype: null,
      progress: 0,
      detail: `${modelLabelFor(modelId)} is not loaded yet.`,
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
    if (!modelId || modelId === this.modelId) {
      return;
    }

    this.modelId = modelId;
    this.#disposeWorker();
    this.#updateState({
      modelId,
      label: modelLabelFor(modelId),
      status: "idle",
      ready: false,
      dtype: null,
      progress: 0,
      error: null,
      detail: `${modelLabelFor(modelId)} is not loaded yet.`
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

    const dtype = gpuInfo.fp16Supported ? "q4f16" : "q4";
    const worker = this.#ensureWorker();
    this.#updateState({
      modelId: this.modelId,
      label: modelLabelFor(this.modelId),
      status: "loading",
      ready: false,
      dtype,
      progress: 0,
      error: null,
      detail: `Loading ${modelLabelFor(this.modelId)} on WebGPU…`
    });

    this.loadPromise = new Promise((resolve, reject) => {
      this.pendingLoad = { resolve, reject, gpuInfo };
      worker.postMessage({
        type: "load",
        data: {
          modelId: this.modelId,
          dtype
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
          messages,
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
          detail: `Loading ${modelLabelFor(message.modelId ?? this.modelId)} on WebGPU…`
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
        this.#updateState({
          modelId: message.modelId ?? this.modelId,
          label: modelLabelFor(message.modelId ?? this.modelId),
          status: "ready",
          ready: true,
          dtype: message.dtype ?? this.state.dtype,
          progress: 100,
          error: null,
          detail: `${modelLabelFor(message.modelId ?? this.modelId)} loaded on WebGPU (${message.dtype ?? this.state.dtype}).`
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
            modelId: this.state.modelId
          });
          this.pendingGeneration = null;
        }
        this.#updateState({
          status: "ready",
          detail: `${this.state.label} loaded on WebGPU (${this.state.dtype}).`
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
          detail: `${this.state.label} loaded on WebGPU (${this.state.dtype}).`
        });
        break;
      }

      case "interrupt-complete": {
        this.#updateState({
          status: this.state.ready ? "ready" : this.state.status,
          detail: this.state.ready
            ? `${this.state.label} loaded on WebGPU (${this.state.dtype}).`
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
