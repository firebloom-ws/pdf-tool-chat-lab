import {
  env,
  AutoTokenizer,
  AutoModelForCausalLM,
  TextStreamer,
  InterruptableStoppingCriteria
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.0.0-next.8";

env.allowLocalModels = false;
if ("useBrowserCache" in env) {
  env.useBrowserCache = true;
}
if (env.backends?.onnx?.wasm) {
  env.backends.onnx.wasm.proxy = false;
}

const stoppingCriteria = new InterruptableStoppingCriteria();

let tokenizer = null;
let model = null;
let currentModelId = null;
let currentDtype = null;
let loadPromise = null;
let generating = false;

function post(type, data = {}) {
  self.postMessage({ type, ...data });
}

function progressHandler(phase, modelId, dtype) {
  return (info) => {
    post("load-progress", {
      phase,
      modelId,
      dtype,
      info
    });
  };
}

async function disposeModel() {
  try {
    await model?.dispose?.();
  } catch (error) {
    console.warn("Failed to dispose WebGPU model cleanly", error);
  }
  model = null;
  tokenizer = null;
}

async function warmupModel() {
  const warmupInputs = tokenizer.apply_chat_template(
    [{ role: "user", content: "Hello" }],
    {
      add_generation_prompt: true,
      return_dict: true,
      enable_thinking: false
    }
  );

  await model.generate({
    ...warmupInputs,
    do_sample: false,
    max_new_tokens: 1
  });
}

async function loadModel({ modelId, dtype }) {
  if (loadPromise) {
    return loadPromise;
  }

  if (model && tokenizer && currentModelId === modelId && currentDtype === dtype) {
    post("load-ready", { modelId, dtype, cached: true });
    return;
  }

  loadPromise = (async () => {
    post("load-start", { modelId, dtype });

    if (currentModelId !== modelId || currentDtype !== dtype) {
      await disposeModel();
      currentModelId = null;
      currentDtype = null;
    }

    tokenizer = await AutoTokenizer.from_pretrained(modelId, {
      progress_callback: progressHandler("tokenizer", modelId, dtype)
    });

    model = await AutoModelForCausalLM.from_pretrained(modelId, {
      device: "webgpu",
      dtype,
      progress_callback: progressHandler("weights", modelId, dtype)
    });

    currentModelId = modelId;
    currentDtype = dtype;

    post("load-progress", {
      phase: "compile",
      modelId,
      dtype,
      info: {
        status: "warming-up"
      }
    });

    await warmupModel();

    post("load-ready", {
      modelId,
      dtype
    });
  })().finally(() => {
    loadPromise = null;
  });

  return loadPromise;
}

async function generate({ requestId, messages, maxNewTokens = 160 }) {
  if (!model || !tokenizer) {
    throw new Error("model-not-ready");
  }
  if (generating) {
    throw new Error("generation-in-progress");
  }

  generating = true;
  stoppingCriteria.reset();

  try {
    const inputs = tokenizer.apply_chat_template(messages, {
      add_generation_prompt: true,
      return_dict: true,
      enable_thinking: false
    });

    let text = "";
    const streamer = new TextStreamer(tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (chunk) => {
        text += chunk;
        post("generate-chunk", {
          requestId,
          chunk,
          text
        });
      }
    });

    post("generate-start", { requestId });

    const sequences = await model.generate({
      ...inputs,
      do_sample: false,
      max_new_tokens: maxNewTokens,
      streamer,
      stopping_criteria: stoppingCriteria
    });

    const promptTokens = Number(inputs.input_ids?.dims?.at(-1) ?? 0);
    const outputTokens = Math.max(Number(sequences?.dims?.at(-1) ?? 0) - promptTokens, 0);

    if (!text.trim()) {
      const decoded = tokenizer.batch_decode(
        sequences.slice(null, [inputs.input_ids.dims.at(-1), null]),
        {
          skip_special_tokens: true
        }
      );
      text = decoded[0] ?? "";
    }

    post("generate-complete", {
      requestId,
      text: text.trim(),
      promptTokens,
      outputTokens,
      maxNewTokens,
      hitTokenLimit: outputTokens >= maxNewTokens
    });
  } finally {
    generating = false;
  }
}

self.addEventListener("message", async (event) => {
  const { type, data } = event.data ?? {};

  try {
    switch (type) {
      case "load":
        await loadModel(data);
        break;

      case "generate":
        await generate(data);
        break;

      case "interrupt":
        stoppingCriteria.interrupt();
        post("interrupt-complete");
        break;

      default:
        break;
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (type === "generate") {
      post("generate-error", {
        requestId: data?.requestId ?? null,
        message
      });
      return;
    }

    post("load-error", {
      modelId: data?.modelId ?? currentModelId,
      dtype: data?.dtype ?? currentDtype,
      message
    });
  }
});
