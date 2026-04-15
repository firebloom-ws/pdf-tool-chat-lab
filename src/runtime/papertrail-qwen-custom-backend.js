export const CUSTOM_QWEN_BACKEND = "papertrail/qwen35-custom-kernel-scaffold";

export const EXPECTED_CUSTOM_KERNELS = [
  "gptq_matvec",
  "gptq_matvec_f16",
  "gptq_matvec_4t",
  "gptq_matvec_4t_f16",
  "bf16_matvec",
  "add_rmsnorm",
  "add_rmsnorm_ro",
  "fused_silu_gptq",
  "fused_sigmoid_gptq",
  "fused_conv_deltanet_norm",
  "fused_split_qknorm_kvstore",
  "argmax"
];

function float32ToHalfBits(value) {
  const floatView = new Float32Array(1);
  const intView = new Uint32Array(floatView.buffer);
  floatView[0] = value;
  const bits = intView[0];

  const sign = (bits >>> 16) & 0x8000;
  const exponent = ((bits >>> 23) & 0xff) - 127 + 15;
  const mantissa = bits & 0x7fffff;

  if (exponent <= 0) {
    if (exponent < -10) {
      return sign;
    }
    const shifted = (mantissa | 0x800000) >>> (1 - exponent);
    return sign | ((shifted + 0x1000) >>> 13);
  }

  if (exponent >= 0x1f) {
    return sign | 0x7c00;
  }

  return sign | (exponent << 10) | ((mantissa + 0x1000) >>> 13);
}

function packHalf2x16(left, right) {
  return float32ToHalfBits(left) | (float32ToHalfBits(right) << 16);
}

function listTensorNames(manifest) {
  return new Set((manifest?.tensors ?? []).map((tensor) => tensor.name));
}

function projectionName(baseName, tensorNames) {
  if (tensorNames.has(`${baseName}.qweight`)) {
    return {
      kind: "gptq",
      qweight: `${baseName}.qweight`,
      qzeros: `${baseName}.qzeros`,
      scales: `${baseName}.scales`
    };
  }
  if (tensorNames.has(`${baseName}.weight`)) {
    return {
      kind: "dense",
      weight: `${baseName}.weight`
    };
  }
  return {
    kind: "missing",
    baseName
  };
}

function buildLayerTypes(textConfig = {}) {
  const numHiddenLayers = Number(textConfig.num_hidden_layers ?? 0);
  const fullAttentionInterval = Number(textConfig.full_attention_interval ?? 1);
  const layerTypes = [];
  for (let index = 0; index < numHiddenLayers; index += 1) {
    const isFullAttention =
      fullAttentionInterval > 0 && (index + 1) % fullAttentionInterval === 0;
    layerTypes.push(isFullAttention ? "full_attention" : "linear_attention");
  }
  return layerTypes;
}

function countFloatOverrides(extraConfig = {}) {
  return Object.entries(extraConfig).filter(([, value]) => {
    return value?.bits === 16 || value?.data_type === "fp";
  }).length;
}

export function analyzeCustomQwenBackend({ config, manifest, shaderKeys = [] } = {}) {
  const textConfig = config?.text_config ?? config ?? {};
  const quantConfig = config?.quantization_config ?? textConfig?.quantization_config ?? {};
  const tensorNames = listTensorNames(manifest);
  const layerTypes = buildLayerTypes(textConfig);
  const fullAttentionLayers = layerTypes.filter((type) => type === "full_attention").length;
  const linearAttentionLayers = layerTypes.length - fullAttentionLayers;

  const missingTensors = [];
  const projectionKinds = {
    gptq: 0,
    dense: 0,
    missing: 0
  };

  const mustHave = [
    "model.language_model.embed_tokens.weight",
    "model.language_model.norm.weight"
  ];

  for (const name of mustHave) {
    if (!tensorNames.has(name)) {
      missingTensors.push(name);
    }
  }

  for (let index = 0; index < layerTypes.length; index += 1) {
    const prefix = `model.language_model.layers.${index}.`;
    const common = [
      `${prefix}input_layernorm.weight`,
      `${prefix}post_attention_layernorm.weight`,
      `${prefix}mlp.gate_proj`,
      `${prefix}mlp.up_proj`,
      `${prefix}mlp.down_proj`
    ];

    for (const baseName of common) {
      const projection = projectionName(baseName, tensorNames);
      projectionKinds[projection.kind] += 1;
      if (projection.kind === "missing") {
        missingTensors.push(`${baseName}.(weight|qweight)`);
      }
    }

    if (layerTypes[index] === "linear_attention") {
      const linearNames = [
        `${prefix}linear_attn.in_proj_qkv`,
        `${prefix}linear_attn.in_proj_z`,
        `${prefix}linear_attn.in_proj_a`,
        `${prefix}linear_attn.in_proj_b`,
        `${prefix}linear_attn.out_proj`
      ];
      for (const baseName of linearNames) {
        const projection = projectionName(baseName, tensorNames);
        projectionKinds[projection.kind] += 1;
        if (projection.kind === "missing") {
          missingTensors.push(`${baseName}.(weight|qweight)`);
        }
      }

      for (const name of [
        `${prefix}linear_attn.dt_bias`,
        `${prefix}linear_attn.A_log`,
        `${prefix}linear_attn.conv1d.weight`,
        `${prefix}linear_attn.norm.weight`
      ]) {
        if (!tensorNames.has(name)) {
          missingTensors.push(name);
        }
      }
    } else {
      const attentionNames = [
        `${prefix}self_attn.q_proj`,
        `${prefix}self_attn.k_proj`,
        `${prefix}self_attn.v_proj`,
        `${prefix}self_attn.o_proj`
      ];
      for (const baseName of attentionNames) {
        const projection = projectionName(baseName, tensorNames);
        projectionKinds[projection.kind] += 1;
        if (projection.kind === "missing") {
          missingTensors.push(`${baseName}.(weight|qweight)`);
        }
      }

      for (const name of [
        `${prefix}self_attn.q_norm.weight`,
        `${prefix}self_attn.k_norm.weight`
      ]) {
        if (!tensorNames.has(name)) {
          missingTensors.push(name);
        }
      }
    }
  }

  const missingKernels = EXPECTED_CUSTOM_KERNELS.filter(
    (kernel) => !shaderKeys.includes(kernel)
  );

  const floatOverrides = countFloatOverrides(quantConfig.extra_config);

  return {
    layerTypes,
    fullAttentionLayers,
    linearAttentionLayers,
    quantization: {
      method: quantConfig.quant_method ?? "unknown",
      packing: quantConfig.packing_format ?? "unknown",
      bits: Number(quantConfig.bits ?? 0),
      groupSize: Number(quantConfig.group_size ?? 0),
      symmetric: Boolean(quantConfig.sym),
      floatOverrides
    },
    manifest: {
      shardCount: Number(manifest?.shardCount ?? 0),
      tensorCount: Number(manifest?.tensors?.length ?? 0)
    },
    projections: projectionKinds,
    missingKernels,
    missingTensors: [...new Set(missingTensors)].sort(),
    missingImplementation: [
      "decoder/full-attention execution graph",
      "decoder/linear-attention execution graph",
      "streaming decode loop on custom graph",
      "weight-cache + activation scheduler integration"
    ]
  };
}

export function formatCustomBackendDetail(
  analysis,
  { selfTestPassed = false, decoderReady = false } = {}
) {
  const method = analysis?.quantization?.method ?? "unknown";
  const bits = analysis?.quantization?.bits ?? "?";
  const groupSize = analysis?.quantization?.groupSize ?? "?";
  const packing = analysis?.quantization?.packing ?? "unknown";
  const full = analysis?.fullAttentionLayers ?? 0;
  const linear = analysis?.linearAttentionLayers ?? 0;
  const tensors = analysis?.manifest?.tensorCount ?? 0;
  const shards = analysis?.manifest?.shardCount ?? 0;
  const floatOverrides = analysis?.quantization?.floatOverrides ?? 0;
  const missingKernelCount = analysis?.missingKernels?.length ?? 0;

  return [
    `Custom backend analyzed ${tensors} tensors across ${shards} shard(s).`,
    `Checkpoint layout: ${method} ${bits}-bit, group size ${groupSize}, packing ${packing}.`,
    `Layer plan: ${linear} linear-attention and ${full} full-attention layers.`,
    floatOverrides
      ? `${floatOverrides} projections remain FP16/BF16 and must stay on the dense path.`
      : "All quantized projection groups matched the expected INT4 path.",
    selfTestPassed
      ? "Kernel self-test passed on WebGPU."
      : "Kernel self-test did not complete cleanly.",
    missingKernelCount
      ? `${missingKernelCount} expected kernels are still unavailable in this build.`
      : "The required low-level kernels are available.",
    decoderReady
      ? "The custom decode loop is active and now runs layer projections on WebGPU with local JS cache/control flow."
      : "The custom decoder graph is still being ported, so chat stays on retrieval fallback for now."
  ].join(" ");
}

function alignTo4(value) {
  return Math.max(4, Math.ceil(value / 4) * 4);
}

function toByteView(data) {
  if (data instanceof Uint8Array) {
    return data;
  }
  if (data instanceof ArrayBuffer) {
    return new Uint8Array(data);
  }
  if (ArrayBuffer.isView(data)) {
    return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  }
  throw new Error("Unsupported GPU buffer input.");
}

function serializeError(error) {
  if (error instanceof Error) {
    return {
      name: error.name,
      message: error.message,
      stack: error.stack ?? null,
      cause:
        error.cause instanceof Error
          ? serializeError(error.cause)
          : error.cause ?? null
    };
  }
  return {
    name: typeof error?.name === "string" ? error.name : "Error",
    message: String(error),
    stack: null,
    cause: null
  };
}

function summarizeLimits(limits) {
  if (!limits) {
    return null;
  }
  return {
    maxBufferSize: Number(limits.maxBufferSize ?? 0),
    maxStorageBufferBindingSize: Number(limits.maxStorageBufferBindingSize ?? 0),
    maxComputeWorkgroupStorageSize: Number(limits.maxComputeWorkgroupStorageSize ?? 0),
    maxComputeInvocationsPerWorkgroup: Number(limits.maxComputeInvocationsPerWorkgroup ?? 0),
    maxComputeWorkgroupSizeX: Number(limits.maxComputeWorkgroupSizeX ?? 0),
    maxComputeWorkgroupsPerDimension: Number(limits.maxComputeWorkgroupsPerDimension ?? 0)
  };
}

class PapertrailGpuRunner {
  constructor({ adapter, device, hasF16 = false, diagnostics = null }) {
    this.adapter = adapter;
    this.device = device;
    this.hasF16 = hasF16;
    this.pipelineCache = new Map();
    this.diagnostics = diagnostics ?? (() => {});
  }

  static async create({ diagnostics = null } = {}) {
    const log = diagnostics ?? (() => {});
    if (!("gpu" in navigator)) {
      throw new Error("navigator.gpu is unavailable in this worker.");
    }

    log("gpu-request-adapter:start");
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance"
    });
    if (!adapter) {
      throw new Error("WebGPU adapter request returned null.");
    }

    const hasF16 = adapter.features?.has?.("shader-f16") ?? false;
    const requestedFeatures = hasF16 ? ["shader-f16"] : [];
    log("gpu-request-adapter:ok", {
      adapterInfo: adapter.info ?? null,
      limits: summarizeLimits(adapter.limits),
      requestedFeatures
    });

    const device = await adapter.requestDevice({
      requiredFeatures: requestedFeatures
    });

    device.addEventListener?.("uncapturederror", (event) => {
      log("gpu-uncapturederror", {
        error: serializeError(event.error)
      });
    });
    device.lost.then((info) => {
      log("gpu-device-lost", {
        reason: info?.reason ?? null,
        message: info?.message ?? null
      });
    });

    log("gpu-request-device:ok", {
      limits: summarizeLimits(device.limits),
      hasF16
    });
    return new PapertrailGpuRunner({
      adapter,
      device,
      hasF16,
      diagnostics: log
    });
  }

  createBuffer(name, byteLength, usage) {
    try {
      return this.device.createBuffer({
        label: name,
        size: alignTo4(byteLength),
        usage
      });
    } catch (error) {
      this.diagnostics("gpu-create-buffer-failed", {
        label: name,
        byteLength,
        usage,
        error: serializeError(error)
      });
      throw error;
    }
  }

  createBufferFromData(name, data, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC) {
    const bytes = toByteView(data);
    try {
      const buffer = this.device.createBuffer({
        label: name,
        size: alignTo4(bytes.byteLength),
        usage,
        mappedAtCreation: true
      });
      new Uint8Array(buffer.getMappedRange()).set(bytes);
      buffer.unmap();
      return buffer;
    } catch (error) {
      this.diagnostics("gpu-create-buffer-from-data-failed", {
        label: name,
        byteLength: bytes.byteLength,
        usage,
        error: serializeError(error)
      });
      throw error;
    }
  }

  async getOrCreateCheckedPipeline(name, shaderCode, entryPoint = "main") {
    const cacheKey = `${name}:${entryPoint}`;
    if (this.pipelineCache.has(cacheKey)) {
      return this.pipelineCache.get(cacheKey);
    }

    const module = this.device.createShaderModule({
      label: `${name}:module`,
      code: shaderCode
    });
    const info = await module.getCompilationInfo?.();
    const messages =
      info?.messages?.map((message) => ({
        type: message.type,
        lineNum: message.lineNum,
        linePos: message.linePos,
        message: message.message
      })) ?? [];
    if (messages.some((message) => message.type === "error")) {
      this.diagnostics("gpu-shader-compilation-failed", {
        name,
        entryPoint,
        messages
      });
      throw new Error(`Shader compilation failed for ${name}.`);
    }

    this.device.pushErrorScope?.("validation");
    this.device.pushErrorScope?.("internal");
    this.device.pushErrorScope?.("out-of-memory");
    let pipeline;
    try {
      pipeline = await this.device.createComputePipelineAsync({
        layout: "auto",
        compute: {
          module,
          entryPoint
        }
      });
    } catch (error) {
      this.diagnostics("gpu-create-pipeline-failed", {
        name,
        entryPoint,
        messages,
        error: serializeError(error)
      });
      throw error;
    }

    const [oomError, internalError, validationError] = await Promise.all([
      this.device.popErrorScope?.(),
      this.device.popErrorScope?.(),
      this.device.popErrorScope?.()
    ]);
    const scopedError = oomError ?? internalError ?? validationError ?? null;
    if (scopedError) {
      this.diagnostics("gpu-create-pipeline-scoped-error", {
        name,
        entryPoint,
        error: serializeError(scopedError)
      });
      throw scopedError;
    }

    this.pipelineCache.set(cacheKey, pipeline);
    return pipeline;
  }

  getOrCreatePipeline(name, shaderCode, entryPoint = "main") {
    const cacheKey = `${name}:${entryPoint}`;
    if (this.pipelineCache.has(cacheKey)) {
      return this.pipelineCache.get(cacheKey);
    }
    const pipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: {
        module: this.device.createShaderModule({
          label: `${name}:module`,
          code: shaderCode
        }),
        entryPoint
      }
    });
    this.pipelineCache.set(cacheKey, pipeline);
    return pipeline;
  }

  createBindGroup(pipeline, index, buffers) {
    return this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(index),
      entries: buffers.map((buffer, binding) => ({
        binding,
        resource: { buffer }
      }))
    });
  }

  dispatch(pipeline, bindGroups, x = 1, y = 1, z = 1) {
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    bindGroups.forEach((bindGroup, index) => {
      pass.setBindGroup(index, bindGroup);
    });
    pass.dispatchWorkgroups(x, y, z);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  async readFloat32Buffer(buffer, count) {
    const byteLength = count * Float32Array.BYTES_PER_ELEMENT;
    const staging = this.createBuffer(
      "pt-readback-f32",
      byteLength,
      GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    );
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, byteLength);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const values = new Float32Array(count);
    values.set(new Float32Array(staging.getMappedRange(), 0, count));
    staging.unmap();
    staging.destroy();
    return values;
  }
}

export class PapertrailQwenKernelHarness {
  constructor({ shaders, diagnostics = null } = {}) {
    this.shaders = shaders ?? {};
    this.gpu = null;
    this.diagnostics = diagnostics ?? (() => {});
  }

  async init() {
    if (this.gpu) {
      return this.gpu;
    }
    this.gpu = await PapertrailGpuRunner.create({
      diagnostics: this.diagnostics
    });
    return this.gpu;
  }

  availableKernelKeys() {
    return Object.keys(this.shaders ?? {});
  }

  async runGptqSelfTest() {
    const gpu = await this.init();
    const hasF16 = Boolean(gpu.hasF16);
    const shaderKey = hasF16 ? "gptq_matvec_4t_f16" : "gptq_matvec_4t";
    const shaderCode = this.shaders?.[shaderKey];
    if (!shaderCode) {
      return {
        ok: false,
        shaderKey,
        message: "missing-gptq-self-test-kernel"
      };
    }

    const K = 32;
    const N = 8;
    const groupSize = 8;
    const input = Float32Array.from({ length: K }, (_, index) => index + 1);
    const qweight = new Uint32Array((K / 8) * N).fill(0x99999999);
    const scales = new Uint32Array((K / groupSize * N) / 2).fill(packHalf2x16(1, 1));
    const output = new Float32Array(N);
    const params = new Uint32Array([K, N, groupSize]);

    const inputBuffer = gpu.createBufferFromData("pt-qwen-selftest-input", input);
    const qweightBuffer = gpu.createBufferFromData("pt-qwen-selftest-qweight", qweight);
    const scalesBuffer = gpu.createBufferFromData("pt-qwen-selftest-scales", scales);
    const outputBuffer = gpu.createBufferFromData("pt-qwen-selftest-output", output);
    const paramsBuffer = gpu.createBufferFromData(
      "pt-qwen-selftest-params",
      params,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    );

    try {
      const pipeline = await gpu.getOrCreateCheckedPipeline(
        "pt-qwen-selftest-gptq",
        shaderCode
      );
      const bindGroup = gpu.createBindGroup(pipeline, 0, [
        inputBuffer,
        qweightBuffer,
        scalesBuffer,
        outputBuffer,
        paramsBuffer
      ]);

      gpu.dispatch(pipeline, [bindGroup], 1);
      const actual = await gpu.readFloat32Buffer(outputBuffer, N);
      const expected = input.reduce((total, value) => total + value, 0);
      const maxError = Math.max(
        ...Array.from(actual, (value) => Math.abs(value - expected))
      );

      return {
        ok: maxError < 1e-2,
        shaderKey,
        maxError,
        expected
      };
    } finally {
      inputBuffer.destroy();
      qweightBuffer.destroy();
      scalesBuffer.destroy();
      outputBuffer.destroy();
      paramsBuffer.destroy();
    }
  }
}
