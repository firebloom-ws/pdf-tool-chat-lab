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

export class PapertrailQwenKernelHarness {
  constructor(GpuOps, shaders) {
    this.GpuOps = GpuOps;
    this.shaders = shaders ?? {};
    this.gpu = null;
  }

  async init() {
    if (this.gpu) {
      return this.gpu;
    }
    this.gpu = new this.GpuOps();
    await this.gpu.init();
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
      const pipeline = gpu.getOrCreatePipeline("pt-qwen-selftest-gptq", shaderCode);
      const bindGroup = gpu.createBindGroup(pipeline, 0, [
        inputBuffer,
        qweightBuffer,
        scalesBuffer,
        outputBuffer,
        paramsBuffer
      ]);

      gpu.dispatch(pipeline, [bindGroup], 1);
      const actual = gpu.readBuffer(outputBuffer, N * 4);
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
