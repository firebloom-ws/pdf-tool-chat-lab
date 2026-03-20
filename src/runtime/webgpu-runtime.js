import { buildMatmulShader, buildRmsNormShader } from "./webgpu-kernels.js";

function alignTo4(value) {
  return Math.max(4, Math.ceil(value / 4) * 4);
}

function toByteView(data) {
  if (data instanceof ArrayBuffer) {
    return new Uint8Array(data);
  }
  return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
}

function epsilonBits(value) {
  return new Uint32Array(new Float32Array([value]).buffer)[0];
}

export class WebGpuRuntime {
  constructor() {
    this.adapter = null;
    this.device = null;
    this.pipelineCache = new Map();
    this.probeResult = null;
  }

  async ensureDevice() {
    const result = await this.probe();
    if (!result.available) {
      throw new Error(result.reason);
    }
    return this.device;
  }

  async probe() {
    if (this.probeResult) {
      return this.probeResult;
    }

    if (!("gpu" in navigator)) {
      this.probeResult = {
        available: false,
        reason: "This browser does not expose navigator.gpu."
      };
      return this.probeResult;
    }

    this.adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance"
    });

    if (!this.adapter) {
      this.probeResult = {
        available: false,
        reason: "No WebGPU adapter was returned by the browser."
      };
      return this.probeResult;
    }

    const optionalFeatures = [];
    if (this.adapter.features.has("shader-f16")) {
      optionalFeatures.push("shader-f16");
    }

    this.device = await this.adapter.requestDevice({
      requiredFeatures: optionalFeatures
    });
    this.probeResult = {
      available: true,
      adapterInfo: this.adapter.info ?? null,
      limits: this.adapter.limits,
      fp16Supported: this.adapter.features.has("shader-f16")
    };

    return this.probeResult;
  }

  getOrCreatePipeline(shaderCode, entryPoint = "main") {
    const key = `${entryPoint}:${shaderCode}`;
    if (this.pipelineCache.has(key)) {
      return this.pipelineCache.get(key);
    }

    const pipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: {
        module: this.device.createShaderModule({ code: shaderCode }),
        entryPoint
      }
    });
    this.pipelineCache.set(key, pipeline);
    return pipeline;
  }

  createStorageBuffer(data, extraUsage = 0) {
    const bytes = toByteView(data);
    const buffer = this.device.createBuffer({
      size: alignTo4(bytes.byteLength),
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC |
        extraUsage,
      mappedAtCreation: true
    });
    new Uint8Array(buffer.getMappedRange()).set(bytes);
    buffer.unmap();
    return buffer;
  }

  createEmptyBuffer(byteLength, extraUsage = 0) {
    return this.device.createBuffer({
      size: alignTo4(byteLength),
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST |
        extraUsage
    });
  }

  createUniformBuffer(data) {
    const bytes = toByteView(data);
    const buffer = this.device.createBuffer({
      size: alignTo4(bytes.byteLength),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });
    new Uint8Array(buffer.getMappedRange()).set(bytes);
    buffer.unmap();
    return buffer;
  }

  async readFloat32Buffer(buffer, count) {
    const byteLength = count * Float32Array.BYTES_PER_ELEMENT;
    const staging = this.device.createBuffer({
      size: alignTo4(byteLength),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, byteLength);
    this.device.queue.submit([encoder.finish()]);

    await staging.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(count);
    result.set(new Float32Array(staging.getMappedRange(), 0, count));
    staging.unmap();
    staging.destroy();
    return result;
  }

  dispatch({ shaderCode, bindings, workgroups, entryPoint = "main" }) {
    const pipeline = this.getOrCreatePipeline(shaderCode, entryPoint);
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: bindings.map((buffer, binding) => ({
        binding,
        resource: { buffer }
      }))
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(...workgroups);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  destroyBuffers(...buffers) {
    for (const buffer of buffers) {
      buffer?.destroy();
    }
  }

  async matmul({ left, right, m, n, k }) {
    await this.ensureDevice();
    const leftBuffer = this.createStorageBuffer(left);
    const rightBuffer = this.createStorageBuffer(right);
    const outputBuffer = this.createEmptyBuffer(m * n * Float32Array.BYTES_PER_ELEMENT);
    const params = this.createUniformBuffer(new Uint32Array([m, n, k]));

    this.dispatch({
      shaderCode: buildMatmulShader(),
      bindings: [leftBuffer, rightBuffer, outputBuffer, params],
      workgroups: [Math.ceil(n / 16), Math.ceil(m / 16), 1]
    });

    const result = await this.readFloat32Buffer(outputBuffer, m * n);
    this.destroyBuffers(leftBuffer, rightBuffer, outputBuffer, params);
    return result;
  }

  async rmsnorm({ input, weight, rows, cols, epsilon = 1e-6 }) {
    await this.ensureDevice();
    const inputBuffer = this.createStorageBuffer(input);
    const weightBuffer = this.createStorageBuffer(weight);
    const outputBuffer = this.createEmptyBuffer(rows * cols * Float32Array.BYTES_PER_ELEMENT);
    const params = this.createUniformBuffer(
      new Uint32Array([rows, cols, epsilonBits(epsilon)])
    );

    this.dispatch({
      shaderCode: buildRmsNormShader(),
      bindings: [inputBuffer, weightBuffer, outputBuffer, params],
      workgroups: [rows, 1, 1]
    });

    const result = await this.readFloat32Buffer(outputBuffer, rows * cols);
    this.destroyBuffers(inputBuffer, weightBuffer, outputBuffer, params);
    return result;
  }

  async smokeTest() {
    const matmul = await this.matmul({
      left: new Float32Array([1, 2, 3, 4]),
      right: new Float32Array([5, 6, 7, 8]),
      m: 2,
      n: 2,
      k: 2
    });

    const rmsnorm = await this.rmsnorm({
      input: new Float32Array([1, 2, 3, 4]),
      weight: new Float32Array([1, 1]),
      rows: 2,
      cols: 2
    });

    return {
      matmul: Array.from(matmul),
      rmsnorm: Array.from(rmsnorm)
    };
  }

  describe() {
    if (!this.probeResult) {
      return "WebGPU has not been probed yet.";
    }
    if (!this.probeResult.available) {
      return this.probeResult.reason;
    }
    return "WebGPU device is available for custom kernels.";
  }
}
