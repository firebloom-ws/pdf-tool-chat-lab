const HEADER_LIMIT_BYTES = 25_000_000;
const SAFETENSORS_INDEX_RE = /\.safetensors\.index\.json$/;
const SAFETENSORS_RE = /\.safetensors$/;

const DTYPE_BYTE_SIZE = {
  BOOL: 1,
  U8: 1,
  I8: 1,
  U16: 2,
  I16: 2,
  F16: 2,
  BF16: 2,
  U32: 4,
  I32: 4,
  F32: 4,
  U64: 8,
  I64: 8,
  F64: 8
};

function elementCount(shape) {
  return shape.reduce((total, size) => total * size, 1);
}

function joinRelativePath(basePath, relativePath) {
  const slash = basePath.lastIndexOf("/");
  const prefix = slash === -1 ? "" : basePath.slice(0, slash + 1);
  return `${prefix}${relativePath}`;
}

function parseHeaderSize(bytes) {
  return Number(new DataView(bytes).getBigUint64(0, true));
}

function halfToFloat32(value) {
  const sign = (value & 0x8000) << 16;
  const exponent = (value & 0x7c00) >> 10;
  const fraction = value & 0x03ff;

  let bits;
  if (exponent === 0) {
    if (fraction === 0) {
      bits = sign;
    } else {
      let mantissa = fraction;
      let shift = 0;
      while ((mantissa & 0x0400) === 0) {
        mantissa <<= 1;
        shift += 1;
      }
      mantissa &= 0x03ff;
      bits = sign | ((127 - 15 - shift) << 23) | (mantissa << 13);
    }
  } else if (exponent === 0x1f) {
    bits = sign | 0x7f800000 | (fraction << 13);
  } else {
    bits = sign | ((exponent + 112) << 23) | (fraction << 13);
  }

  return new Float32Array(new Uint32Array([bits]).buffer)[0];
}

function bfloat16ToFloat32(value) {
  const bits = value << 16;
  return new Float32Array(new Uint32Array([bits]).buffer)[0];
}

export function decodeTensorBytesToFloat32(dtype, arrayBuffer) {
  switch (dtype) {
    case "F32":
      return new Float32Array(arrayBuffer);
    case "F16": {
      const source = new Uint16Array(arrayBuffer);
      const output = new Float32Array(source.length);
      for (let index = 0; index < source.length; index += 1) {
        output[index] = halfToFloat32(source[index]);
      }
      return output;
    }
    case "BF16": {
      const source = new Uint16Array(arrayBuffer);
      const output = new Float32Array(source.length);
      for (let index = 0; index < source.length; index += 1) {
        output[index] = bfloat16ToFloat32(source[index]);
      }
      return output;
    }
    case "I32":
      return Float32Array.from(new Int32Array(arrayBuffer));
    case "U32":
      return Float32Array.from(new Uint32Array(arrayBuffer));
    case "I16":
      return Float32Array.from(new Int16Array(arrayBuffer));
    case "U16":
      return Float32Array.from(new Uint16Array(arrayBuffer));
    case "I8":
      return Float32Array.from(new Int8Array(arrayBuffer));
    case "U8":
    case "BOOL":
      return Float32Array.from(new Uint8Array(arrayBuffer));
    default:
      throw new Error(`Unsupported tensor dtype for Float32 decode: ${dtype}`);
  }
}

export async function readSafetensorsHeader(client, repo, path) {
  const lengthBytes = await client.fetchRange(repo, path, 0, 8);
  if (!lengthBytes) {
    throw new Error(`Could not read safetensors header length for ${path}`);
  }

  const headerByteLength = parseHeaderSize(lengthBytes);
  if (headerByteLength <= 0 || headerByteLength > HEADER_LIMIT_BYTES) {
    throw new Error(`Refusing suspicious safetensors header size for ${path}`);
  }

  const headerBytes = await client.fetchRange(repo, path, 8, 8 + headerByteLength);
  if (!headerBytes) {
    throw new Error(`Could not read safetensors header for ${path}`);
  }
  const header = JSON.parse(new TextDecoder().decode(headerBytes));
  const payloadOffset = 8 + headerByteLength;

  return { header, payloadOffset };
}

function buildTensorDescriptors(path, header, payloadOffset) {
  return Object.entries(header)
    .filter(([name]) => name !== "__metadata__")
    .map(([name, tensor]) => {
      const [start, end] = tensor.data_offsets;
      return {
        name,
        filePath: path,
        dtype: tensor.dtype,
        shape: tensor.shape,
        elementCount: elementCount(tensor.shape),
        byteStart: payloadOffset + start,
        byteEnd: payloadOffset + end,
        byteLength: end - start
      };
    });
}

export async function inspectModelSafetensors(client, repo) {
  const files = await client.listModelFiles(repo);
  const paths = files.map((file) => file.path);
  const indexPath =
    paths.find((path) => path === "model.safetensors.index.json") ??
    paths.find((path) => SAFETENSORS_INDEX_RE.test(path));

  if (indexPath) {
    const indexJson = await client.fetchJson(repo, indexPath);
    const shardNames = [...new Set(Object.values(indexJson.weight_map))];
    const shards = [];
    const tensors = [];

    for (const shardName of shardNames) {
      const shardPath = joinRelativePath(indexPath, shardName);
      const { header, payloadOffset } = await readSafetensorsHeader(client, repo, shardPath);
      shards.push({ path: shardPath, tensorCount: Object.keys(header).length - 1 });
      const descriptors = buildTensorDescriptors(shardPath, header, payloadOffset);
      for (const descriptor of descriptors) {
        tensors.push(descriptor);
      }
    }

    return {
      sharded: true,
      indexPath,
      shardCount: shards.length,
      shards,
      tensors
    };
  }

  const tensorPaths = paths.filter((path) => SAFETENSORS_RE.test(path)).sort();
  const shards = [];
  const tensors = [];

  for (const path of tensorPaths) {
    const { header, payloadOffset } = await readSafetensorsHeader(client, repo, path);
    shards.push({ path, tensorCount: Object.keys(header).length - 1 });
    tensors.push(...buildTensorDescriptors(path, header, payloadOffset));
  }

  return {
    sharded: false,
    indexPath: null,
    shardCount: shards.length,
    shards,
    tensors
  };
}

export async function readTensorPreview(client, repo, tensor, sampleElements = 64) {
  const bytesPerElement = DTYPE_BYTE_SIZE[tensor.dtype];
  if (!bytesPerElement) {
    throw new Error(`Unsupported preview dtype: ${tensor.dtype}`);
  }

  const byteLength = Math.min(tensor.byteLength, sampleElements * bytesPerElement);
  const range = await client.fetchRange(
    repo,
    tensor.filePath,
    tensor.byteStart,
    tensor.byteStart + byteLength
  );
  if (!range) {
    throw new Error(`Could not read tensor preview for ${tensor.name}`);
  }

  return decodeTensorBytesToFloat32(tensor.dtype, range);
}
