import { OpfsCache } from "./opfs-cache.js";

const CACHE_VERSION = 1;

function modelSlug(modelId) {
  return String(modelId).replace(/[^a-z0-9]+/gi, "-").replace(/^-+|-+$/g, "").toLowerCase();
}

function manifestKey(modelId, revision) {
  return `tb-cache:${modelSlug(modelId)}:${revision}:manifest`;
}

function shardMetaKey(modelId, revision, shardIndex) {
  return `tb-cache:${modelSlug(modelId)}:${revision}:shard:${shardIndex}:meta`;
}

function shardDataKey(modelId, revision, shardIndex) {
  return `tb-cache:${modelSlug(modelId)}:${revision}:shard:${shardIndex}:data`;
}

function cloneShape(shape) {
  return Array.isArray(shape) ? shape.map((value) => Number(value)) : [];
}

function packShard(shard) {
  const entries = Object.entries(shard ?? {});
  const tensors = [];
  let totalBytes = 0;

  for (const [name, tensor] of entries) {
    const bytes = tensor?.data
      ? new Uint8Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.byteLength)
      : new Uint8Array();
    tensors.push({
      name,
      dtype: tensor?.dtype ?? "U8",
      shape: cloneShape(tensor?.shape),
      byteLength: bytes.byteLength,
      bytes
    });
    totalBytes += bytes.byteLength;
  }

  const payload = new Uint8Array(totalBytes);
  const meta = [];
  let offset = 0;

  for (const tensor of tensors) {
    payload.set(tensor.bytes, offset);
    meta.push({
      name: tensor.name,
      dtype: tensor.dtype,
      shape: tensor.shape,
      offset,
      byteLength: tensor.byteLength
    });
    offset += tensor.byteLength;
  }

  return { payload, meta };
}

function unpackShard(payload, meta) {
  const shard = {};
  const payloadBytes =
    payload instanceof Uint8Array ? payload : new Uint8Array(payload ?? new ArrayBuffer(0));

  for (const entry of meta ?? []) {
    shard[entry.name] = {
      dtype: entry.dtype,
      shape: cloneShape(entry.shape),
      data: new Uint8Array(
        payloadBytes.buffer,
        payloadBytes.byteOffset + entry.offset,
        entry.byteLength
      )
    };
  }

  return shard;
}

export class TensorbendPackedCache {
  constructor(namespace = "papertrail-tensorbend") {
    this.cache = new OpfsCache(namespace);
  }

  async readManifest(modelId, revision) {
    const manifest = await this.cache.readJson(manifestKey(modelId, revision));
    if (!manifest || manifest.version !== CACHE_VERSION) {
      return null;
    }
    return manifest;
  }

  async writeManifest(modelId, revision, manifest) {
    await this.cache.writeJson(manifestKey(modelId, revision), {
      version: CACHE_VERSION,
      modelId,
      revision,
      createdAt: Date.now(),
      ...manifest
    });
  }

  async writeShard(modelId, revision, shardIndex, shard, extra = {}) {
    const { payload, meta } = packShard(shard);
    await this.cache.writeBytes(shardDataKey(modelId, revision, shardIndex), payload);
    await this.cache.writeJson(shardMetaKey(modelId, revision, shardIndex), {
      shardIndex,
      tensorCount: meta.length,
      byteLength: payload.byteLength,
      tensors: meta,
      ...extra
    });
    return {
      tensorCount: meta.length,
      byteLength: payload.byteLength
    };
  }

  async readShard(modelId, revision, shardIndex) {
    const [meta, payload] = await Promise.all([
      this.cache.readJson(shardMetaKey(modelId, revision, shardIndex)),
      this.cache.readBytes(shardDataKey(modelId, revision, shardIndex))
    ]);

    if (!meta || !payload) {
      return null;
    }

    return unpackShard(payload, meta.tensors);
  }

  async clearModel(modelId, revision, manifest = null) {
    const resolvedManifest = manifest ?? (await this.readManifest(modelId, revision));
    if (resolvedManifest?.shards?.length) {
      for (const shard of resolvedManifest.shards) {
        const shardIndex = Number(shard.shardIndex);
        await Promise.all([
          this.cache.delete(shardMetaKey(modelId, revision, shardIndex)),
          this.cache.delete(shardDataKey(modelId, revision, shardIndex))
        ]);
      }
    }
    await this.cache.delete(manifestKey(modelId, revision));
  }
}
