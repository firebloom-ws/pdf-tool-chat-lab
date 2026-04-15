import { OpfsCache } from "./opfs-cache.js";
import { inspectModelSafetensors } from "./safetensors.js";

const SMALL_CACHE_LIMIT = 8_000_000;

let hubModulePromise = null;
let jinjaModulePromise = null;
const resolvedRangeUrls = new Map();

async function loadHubModule() {
  if (!hubModulePromise) {
    hubModulePromise = import("@huggingface/hub");
  }
  return hubModulePromise;
}

async function loadJinjaModule() {
  if (!jinjaModulePromise) {
    jinjaModulePromise = import("@huggingface/jinja");
  }
  return jinjaModulePromise;
}

function encodeHubPath(path) {
  return path
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
}

export class HfHubClient {
  constructor() {
    this.cache = new OpfsCache("papertrail-lab");
    this.listCache = new Map();
    this.chatTemplateCache = new Map();
  }

  async fetchBlob(repo, path, { cache = true, type = "" } = {}) {
    const { downloadFile } = await loadHubModule();
    const cacheKey = `${repo}:${path}`;
    if (cache) {
      const cached = await this.cache.readBlob(cacheKey, type);
      if (cached) {
        return cached;
      }
    }

    const blob = await downloadFile({ repo, path });
    if (!blob) {
      return null;
    }

    if (cache && blob.size <= SMALL_CACHE_LIMIT) {
      await this.cache.writeBlob(cacheKey, blob);
    }

    return blob;
  }

  async fetchRange(repo, path, start, end) {
    const key = `${repo}:main:${path}`;
    const initialUrl =
      resolvedRangeUrls.get(key) ??
      `https://huggingface.co/${repo}/resolve/main/${encodeHubPath(path)}?download=1`;
    let response = await fetch(initialUrl, {
      headers: {
        Range: `bytes=${start}-${Math.max(start, end - 1)}`
      }
    });
    if (!(response.ok || response.status === 206)) {
      return null;
    }
    if (response.url) {
      resolvedRangeUrls.set(key, response.url);
    }
    if (response.status !== 206 && start > 0 && response.url && response.url !== initialUrl) {
      const retry = await fetch(response.url, {
        headers: {
          Range: `bytes=${start}-${Math.max(start, end - 1)}`
        }
      });
      if (retry.ok || retry.status === 206) {
        response = retry;
      }
    }
    const total = Math.max(0, end - start);
    if (total === 0) {
      return new ArrayBuffer(0);
    }

    if (!response.body) {
      const bytes = new Uint8Array(await response.arrayBuffer());
      const sliced =
        response.status === 206
          ? bytes.slice(0, Math.min(total, bytes.byteLength))
          : bytes.slice(start, Math.min(end, bytes.byteLength));
      return sliced.buffer.slice(sliced.byteOffset, sliced.byteOffset + sliced.byteLength);
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
      if (loaded >= total) {
        await reader.cancel("range-complete");
        break;
      }
    }

    const sliced = loaded === total ? bytes : bytes.slice(0, loaded);
    return sliced.buffer.slice(sliced.byteOffset, sliced.byteOffset + sliced.byteLength);
  }

  async fetchText(repo, path, options) {
    const blob = await this.fetchBlob(repo, path, options);
    if (!blob) {
      return null;
    }
    return blob.text();
  }

  async fetchJson(repo, path) {
    const text = await this.fetchText(repo, path, { type: "application/json" });
    if (!text) {
      return null;
    }
    return JSON.parse(text);
  }

  async listModelFiles(repo) {
    if (this.listCache.has(repo)) {
      return this.listCache.get(repo);
    }

    const { listFiles } = await loadHubModule();
    const files = [];
    for await (const file of listFiles({ repo, recursive: true })) {
      files.push(file);
    }
    this.listCache.set(repo, files);
    return files;
  }

  async describeModel(name, keyFiles = []) {
    const { modelInfo } = await loadHubModule();
    const [info, fileEntries] = await Promise.all([
      modelInfo({ name }),
      this.listModelFiles(name)
    ]);
    const presentPaths = new Set(fileEntries.map((entry) => entry.path));
    const files = await Promise.all(
      keyFiles.map(async (path) => ({ path, exists: presentPaths.has(path) }))
    );

    return {
      name: info.name,
      downloads: info.downloads,
      likes: info.likes,
      updatedAt: info.updatedAt,
      task: info.task,
      fileCount: fileEntries.length,
      files
    };
  }

  async loadTokenizerBundle(repo) {
    const [tokenizerConfig, tokenizerJson, vocabJson, mergesText, chatTemplate] = await Promise.all([
      this.fetchJson(repo, "tokenizer_config.json"),
      this.fetchJson(repo, "tokenizer.json"),
      this.fetchJson(repo, "vocab.json"),
      this.fetchText(repo, "merges.txt"),
      this.fetchText(repo, "chat_template.jinja")
    ]);

    return {
      tokenizerConfig,
      tokenizerJson,
      vocabJson,
      mergesText,
      chatTemplate
    };
  }

  async inspectSafetensors(repo) {
    return inspectModelSafetensors(this, repo);
  }

  async loadChatTemplate(repo) {
    if (this.chatTemplateCache.has(repo)) {
      return this.chatTemplateCache.get(repo);
    }

    const promise = (async () => {
      const [tokenizerConfig, chatTemplateFile] = await Promise.all([
        this.fetchJson(repo, "tokenizer_config.json"),
        this.fetchText(repo, "chat_template.jinja")
      ]);
      return chatTemplateFile ?? tokenizerConfig?.chat_template ?? null;
    })();

    this.chatTemplateCache.set(repo, promise);
    try {
      return await promise;
    } catch (error) {
      this.chatTemplateCache.delete(repo);
      throw error;
    }
  }

  async renderChatPrompt(repo, messages, extraContext = {}) {
    const [tokenizerConfig, chatTemplate] = await Promise.all([
      this.fetchJson(repo, "tokenizer_config.json"),
      this.loadChatTemplate(repo)
    ]);

    if (chatTemplate) {
      try {
        const { Template } = await loadJinjaModule();
        const template = new Template(chatTemplate);
        return template.render({
          messages,
          bos_token: tokenizerConfig.bos_token ?? "",
          eos_token: tokenizerConfig.eos_token ?? "",
          ...extraContext
        });
      } catch {
        // Fall through to plain formatting.
      }
    }

    return messages
      .map((message) => `<${message.role}>\n${message.content}\n</${message.role}>`)
      .join("\n");
  }
}
