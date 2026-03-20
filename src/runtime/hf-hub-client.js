import { OpfsCache } from "./opfs-cache.js";
import { inspectModelSafetensors } from "./safetensors.js";

const SMALL_CACHE_LIMIT = 8_000_000;

let hubModulePromise = null;
let jinjaModulePromise = null;

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
    const response = await fetch(
      `https://huggingface.co/${repo}/resolve/main/${encodeHubPath(path)}?download=1`,
      {
        headers: {
          Range: `bytes=${start}-${Math.max(start, end - 1)}`
        }
      }
    );
    if (!response.ok && response.status !== 206) {
      return null;
    }
    return response.arrayBuffer();
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
    const [tokenizerConfig, tokenizerJson, vocabJson, mergesText] = await Promise.all([
      this.fetchJson(repo, "tokenizer_config.json"),
      this.fetchJson(repo, "tokenizer.json"),
      this.fetchJson(repo, "vocab.json"),
      this.fetchText(repo, "merges.txt")
    ]);

    return {
      tokenizerConfig,
      tokenizerJson,
      vocabJson,
      mergesText
    };
  }

  async inspectSafetensors(repo) {
    return inspectModelSafetensors(this, repo);
  }

  async renderChatPrompt(repo, messages, extraContext = {}) {
    const tokenizerConfig = await this.fetchJson(repo, "tokenizer_config.json");
    if (tokenizerConfig?.chat_template) {
      try {
        const { Template } = await loadJinjaModule();
        const template = new Template(tokenizerConfig.chat_template);
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
