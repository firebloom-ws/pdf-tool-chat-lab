import { buildDocumentIndex } from "../models/document-indexer.js";
import { HashLayoutEmbedder } from "../models/hash-embedder.js";
import {
  buildSectionChunks,
  toOpenDataLoaderLikeDocument
} from "../models/opendataloader-adapter.js";
import { loadPdfBundle } from "../pdf/pdf-service.js";
import { PdfViewer } from "../pdf/pdf-viewer.js";
import { HfHubClient } from "../runtime/hf-hub-client.js";
import { LightOnOCRRuntime, QwenToolRuntime } from "../runtime/model-runtimes.js";
import { WebGpuRuntime } from "../runtime/webgpu-runtime.js";
import { SessionStore } from "../storage/session-store.js";
import { ToolRegistry } from "../tools/tool-registry.js";
import { renderMarkdownToHtml } from "../ui/markdown-renderer.js";
import { VectorDatabase } from "../vdb/vector-db.js";

/* ─── helpers ─────────────────────────────────────────────── */

function serializeChunks(chunks) {
  return chunks.map((c) => ({
    ...c,
    vector: Array.from(c.vector),
    signature: Array.from(c.signature)
  }));
}

function hydrateChunks(chunks) {
  return chunks.map((c) => ({
    ...c,
    vector: new Float32Array(c.vector),
    signature: new Uint32Array(c.signature)
  }));
}

function resolveDocumentTitle(bundle, fallback = "document.pdf") {
  const title = bundle?.metadata?.info?.Title?.trim();
  if (title) return title;
  if (bundle?.file?.name) return bundle.file.name;
  return fallback;
}

function createReferenceChip(reference) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "doc-ref-chip";
  button.dataset.refId = reference.id;
  if (Number.isFinite(reference.pageNumber)) {
    button.dataset.pageNumber = String(reference.pageNumber);
  }
  if (reference.bbox) {
    button.dataset.bboxX = String(reference.bbox.x);
    button.dataset.bboxY = String(reference.bbox.y);
    button.dataset.bboxWidth = String(reference.bbox.width);
    button.dataset.bboxHeight = String(reference.bbox.height);
  }
  button.textContent = reference.label ?? reference.id;
  button.title = reference.snippet ?? reference.label ?? reference.id;
  return button;
}

function createMessageElement(role, text, trace = [], references = []) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  if (role === "assistant") {
    const body = document.createElement("div");
    body.className = "message-content message-markdown";
    body.innerHTML = renderMarkdownToHtml(text, { references });
    article.append(body);

    if (references.length) {
      const title = document.createElement("div");
      title.className = "tool-trace-title";
      title.textContent = "Highlights";

      const strip = document.createElement("div");
      strip.className = "message-highlights";
      for (const reference of references.slice(0, 4)) {
        strip.append(createReferenceChip(reference));
      }
      article.append(title);
      article.append(strip);
    }
  } else {
    const p = document.createElement("p");
    p.className = "message-content";
    p.textContent = text;
    article.append(p);
  }

  if (trace.length) {
    const title = document.createElement("div");
    title.className = "tool-trace-title";
    title.textContent = "Tools used";

    const ul = document.createElement("ul");
    ul.className = "tool-trace";
    for (const item of trace) {
      const li = document.createElement("li");
      li.textContent = item;
      ul.append(li);
    }
    article.append(title);
    article.append(ul);
  }

  return article;
}

function createTypingIndicator() {
  const el = document.createElement("div");
  el.className = "typing-indicator";
  el.innerHTML =
    '<span class="typing-dot"></span>' +
    '<span class="typing-dot"></span>' +
    '<span class="typing-dot"></span>';
  return el;
}

function withInactivityTimeout(startTask, {
  timeoutMs = 90000,
  label = "request-stalled",
  onTimeout = null
} = {}) {
  return new Promise((resolve, reject) => {
    let settled = false;
    let lastActivity = Date.now();

    const touch = () => {
      lastActivity = Date.now();
    };

    const interval = setInterval(() => {
      if (settled) {
        return;
      }
      if (Date.now() - lastActivity < timeoutMs) {
        return;
      }
      settled = true;
      clearInterval(interval);
      Promise.resolve(onTimeout?.())
        .catch(() => {})
        .finally(() => reject(new Error(label)));
    }, 1000);

    Promise.resolve(startTask(touch))
      .then((value) => {
        if (settled) {
          return;
        }
        settled = true;
        clearInterval(interval);
        resolve(value);
      })
      .catch((error) => {
        if (settled) {
          return;
        }
        settled = true;
        clearInterval(interval);
        reject(error);
      });
  });
}

/* ─── AppController ───────────────────────────────────────── */

export class AppController {
  constructor(documentRef) {
    this.document = documentRef;

    this.elements = {
      upload: documentRef.getElementById("pdf-upload"),
      searchForm: documentRef.getElementById("search-form"),
      searchInput: documentRef.getElementById("search-input"),
      searchResults: documentRef.getElementById("search-results"),
      chatForm: documentRef.getElementById("chat-form"),
      chatInput: documentRef.getElementById("chat-input"),
      chatLog: documentRef.getElementById("chat-log"),
      stats: documentRef.getElementById("document-stats"),
      pageRail: documentRef.getElementById("page-rail"),
      pageLabel: documentRef.getElementById("page-label"),
      zoomLabel: documentRef.getElementById("zoom-label"),
      statusPill: documentRef.getElementById("status-pill"),
      savedSnapshots: documentRef.getElementById("saved-snapshots"),
      probeButton: documentRef.getElementById("probe-models-button"),
      exportButton: documentRef.getElementById("export-layout-button"),
      ocrBadge: documentRef.getElementById("ocr-runtime-badge"),
      ocrCopy: documentRef.getElementById("ocr-runtime-copy"),
      qwenBadge: documentRef.getElementById("qwen-runtime-badge"),
      qwenCopy: documentRef.getElementById("qwen-runtime-copy"),
      modelSelect: documentRef.getElementById("model-select"),
      loadModelButton: documentRef.getElementById("load-model-button"),
      modelStatusText: documentRef.getElementById("model-status-text"),
      prevPageButton: documentRef.getElementById("prev-page-button"),
      nextPageButton: documentRef.getElementById("next-page-button"),
      zoomOutButton: documentRef.getElementById("zoom-out-button"),
      zoomInButton: documentRef.getElementById("zoom-in-button"),
      processingLabel: documentRef.getElementById("processing-label"),
      processingDetail: documentRef.getElementById("processing-detail"),
      bgProgress: documentRef.getElementById("bg-progress"),
      docHeader: documentRef.getElementById("doc-header"),
      docTitleBar: documentRef.getElementById("doc-title-bar"),
      dropZone: documentRef.getElementById("drop-zone"),
      recentDocsLanding: documentRef.getElementById("recent-docs-landing")
    };

    this.viewer = new PdfViewer({
      canvas: documentRef.getElementById("pdf-canvas"),
      overlay: documentRef.getElementById("bbox-overlay"),
      frame: documentRef.getElementById("viewer-frame"),
      pageRail: this.elements.pageRail,
      pageLabel: this.elements.pageLabel,
      zoomLabel: this.elements.zoomLabel
    });

    this.sessionStore = new SessionStore();
    this.hfHubClient = new HfHubClient();
    this.webgpuRuntime = new WebGpuRuntime();
    this.ocrRuntime = new LightOnOCRRuntime(this.hfHubClient, this.webgpuRuntime);
    this.qwenRuntime = new QwenToolRuntime(this.hfHubClient, this.webgpuRuntime);

    this.bundle = null;
    this.index = null;
    this.vectorDatabase = null;
    this.toolRegistry = null;
    this.messages = [];

    // Document-profile state subscription — updates the bg-progress pill
    this._profileUnsub = this.qwenRuntime.subscribeDocumentProfile((profileState) => {
      this._onDocumentProfileStateChange(profileState);
    });
  }

  /* ── lifecycle ──────────────────────────────────────────── */

  mount() {
    this.setAppState("landing");
    this.populateModelOptions();
    this.unsubscribeQwenState = this.qwenRuntime.subscribe((state) => {
      this.updateQwenModelState(state);
    });
    this.bindEvents();
    this.refreshSnapshotList().catch(console.error);
    window.addEventListener("resize", () => {
      this.viewer.render().catch(() => {});
    });
  }

  /* ── state machine ──────────────────────────────────────── */

  setAppState(state) {
    this.document.querySelector(".app-shell").dataset.state = state;
  }

  setProcessingStatus(title, detail = "") {
    if (this.elements.processingLabel) {
      this.elements.processingLabel.textContent = title;
    }
    if (this.elements.processingDetail) {
      this.elements.processingDetail.textContent = detail;
    }
  }

  setBackgroundProgress(text) {
    const el = this.elements.bgProgress;
    if (!el) return;
    if (!text) {
      el.hidden = true;
      el.textContent = "";
    } else {
      el.hidden = false;
      el.textContent = text;
    }
  }

  // kept for internal/compat use — not shown in the redesigned UI
  setStatus(text, tone = "neutral") {
    if (this.elements.statusPill) {
      this.elements.statusPill.textContent = text;
      this.elements.statusPill.className = `pill pill-${tone}`;
    }
  }

  /* ── document header ────────────────────────────────────── */

  updateDocHeader() {
    if (!this.bundle || !this.index) return;

    const title = resolveDocumentTitle(this.bundle).replace(/\.pdf$/i, "");
    const pages = this.bundle.pageCount;
    const chunks = this.index.chunks.length;
    const profileStatus = this.qwenRuntime.documentProfileStatus;
    const profile = this.qwenRuntime.documentProfile;

    if (this.elements.docHeader) {
      const profileChip = profile
        ? `<span class="lora-chip lora-chip--ready" title="${profile.describe()}">Profile ✓</span>`
        : profileStatus
          ? `<span class="lora-chip lora-chip--encoding">Profile…</span>`
          : "";
      this.elements.docHeader.innerHTML = `
        <strong>${title}</strong>
        <span>${pages} pages &middot; ${chunks} sections indexed ${profileChip}</span>
      `;
    }

    if (this.elements.docTitleBar) {
      this.elements.docTitleBar.innerHTML = `
        <strong>${title}</strong>
        <span>${pages} pages</span>
      `;
    }
  }

  populateModelOptions() {
    const select = this.elements.modelSelect;
    if (!select) return;

    select.innerHTML = "";
    for (const option of this.qwenRuntime.getModelOptions()) {
      const el = document.createElement("option");
      el.value = option.id;
      el.textContent = `${option.label} · ${option.description}`;
      select.append(el);
    }
    select.value = this.qwenRuntime.getModelState().modelId;
  }

  updateQwenModelState(state) {
    const select = this.elements.modelSelect;
    const button = this.elements.loadModelButton;
    const text = this.elements.modelStatusText;

    if (select && select.value !== state.modelId) {
      select.value = state.modelId;
    }

    if (button) {
      button.disabled = state.status === "loading" || state.ready;
      button.textContent =
        state.status === "loading"
          ? "Loading…"
          : state.ready
            ? "Loaded"
            : "Load Model";
    }

    if (text) {
      if (state.status === "loading" || state.status === "generating") {
        text.textContent = state.detail;
      } else if (state.status === "ready") {
        text.textContent = `${state.label} is live on WebGPU using ${state.dtype}.`;
      } else if (state.status === "error") {
        text.textContent = `Model load failed: ${state.error}`;
      } else if (state.status === "unsupported") {
        text.textContent = state.detail;
      } else {
        text.textContent = `${state.label} is not loaded yet. Search fallback remains available.`;
      }
    }

    if (state.status === "loading") {
      this.setBackgroundProgress(state.detail);
    }

    if (this.elements.qwenBadge) {
      this.elements.qwenBadge.textContent = state.ready
        ? "WebGPU Ready"
        : state.status === "loading"
          ? "Loading"
          : "Search Fallback";
      this.elements.qwenBadge.className = state.ready ? "pill pill-success" : "pill pill-warning";
    }
    if (this.elements.qwenCopy) {
      this.elements.qwenCopy.textContent = text?.textContent ?? state.detail;
    }
  }

  /* ── event binding ──────────────────────────────────────── */

  bindEvents() {
    // File upload via input
    this.elements.upload.addEventListener("change", async (e) => {
      const [file] = e.target.files ?? [];
      if (file) await this.loadFile(file);
    });

    // Drag & drop on the landing zone
    const dz = this.elements.dropZone;
    if (dz) {
      dz.addEventListener("dragover", (e) => {
        e.preventDefault();
        dz.classList.add("is-dragover");
      });
      dz.addEventListener("dragleave", () => dz.classList.remove("is-dragover"));
      dz.addEventListener("drop", async (e) => {
        e.preventDefault();
        dz.classList.remove("is-dragover");
        const file = e.dataTransfer?.files?.[0];
        if (file?.type === "application/pdf") await this.loadFile(file);
      });
    }

    // Chat form submit
    this.elements.chatForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const message = this.elements.chatInput.value.trim();
      if (!message) return;
      this.elements.chatInput.value = "";
      this.elements.chatInput.style.height = "auto";
      await this.sendChatMessage(message);
    });

    // Enter = submit (Shift+Enter = newline)
    this.elements.chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        if (this.elements.chatInput.value.trim()) {
          this.elements.chatForm.requestSubmit();
        }
      }
    });

    // Auto-resize textarea
    this.elements.chatInput.addEventListener("input", () => {
      const el = this.elements.chatInput;
      el.style.height = "auto";
      el.style.height = `${Math.min(el.scrollHeight, 120)}px`;
    });

    this.elements.chatLog.addEventListener("click", async (event) => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) {
        return;
      }
      const pageChip = target.closest("[data-page-number]");
      if (!(pageChip instanceof HTMLElement)) {
        return;
      }
      const pageNumber = Number(pageChip.dataset.pageNumber);
      if (!Number.isFinite(pageNumber) || !this.toolRegistry) {
        return;
      }
      const bbox = ["bboxX", "bboxY", "bboxWidth", "bboxHeight"].every((key) => key in pageChip.dataset)
        ? {
            x: Number(pageChip.dataset.bboxX),
            y: Number(pageChip.dataset.bboxY),
            width: Number(pageChip.dataset.bboxWidth),
            height: Number(pageChip.dataset.bboxHeight)
          }
        : null;
      await this.toolRegistry.openPage({ pageNumber, bbox });
    });

    // Viewer controls
    this.elements.prevPageButton.addEventListener("click", () => this.viewer.stepPage(-1));
    this.elements.nextPageButton.addEventListener("click", () => this.viewer.stepPage(1));
    this.elements.zoomOutButton.addEventListener("click", () => {
      this.viewer.setZoom(this.viewer.zoom - 0.1);
    });
    this.elements.zoomInButton.addEventListener("click", () => {
      this.viewer.setZoom(this.viewer.zoom + 0.1);
    });

    // Export
    this.elements.exportButton.addEventListener("click", () => this.exportLayoutJson());

    if (this.elements.modelSelect) {
      this.elements.modelSelect.addEventListener("change", () => {
        this.qwenRuntime.setModel(this.elements.modelSelect.value);
      });
    }

    if (this.elements.loadModelButton) {
      this.elements.loadModelButton.addEventListener("click", () => {
        this.probeModels().catch((error) => {
          console.error("Model load failed", error);
          this.setBackgroundProgress("");
          this.updateQwenModelState(this.qwenRuntime.getModelState());
        });
      });
    }

    // Hidden probe button (kept for compat)
    this.elements.probeButton.addEventListener("click", () => {
      this.probeModels().catch(() => this.setStatus("Model load failed", "warning"));
    });

    // Mobile tab bar
    const shell = this.document.querySelector(".app-shell");
    this.document.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        shell.dataset.tab = btn.dataset.tab;
        this.document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("is-active"));
        btn.classList.add("is-active");
        if (btn.dataset.tab === "viewer") this.viewer.render().catch(() => {});
      });
    });
  }

  /* ── chat ───────────────────────────────────────────────── */

  resetChat() {
    this.messages = [];
    this.elements.chatLog.innerHTML = "";
    if (this.bundle) {
      const title = resolveDocumentTitle(this.bundle).replace(/\.pdf$/i, "");
      this.elements.chatLog.append(
        createMessageElement("assistant", `"${title}" is ready. What would you like to know?`)
      );
    }
  }

  scrollChatToBottom() {
    const log = this.elements.chatLog;
    log.scrollTop = log.scrollHeight;
  }

  async sendChatMessage(message) {
    this.messages.push({ role: "user", content: message });
    this.elements.chatLog.append(createMessageElement("user", message));
    this.scrollChatToBottom();

    const typing = createTypingIndicator();
    this.elements.chatLog.append(typing);
    this.scrollChatToBottom();
    let draft = null;

    const sendBtn = this.elements.chatForm.querySelector(".send-btn");
    this.elements.chatInput.disabled = true;
    if (sendBtn) sendBtn.disabled = true;

    try {
      const response = await withInactivityTimeout(
        (touch) => this.qwenRuntime.run(this.messages, {
          onPartial: (partialText) => {
            touch();
            if (!partialText?.trim()) {
              return;
            }
            if (!draft) {
              draft = createMessageElement("assistant", "");
              typing.replaceWith(draft);
            }
            const body = draft.querySelector(".message-content");
            if (body) {
              body.textContent = partialText;
            }
            this.scrollChatToBottom();
          }
        }),
        {
          timeoutMs: 90000,
          label: "chat-stalled",
          onTimeout: () => this.qwenRuntime.interrupt?.()
        }
      );
      this.messages.push({ role: "assistant", content: response.text });
      const responseEl = createMessageElement(
        "assistant",
        response.text,
        response.trace,
        response.references ?? []
      );
      if (draft) {
        draft.replaceWith(responseEl);
      } else {
        typing.replaceWith(responseEl);
      }
      this.scrollChatToBottom();
    } catch (err) {
      console.error("Chat error", err);
      const errorEl = createMessageElement(
        "assistant",
        err instanceof Error && err.message === "chat-stalled"
          ? "The model stopped making progress, so the app interrupted generation. Try again, or ask a narrower question."
          : "Something went wrong. Please try again."
      );
      if (draft) {
        draft.replaceWith(errorEl);
      } else {
        typing.replaceWith(errorEl);
      }
      this.scrollChatToBottom();
    } finally {
      this.elements.chatInput.disabled = false;
      if (sendBtn) sendBtn.disabled = false;
      this.elements.chatInput.focus();
    }
  }

  /* ── document loading ───────────────────────────────────── */

  async loadFile(file) {
    this.setAppState("processing");
    this.setProcessingStatus("Reading document\u2026", file.name);

    try {
      this.bundle = await loadPdfBundle(file);
    } catch (error) {
      console.error("Failed to load PDF", error);
      this.setAppState("landing");
      return;
    }

    this.setProcessingStatus(
      `Indexing ${this.bundle.pageCount} pages\u2026`,
      "Building search index"
    );

    try {
      this.index = await buildDocumentIndex(this.bundle, {
        ocrRuntime: this.ocrRuntime,
        onProgress: (label) => {
          this.setProcessingStatus(label, "");
        }
      });
    } catch (error) {
      console.error("Failed to index PDF", error);
      this.setAppState("landing");
      return;
    }

    this.vectorDatabase = new VectorDatabase(this.index.dimensions);
    await this.vectorDatabase.hydrate(this.index.chunks);

    this.toolRegistry = new ToolRegistry({
      viewer: this.viewer,
      vectorDatabase: this.vectorDatabase,
      index: this.index
    });
    this.qwenRuntime.setToolRegistry(this.toolRegistry);
    this.qwenRuntime.resetDocumentProfile();

    await this.viewer.attachDocument(this.bundle, this.index.pages);
    await this.persistCurrentSnapshot();
    await this.refreshSnapshotList();

    this.resetChat();
    this.setAppState("loaded");
    this.updateDocHeader();
    this.setStatus("Ready", "success");

    // Kick off background work — fire & forget
    this.runBackgroundWork().catch(console.error);
  }

  /* ── Document Profile ────────────────────────────────────── */

  _onDocumentProfileStateChange(profileState) {
    const { state, detail } = profileState;
    if (state === "encoding") {
      this.setBackgroundProgress(`⚙ Profile: ${detail}`);
      this.updateDocHeader();
    } else if (state === "ready") {
      this.setBackgroundProgress("✓ Profile ready");
      this.updateDocHeader();
      // Clear the pill after 3 s so it doesn't persist indefinitely
      clearTimeout(this._loraPillTimer);
      this._loraPillTimer = setTimeout(() => {
        this.setBackgroundProgress("");
      }, 3000);
    } else if (state === "error") {
      this.setBackgroundProgress(`Profile: ${detail}`);
      clearTimeout(this._loraPillTimer);
      this._loraPillTimer = setTimeout(() => this.setBackgroundProgress(""), 6000);
    }
  }

  /**
   * Kick off document profiling for the currently loaded document.
   * Safe to call multiple times — the encoder serialises concurrent calls.
   */
  async encodeDocumentProfile() {
    if (!this.index || !this.bundle) return;

    const title = this.bundle?.metadata?.info?.Title?.trim()
      || this.bundle?.file?.name
      || "document";

    try {
      await this.qwenRuntime.encodeDocument(this.index, {
        title,
        onProgress: (fraction, message) => {
          const pct = Math.round(fraction * 100);
          this.setBackgroundProgress(`⚙ Profile ${pct}% — ${message}`);
        }
      });
    } catch (err) {
      console.error("[AppController] Document profile build failed:", err);
    }
  }

  /* ── background work (page scan + model probe + document profile) ─── */

  async runBackgroundWork() {
    if (!this.bundle) return;
    const pageCount = this.bundle.pageCount;

    // Phase 1: per-page scanning progress
    for (let i = 1; i <= pageCount; i++) {
      this.setBackgroundProgress(`Analyzing ${i} / ${pageCount} pages…`);
      await new Promise((r) => setTimeout(r, 18));
    }

    // Phase 2: load AI model assets
    this.setBackgroundProgress("Loading AI model\u2026");
    try {
      await this.probeModels();
    } catch {
      // model loading is best-effort
    }

    // Phase 3: compile a bounded document profile after model load
    await this.encodeDocumentProfile();

    this.setBackgroundProgress("");
  }

  /* ── model probing (also called by hidden probe button) ─── */

  async probeModels() {
    this.setStatus("Loading model assets", "working");
    const [ocrInfo, qwenInfo, gpuInfo] = await Promise.all([
      this.ocrRuntime.probe(),
      this.qwenRuntime.probe(),
      this.webgpuRuntime.probe()
    ]);

    const gpuNote = gpuInfo.available
      ? " WebGPU available."
      : ` ${gpuInfo.reason}`;

    const ocrPreview = ocrInfo.previewTensor
      ? ` Sampled ${ocrInfo.previewTensor.name} (${ocrInfo.previewTensor.dtype}).`
      : "";

    if (this.elements.ocrBadge) {
      this.elements.ocrBadge.textContent = "Hub Ready";
      this.elements.ocrBadge.className = "pill pill-warning";
    }
    if (this.elements.ocrCopy) {
      this.elements.ocrCopy.textContent =
        `${ocrInfo.model.files.filter((f) => f.exists).length}/${ocrInfo.model.files.length} files.` +
        ocrPreview + gpuNote;
    }
    this.updateQwenModelState(this.qwenRuntime.getModelState());
    if (this.elements.qwenCopy) {
      this.elements.qwenCopy.textContent =
        `${qwenInfo.label} via ${qwenInfo.backend}.` +
        (qwenInfo.ready
          ? ` Loaded on WebGPU with ${qwenInfo.dtype}.`
          : ` ${qwenInfo.detail}`) +
        gpuNote;
    }

    this.setBackgroundProgress("");
    this.setStatus("Ready", "success");
    return { ocrInfo, qwenInfo, gpuInfo };
  }

  /* ── snapshots ──────────────────────────────────────────── */

  async persistCurrentSnapshot() {
    if (!this.bundle || !this.index) return;
    await this.sessionStore.saveSnapshot({
      id: crypto.randomUUID(),
      savedAt: Date.now(),
      title: resolveDocumentTitle(this.bundle),
      pdfBlob: this.bundle.file,
      metadata: this.bundle.metadata,
      pageCount: this.bundle.pageCount,
      pages: this.index.pages,
      chunks: serializeChunks(this.index.chunks)
    });
  }

  async refreshSnapshotList() {
    const snapshots = await this.sessionStore.listSnapshots();
    const recentEl = this.elements.recentDocsLanding;

    // Update hidden compat element
    if (!snapshots.length) {
      this.elements.savedSnapshots.className = "results-list empty-state";
      this.elements.savedSnapshots.textContent = "No saved files yet.";
      if (recentEl) recentEl.innerHTML = "";
      return;
    }

    this.elements.savedSnapshots.className = "results-list";
    this.elements.savedSnapshots.innerHTML = "";

    // Landing screen: "Recent" section
    if (recentEl) {
      recentEl.innerHTML = "";
      const label = document.createElement("span");
      label.className = "recent-docs-label";
      label.textContent = "Recent";
      recentEl.append(label);

      for (const snap of snapshots.slice(0, 3)) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "recent-doc-btn";
        btn.innerHTML = `
          <strong>${snap.title.replace(/\.pdf$/i, "")}</strong>
          <span>${snap.pageCount}p</span>
        `;
        btn.addEventListener("click", () => this.loadSnapshot(snap.id));
        recentEl.append(btn);
      }
    }

    // Compat list
    for (const snap of snapshots.slice(0, 5)) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "result-card";
      btn.innerHTML = `
        <div class="result-head">
          <strong>${snap.title}</strong>
          <span>${snap.pageCount}p</span>
        </div>
        <p>${new Date(snap.savedAt).toLocaleString()}</p>
      `;
      btn.addEventListener("click", () => this.loadSnapshot(snap.id));
      this.elements.savedSnapshots.append(btn);
    }
  }

  async loadSnapshot(id) {
    const snapshot = await this.sessionStore.loadSnapshot(id);
    if (!snapshot) return;

    this.setAppState("processing");
    this.setProcessingStatus("Opening document\u2026", snapshot.title ?? "");

    const restoredFile =
      snapshot.pdfBlob instanceof File
        ? snapshot.pdfBlob
        : new File([snapshot.pdfBlob], snapshot.title ?? "document.pdf", {
            type: "application/pdf"
          });

    this.bundle = await loadPdfBundle(restoredFile);
    this.index = {
      dimensions: snapshot.chunks[0]?.vector.length ?? 384,
      embedder: this.index?.embedder ?? new HashLayoutEmbedder(),
      pages: snapshot.pages,
      chunks: hydrateChunks(snapshot.chunks)
    };

    this.vectorDatabase = new VectorDatabase(this.index.dimensions);
    await this.vectorDatabase.hydrate(this.index.chunks);
    this.toolRegistry = new ToolRegistry({
      viewer: this.viewer,
      vectorDatabase: this.vectorDatabase,
      index: this.index
    });
    this.qwenRuntime.setToolRegistry(this.toolRegistry);
    this.qwenRuntime.resetDocumentProfile();

    await this.viewer.attachDocument(this.bundle, this.index.pages);

    this.resetChat();
    this.setAppState("loaded");
    this.updateDocHeader();
    this.setStatus("Ready", "success");

    this.runBackgroundWork().catch(console.error);
  }

  /* ── search results (compat — results surface via chat) ─── */

  renderDocumentStats() {
    this.updateDocHeader();
  }

  renderSearchResults(results, query) {
    // Results now surface through the chat interface.
    // Update the hidden compat element to preserve any listeners.
    if (!results.length) {
      this.elements.searchResults.className = "results-list empty-state";
      this.elements.searchResults.textContent = query
        ? "No strong matches. Try a phrase copied from the PDF."
        : "";
      return;
    }
    this.elements.searchResults.className = "results-list";
    this.elements.searchResults.innerHTML = "";
    for (const result of results) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "result-card";
      btn.innerHTML = `
        <div class="result-head">
          <strong>Page ${result.pageNumber}</strong>
          <span>${result.similarity.toFixed(3)}</span>
        </div>
        <p>${result.snippet}</p>
      `;
      btn.addEventListener("click", async () => {
        await this.toolRegistry.openPage({ pageNumber: result.pageNumber, bbox: result.bbox });
      });
      this.elements.searchResults.append(btn);
    }
  }

  /* ── export ─────────────────────────────────────────────── */

  exportLayoutJson() {
    if (!this.index || !this.bundle) return;

    const payload = {
      title: resolveDocumentTitle(this.bundle),
      pageCount: this.bundle.pageCount,
      savedAt: new Date().toISOString(),
      sectionChunks: buildSectionChunks(this.index),
      opendataloaderLike: toOpenDataLoaderLikeDocument(this.bundle, this.index),
      pages: this.index.pages,
      chunks: this.index.chunks.map((c) => ({
        id: c.id,
        pageNumber: c.pageNumber,
        snippet: c.snippet,
        text: c.text,
        bbox: c.bbox,
        elementType: c.elementType,
        headingLevel: c.headingLevel,
        layoutSource: c.layoutSource
      }))
    };

    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${resolveDocumentTitle(this.bundle).replace(/\.pdf$/i, "")}.layout.json`;
    link.click();
    URL.revokeObjectURL(url);
  }
}
