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

function serializeReferences(references = []) {
  return references.map((reference) => ({
    id: reference.id,
    label: reference.label ?? reference.id,
    snippet: reference.snippet ?? "",
    pageNumber: Number.isFinite(reference.pageNumber) ? reference.pageNumber : null,
    bbox: reference.bbox
      ? {
          x: reference.bbox.x,
          y: reference.bbox.y,
          width: reference.bbox.width,
          height: reference.bbox.height
        }
      : null
  }));
}

function serializeMessages(messages = []) {
  return messages.map((message) => ({
    role: message.role,
    content: message.content,
    trace: Array.isArray(message.trace) ? [...message.trace] : [],
    references: serializeReferences(message.references ?? [])
  }));
}

function hydrateMessages(messages = []) {
  return messages
    .filter((message) => message?.role && typeof message?.content === "string")
    .map((message) => ({
      role: message.role,
      content: message.content,
      trace: Array.isArray(message.trace) ? [...message.trace] : [],
      references: serializeReferences(message.references ?? [])
    }));
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
    '<div class="typing-dots" aria-hidden="true">' +
    '<span class="typing-dot"></span>' +
    '<span class="typing-dot"></span>' +
    '<span class="typing-dot"></span>' +
    "</div>" +
    '<span class="typing-label">Thinking…</span>';
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

const CHAT_LAYOUT_STORAGE_KEY = "papertrail-chat-layout-v1";
const CHAT_NODE_DEFAULTS = {
  doc: { x: 20, y: 28, w: 248, h: 138, collapsed: false },
  model: { x: 320, y: 46, w: 286, h: 228, collapsed: false },
  config: { x: 24, y: 238, w: 286, h: 472, collapsed: false },
  conversation: { x: 332, y: 324, w: 548, h: 448, collapsed: false }
};
const CHAT_NODE_MIN_SIZES = {
  doc: { w: 180, h: 112 },
  model: { w: 240, h: 176 },
  config: { w: 240, h: 260 },
  conversation: { w: 320, h: 260 }
};
const CHAT_NODE_COLLAPSED_SIZE = { w: 176, h: 52 };
const CHAT_DRAG_CLICK_THRESHOLD = 5;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function normalizeChatNodeLayout(id, layout = {}) {
  const defaults = CHAT_NODE_DEFAULTS[id] ?? { x: 16, y: 16, w: 260, h: 180, collapsed: false };
  return {
    x: Number.isFinite(layout.x) ? layout.x : defaults.x,
    y: Number.isFinite(layout.y) ? layout.y : defaults.y,
    w: Number.isFinite(layout.w) ? layout.w : defaults.w,
    h: Number.isFinite(layout.h) ? layout.h : defaults.h,
    collapsed: Boolean(layout.collapsed)
  };
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
      modelProgressPanel: documentRef.getElementById("model-progress-panel"),
      modelProgressPhase: documentRef.getElementById("model-progress-phase"),
      modelProgressPercent: documentRef.getElementById("model-progress-percent"),
      modelProgressFill: documentRef.getElementById("model-progress-fill"),
      prevPageButton: documentRef.getElementById("prev-page-button"),
      nextPageButton: documentRef.getElementById("next-page-button"),
      zoomOutButton: documentRef.getElementById("zoom-out-button"),
      zoomInButton: documentRef.getElementById("zoom-in-button"),
      processingLabel: documentRef.getElementById("processing-label"),
      processingDetail: documentRef.getElementById("processing-detail"),
      bgProgress: documentRef.getElementById("bg-progress"),
      workspace: documentRef.querySelector(".workspace"),
      chatResizeHandle: documentRef.getElementById("chat-resize-handle"),
      chatBoard: documentRef.getElementById("chat-board"),
      chatBoardWires: documentRef.getElementById("chat-board-wires"),
      chatNodes: Array.from(documentRef.querySelectorAll("[data-chat-node]")),
      docHeader: documentRef.getElementById("doc-header"),
      docTitleBar: documentRef.getElementById("doc-title-bar"),
      dropZone: documentRef.getElementById("drop-zone"),
      recentDocsLanding: documentRef.getElementById("recent-docs-landing"),
      configSearchLimit: documentRef.getElementById("config-search-limit"),
      configSearchLimitValue: documentRef.getElementById("config-search-limit-value"),
      configMaxNewTokens: documentRef.getElementById("config-max-new-tokens"),
      configMaxNewTokensValue: documentRef.getElementById("config-max-new-tokens-value"),
      configContinuationTokens: documentRef.getElementById("config-continuation-tokens"),
      configContinuationTokensValue: documentRef.getElementById("config-continuation-tokens-value"),
      configMaxPasses: documentRef.getElementById("config-max-passes"),
      configProfileBroadLines: documentRef.getElementById("config-profile-broad-lines"),
      configProfileBroadLinesValue: documentRef.getElementById("config-profile-broad-lines-value"),
      configProfileFocusedLines: documentRef.getElementById("config-profile-focused-lines"),
      configProfileFocusedLinesValue: documentRef.getElementById("config-profile-focused-lines-value"),
      configProfileFallbackLines: documentRef.getElementById("config-profile-fallback-lines"),
      configProfileFallbackLinesValue: documentRef.getElementById("config-profile-fallback-lines-value"),
      configStallTimeout: documentRef.getElementById("config-stall-timeout"),
      configStallTimeoutValue: documentRef.getElementById("config-stall-timeout-value"),
      configProfileEnabled: documentRef.getElementById("config-profile-enabled"),
      configAutoNavigate: documentRef.getElementById("config-auto-navigate"),
      resetChatLayoutButton: documentRef.getElementById("reset-chat-layout-button")
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
    this.currentSnapshotId = null;
    this.ocrProbePromise = null;
    this.lastOcrProbe = null;
    this.chatNodePositions = { ...CHAT_NODE_DEFAULTS };
    this.chatDragState = null;
    this.chatResizeState = null;

    // Document-profile state subscription — updates the bg-progress pill
    this._profileUnsub = this.qwenRuntime.subscribeDocumentProfile((profileState) => {
      this._onDocumentProfileStateChange(profileState);
    });
  }

  /* ── lifecycle ──────────────────────────────────────────── */

  mount() {
    this.setAppState("landing");
    this.populateModelOptions();
    this.restoreChatLayout();
    this.applyChatLayout();
    this.updateChatBoardWires();
    this.unsubscribeQwenState = this.qwenRuntime.subscribe((state) => {
      this.updateQwenModelState(state);
    });
    this.unsubscribeQwenSettings = this.qwenRuntime.subscribeSettings((settings) => {
      this.updateChatSettingsUi(settings);
    });
    this.bindEvents();
    this.refreshSnapshotList().catch(console.error);
    window.addEventListener("resize", () => {
      this.viewer.render().catch(() => {});
      this.applyChatLayout();
      this.updateChatBoardWires();
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

  restoreChatLayout() {
    try {
      const raw = window.localStorage.getItem(CHAT_LAYOUT_STORAGE_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw);
      if (parsed?.sidebarWidth) {
        this.document.querySelector(".app-shell")?.style.setProperty("--chat-w", `${parsed.sidebarWidth}px`);
      }
      if (parsed?.positions && typeof parsed.positions === "object") {
        this.chatNodePositions = Object.fromEntries(
          Object.keys(CHAT_NODE_DEFAULTS).map((id) => [
            id,
            normalizeChatNodeLayout(id, parsed.positions[id])
          ])
        );
      }
    } catch {}
  }

  persistChatLayout() {
    const shell = this.document.querySelector(".app-shell");
    const widthValue = shell ? parseFloat(getComputedStyle(shell).getPropertyValue("--chat-w")) : null;
    try {
      window.localStorage.setItem(
        CHAT_LAYOUT_STORAGE_KEY,
        JSON.stringify({
          sidebarWidth: Number.isFinite(widthValue) ? widthValue : null,
          positions: this.chatNodePositions
        })
      );
    } catch {}
  }

  applyChatLayout() {
    const board = this.elements.chatBoard;
    if (!board || window.matchMedia("(max-width: 640px)").matches) {
      return;
    }

    const boardWidth = board.clientWidth || board.getBoundingClientRect().width;
    const boardHeight = board.clientHeight || board.getBoundingClientRect().height;
    for (const node of this.elements.chatNodes) {
      const id = node.dataset.chatNode;
      if (!id) continue;
      const layout = normalizeChatNodeLayout(id, this.chatNodePositions[id]);
      const minSize = CHAT_NODE_MIN_SIZES[id] ?? { w: 180, h: 120 };
      const width = layout.collapsed
        ? CHAT_NODE_COLLAPSED_SIZE.w
        : clamp(layout.w, minSize.w, Math.max(minSize.w, boardWidth - 16));
      const height = layout.collapsed
        ? CHAT_NODE_COLLAPSED_SIZE.h
        : clamp(layout.h, minSize.h, Math.max(minSize.h, boardHeight - 16));
      const x = clamp(layout.x, 8, Math.max(8, boardWidth - width - 8));
      const y = clamp(layout.y, 8, Math.max(8, boardHeight - height - 8));
      this.chatNodePositions[id] = {
        ...layout,
        x,
        y,
        w: layout.w,
        h: layout.h
      };
      node.style.left = `${x}px`;
      node.style.top = `${y}px`;
      node.style.width = `${width}px`;
      node.style.height = `${height}px`;
      node.classList.toggle("is-collapsed", layout.collapsed);
      const header = node.querySelector(".chat-node-header");
      header?.setAttribute("aria-expanded", layout.collapsed ? "false" : "true");
      const headerMeta = node.querySelector(".chat-node-header-meta");
      if (headerMeta) {
        headerMeta.textContent = layout.collapsed ? "click to expand" : "drag / click";
      }
    }
    this.persistChatLayout();
  }

  resetChatLayout() {
    const shell = this.document.querySelector(".app-shell");
    this.chatNodePositions = Object.fromEntries(
      Object.keys(CHAT_NODE_DEFAULTS).map((id) => [
        id,
        normalizeChatNodeLayout(id, CHAT_NODE_DEFAULTS[id])
      ])
    );
    shell?.style.setProperty("--chat-w", "560px");
    this.persistChatLayout();
    this.applyChatLayout();
    this.updateChatBoardWires();
  }

  getChatNodeLayout(id) {
    const layout = normalizeChatNodeLayout(id, this.chatNodePositions[id]);
    this.chatNodePositions[id] = layout;
    return layout;
  }

  setChatNodeLayout(id, patch = {}) {
    this.chatNodePositions[id] = {
      ...this.getChatNodeLayout(id),
      ...patch
    };
  }

  toggleChatNodeCollapsed(id) {
    const layout = this.getChatNodeLayout(id);
    this.chatNodePositions[id] = {
      ...layout,
      collapsed: !layout.collapsed
    };
    this.applyChatLayout();
    this.updateChatBoardWires();
  }

  finishChatNodeInteraction(pointerId = null, { cancelled = false } = {}) {
    const state = this.chatDragState;
    if (!state) {
      return;
    }
    if (pointerId !== null && state.pointerId !== pointerId) {
      return;
    }

    const target = state.target;
    if (target?.hasPointerCapture?.(state.pointerId)) {
      try {
        target.releasePointerCapture(state.pointerId);
      } catch {}
    }
    target?.classList?.remove("is-active");

    const shouldToggle =
      !cancelled &&
      state.kind === "move-node" &&
      !state.moved;

    this.chatDragState = null;

    if (shouldToggle) {
      this.toggleChatNodeCollapsed(state.id);
      return;
    }

    this.persistChatLayout();
    this.applyChatLayout();
    this.updateChatBoardWires();
  }

  finishSidebarResize(pointerId = null, { cancelled = false } = {}) {
    if (!this.chatResizeState) {
      return;
    }
    if (pointerId !== null && this.chatResizeState.pointerId !== pointerId) {
      return;
    }

    const handle = this.elements.chatResizeHandle;
    if (handle?.hasPointerCapture?.(this.chatResizeState.pointerId)) {
      try {
        handle.releasePointerCapture(this.chatResizeState.pointerId);
      } catch {}
    }

    this.chatResizeState = null;
    handle?.classList.remove("is-active");

    if (!cancelled) {
      this.persistChatLayout();
      this.applyChatLayout();
      this.updateChatBoardWires();
    }
  }

  updateChatBoardWires() {
    const board = this.elements.chatBoard;
    const svg = this.elements.chatBoardWires;
    if (!board || !svg || window.matchMedia("(max-width: 640px)").matches) {
      if (svg) svg.innerHTML = "";
      return;
    }

    const boardRect = board.getBoundingClientRect();
    const anchors = new Map();
    for (const node of this.elements.chatNodes) {
      const id = node.dataset.chatNode;
      if (!id) continue;
      const rect = node.getBoundingClientRect();
      anchors.set(id, {
        left: rect.left - boardRect.left,
        right: rect.right - boardRect.left,
        top: rect.top - boardRect.top,
        bottom: rect.bottom - boardRect.top,
        centerX: rect.left - boardRect.left + rect.width / 2,
        centerY: rect.top - boardRect.top + rect.height / 2
      });
    }

    const pathFor = (fromId, toId) => {
      const from = anchors.get(fromId);
      const to = anchors.get(toId);
      if (!from || !to) return "";
      const startX = from.centerX < to.centerX ? from.right : from.left;
      const startY = from.centerY;
      const endX = from.centerX < to.centerX ? to.left : to.right;
      const endY = to.centerY;
      const dx = endX - startX;
      const c1x = startX + dx * 0.35;
      const c2x = endX - dx * 0.35;
      return `M ${startX} ${startY} C ${c1x} ${startY}, ${c2x} ${endY}, ${endX} ${endY}`;
    };

    svg.setAttribute("viewBox", `0 0 ${Math.max(1, boardRect.width)} ${Math.max(1, boardRect.height)}`);
    svg.innerHTML = [
      pathFor("doc", "conversation"),
      pathFor("model", "conversation"),
      pathFor("config", "conversation")
    ]
      .filter(Boolean)
      .map((d) => `<path d="${d}"></path>`)
      .join("");
  }

  updateChatSettingsUi(settings) {
    if (this.elements.configSearchLimit) {
      this.elements.configSearchLimit.value = String(settings.searchLimit);
    }
    if (this.elements.configSearchLimitValue) {
      this.elements.configSearchLimitValue.value = String(settings.searchLimit);
      this.elements.configSearchLimitValue.textContent = String(settings.searchLimit);
    }
    if (this.elements.configMaxNewTokens) {
      this.elements.configMaxNewTokens.value = String(settings.maxNewTokens);
    }
    if (this.elements.configMaxNewTokensValue) {
      this.elements.configMaxNewTokensValue.value = String(settings.maxNewTokens);
      this.elements.configMaxNewTokensValue.textContent = String(settings.maxNewTokens);
    }
    if (this.elements.configContinuationTokens) {
      this.elements.configContinuationTokens.value = String(settings.continuationTokens);
    }
    if (this.elements.configContinuationTokensValue) {
      this.elements.configContinuationTokensValue.value = String(settings.continuationTokens);
      this.elements.configContinuationTokensValue.textContent = String(settings.continuationTokens);
    }
    if (this.elements.configMaxPasses) {
      this.elements.configMaxPasses.value = String(settings.maxPasses);
    }
    if (this.elements.configProfileBroadLines) {
      this.elements.configProfileBroadLines.value = String(settings.profileBroadLines);
    }
    if (this.elements.configProfileBroadLinesValue) {
      this.elements.configProfileBroadLinesValue.value = String(settings.profileBroadLines);
      this.elements.configProfileBroadLinesValue.textContent = String(settings.profileBroadLines);
    }
    if (this.elements.configProfileFocusedLines) {
      this.elements.configProfileFocusedLines.value = String(settings.profileFocusedLines);
    }
    if (this.elements.configProfileFocusedLinesValue) {
      this.elements.configProfileFocusedLinesValue.value = String(settings.profileFocusedLines);
      this.elements.configProfileFocusedLinesValue.textContent = String(settings.profileFocusedLines);
    }
    if (this.elements.configProfileFallbackLines) {
      this.elements.configProfileFallbackLines.value = String(settings.profileFallbackLines);
    }
    if (this.elements.configProfileFallbackLinesValue) {
      this.elements.configProfileFallbackLinesValue.value = String(settings.profileFallbackLines);
      this.elements.configProfileFallbackLinesValue.textContent = String(settings.profileFallbackLines);
    }
    if (this.elements.configStallTimeout) {
      this.elements.configStallTimeout.value = String(Math.round(settings.stallTimeoutMs / 1000));
    }
    if (this.elements.configStallTimeoutValue) {
      const seconds = Math.round(settings.stallTimeoutMs / 1000);
      this.elements.configStallTimeoutValue.value = String(seconds);
      this.elements.configStallTimeoutValue.textContent = `${seconds}s`;
    }
    if (this.elements.configProfileEnabled) {
      this.elements.configProfileEnabled.checked = Boolean(settings.profileEnabled);
    }
    if (this.elements.configAutoNavigate) {
      this.elements.configAutoNavigate.checked = Boolean(settings.autoNavigate);
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

    this.applyChatLayout();
    this.updateChatBoardWires();
  }

  populateModelOptions() {
    const select = this.elements.modelSelect;
    if (!select) return;

    select.innerHTML = "";
    const options = this.qwenRuntime.getModelOptions();
    for (const option of options) {
      const el = document.createElement("option");
      el.value = option.id;
      el.textContent = `${option.label} · ${option.description}`;
      select.append(el);
    }
    select.value = this.qwenRuntime.getModelState().modelId;
    select.disabled = options.length <= 1;
  }

  updateQwenModelState(state) {
    const select = this.elements.modelSelect;
    const button = this.elements.loadModelButton;
    const text = this.elements.modelStatusText;
    const progressPanel = this.elements.modelProgressPanel;
    const progressPhase = this.elements.modelProgressPhase;
    const progressPercent = this.elements.modelProgressPercent;
    const progressFill = this.elements.modelProgressFill;

    if (select && select.value !== state.modelId) {
      select.value = state.modelId;
    }

    if (button) {
      button.disabled = state.status === "loading" || state.ready;
      button.textContent =
        state.status === "loading"
          ? "Loading…"
          : state.status === "parked"
            ? "Analyzed"
          : state.ready
            ? "Loaded"
            : "Load Model";
    }

    if (text) {
      if (state.status === "loading" || state.status === "generating") {
        text.textContent = state.detail;
      } else if (state.status === "ready") {
        text.textContent = `${state.label} is live on WebGPU using ${state.dtype}.`;
      } else if (state.status === "parked") {
        text.textContent = state.detail;
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

    if (progressPanel && progressPhase && progressPercent && progressFill) {
      const progressValue = Math.max(
        0,
        Math.min(
          100,
          Number.isFinite(Number(state.progress)) ? Number(state.progress) : 0
        )
      );

      if (state.status === "loading") {
        progressPanel.hidden = false;
        progressPanel.dataset.state = "loading";
        progressPhase.textContent = state.detail || "Preparing model…";
        progressPercent.textContent = `${Math.round(progressValue)}%`;
        progressFill.style.width = `${progressValue}%`;
      } else if (state.status === "parked") {
        progressPanel.hidden = false;
        progressPanel.dataset.state = "parked";
        progressPhase.textContent = "Custom backend analyzed";
        progressPercent.textContent = "parked";
        progressFill.style.width = "100%";
      } else if (state.ready) {
        progressPanel.hidden = false;
        progressPanel.dataset.state = "ready";
        progressPhase.textContent = "Model ready";
        progressPercent.textContent = "100%";
        progressFill.style.width = "100%";
      } else if (state.status === "error") {
        progressPanel.hidden = false;
        progressPanel.dataset.state = "error";
        progressPhase.textContent = state.error || "Model load failed";
        progressPercent.textContent = "!";
        progressFill.style.width = "100%";
      } else {
        progressPanel.hidden = true;
        progressPanel.dataset.state = "idle";
        progressPhase.textContent = "";
        progressPercent.textContent = "0%";
        progressFill.style.width = "0%";
      }
    }

    if (this.elements.qwenBadge) {
      this.elements.qwenBadge.textContent = state.ready
        ? "WebGPU Ready"
        : state.status === "parked"
          ? "Custom Parked"
        : state.status === "loading"
          ? "Loading"
          : "Search Fallback";
      this.elements.qwenBadge.className =
        state.ready
          ? "pill pill-success"
          : state.status === "parked"
            ? "pill pill-warning"
            : "pill pill-warning";
    }
    if (this.elements.qwenCopy) {
      this.elements.qwenCopy.textContent = text?.textContent ?? state.detail;
    }

    this.applyChatLayout();
    this.updateChatBoardWires();
  }

  updateOcrProbeState(ocrInfo, gpuInfo = this.webgpuRuntime.getState?.() ?? null) {
    if (!ocrInfo) {
      return;
    }

    const gpuNote = gpuInfo?.available
      ? " WebGPU available."
      : gpuInfo?.reason
        ? ` ${gpuInfo.reason}`
        : "";

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
        ocrPreview +
        gpuNote;
    }
  }

  ensureOcrProbe(gpuInfo) {
    if (this.lastOcrProbe) {
      this.updateOcrProbeState(this.lastOcrProbe, gpuInfo);
      return Promise.resolve(this.lastOcrProbe);
    }

    if (!this.ocrProbePromise) {
      this.ocrProbePromise = this.ocrRuntime
        .probe()
        .then((ocrInfo) => {
          this.lastOcrProbe = ocrInfo;
          this.updateOcrProbeState(ocrInfo, gpuInfo);
          return ocrInfo;
        })
        .catch((error) => {
          console.error("OCR probe failed", error);
          if (this.elements.ocrBadge) {
            this.elements.ocrBadge.textContent = "Probe Failed";
            this.elements.ocrBadge.className = "pill pill-warning";
          }
          if (this.elements.ocrCopy) {
            this.elements.ocrCopy.textContent = "OCR assets could not be inspected right now.";
          }
          return null;
        })
        .finally(() => {
          this.ocrProbePromise = null;
        });
    }

    return this.ocrProbePromise;
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

    const updateChatSettings = () => {
      this.qwenRuntime.updateSettings({
        searchLimit: Number(this.elements.configSearchLimit?.value ?? 12),
        maxNewTokens: Number(this.elements.configMaxNewTokens?.value ?? 160),
        continuationTokens: Number(this.elements.configContinuationTokens?.value ?? 96),
        maxPasses: Number(this.elements.configMaxPasses?.value ?? 3),
        profileBroadLines: Number(this.elements.configProfileBroadLines?.value ?? 4),
        profileFocusedLines: Number(this.elements.configProfileFocusedLines?.value ?? 2),
        profileFallbackLines: Number(this.elements.configProfileFallbackLines?.value ?? 3),
        profileEnabled: Boolean(this.elements.configProfileEnabled?.checked),
        autoNavigate: Boolean(this.elements.configAutoNavigate?.checked),
        stallTimeoutMs: Number(this.elements.configStallTimeout?.value ?? 90) * 1000
      });
    };

    [
      this.elements.configSearchLimit,
      this.elements.configMaxNewTokens,
      this.elements.configContinuationTokens,
      this.elements.configMaxPasses,
      this.elements.configProfileBroadLines,
      this.elements.configProfileFocusedLines,
      this.elements.configProfileFallbackLines,
      this.elements.configStallTimeout,
      this.elements.configProfileEnabled,
      this.elements.configAutoNavigate
    ].forEach((control) => {
      control?.addEventListener("input", updateChatSettings);
      control?.addEventListener("change", updateChatSettings);
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

    this.elements.resetChatLayoutButton?.addEventListener("click", () => {
      this.resetChatLayout();
    });

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

    this.elements.chatNodes.forEach((node) => {
      const header = node.querySelector(".chat-node-header");
      const resizeHandle = node.querySelector(".chat-node-resize-handle");
      const nodeId = node.dataset.chatNode;
      if (!header || !nodeId) return;
      header.addEventListener("pointerdown", (event) => {
        if (window.matchMedia("(max-width: 640px)").matches) {
          return;
        }
        event.preventDefault();
        event.stopPropagation();
        header.setPointerCapture?.(event.pointerId);
        const layout = this.getChatNodeLayout(nodeId);
        this.chatDragState = {
          kind: "move-node",
          id: nodeId,
          pointerId: event.pointerId,
          target: header,
          startClientX: event.clientX,
          startClientY: event.clientY,
          startX: layout.x,
          startY: layout.y,
          moved: false
        };
      });

      header.addEventListener("lostpointercapture", (event) => {
        this.finishChatNodeInteraction(event.pointerId, { cancelled: true });
      });

      resizeHandle?.addEventListener("pointerdown", (event) => {
        if (window.matchMedia("(max-width: 640px)").matches) {
          return;
        }
        event.preventDefault();
        event.stopPropagation();
        resizeHandle.classList.add("is-active");
        resizeHandle.setPointerCapture?.(event.pointerId);
        const layout = this.getChatNodeLayout(nodeId);
        const currentWidth = node.offsetWidth || layout.w;
        const currentHeight = node.offsetHeight || layout.h;
        this.chatDragState = {
          kind: "resize-node",
          id: nodeId,
          pointerId: event.pointerId,
          target: resizeHandle,
          startClientX: event.clientX,
          startClientY: event.clientY,
          startX: layout.x,
          startY: layout.y,
          startW: currentWidth,
          startH: currentHeight,
          moved: false
        };
      });

      resizeHandle?.addEventListener("lostpointercapture", (event) => {
        this.finishChatNodeInteraction(event.pointerId, { cancelled: true });
      });
    });

    this.elements.chatResizeHandle?.addEventListener("pointerdown", (event) => {
      if (window.matchMedia("(max-width: 640px)").matches) {
        return;
      }
      event.preventDefault();
      this.elements.chatResizeHandle.classList.add("is-active");
      this.chatResizeState = { pointerId: event.pointerId };
      this.elements.chatResizeHandle.setPointerCapture?.(event.pointerId);
    });

    this.elements.chatResizeHandle?.addEventListener("lostpointercapture", (event) => {
      this.finishSidebarResize(event.pointerId, { cancelled: true });
    });

    this.elements.chatResizeHandle?.addEventListener("keydown", (event) => {
      const shellEl = this.document.querySelector(".app-shell");
      if (!shellEl) {
        return;
      }
      const currentWidth = parseFloat(getComputedStyle(shellEl).getPropertyValue("--chat-w")) || 540;
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") {
        return;
      }
      event.preventDefault();
      const boundedWidth = clamp(currentWidth + (event.key === "ArrowLeft" ? -24 : 24), 380, 760);
      shellEl.style.setProperty("--chat-w", `${boundedWidth}px`);
      this.persistChatLayout();
      this.applyChatLayout();
      this.updateChatBoardWires();
    });

    window.addEventListener("pointermove", (event) => {
      if (this.chatDragState) {
        const boardRect = this.elements.chatBoard?.getBoundingClientRect();
        const node = this.elements.chatNodes.find((item) => item.dataset.chatNode === this.chatDragState.id);
        if (!boardRect || !node || event.pointerId !== this.chatDragState.pointerId) return;
        const dx = event.clientX - this.chatDragState.startClientX;
        const dy = event.clientY - this.chatDragState.startClientY;
        if (Math.abs(dx) >= CHAT_DRAG_CLICK_THRESHOLD || Math.abs(dy) >= CHAT_DRAG_CLICK_THRESHOLD) {
          this.chatDragState.moved = true;
        }
        if (this.chatDragState.kind === "move-node") {
          const width = node.offsetWidth || this.getChatNodeLayout(this.chatDragState.id).w;
          const height = node.offsetHeight || this.getChatNodeLayout(this.chatDragState.id).h;
          const x = clamp(
            this.chatDragState.startX + dx,
            8,
            Math.max(8, boardRect.width - width - 8)
          );
          const y = clamp(
            this.chatDragState.startY + dy,
            8,
            Math.max(8, boardRect.height - height - 8)
          );
          this.setChatNodeLayout(this.chatDragState.id, { x, y });
          node.style.left = `${x}px`;
          node.style.top = `${y}px`;
        } else if (this.chatDragState.kind === "resize-node") {
          const minSize = CHAT_NODE_MIN_SIZES[this.chatDragState.id] ?? { w: 180, h: 120 };
          const width = clamp(
            this.chatDragState.startW + dx,
            minSize.w,
            Math.max(minSize.w, boardRect.width - this.chatDragState.startX - 8)
          );
          const height = clamp(
            this.chatDragState.startH + dy,
            minSize.h,
            Math.max(minSize.h, boardRect.height - this.chatDragState.startY - 8)
          );
          this.setChatNodeLayout(this.chatDragState.id, {
            w: width,
            h: height,
            collapsed: false
          });
          node.style.width = `${width}px`;
          node.style.height = `${height}px`;
        }
        this.updateChatBoardWires();
      }

      if (this.chatResizeState) {
        const workspaceRect = this.elements.workspace?.getBoundingClientRect();
        const shellEl = this.document.querySelector(".app-shell");
        if (!workspaceRect || !shellEl) return;
        const width = clamp(workspaceRect.right - event.clientX, 380, 760);
        shellEl.style.setProperty("--chat-w", `${width}px`);
      }
    });

    window.addEventListener("pointerup", (event) => {
      this.finishChatNodeInteraction(event.pointerId);
      this.finishSidebarResize(event.pointerId);
    });

    window.addEventListener("pointercancel", (event) => {
      this.finishChatNodeInteraction(event.pointerId, { cancelled: true });
      this.finishSidebarResize(event.pointerId, { cancelled: true });
    });

    window.addEventListener("blur", () => {
      this.finishChatNodeInteraction(null, { cancelled: true });
      this.finishSidebarResize(null, { cancelled: true });
    });
  }

  /* ── chat ───────────────────────────────────────────────── */

  renderChatHistory() {
    this.elements.chatLog.innerHTML = "";
    if (!this.messages.length) {
      if (this.bundle) {
        const title = resolveDocumentTitle(this.bundle).replace(/\.pdf$/i, "");
        this.elements.chatLog.append(
          createMessageElement("assistant", `"${title}" is ready. What would you like to know?`)
        );
      }
      this.scrollChatToBottom();
      return;
    }

    for (const message of this.messages) {
      this.elements.chatLog.append(
        createMessageElement(
          message.role,
          message.content,
          message.trace ?? [],
          message.references ?? []
        )
      );
    }
    this.scrollChatToBottom();
  }

  resetChat() {
    this.messages = [];
    this.renderChatHistory();
    if (this.bundle && this.index) {
      void this.persistCurrentSnapshot().catch(console.error);
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
      const chatSettings = this.qwenRuntime.getSettings();
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
          },
          onProgress: (detail) => {
            touch();
            const label = typing.querySelector(".typing-label");
            if (label && detail) {
              label.textContent = detail;
            }
          }
        }),
        {
          timeoutMs: chatSettings.stallTimeoutMs,
          label: "chat-stalled",
          onTimeout: () => this.qwenRuntime.interrupt?.()
        }
      );
      this.messages.push({
        role: "assistant",
        content: response.text,
        trace: response.trace ?? [],
        references: response.references ?? []
      });
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
      await this.persistCurrentSnapshot().catch(console.error);
    }
  }

  /* ── document loading ───────────────────────────────────── */

  async loadFile(file) {
    this.currentSnapshotId = null;
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
    let qwenInfo = null;
    let gpuInfo = null;

    try {
      qwenInfo = await this.qwenRuntime.probe();
      gpuInfo = qwenInfo?.gpuInfo ?? this.webgpuRuntime.getState?.() ?? null;
    } catch (error) {
      gpuInfo = this.webgpuRuntime.getState?.() ?? null;
      this.updateQwenModelState(this.qwenRuntime.getModelState());
      throw error;
    }

    const gpuNote = gpuInfo?.available
      ? " WebGPU available."
      : gpuInfo?.reason
        ? ` ${gpuInfo.reason}`
        : "";
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

    void this.ensureOcrProbe(gpuInfo);

    return { ocrInfo: this.lastOcrProbe, qwenInfo, gpuInfo };
  }

  /* ── snapshots ──────────────────────────────────────────── */

  async persistCurrentSnapshot() {
    if (!this.bundle || !this.index) return;
    this.currentSnapshotId ??= crypto.randomUUID();
    await this.sessionStore.saveSnapshot({
      id: this.currentSnapshotId,
      savedAt: Date.now(),
      title: resolveDocumentTitle(this.bundle),
      pdfBlob: this.bundle.file,
      metadata: this.bundle.metadata,
      pageCount: this.bundle.pageCount,
      pages: this.index.pages,
      chunks: serializeChunks(this.index.chunks),
      messages: serializeMessages(this.messages)
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
    this.currentSnapshotId = id;

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

    this.messages = hydrateMessages(snapshot.messages ?? []);
    this.renderChatHistory();
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
