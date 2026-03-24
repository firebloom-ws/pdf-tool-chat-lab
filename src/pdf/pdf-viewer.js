import { renderPdfPage } from "./pdf-service.js";

function clearChildren(node) {
  while (node.firstChild) {
    node.firstChild.remove();
  }
}

function renderFocusBoxes(overlay, boxes) {
  clearChildren(overlay);
  for (const bbox of boxes) {
    const marker = document.createElement("div");
    marker.className = "bbox-marker";
    marker.style.left = `${bbox.x * 100}%`;
    marker.style.top = `${bbox.y * 100}%`;
    marker.style.width = `${bbox.width * 100}%`;
    marker.style.height = `${bbox.height * 100}%`;
    overlay.append(marker);
  }
}

export class PdfViewer {
  constructor({ canvas, overlay, frame, pageRail, pageLabel, zoomLabel }) {
    this.canvas = canvas;
    this.overlay = overlay;
    this.frame = frame;
    this.pageRail = pageRail;
    this.pageLabel = pageLabel;
    this.zoomLabel = zoomLabel;
    this.bundle = null;
    this.pages = [];
    this.currentPage = 1;
    this.zoom = 1;
    this.currentBoxes = [];
    this.renderAbortController = null;
    this.renderRequestId = 0;
  }

  async attachDocument(bundle, pages = []) {
    this.bundle = bundle;
    this.pages = pages;
    this.currentPage = 1;
    this.zoom = 1;
    this.renderPageRail();
    await this.render();
  }

  async openPage(pageNumber, boxes = []) {
    if (!this.bundle) {
      return;
    }
    this.currentPage = Math.max(1, Math.min(this.bundle.pageCount, pageNumber));
    this.currentBoxes = boxes;
    await this.render();
  }

  async stepPage(delta) {
    if (!this.bundle) {
      return;
    }
    await this.openPage(this.currentPage + delta, []);
  }

  async setZoom(nextZoom) {
    this.zoom = Math.max(0.65, Math.min(2.25, nextZoom));
    await this.render();
  }

  renderPageRail() {
    clearChildren(this.pageRail);
    if (!this.bundle) {
      return;
    }

    for (let pageNumber = 1; pageNumber <= this.bundle.pageCount; pageNumber += 1) {
      const summary = this.pages.find((page) => page.pageNumber === pageNumber)?.summary ?? "Untitled page";
      const button = document.createElement("button");
      button.type = "button";
      button.className = "page-chip";
      button.dataset.pageNumber = String(pageNumber);
      button.innerHTML = `<strong>${pageNumber}</strong><span>${summary}</span>`;
      button.addEventListener("click", () => {
        this.openPage(pageNumber, []);
      });
      this.pageRail.append(button);
    }
  }

  async render() {
    if (!this.bundle) {
      return;
    }

    const requestId = ++this.renderRequestId;
    this.renderAbortController?.abort();
    const abortController = new AbortController();
    this.renderAbortController = abortController;

    const targetPage = this.currentPage;
    const targetZoom = this.zoom;
    const targetBoxes = [...this.currentBoxes];

    let rendering;
    try {
      rendering = await renderPdfPage({
        pdfDocument: this.bundle.pdfDocument,
        pageNumber: targetPage,
        canvas: this.canvas,
        zoom: targetZoom,
        containerWidth: Math.max(280, this.frame.clientWidth - 48),
        signal: abortController.signal
      });
    } catch (error) {
      if (
        abortController.signal.aborted ||
        error?.name === "RenderingCancelledException" ||
        error?.message === "pdf-render-cancelled"
      ) {
        return;
      }
      throw error;
    }

    if (requestId !== this.renderRequestId) {
      return;
    }

    if (this.renderAbortController === abortController) {
      this.renderAbortController = null;
    }

    renderFocusBoxes(this.overlay, targetBoxes);
    this.pageLabel.textContent = `Page ${targetPage} / ${this.bundle.pageCount}`;
    this.zoomLabel.textContent = `${Math.round(targetZoom * 100)}%`;

    for (const chip of this.pageRail.querySelectorAll(".page-chip")) {
      chip.classList.toggle(
        "is-active",
        Number(chip.dataset.pageNumber) === targetPage
      );
    }

    this.overlay.style.width = `${rendering.width}px`;
    this.overlay.style.height = `${rendering.height}px`;
  }
}
