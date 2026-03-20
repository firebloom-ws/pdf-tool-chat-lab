let pdfJsPromise = null;

async function loadPdfJs() {
  if (!pdfJsPromise) {
    pdfJsPromise = import("pdfjs-dist").then((module) => {
      module.GlobalWorkerOptions.workerSrc =
        "https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/pdf.worker.mjs";
      return module;
    });
  }
  return pdfJsPromise;
}

function mergeBbox(a, b) {
  if (!a) {
    return { ...b };
  }
  const x = Math.min(a.x, b.x);
  const y = Math.min(a.y, b.y);
  const width = Math.max(a.x + a.width, b.x + b.width) - x;
  const height = Math.max(a.y + a.height, b.y + b.height) - y;
  return { x, y, width, height };
}

function toNormalizedBbox(viewport, x, y, width, height) {
  return {
    x: Math.max(0, Math.min(1, x / viewport.width)),
    y: Math.max(0, Math.min(1, y / viewport.height)),
    width: Math.max(0, Math.min(1, width / viewport.width)),
    height: Math.max(0, Math.min(1, height / viewport.height))
  };
}

function normalizeString(value) {
  return value.replace(/\s+/g, " ").trim();
}

function textItemToLayoutItem(item, viewport, pdfJs) {
  const transform = pdfJs.Util.transform(viewport.transform, item.transform);
  const x = transform[4];
  const fontHeight = Math.max(Math.hypot(transform[2], transform[3]), item.height ?? 0);
  const y = transform[5] - fontHeight;
  const width = item.width ?? fontHeight;
  const height = fontHeight;

  return {
    text: normalizeString(item.str ?? ""),
    bbox: toNormalizedBbox(viewport, x, y, width, height)
  };
}

export async function loadPdfBundle(file) {
  const pdfJs = await loadPdfJs();
  const bytes = new Uint8Array(await file.arrayBuffer());
  const loadingTask = pdfJs.getDocument({
    data: bytes,
    useSystemFonts: true,
    isEvalSupported: false,
    enableXfa: true
  });

  const pdfDocument = await loadingTask.promise;
  const metadata = await pdfDocument.getMetadata().catch(() => null);

  return {
    file,
    bytes,
    pdfDocument,
    metadata,
    pageCount: pdfDocument.numPages
  };
}

export async function extractPageRecord(pdfDocument, pageNumber) {
  const pdfJs = await loadPdfJs();
  const page = await pdfDocument.getPage(pageNumber);
  const viewport = page.getViewport({ scale: 1 });
  const textContent = await page.getTextContent();
  const textItems = [];
  let aggregateBox = null;

  for (const rawItem of textContent.items) {
    const item = textItemToLayoutItem(rawItem, viewport, pdfJs);
    if (!item.text) {
      continue;
    }
    aggregateBox = mergeBbox(aggregateBox, item.bbox);
    textItems.push(item);
  }

  return {
    pageNumber,
    width: viewport.width,
    height: viewport.height,
    textItems,
    aggregateBox,
    rawText: textItems.map((item) => item.text).join(" ")
  };
}

export async function renderPdfPage({
  pdfDocument,
  pageNumber,
  canvas,
  zoom = 1,
  containerWidth
}) {
  await loadPdfJs();
  const page = await pdfDocument.getPage(pageNumber);
  const baseViewport = page.getViewport({ scale: 1 });
  const fitScale = containerWidth ? containerWidth / baseViewport.width : 1.2;
  const viewport = page.getViewport({ scale: fitScale * zoom });
  const outputScale = window.devicePixelRatio || 1;

  canvas.width = Math.floor(viewport.width * outputScale);
  canvas.height = Math.floor(viewport.height * outputScale);
  canvas.style.width = `${Math.floor(viewport.width)}px`;
  canvas.style.height = `${Math.floor(viewport.height)}px`;

  const context = canvas.getContext("2d", { alpha: false });
  const transform =
    outputScale === 1 ? null : [outputScale, 0, 0, outputScale, 0, 0];

  await page.render({
    canvasContext: context,
    viewport,
    transform
  }).promise;

  return {
    width: viewport.width,
    height: viewport.height,
    scale: fitScale * zoom
  };
}

export async function renderPageToImageData(
  pdfDocument,
  pageNumber,
  { maxSide = 1008 } = {}
) {
  await loadPdfJs();
  const page = await pdfDocument.getPage(pageNumber);
  const baseViewport = page.getViewport({ scale: 1 });
  const scale = maxSide / Math.max(baseViewport.width, baseViewport.height);
  const viewport = page.getViewport({ scale });

  const width = Math.max(1, Math.floor(viewport.width));
  const height = Math.max(1, Math.floor(viewport.height));
  const canvas =
    typeof OffscreenCanvas !== "undefined"
      ? new OffscreenCanvas(width, height)
      : (() => {
          const element = document.createElement("canvas");
          element.width = width;
          element.height = height;
          return element;
        })();

  const context = canvas.getContext("2d", { willReadFrequently: true });
  await page.render({
    canvasContext: context,
    viewport
  }).promise;

  const imageData = context.getImageData(0, 0, width, height);
  return {
    width,
    height,
    data: imageData.data
  };
}
