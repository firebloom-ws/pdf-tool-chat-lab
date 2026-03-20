import { EMBEDDING_DIMENSIONS, HashLayoutEmbedder } from "./hash-embedder.js";
import { extractPageRecord, renderPageToImageData } from "../pdf/pdf-service.js";

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

function groupItemsIntoLines(textItems) {
  const sorted = [...textItems].sort((left, right) => {
    const yDelta = left.bbox.y - right.bbox.y;
    if (Math.abs(yDelta) > 0.012) {
      return yDelta;
    }
    return left.bbox.x - right.bbox.x;
  });

  const lines = [];
  for (const item of sorted) {
    const line = lines.at(-1);
    if (!line || Math.abs(item.bbox.y - line.anchorY) > 0.018) {
      lines.push({
        anchorY: item.bbox.y,
        items: [item],
        bbox: { ...item.bbox }
      });
      continue;
    }

    line.items.push(item);
    line.bbox = mergeBbox(line.bbox, item.bbox);
  }

  return lines.map((line) => ({
    text: line.items.map((item) => item.text).join(" ").replace(/\s+/g, " ").trim(),
    bbox: line.bbox
  }));
}

function linesToChunks(lines, pageNumber) {
  if (!lines.length) {
    return [
      {
        pageNumber,
        text: "[Image page — no text layer detected]",
        bbox: { x: 0.08, y: 0.08, width: 0.84, height: 0.84 },
        blockCount: 1,
        lineCount: 0,
        lines: [],
        isFallback: true
      }
    ];
  }

  const chunks = [];
  let current = {
    pageNumber,
    text: "",
    bbox: null,
    lines: [],
    blockCount: 0
  };

  for (const line of lines) {
    const nextText = current.text ? `${current.text}\n${line.text}` : line.text;
    const gapTooLarge =
      current.lines.length > 0 &&
      Math.abs(line.bbox.y - current.lines.at(-1).bbox.y) > 0.08;
    const chunkTooLong = nextText.length > 620;

    if ((gapTooLarge || chunkTooLong) && current.lines.length > 0) {
      chunks.push({
        pageNumber,
        text: current.text,
        bbox: current.bbox,
        blockCount: current.blockCount,
        lineCount: current.lines.length,
        lines: [...current.lines],
        isFallback: false
      });
      current = {
        pageNumber,
        text: line.text,
        bbox: { ...line.bbox },
        lines: [line],
        blockCount: 1
      };
      continue;
    }

    current.text = nextText;
    current.bbox = current.bbox ? mergeBbox(current.bbox, line.bbox) : { ...line.bbox };
    current.lines.push(line);
    current.blockCount += 1;
  }

  if (current.lines.length > 0 || current.text) {
    chunks.push({
      pageNumber,
      text: current.text,
      bbox: current.bbox ?? { x: 0.08, y: 0.08, width: 0.84, height: 0.84 },
      blockCount: current.blockCount,
      lineCount: current.lines.length,
      lines: [...current.lines],
      isFallback: false
    });
  }

  return chunks;
}

function buildSnippet(text) {
  return text.replace(/\s+/g, " ").trim().slice(0, 180);
}

function isListLike(lines) {
  if (lines.length < 2) {
    return false;
  }
  const matches = lines.filter((line) =>
    /^(\u2022|-|\*|\d+[\.\)])\s+/.test(line.text)
  ).length;
  return matches >= Math.max(2, Math.ceil(lines.length * 0.6));
}

function isHeadingLike(text, bbox, lineCount) {
  const compact = text.replace(/\s+/g, " ").trim();
  if (!compact || compact.length > 120 || lineCount > 3) {
    return false;
  }
  if (/[.!?:;]$/.test(compact)) {
    return false;
  }
  const words = compact.split(/\s+/);
  if (words.length > 14) {
    return false;
  }

  const upperCount = (compact.match(/[A-Z]/g) ?? []).length;
  const alphaCount = (compact.match(/[A-Za-z]/g) ?? []).length || 1;
  const upperRatio = upperCount / alphaCount;
  const topWeighted = bbox.y < 0.28;
  const titleCase =
    words.length > 1 &&
    words.filter((word) => /^[A-Z][a-z]/.test(word)).length >= Math.ceil(words.length * 0.6);

  return upperRatio > 0.24 || (titleCase && topWeighted);
}

function inferHeadingLevel(bbox) {
  if (bbox.y < 0.12) {
    return 1;
  }
  if (bbox.y < 0.25) {
    return 2;
  }
  return 3;
}

function inferElementType(chunk) {
  const compact = chunk.text.replace(/\s+/g, " ").trim();
  if (chunk.isFallback) {
    return { type: "image", headingLevel: null };
  }
  if (/^(figure|fig\.?)\s+\d+[:.\-]?\s+/i.test(compact)) {
    return { type: "caption", headingLevel: null };
  }
  if (/^table\s+\d+[:.\-]?\s+/i.test(compact)) {
    return { type: "caption", headingLevel: null };
  }
  if (isListLike(chunk.lines ?? [])) {
    return { type: "list", headingLevel: null };
  }
  if (isHeadingLike(chunk.text, chunk.bbox, chunk.lineCount)) {
    return { type: "heading", headingLevel: inferHeadingLevel(chunk.bbox) };
  }
  return { type: "paragraph", headingLevel: null };
}

export async function buildDocumentIndex(pdfBundle, { onProgress, ocrRuntime } = {}) {
  const embedder = new HashLayoutEmbedder();
  const chunks = [];
  const pages = [];

  for (let pageNumber = 1; pageNumber <= pdfBundle.pageCount; pageNumber += 1) {
    onProgress?.(
      `Indexing page ${pageNumber} of ${pdfBundle.pageCount}`,
      pageNumber / pdfBundle.pageCount
    );

    const pageRecord = await extractPageRecord(pdfBundle.pdfDocument, pageNumber);
    const analysis = await ocrRuntime.analyzePage(pageRecord, {
      pageCount: pdfBundle.pageCount,
      renderPageImage: () =>
        renderPageToImageData(pdfBundle.pdfDocument, pageNumber, {
          maxSide: 1008
        })
    });
    const lines = analysis.blocks ? groupItemsIntoLines(analysis.blocks) : [];
    const pageChunks = linesToChunks(lines, pageNumber);
    const summary =
      buildSnippet(pageChunks[0]?.text ?? pageRecord.rawText ?? "") || "Image-heavy page";

    pages.push({
      pageNumber,
      width: pageRecord.width,
      height: pageRecord.height,
      summary,
      layoutSource: analysis.source,
      blockCount: analysis.blocks.length
    });

    for (const [chunkIndex, chunk] of pageChunks.entries()) {
      const id = `p${pageNumber}-c${chunkIndex + 1}`;
      const semantic = inferElementType(chunk);
      const vector = embedder.encodeText(chunk.text, {
        bbox: chunk.bbox,
        blockCount: chunk.blockCount,
        lineCount: chunk.lineCount,
        pageNumber,
        pageCount: pdfBundle.pageCount,
        isFallback: chunk.isFallback
      });

      chunks.push({
        id,
        pageNumber,
        text: chunk.text,
        snippet: buildSnippet(chunk.text),
        bbox: chunk.bbox,
        blockCount: chunk.blockCount,
        lineCount: chunk.lineCount,
        lines: chunk.lines,
        elementType: semantic.type,
        headingLevel: semantic.headingLevel,
        layoutSource: analysis.source,
        vector,
        signature: embedder.packSignature(vector)
      });
    }
  }

  return {
    dimensions: EMBEDDING_DIMENSIONS,
    embedder,
    pages,
    chunks
  };
}
