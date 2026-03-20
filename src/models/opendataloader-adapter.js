function absoluteBoundingBox(chunk, page) {
  const left = chunk.bbox.x * page.width;
  const right = (chunk.bbox.x + chunk.bbox.width) * page.width;
  const topFromTop = chunk.bbox.y * page.height;
  const bottomFromTop = (chunk.bbox.y + chunk.bbox.height) * page.height;

  return [
    Number(left.toFixed(1)),
    Number((page.height - bottomFromTop).toFixed(1)),
    Number(right.toFixed(1)),
    Number((page.height - topFromTop).toFixed(1))
  ];
}

function makeBaseElement(chunk, page, id) {
  return {
    type: chunk.elementType,
    id,
    "page number": chunk.pageNumber,
    "bounding box": absoluteBoundingBox(chunk, page)
  };
}

function toKidElement(chunk, page, id) {
  const base = makeBaseElement(chunk, page, id);
  const content = chunk.text.replace(/\s+/g, " ").trim();

  if (chunk.elementType === "heading") {
    return {
      ...base,
      "heading level": chunk.headingLevel ?? 2,
      font: "Unknown",
      "font size": 0,
      "text color": "[0,0,0]",
      content
    };
  }

  if (chunk.elementType === "caption") {
    return {
      ...base,
      font: "Unknown",
      "font size": 0,
      "text color": "[0,0,0]",
      content
    };
  }

  if (chunk.elementType === "list") {
    return {
      ...base,
      content,
      kids: content
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean)
        .map((line, index) => ({
          type: "paragraph",
          id: Number(`${id}${index + 1}`),
          "page number": chunk.pageNumber,
          "bounding box": absoluteBoundingBox(chunk, page),
          font: "Unknown",
          "font size": 0,
          "text color": "[0,0,0]",
          content: line
        }))
    };
  }

  if (chunk.elementType === "image") {
    return {
      ...base,
      alt: content || "Image region"
    };
  }

  return {
    ...base,
    font: "Unknown",
    "font size": 0,
    "text color": "[0,0,0]",
    content
  };
}

export function buildSectionChunks(index) {
  const sections = [];
  let current = null;

  for (const chunk of index.chunks) {
    if (chunk.elementType === "heading") {
      if (current) {
        sections.push(current);
      }
      current = {
        heading: chunk.text.replace(/\s+/g, " ").trim(),
        pageNumber: chunk.pageNumber,
        text: chunk.text.trim(),
        chunkIds: [chunk.id]
      };
      continue;
    }

    if (!current) {
      current = {
        heading: null,
        pageNumber: chunk.pageNumber,
        text: chunk.text.trim(),
        chunkIds: [chunk.id]
      };
      continue;
    }

    current.text = `${current.text}\n${chunk.text.trim()}`.trim();
    current.chunkIds.push(chunk.id);
  }

  if (current) {
    sections.push(current);
  }

  return sections;
}

export function toOpenDataLoaderLikeDocument(bundle, index) {
  const pageMap = new Map(index.pages.map((page) => [page.pageNumber, page]));
  const kids = index.chunks.map((chunk, indexNumber) =>
    toKidElement(chunk, pageMap.get(chunk.pageNumber), indexNumber + 1)
  );

  return {
    "file name": bundle.file.name,
    "number of pages": bundle.pageCount,
    author: bundle.metadata?.info?.Author ?? null,
    title: bundle.metadata?.info?.Title ?? null,
    "creation date": bundle.metadata?.info?.CreationDate ?? null,
    "modification date": bundle.metadata?.info?.ModDate ?? null,
    kids
  };
}
