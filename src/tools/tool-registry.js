function snippetForQuery(text, query) {
  const compact = text.replace(/\s+/g, " ").trim();
  if (!query) {
    return compact.slice(0, 180);
  }

  const lowerText = compact.toLowerCase();
  const lowerQuery = query.toLowerCase();
  const foundAt = lowerText.indexOf(lowerQuery);

  if (foundAt === -1) {
    return compact.slice(0, 180);
  }

  const start = Math.max(0, foundAt - 60);
  const end = Math.min(compact.length, foundAt + lowerQuery.length + 90);
  return `${start > 0 ? "..." : ""}${compact.slice(start, end)}${end < compact.length ? "..." : ""}`;
}

export class ToolRegistry {
  constructor({ viewer, vectorDatabase, index }) {
    this.viewer = viewer;
    this.vectorDatabase = vectorDatabase;
    this.index = index;
  }

  describeTools() {
    return [
      {
        name: "search",
        description: "Searches the indexed PDF chunks for relevant passages."
      },
      {
        name: "openPage",
        description: "Opens a page in the viewer and optionally focuses a bounding box."
      },
      {
        name: "openChunk",
        description: "Opens the exact indexed chunk region for a chunk id and highlights its bounding box."
      },
      {
        name: "peekPage",
        description: "Returns a short preview summary for a given page number."
      }
    ];
  }

  async search(query, { limit = 6 } = {}) {
    const vector = this.index.embedder.encodeQuery(query);
    const queryTerms = this.index.embedder.getQueryTerms(query);
    const results = await this.vectorDatabase.search(vector, {
      limit,
      queryTerms
    });

    return results.map((result) => ({
      ...result,
      snippet: snippetForQuery(result.text, query)
    }));
  }

  async openPage({ pageNumber, bbox }) {
    await this.viewer.openPage(pageNumber, bbox ? [bbox] : []);
    return { pageNumber, bbox };
  }

  async openChunk({ chunkId }) {
    const chunk = this.index.chunks.find((entry) => entry.id === chunkId);
    if (!chunk) {
      return null;
    }
    await this.viewer.openPage(chunk.pageNumber, chunk.bbox ? [chunk.bbox] : []);
    return {
      chunkId: chunk.id,
      pageNumber: chunk.pageNumber,
      bbox: chunk.bbox
    };
  }

  peekPage(pageNumber) {
    return this.index.pages.find((page) => page.pageNumber === pageNumber) ?? null;
  }
}
