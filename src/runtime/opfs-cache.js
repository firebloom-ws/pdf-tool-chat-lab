function opfsSupported() {
  return (
    typeof navigator !== "undefined" &&
    "storage" in navigator &&
    "getDirectory" in navigator.storage
  );
}

function encodeKey(value) {
  const bytes = new TextEncoder().encode(value);
  return `blob-${Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("")}`;
}

export class OpfsCache {
  constructor(namespace = "papertrail-lab") {
    this.namespace = namespace;
    this.rootPromise = null;
  }

  async getRoot() {
    if (!opfsSupported()) {
      return null;
    }
    if (!this.rootPromise) {
      this.rootPromise = navigator.storage
        .getDirectory()
        .then((root) => root.getDirectoryHandle(this.namespace, { create: true }));
    }
    return this.rootPromise;
  }

  async writeBytes(key, bytes) {
    const root = await this.getRoot();
    if (!root) {
      return;
    }
    const handle = await root.getFileHandle(encodeKey(key), { create: true });
    const writable = await handle.createWritable();
    await writable.write(bytes);
    await writable.close();
  }

  async readBytes(key) {
    const root = await this.getRoot();
    if (!root) {
      return null;
    }
    try {
      const handle = await root.getFileHandle(encodeKey(key));
      const file = await handle.getFile();
      return new Uint8Array(await file.arrayBuffer());
    } catch (error) {
      if (error instanceof DOMException && error.name === "NotFoundError") {
        return null;
      }
      throw error;
    }
  }

  async writeBlob(key, blob) {
    const bytes = new Uint8Array(await blob.arrayBuffer());
    await this.writeBytes(key, bytes);
  }

  async readBlob(key, type = "") {
    const bytes = await this.readBytes(key);
    if (!bytes) {
      return null;
    }
    return new Blob([bytes], { type });
  }

  async writeJson(key, value) {
    const bytes = new TextEncoder().encode(JSON.stringify(value));
    await this.writeBytes(key, bytes);
  }

  async readJson(key) {
    const bytes = await this.readBytes(key);
    if (!bytes) {
      return null;
    }
    return JSON.parse(new TextDecoder().decode(bytes));
  }
}
