function opfsSupported() {
  return (
    typeof navigator !== "undefined" &&
    "storage" in navigator &&
    "getDirectory" in navigator.storage
  );
}

function isBestEffortWriteError(error) {
  return (
    error instanceof DOMException &&
    (
      error.name === "UnknownError" ||
      error.name === "QuotaExceededError" ||
      error.name === "InvalidStateError" ||
      error.name === "NoModificationAllowedError" ||
      error.name === "NotAllowedError" ||
      error.name === "SecurityError"
    )
  );
}

function isBestEffortStorageError(error) {
  return isBestEffortWriteError(error) || (
    error instanceof DOMException &&
    (
      error.name === "AbortError" ||
      error.name === "InvalidAccessError"
    )
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
    this.writeDisabled = false;
    this.lastWriteError = null;
    this.readDisabled = false;
    this.lastStorageError = null;
  }

  #disableStorage(error) {
    this.writeDisabled = true;
    this.readDisabled = true;
    this.lastStorageError = error ?? null;
    this.rootPromise = Promise.resolve(null);
  }

  async getRoot() {
    if (!opfsSupported()) {
      return null;
    }
    if (this.readDisabled) {
      return null;
    }
    if (!this.rootPromise) {
      this.rootPromise = navigator.storage
        .getDirectory()
        .then((root) => root.getDirectoryHandle(this.namespace, { create: true }))
        .catch((error) => {
          if (isBestEffortStorageError(error)) {
            this.#disableStorage(error);
            return null;
          }
          throw error;
        });
    }
    return this.rootPromise;
  }

  async writeBytes(key, bytes) {
    const root = await this.getRoot();
    if (!root || this.writeDisabled) {
      return false;
    }
    try {
      const handle = await root.getFileHandle(encodeKey(key), { create: true });
      const writable = await handle.createWritable();
      await writable.write(bytes);
      await writable.close();
      return true;
    } catch (error) {
      if (isBestEffortStorageError(error)) {
        this.#disableStorage(error);
        this.lastWriteError = error;
        return false;
      }
      throw error;
    }
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
      if (isBestEffortStorageError(error)) {
        this.#disableStorage(error);
        return null;
      }
      throw error;
    }
  }

  async readBytesRange(key, start = 0, end = null) {
    const root = await this.getRoot();
    if (!root) {
      return null;
    }
    try {
      const handle = await root.getFileHandle(encodeKey(key));
      const file = await handle.getFile();
      const blob = file.slice(start, end ?? file.size);
      return new Uint8Array(await blob.arrayBuffer());
    } catch (error) {
      if (error instanceof DOMException && error.name === "NotFoundError") {
        return null;
      }
      if (isBestEffortStorageError(error)) {
        this.#disableStorage(error);
        return null;
      }
      throw error;
    }
  }

  async writeBlob(key, blob) {
    const bytes = new Uint8Array(await blob.arrayBuffer());
    return this.writeBytes(key, bytes);
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
    return this.writeBytes(key, bytes);
  }

  async readJson(key) {
    const bytes = await this.readBytes(key);
    if (!bytes) {
      return null;
    }
    return JSON.parse(new TextDecoder().decode(bytes));
  }

  async delete(key) {
    const root = await this.getRoot();
    if (!root) {
      return;
    }
    try {
      await root.removeEntry(encodeKey(key));
    } catch (error) {
      if (error instanceof DOMException && error.name === "NotFoundError") {
        return;
      }
      if (isBestEffortStorageError(error)) {
        this.#disableStorage(error);
        return;
      }
      throw error;
    }
  }
}
