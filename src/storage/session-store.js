const DATABASE_NAME = "papertrail-lab";
const STORE_NAME = "documents";
const DATABASE_VERSION = 1;

function normalizeIndexedDbError(error, fallbackMessage) {
  if (error instanceof Error) {
    return error;
  }
  if (error && typeof error === "object") {
    const wrapped = new Error(
      typeof error.message === "string" && error.message ? error.message : fallbackMessage
    );
    if (typeof error.name === "string" && error.name) {
      wrapped.name = error.name;
    }
    return wrapped;
  }
  return new Error(fallbackMessage);
}

function requestToPromise(request) {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(normalizeIndexedDbError(request.error, "indexeddb-request-failed"));
  });
}

function transactionDone(transaction) {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onerror = () =>
      reject(normalizeIndexedDbError(transaction.error, "indexeddb-transaction-failed"));
    transaction.onabort = () =>
      reject(normalizeIndexedDbError(transaction.error, "indexeddb-transaction-aborted"));
  });
}

function openDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(STORE_NAME)) {
        database.createObjectStore(STORE_NAME, { keyPath: "id" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(normalizeIndexedDbError(request.error, "indexeddb-open-failed"));
  });
}

export class SessionStore {
  async saveSnapshot(snapshot) {
    const database = await openDatabase();
    try {
      const transaction = database.transaction(STORE_NAME, "readwrite");
      const store = transaction.objectStore(STORE_NAME);
      let request;
      try {
        request = store.put(snapshot);
      } catch (error) {
        throw normalizeIndexedDbError(error, "indexeddb-put-failed");
      }
      await requestToPromise(request);
      await transactionDone(transaction);
    } finally {
      database.close();
    }
  }

  async listSnapshots() {
    const database = await openDatabase();
    try {
      const transaction = database.transaction(STORE_NAME, "readonly");
      const store = transaction.objectStore(STORE_NAME);
      const snapshots = await requestToPromise(store.getAll());
      return snapshots.sort((left, right) => right.savedAt - left.savedAt);
    } finally {
      database.close();
    }
  }

  async loadSnapshot(id) {
    const database = await openDatabase();
    try {
      const transaction = database.transaction(STORE_NAME, "readonly");
      const store = transaction.objectStore(STORE_NAME);
      const snapshot = await requestToPromise(store.get(id));
      return snapshot;
    } finally {
      database.close();
    }
  }
}
