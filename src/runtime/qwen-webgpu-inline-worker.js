import { createQwenBackendController } from "./qwen-webgpu-backend-controller.js";

function invokeListener(listener, event) {
  if (typeof listener === "function") {
    listener(event);
    return;
  }
  listener?.handleEvent?.(event);
}

export function createInProcessQwenWorker() {
  const messageListeners = new Set();
  const errorListeners = new Set();
  let terminated = false;

  const controller = createQwenBackendController({
    post(message) {
      if (terminated) {
        return;
      }
      queueMicrotask(() => {
        if (terminated) {
          return;
        }
        const event = { data: message };
        for (const listener of messageListeners) {
          try {
            invokeListener(listener, event);
          } catch (error) {
            const errorEvent = {
              message: error instanceof Error ? error.message : String(error),
              error
            };
            for (const errorListener of errorListeners) {
              invokeListener(errorListener, errorEvent);
            }
          }
        }
      });
    }
  });

  return {
    postMessage(message) {
      if (terminated) {
        return;
      }
      queueMicrotask(() => {
        if (terminated) {
          return;
        }
        controller.handleMessage(message).catch((error) => {
          const event = {
            message: error instanceof Error ? error.message : String(error),
            error
          };
          for (const listener of errorListeners) {
            invokeListener(listener, event);
          }
        });
      });
    },
    addEventListener(type, listener) {
      if (type === "message") {
        messageListeners.add(listener);
      } else if (type === "error") {
        errorListeners.add(listener);
      }
    },
    removeEventListener(type, listener) {
      if (type === "message") {
        messageListeners.delete(listener);
      } else if (type === "error") {
        errorListeners.delete(listener);
      }
    },
    terminate() {
      terminated = true;
      controller.terminate();
      messageListeners.clear();
      errorListeners.clear();
    }
  };
}
