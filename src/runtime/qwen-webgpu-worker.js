import { createQwenBackendController } from "./qwen-webgpu-backend-controller.js";

const controller = createQwenBackendController({
  post(message) {
    self.postMessage(message);
  }
});

self.addEventListener("message", (event) => {
  controller.handleMessage(event.data ?? {});
});
