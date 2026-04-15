import { APP_BUILD_ID } from "./build-id.js";
import { AppController } from "./app/app-controller.js";

async function clearLocalhostShellCaches() {
  const isLocalhost =
    window.location.hostname === "localhost" ||
    window.location.hostname === "127.0.0.1" ||
    window.location.hostname === "::1";
  if (!isLocalhost || !("serviceWorker" in navigator)) {
    return false;
  }

  let changed = false;

  try {
    const registrations = await navigator.serviceWorker.getRegistrations();
    if (registrations.length) {
      const results = await Promise.all(
        registrations.map((registration) => registration.unregister())
      );
      changed ||= results.some(Boolean);
    }
  } catch (error) {
    console.error("Papertrail localhost service worker cleanup failed", error);
  }

  if ("caches" in window) {
    try {
      const keys = await caches.keys();
      const shellKeys = keys.filter((key) => key.startsWith("papertrail-shell-"));
      if (shellKeys.length) {
        const results = await Promise.all(shellKeys.map((key) => caches.delete(key)));
        changed ||= results.some(Boolean);
      }
    } catch (error) {
      console.error("Papertrail localhost cache cleanup failed", error);
    }
  }

  if (!changed) {
    return false;
  }

  const reloadMarkerKey = "papertrail-localhost-cache-reset";
  if (window.sessionStorage.getItem(reloadMarkerKey) === APP_BUILD_ID) {
    return false;
  }
  window.sessionStorage.setItem(reloadMarkerKey, APP_BUILD_ID);
  window.location.reload();
  return true;
}

async function registerOfflineShell() {
  if (!("serviceWorker" in navigator) || window.location.protocol === "file:") {
    return;
  }
  const isLocalhost =
    window.location.hostname === "localhost" ||
    window.location.hostname === "127.0.0.1" ||
    window.location.hostname === "::1";
  if (isLocalhost) {
    try {
      const registrations = await navigator.serviceWorker.getRegistrations();
      await Promise.all(registrations.map((registration) => registration.unregister()));
    } catch (error) {
      console.error("Papertrail localhost service worker cleanup failed", error);
    }
    return;
  }
  try {
    await navigator.serviceWorker.register("./service-worker.js", { scope: "./" });
  } catch (error) {
    console.error("Papertrail service worker registration failed", error);
  }
}

async function main() {
  if (window.location.protocol === "file:") {
    console.error(
      "Papertrail must be opened over http://localhost. Opening index.html directly via file:// is unsupported."
    );
    const pill = document.getElementById("status-pill");
    const stats = document.getElementById("document-stats");
    if (pill) {
      pill.textContent = "Use localhost";
      pill.className = "pill pill-warning";
    }
    if (stats) {
      stats.innerHTML =
        "<p>Open this app through the Bun local server, usually <code>http://localhost:4173</code>, not by double-clicking <code>index.html</code>.</p>";
    }
  }

  if (await clearLocalhostShellCaches()) {
    return;
  }

  const app = new AppController(document);
  app.mount();
  console.info(`Papertrail build ${APP_BUILD_ID}`);
  await registerOfflineShell();

  window.addEventListener("error", (event) => {
    console.error("Papertrail boot error", event.error ?? event.message);
  });
  window.addEventListener("unhandledrejection", (event) => {
    console.error("Papertrail unhandled rejection", event.reason);
  });
}

main().catch((error) => {
  console.error("Papertrail failed to boot", error);
  const pill = document.getElementById("status-pill");
  if (pill) {
    pill.textContent = "Boot failed";
    pill.className = "pill pill-warning";
  }
});
