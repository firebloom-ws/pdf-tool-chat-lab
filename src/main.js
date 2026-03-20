import { AppController } from "./app/app-controller.js";

try {
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

  const app = new AppController(document);
  app.mount();

  window.addEventListener("error", (event) => {
    console.error("Papertrail boot error", event.error ?? event.message);
  });
  window.addEventListener("unhandledrejection", (event) => {
    console.error("Papertrail unhandled rejection", event.reason);
  });
} catch (error) {
  console.error("Papertrail failed to boot", error);
  const pill = document.getElementById("status-pill");
  if (pill) {
    pill.textContent = "Boot failed";
    pill.className = "pill pill-warning";
  }
}
