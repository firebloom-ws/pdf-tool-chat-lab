import { createReadStream } from "node:fs";
import { access, stat } from "node:fs/promises";
import { createServer } from "node:http";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

const rootDirectory = fileURLToPath(new URL(".", import.meta.url));
const requestedPort = Bun.env.PORT ? Number(Bun.env.PORT) : null;

const MIME_TYPES = new Map([
  [".css", "text/css; charset=utf-8"],
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".mjs", "text/javascript; charset=utf-8"],
  [".pdf", "application/pdf"],
  [".svg", "image/svg+xml"],
  [".txt", "text/plain; charset=utf-8"],
  [".wasm", "application/wasm"]
]);

function resolvePathname(pathname: string) {
  const decoded = decodeURIComponent(pathname);
  const relativePath =
    decoded === "/" ? "index.html" : normalize(decoded).replace(/^\/+/, "");
  const filePath = join(rootDirectory, relativePath);
  if (!filePath.startsWith(rootDirectory)) {
    return null;
  }
  return filePath;
}

function sendNotFound(response: import("node:http").ServerResponse) {
  response.writeHead(404, {
    "Cache-Control": "no-store",
    "Content-Type": "text/plain; charset=utf-8"
  });
  response.end("Not found");
}

async function streamFile(
  filePath: string,
  response: import("node:http").ServerResponse
) {
  try {
    await access(filePath);
    const details = await stat(filePath);
    if (!details.isFile()) {
      sendNotFound(response);
      return;
    }
  } catch {
    sendNotFound(response);
    return;
  }

  response.writeHead(200, {
    "Cache-Control": "no-store",
    "Content-Length": String((await stat(filePath)).size),
    "Content-Type":
      MIME_TYPES.get(extname(filePath)) ?? "application/octet-stream"
  });
  createReadStream(filePath).pipe(response);
}

async function startServer() {
  const candidatePorts = requestedPort
    ? [requestedPort]
    : [4173, 4174, 4175, 3000, 8080, 0];
  let lastError: unknown = null;

  for (const port of candidatePorts) {
    const server = createServer(async (request, response) => {
      const url = new URL(request.url ?? "/", `http://${request.headers.host ?? "localhost"}`);
      const filePath = resolvePathname(url.pathname);
      if (!filePath) {
        sendNotFound(response);
        return;
      }
      await streamFile(filePath, response);
    });

    try {
      await new Promise<void>((resolve, reject) => {
        server.once("error", reject);
        server.listen(port, () => {
          server.removeListener("error", reject);
          resolve();
        });
      });
      const address = server.address();
      const resolvedPort =
        typeof address === "object" && address ? address.port : port;
      return { server, port: resolvedPort };
    } catch (error) {
      lastError = error;
      server.close();
      if (requestedPort) {
        throw error;
      }
    }
  }

  throw lastError ?? new Error("Unable to start the local server.");
}

const { port } = await startServer();
console.log(`Papertrail Lab local server: http://localhost:${port}`);
