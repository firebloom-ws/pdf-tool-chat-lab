# Papertrail Lab

`Papertrail Lab` is a standalone GitHub Pages friendly PDF viewer and grounded chat shell. It lives entirely inside [`/Users/typealias/Downloads/project/project`](/Users/typealias/Downloads/project/project), parses PDFs with `pdf.js`, indexes layout chunks locally, stores snapshots in IndexedDB, and exposes a tool-driven chat loop with `search`, `openPage`, and `peekPage`.

## What works now

- Upload a PDF and render it in-browser with `pdf.js`
- Extract the PDF text layer with normalized bounding boxes
- Build a custom hashed embedding for each grounded chunk
- Search the document using a custom JS/WASM vector database
- Open matched pages with bounding box overlays
- Cache uploaded PDFs and their layout index locally in IndexedDB
- Export a `.layout.json` file with saved chunk text and bounding boxes
- Download model metadata and weight headers from the official Hugging Face Hub API
- Inspect safetensors shard layouts in-browser before pulling heavy weights
- Cache small Hub assets locally in OPFS
- Load a browser-side BPE tokenizer from Hub tokenizer assets
- Run a custom WebGPU smoke-test path for matmul and RMSNorm
- Prepare rendered PDF pages into Pixtral-style vision patch inputs for the LightOn OCR path
- Run a batch-1 Qwen 3.5 text graph with exact hybrid linear/full-attention layer logic in local JS

## Important reality check

The UI, retrieval layer, and tool loop are working today. The full browser-side execution of these two exact models is **not** complete yet:

- `lightonai/LightOnOCR-2-1B-bbox-soup`
- `Qwen/Qwen3.5-0.8B`

Both models need a much larger custom runtime than a normal static site scaffold:

- LightOnOCR is a multimodal OCR/document model and needs custom image preprocessing, safetensors loading, attention kernels, KV cache management, and decoder logic.
- Qwen 3.5 likewise needs a custom tokenizer + transformer runtime + sampling stack in WebGPU if you want truly local inference with no third-party runtime.

This project keeps those boundaries explicit instead of pretending they are solved. The current app now uses:

- `pdf.js` for PDF parsing and rendering
- `@huggingface/hub` for official file listing and downloads
- `@huggingface/jinja` for chat-template rendering when the model ships one
- OPFS for small browser-side model asset caching
- custom JS safetensors inspection and dtype decoding
- custom JS BytePair tokenizer loading for tokenizer assets
- custom WebGPU kernels for matmul and RMSNorm smoke tests
- a Pixtral-style page-image preprocessor ahead of the LightOn OCR path
- a hybrid Qwen path: deterministic document tools plus a local Qwen 3.5 text graph for grounded answer synthesis

The document-internalization feature in the shipped app is a bounded
document profile, not true Doc-to-LoRA weight patching. The live WebGPU
worker runtime cannot inject generated LoRA weights today, so the profile
is compiled into a compact prompt prefix instead.

## Structure

- [`index.html`](/Users/typealias/Downloads/project/project/index.html): static app shell and import map
- [`src/app/app-controller.js`](/Users/typealias/Downloads/project/project/src/app/app-controller.js): orchestration, persistence, UI wiring
- [`src/pdf/pdf-service.js`](/Users/typealias/Downloads/project/project/src/pdf/pdf-service.js): PDF loading, extraction, and rendering
- [`src/models/document-indexer.js`](/Users/typealias/Downloads/project/project/src/models/document-indexer.js): chunking and layout grounding
- [`src/models/hash-embedder.js`](/Users/typealias/Downloads/project/project/src/models/hash-embedder.js): custom document/query embedding
- [`src/vdb/vector-db.js`](/Users/typealias/Downloads/project/project/src/vdb/vector-db.js): JS/WASM retrieval with binary Hamming prefilter and dense JS rerank
- [`src/vdb/vector-kernel.c`](/Users/typealias/Downloads/project/project/src/vdb/vector-kernel.c): future dense-kernel starting point for a compiler-backed WASM path
- [`src/runtime/hf-hub-client.js`](/Users/typealias/Downloads/project/project/src/runtime/hf-hub-client.js): official HF Hub listing/download/render helpers
- [`src/runtime/opfs-cache.js`](/Users/typealias/Downloads/project/project/src/runtime/opfs-cache.js): browser-side OPFS asset cache
- [`src/runtime/safetensors.js`](/Users/typealias/Downloads/project/project/src/runtime/safetensors.js): safetensors shard/header inspection and preview reads
- [`src/runtime/tokenizer-bpe.js`](/Users/typealias/Downloads/project/project/src/runtime/tokenizer-bpe.js): browser-side ByteLevel/BPE tokenizer loader
- [`src/runtime/model-runtimes.js`](/Users/typealias/Downloads/project/project/src/runtime/model-runtimes.js): LightOnOCR/Qwen runtime boundaries and fallback behavior
- [`src/runtime/webgpu-runtime.js`](/Users/typealias/Downloads/project/project/src/runtime/webgpu-runtime.js): WebGPU device, buffers, dispatch, and smoke tests
- [`src/runtime/webgpu-kernels.js`](/Users/typealias/Downloads/project/project/src/runtime/webgpu-kernels.js): hand-authored WGSL kernels

## Local notes

- The app is written as native browser modules so it can be hosted on GitHub Pages.
- Do not open [`index.html`](/Users/typealias/Downloads/project/project/index.html) directly with `file://`. Run it through `http://localhost` so OPFS, module loading, and worker fetches behave correctly.
- `pdf.js`, `@huggingface/hub`, and `@huggingface/jinja` are loaded via browser ESM/CDN imports.
- The current app already uses an embedded WASM Hamming-distance kernel for approximate search.
- Model manifests and small tokenizer/config files are pulled from the official Hugging Face Hub API in the browser.
- A future compiler-backed dense kernel can be rebuilt with:

```bash
npm run build:wasm
```

That rebuild step depends on having a local toolchain with `wasm32` support.

Run the app locally with Bun:

```bash
cd /Users/typealias/Downloads/project/project
bun install
bun run dev
```

Then open the URL printed by the server. It will usually be `http://localhost:4173`, but it can move to another free local port if that one is already taken.

If you want to force a specific port:

```bash
cd /Users/typealias/Downloads/project/project
PORT=3000 bun run dev
```

If that port is busy, the command will fail and you can pick another one:

```bash
cd /Users/typealias/Downloads/project/project
PORT=4174 bun run dev
```

## GitHub Pages

This repo now includes a GitHub Pages workflow at
[`/.github/workflows/pages.yml`](/Users/typealias/Downloads/project/project/.github/workflows/pages.yml).
Once the repository is pushed to GitHub and Pages is enabled for GitHub Actions,
every push to `main` will deploy the static site.

The app is already structured to work from a repository subpath:

- `index.html` is at the repo root
- browser imports are relative or CDN-based
- no extra bundling step is required for Pages hosting

After the first push:

1. Open the repository on GitHub.
2. Go to `Settings` -> `Pages`.
3. Set `Source` to `GitHub Actions`.
4. Push to `main` or re-run the `Deploy GitHub Pages` workflow.

## Next steps for the full model goal

1. Expand the WebGPU kernel set from smoke-test ops to transformer-grade kernels: RoPE, attention, KV cache update, SwiGLU, logits, and sampling.
2. Add paged weight loading from safetensors shards into GPU buffers instead of preview slices only.
3. Finish the exact Qwen 3.5 text stack, especially unsupported architecture pieces like `linear_attention` if present in the target config.
4. Finish the LightOnOCR image encoder/decoder path and emit real OCR bbox blocks instead of the current vision-prepared placeholder path.
5. Replace the deterministic tool routing with full Qwen tool-calling once the exact generation graph is reliable enough for interactive use.

## Architecture references

- [`mlx-lm/mlx_lm/models/qwen3_5.py`](/Users/typealias/Downloads/project/mlx-lm/mlx_lm/models/qwen3_5.py) is the best text-side reference for the exact Qwen 3.5 family: alternating full-attention and `GatedDeltaNet` layers, custom cache shapes, grouped KV attention, and the weight-sanitizing rules needed for converted checkpoints.
- [`mlx-lm/mlx_lm/models/qwen3.py`](/Users/typealias/Downloads/project/mlx-lm/mlx_lm/models/qwen3.py) is still a useful simpler baseline for standard Qwen blocks: RMSNorm, RoPE, grouped KV attention, SwiGLU MLP, and KV-cache decoding.
- [`mlx-vlm/mlx_vlm/models/qwen3_5/language.py`](/Users/typealias/Downloads/project/mlx-vlm/mlx_vlm/models/qwen3_5/language.py) shows the heavier multimodal Qwen 3.5 stack: interleaved multimodal RoPE, gated `q_proj`, periodic full attention, and `Qwen3_5GatedDeltaNet` linear-attention layers.
- [`mlx-vlm/mlx_vlm/models/deepseekocr/deepseekocr.py`](/Users/typealias/Downloads/project/mlx-vlm/mlx_vlm/models/deepseekocr/deepseekocr.py) is a useful OCR-VLM integration reference for LightOn-style work, especially the vision encoder plus projector plus image-token merge path.
- I did not find a direct `LightOnOCR` implementation in `mlx-vlm`, so that repo is best used as an OCR/VLM design reference rather than a drop-in LightOn execution map.
