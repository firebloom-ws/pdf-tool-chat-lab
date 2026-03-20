function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function sanitizeUrl(rawUrl) {
  try {
    const url = new URL(rawUrl, window.location.href);
    if (["http:", "https:", "mailto:"].includes(url.protocol)) {
      return url.href;
    }
  } catch {
    return null;
  }
  return null;
}

const GREEK_SYMBOLS = {
  alpha: "alpha",
  beta: "beta",
  gamma: "gamma",
  delta: "delta",
  eta: "eta",
  epsilon: "epsilon",
  zeta: "zeta",
  theta: "theta",
  iota: "iota",
  kappa: "kappa",
  lambda: "lambda",
  mu: "mu",
  nu: "nu",
  xi: "xi",
  pi: "pi",
  rho: "rho",
  sigma: "sigma",
  tau: "tau",
  phi: "phi",
  chi: "chi",
  psi: "psi",
  omega: "omega"
};

const OPERATOR_COMMANDS = {
  cdot: "&#x22C5;",
  cdots: "&#x22EF;",
  ldots: "&#x2026;",
  times: "&#x00D7;",
  pm: "&#x00B1;",
  leq: "&#x2264;",
  le: "&#x2264;",
  geq: "&#x2265;",
  ge: "&#x2265;",
  neq: "&#x2260;",
  ne: "&#x2260;",
  approx: "&#x2248;",
  to: "&#x2192;",
  rightarrow: "&#x2192;",
  leftarrow: "&#x2190;",
  mapsto: "&#x21A6;",
  lceil: "&#x2308;",
  rceil: "&#x2309;",
  lfloor: "&#x230A;",
  rfloor: "&#x230B;",
  langle: "&#x27E8;",
  rangle: "&#x27E9;",
  infty: "&#x221E;",
  sum: "&#x2211;",
  prod: "&#x220F;",
  int: "&#x222B;",
  partial: "&#x2202;",
  mid: "|",
  forall: "&#x2200;",
  exists: "&#x2203;",
  in: "&#x2208;",
  notin: "&#x2209;",
  subseteq: "&#x2286;",
  supseteq: "&#x2287;",
  cup: "&#x222A;",
  cap: "&#x2229;",
  land: "&#x2227;",
  lor: "&#x2228;"
};

const NORMAL_FUNCTIONS = new Set([
  "log",
  "ln",
  "sin",
  "cos",
  "tan",
  "max",
  "min",
  "exp",
  "BP"
]);

const STYLED_IDENTIFIER_COMMANDS = {
  mathrm: "normal",
  operatorname: "normal",
  mathit: "italic",
  mathbf: "bold",
  mathbb: "double-struck",
  mathcal: "script"
};

function wrapMathRow(nodes, force = false) {
  if (!nodes.length) {
    return "<mrow></mrow>";
  }
  if (nodes.length === 1 && !force) {
    return nodes[0];
  }
  return `<mrow>${nodes.join("")}</mrow>`;
}

function tokenizeMath(input) {
  const tokens = [];
  for (let i = 0; i < input.length;) {
    const char = input[i];
    if (/\s/.test(char)) {
      i += 1;
      continue;
    }
    if (char === "\\") {
      let j = i + 1;
      while (j < input.length && /[A-Za-z]/.test(input[j])) {
        j += 1;
      }
      if (j === i + 1 && j < input.length) {
        j += 1;
      }
      tokens.push({ type: "command", value: input.slice(i + 1, j) });
      i = j;
      continue;
    }
    if (/[0-9]/.test(char)) {
      let j = i + 1;
      while (j < input.length && /[0-9.]/.test(input[j])) {
        j += 1;
      }
      tokens.push({ type: "number", value: input.slice(i, j) });
      i = j;
      continue;
    }
    if (/[A-Za-z]/.test(char)) {
      let j = i + 1;
      while (j < input.length && /[A-Za-z]/.test(input[j])) {
        j += 1;
      }
      tokens.push({ type: "identifier", value: input.slice(i, j) });
      i = j;
      continue;
    }
    tokens.push({ type: "symbol", value: char });
    i += 1;
  }
  return tokens;
}

function readTextArgument(state) {
  if (!state.tokens[state.index] || state.tokens[state.index].value !== "{") {
    return "";
  }
  state.index += 1;
  const parts = [];
  let depth = 1;
  while (state.index < state.tokens.length && depth > 0) {
    const token = state.tokens[state.index];
    state.index += 1;
    if (token.value === "{") {
      depth += 1;
      parts.push(token.value);
      continue;
    }
    if (token.value === "}") {
      depth -= 1;
      if (depth > 0) {
        parts.push(token.value);
      }
      continue;
    }
    parts.push(token.value);
  }
  return parts.join(" ").replace(/\s+([.,;:!?])/g, "$1");
}

function parseMathArgument(state) {
  const token = state.tokens[state.index];
  if (!token) {
    return "<mrow></mrow>";
  }
  if (token.value === "{") {
    state.index += 1;
    const nodes = [];
    while (state.index < state.tokens.length && state.tokens[state.index].value !== "}") {
      nodes.push(parseMathNode(state));
    }
    if (state.tokens[state.index]?.value === "}") {
      state.index += 1;
    }
    return wrapMathRow(nodes, true);
  }
  return parseMathPrimary(state);
}

function parseMathPrimary(state) {
  const token = state.tokens[state.index];
  if (!token) {
    return "<mrow></mrow>";
  }

  state.index += 1;

  if (token.type === "number") {
    return `<mn>${escapeHtml(token.value)}</mn>`;
  }

  if (token.type === "identifier") {
    if (NORMAL_FUNCTIONS.has(token.value)) {
      return `<mi mathvariant="normal">${escapeHtml(token.value)}</mi>`;
    }
    return `<mi>${escapeHtml(token.value)}</mi>`;
  }

  if (token.type === "command") {
    if (token.value === "frac") {
      const numerator = parseMathArgument(state);
      const denominator = parseMathArgument(state);
      return `<mfrac>${numerator}${denominator}</mfrac>`;
    }
    if (token.value === "sqrt") {
      const radicand = parseMathArgument(state);
      return `<msqrt>${radicand}</msqrt>`;
    }
    if (token.value === "text") {
      return `<mtext>${escapeHtml(readTextArgument(state))}</mtext>`;
    }
    if (token.value === "left" || token.value === "right") {
      return parseMathPrimary(state);
    }
    if (token.value in STYLED_IDENTIFIER_COMMANDS) {
      const variant = STYLED_IDENTIFIER_COMMANDS[token.value];
      const content = readTextArgument(state).trim();
      if (!content) {
        return "<mrow></mrow>";
      }
      return `<mi mathvariant="${variant}">${escapeHtml(content)}</mi>`;
    }
    if (token.value in GREEK_SYMBOLS) {
      return `<mi>&${GREEK_SYMBOLS[token.value]};</mi>`;
    }
    if (token.value in OPERATOR_COMMANDS) {
      return `<mo>${OPERATOR_COMMANDS[token.value]}</mo>`;
    }
    if (NORMAL_FUNCTIONS.has(token.value)) {
      return `<mi mathvariant="normal">${escapeHtml(token.value)}</mi>`;
    }
    return `<mi>${escapeHtml(token.value)}</mi>`;
  }

  if (token.value === "{") {
    const nodes = [];
    while (state.index < state.tokens.length && state.tokens[state.index].value !== "}") {
      nodes.push(parseMathNode(state));
    }
    if (state.tokens[state.index]?.value === "}") {
      state.index += 1;
    }
    return wrapMathRow(nodes, true);
  }

  if (token.value === "'") {
    return "<mo>&#x2032;</mo>";
  }

  if ("()[]|,=+-*/<>".includes(token.value)) {
    return `<mo>${escapeHtml(token.value)}</mo>`;
  }

  return `<mo>${escapeHtml(token.value)}</mo>`;
}

function parseMathNode(state) {
  let base = parseMathPrimary(state);
  let subscript = null;
  let superscript = null;

  while (
    state.index < state.tokens.length &&
    (state.tokens[state.index].value === "_" || state.tokens[state.index].value === "^")
  ) {
    const kind = state.tokens[state.index].value;
    state.index += 1;
    const argument = parseMathArgument(state);
    if (kind === "_") {
      subscript = argument;
    } else {
      superscript = argument;
    }
  }

  if (subscript && superscript) {
    return `<msubsup>${base}${subscript}${superscript}</msubsup>`;
  }
  if (subscript) {
    return `<msub>${base}${subscript}</msub>`;
  }
  if (superscript) {
    return `<msup>${base}${superscript}</msup>`;
  }
  return base;
}

function texToMathMl(expression, display = false) {
  try {
    const state = {
      tokens: tokenizeMath(expression),
      index: 0
    };
    const nodes = [];
    while (state.index < state.tokens.length) {
      nodes.push(parseMathNode(state));
    }
    const body = wrapMathRow(nodes, true);
    return `<math xmlns="http://www.w3.org/1998/Math/MathML"${display ? ' display="block"' : ""}>${body}</math>`;
  } catch {
    return `<code>${escapeHtml(expression)}</code>`;
  }
}

function buildReferenceLookup(references = []) {
  return new Map(
    references
      .filter((reference) => reference?.id)
      .map((reference) => [String(reference.id).toLowerCase(), reference])
  );
}

function renderReferenceChip(reference) {
  const attrs = [
    'class="page-chip doc-ref-chip"',
    'type="button"',
    `data-ref-id="${escapeHtml(reference.id)}"`
  ];

  if (Number.isFinite(reference.pageNumber)) {
    attrs.push(`data-page-number="${reference.pageNumber}"`);
  }
  if (reference.bbox) {
    attrs.push(`data-bbox-x="${escapeHtml(String(reference.bbox.x))}"`);
    attrs.push(`data-bbox-y="${escapeHtml(String(reference.bbox.y))}"`);
    attrs.push(`data-bbox-width="${escapeHtml(String(reference.bbox.width))}"`);
    attrs.push(`data-bbox-height="${escapeHtml(String(reference.bbox.height))}"`);
  }

  const label = reference.label ?? reference.id;
  const title = reference.snippet ? ` title="${escapeHtml(reference.snippet)}"` : "";
  return `<button ${attrs.join(" ")}${title}>${escapeHtml(label)}</button>`;
}

function renderInlineMarkdown(text, options = {}) {
  const referenceLookup = buildReferenceLookup(options.references);
  const tokens = [];
  const stash = (html) => {
    const key = `\u0000${tokens.length}\u0000`;
    tokens.push(html);
    return key;
  };

  let html = String(text);

  html = html.replace(/\\\[([\s\S]+?)\\\]/g, (_, expr) =>
    stash(`<span class="math-block">${texToMathMl(expr, true)}</span>`)
  );

  html = html.replace(/\\\(([^)\n]+?)\\\)/g, (_, expr) =>
    stash(texToMathMl(expr, false))
  );

  html = html.replace(/\$\$([\s\S]+?)\$\$/g, (_, expr) =>
    stash(`<span class="math-block">${texToMathMl(expr, true)}</span>`)
  );

  html = html.replace(/(^|[^\\])\$([^$\n]+?)\$/g, (_, prefix, expr) =>
    `${prefix}${stash(texToMathMl(expr, false))}`
  );

  html = escapeHtml(html);

  html = html.replace(/\[ref:([A-Za-z0-9_-]+)\]/gi, (_, refId) => {
    const reference = referenceLookup.get(String(refId).toLowerCase());
    if (!reference) {
      return escapeHtml(`[ref:${refId}]`);
    }
    return stash(renderReferenceChip(reference));
  });

  html = html.replace(/\[(p\.(\d+))\]/gi, (_, label, pageNumber) =>
    stash(
      `<button class="page-chip" type="button" data-page-number="${pageNumber}">${escapeHtml(label)}</button>`
    )
  );

  html = html.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, label, rawUrl) => {
    const safeUrl = sanitizeUrl(rawUrl);
    if (!safeUrl) {
      return escapeHtml(label);
    }
    return stash(
      `<a href="${escapeHtml(safeUrl)}" target="_blank" rel="noreferrer">${renderInlineMarkdown(label, options)}</a>`
    );
  });

  html = html.replace(/`([^`]+)`/g, (_, code) => stash(`<code>${escapeHtml(code)}</code>`));
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  html = html.replace(/~~([^~]+)~~/g, "<del>$1</del>");

  return html.replace(/\u0000(\d+)\u0000/g, (_, index) => tokens[Number(index)] ?? "");
}

function isUnorderedListItem(line) {
  return /^\s*[-*+]\s+/.test(line);
}

function isOrderedListItem(line) {
  return /^\s*\d+\.\s+/.test(line);
}

function isBlockBoundary(line) {
  return (
    /^\s*$/.test(line) ||
    /^#{1,6}\s+/.test(line) ||
    /^```/.test(line) ||
    /^\s*>\s?/.test(line) ||
    isUnorderedListItem(line) ||
    isOrderedListItem(line) ||
    /^---+$/.test(line.trim())
  );
}

export function renderMarkdownToHtml(markdown, options = {}) {
  const source = String(markdown ?? "").replace(/\r\n?/g, "\n").trim();
  if (!source) {
    return "";
  }

  const lines = source.split("\n");
  const blocks = [];

  for (let i = 0; i < lines.length;) {
    const line = lines[i];

    if (/^\s*$/.test(line)) {
      i += 1;
      continue;
    }

    if (/^```/.test(line)) {
      const language = line.slice(3).trim();
      const code = [];
      i += 1;
      while (i < lines.length && !/^```/.test(lines[i])) {
        code.push(lines[i]);
        i += 1;
      }
      if (i < lines.length) {
        i += 1;
      }
      blocks.push(
        `<pre><code${language ? ` class="language-${escapeHtml(language)}"` : ""}>${escapeHtml(code.join("\n"))}</code></pre>`
      );
      continue;
    }

    if (/^#{1,6}\s+/.test(line)) {
      const match = line.match(/^(#{1,6})\s+(.*)$/);
      const level = Math.min(match?.[1]?.length ?? 1, 6);
      const content = renderInlineMarkdown(match?.[2] ?? "", options);
      blocks.push(`<h${level}>${content}</h${level}>`);
      i += 1;
      continue;
    }

    if (/^\s*>\s?/.test(line)) {
      const quoteLines = [];
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) {
        quoteLines.push(lines[i].replace(/^\s*>\s?/, ""));
        i += 1;
      }
      blocks.push(`<blockquote>${renderMarkdownToHtml(quoteLines.join("\n"), options)}</blockquote>`);
      continue;
    }

    if (isUnorderedListItem(line)) {
      const items = [];
      while (i < lines.length && isUnorderedListItem(lines[i])) {
        items.push(
          `<li>${renderInlineMarkdown(lines[i].replace(/^\s*[-*+]\s+/, ""), options)}</li>`
        );
        i += 1;
      }
      blocks.push(`<ul>${items.join("")}</ul>`);
      continue;
    }

    if (isOrderedListItem(line)) {
      const items = [];
      while (i < lines.length && isOrderedListItem(lines[i])) {
        items.push(
          `<li>${renderInlineMarkdown(lines[i].replace(/^\s*\d+\.\s+/, ""), options)}</li>`
        );
        i += 1;
      }
      blocks.push(`<ol>${items.join("")}</ol>`);
      continue;
    }

    if (/^---+$/.test(line.trim())) {
      blocks.push("<hr>");
      i += 1;
      continue;
    }

    const paragraph = [];
    while (i < lines.length && !isBlockBoundary(lines[i])) {
      paragraph.push(lines[i].trim());
      i += 1;
    }
    blocks.push(`<p>${renderInlineMarkdown(paragraph.join(" "), options)}</p>`);
  }

  return blocks.join("");
}
