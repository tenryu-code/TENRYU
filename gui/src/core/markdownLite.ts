/**
 * Dependency-free parser for the Markdown subset that assistant answers use:
 * headings, paragraphs, lists, fenced code, pipe tables, rules, inline code and
 * bold. Everything else (emphasis, links, `\( \)` math) is left as plain text,
 * and backslash is never an escape so LaTeX survives intact.
 */

export type MdInline =
  | { type: "text"; text: string }
  | { type: "code"; text: string }
  | { type: "strong"; children: MdInline[] };

export type MdBlock =
  | { type: "heading"; level: number; children: MdInline[] }
  | { type: "paragraph"; children: MdInline[] }
  | { type: "list"; ordered: boolean; items: MdInline[][] }
  | { type: "code"; lang: string; text: string }
  | { type: "table"; header: MdInline[][]; rows: MdInline[][][] }
  | { type: "rule" };

const FENCE = "```";
const HEADING_RE = /^(#{1,6})\s+(.+?)\s*#*\s*$/;
const RULE_RE = /^\s*(?:-{3,}|\*{3,}|_{3,})\s*$/;
const LIST_RE = /^(\s*)([-*+]|\d+[.)])\s+(.*)$/;
const ORDERED_MARKER_RE = /^\d+[.)]$/;
const TABLE_SEPARATOR_RE = /^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?\s*$/;

/** Inline code spans only — used for the children of a bold span. */
function parseCodeSpans(text: string): MdInline[] {
  const nodes: MdInline[] = [];
  let buffer = "";
  let index = 0;

  const flush = (): void => {
    if (buffer.length > 0) {
      nodes.push({ type: "text", text: buffer });
      buffer = "";
    }
  };

  while (index < text.length) {
    if (text[index] === "`") {
      const end = text.indexOf("`", index + 1);
      if (end > index && !text.slice(index + 1, end).includes("\n")) {
        flush();
        nodes.push({ type: "code", text: text.slice(index + 1, end) });
        index = end + 1;
        continue;
      }
    }
    buffer += text[index];
    index += 1;
  }
  flush();
  return nodes;
}

export function parseInlines(text: string): MdInline[] {
  const nodes: MdInline[] = [];
  let buffer = "";
  let index = 0;

  const flush = (): void => {
    if (buffer.length > 0) {
      nodes.push({ type: "text", text: buffer });
      buffer = "";
    }
  };

  while (index < text.length) {
    if (text[index] === "`") {
      const end = text.indexOf("`", index + 1);
      if (end > index && !text.slice(index + 1, end).includes("\n")) {
        flush();
        nodes.push({ type: "code", text: text.slice(index + 1, end) });
        index = end + 1;
        continue;
      }
    } else if (text.startsWith("**", index)) {
      const end = text.indexOf("**", index + 2);
      if (end > index + 1) {
        flush();
        nodes.push({ type: "strong", children: parseCodeSpans(text.slice(index + 2, end)) });
        index = end + 2;
        continue;
      }
    }
    buffer += text[index];
    index += 1;
  }
  flush();
  return nodes;
}

/** A list block still being collected: its marker kind plus the raw item texts. */
interface OpenList {
  ordered: boolean;
  items: string[];
}

/** Strip one leading and one trailing pipe, then split and trim the cells. */
function parseTableRow(line: string): MdInline[][] {
  let body = line.trim();
  if (body.startsWith("|")) body = body.slice(1);
  if (body.endsWith("|")) body = body.slice(0, -1);
  return body.split("|").map((cell) => parseInlines(cell.trim()));
}

export function parseMarkdownLite(text: string): MdBlock[] {
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  const blocks: MdBlock[] = [];

  let paragraph: string[] = [];
  let list: OpenList | null = null;

  const flushParagraph = (): void => {
    if (paragraph.length === 0) return;
    blocks.push({ type: "paragraph", children: parseInlines(paragraph.join("\n")) });
    paragraph = [];
  };
  const flushList = (): void => {
    if (list === null) return;
    blocks.push({
      type: "list",
      ordered: list.ordered,
      items: list.items.map((item) => parseInlines(item)),
    });
    list = null;
  };
  const flushAll = (): void => {
    flushParagraph();
    flushList();
  };

  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    const trimmed = line.trim();

    if (trimmed.startsWith(FENCE)) {
      flushAll();
      const lang = trimmed.slice(FENCE.length).trim();
      const body: string[] = [];
      index += 1;
      while (index < lines.length && !lines[index].trim().startsWith(FENCE)) {
        body.push(lines[index]);
        index += 1;
      }
      // Skip the closing fence; an unclosed fence simply ran to the end.
      if (index < lines.length) index += 1;
      blocks.push({ type: "code", lang, text: body.join("\n") });
      continue;
    }

    if (trimmed.length === 0) {
      flushAll();
      index += 1;
      continue;
    }

    const heading = HEADING_RE.exec(line);
    if (heading !== null) {
      flushAll();
      blocks.push({ type: "heading", level: heading[1].length, children: parseInlines(heading[2]) });
      index += 1;
      continue;
    }

    if (RULE_RE.test(line)) {
      flushAll();
      blocks.push({ type: "rule" });
      index += 1;
      continue;
    }

    if (
      trimmed.startsWith("|") &&
      index + 1 < lines.length &&
      TABLE_SEPARATOR_RE.test(lines[index + 1])
    ) {
      flushAll();
      const header = parseTableRow(line);
      const rows: MdInline[][][] = [];
      index += 2;
      while (index < lines.length && lines[index].trim().startsWith("|")) {
        const row = parseTableRow(lines[index]);
        while (row.length < header.length) row.push([]);
        rows.push(row.slice(0, header.length));
        index += 1;
      }
      blocks.push({ type: "table", header, rows });
      continue;
    }

    const item = LIST_RE.exec(line);
    if (item !== null) {
      flushParagraph();
      const open: OpenList = list ?? { ordered: ORDERED_MARKER_RE.test(item[2]), items: [] };
      open.items.push(item[3]);
      list = open;
      index += 1;
      continue;
    }

    const open = list;
    if (open !== null && /^\s/.test(line)) {
      open.items[open.items.length - 1] = `${open.items[open.items.length - 1]} ${trimmed}`;
      index += 1;
      continue;
    }

    flushList();
    paragraph.push(line);
    index += 1;
  }

  flushAll();
  return blocks;
}
