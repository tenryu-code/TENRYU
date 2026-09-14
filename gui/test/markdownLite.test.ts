import { describe, expect, it } from "vitest";
import {
  parseInlines,
  parseMarkdownLite,
  type MdBlock,
  type MdInline,
} from "../src/core/markdownLite";

function textOf(nodes: MdInline[]): string {
  return nodes
    .map((node) =>
      node.type === "text" ? node.text : node.type === "code" ? node.text : textOf(node.children),
    )
    .join("");
}

describe("parseMarkdownLite headings", () => {
  it("reads levels 1-3 and strips trailing hashes", () => {
    const blocks = parseMarkdownLite("# One\n\n## Two ##\n\n### Three   ###");
    expect(blocks.map((b) => b.type)).toEqual(["heading", "heading", "heading"]);
    expect(blocks[0]).toEqual({ type: "heading", level: 1, children: [{ type: "text", text: "One" }] });
    expect(blocks[1]).toEqual({ type: "heading", level: 2, children: [{ type: "text", text: "Two" }] });
    expect(blocks[2]).toEqual({
      type: "heading",
      level: 3,
      children: [{ type: "text", text: "Three" }],
    });
  });

  it("keeps interior hashes in the heading text", () => {
    const blocks = parseMarkdownLite("### A # B ###");
    expect(blocks[0]).toEqual({
      type: "heading",
      level: 3,
      children: [{ type: "text", text: "A # B" }],
    });
  });
});

describe("parseMarkdownLite paragraphs", () => {
  it("joins consecutive lines with a newline and splits on a blank line", () => {
    const blocks = parseMarkdownLite("first line\nsecond line\n\nnext paragraph");
    expect(blocks).toHaveLength(2);
    expect(blocks[0]).toEqual({
      type: "paragraph",
      children: [{ type: "text", text: "first line\nsecond line" }],
    });
    expect(blocks[1]).toEqual({
      type: "paragraph",
      children: [{ type: "text", text: "next paragraph" }],
    });
  });

  it("normalizes CRLF input", () => {
    const blocks = parseMarkdownLite("alpha\r\nbeta\r\n");
    expect(blocks).toEqual([
      { type: "paragraph", children: [{ type: "text", text: "alpha\nbeta" }] },
    ]);
  });
});

describe("parseMarkdownLite lists", () => {
  it("collects dash and star markers with a continuation and a flattened nested item", () => {
    const blocks = parseMarkdownLite("- first\n  continued\n  - nested\n* third");
    expect(blocks).toHaveLength(1);
    const list = blocks[0];
    expect(list.type).toBe("list");
    if (list.type !== "list") throw new Error("expected a list block");
    expect(list.ordered).toBe(false);
    expect(list.items.map(textOf)).toEqual(["first continued", "nested", "third"]);
  });

  it("is ordered when the first marker is numeric, for both . and ) markers", () => {
    const blocks = parseMarkdownLite("1. one\n2) two");
    const list = blocks[0];
    if (list.type !== "list") throw new Error("expected a list block");
    expect(list.ordered).toBe(true);
    expect(list.items.map(textOf)).toEqual(["one", "two"]);
  });

  it("ends the list at a blank line", () => {
    const blocks = parseMarkdownLite("- a\n\nplain text");
    expect(blocks.map((b) => b.type)).toEqual(["list", "paragraph"]);
  });
});

describe("parseMarkdownLite fenced code", () => {
  it("records the language and the verbatim body", () => {
    const blocks = parseMarkdownLite("```python\nx = 1\n  y = 2\n```\nafter");
    expect(blocks[0]).toEqual({ type: "code", lang: "python", text: "x = 1\n  y = 2" });
    expect(blocks[1]).toEqual({ type: "paragraph", children: [{ type: "text", text: "after" }] });
  });

  it("runs an unclosed fence to the end of the input", () => {
    const blocks = parseMarkdownLite("intro\n\n```\nnever closed\n# not a heading");
    expect(blocks.map((b) => b.type)).toEqual(["paragraph", "code"]);
    expect(blocks[1]).toEqual({ type: "code", lang: "", text: "never closed\n# not a heading" });
  });
});

describe("parseInlines", () => {
  it("extracts inline code and bold, including bold that contains code", () => {
    expect(parseInlines("use `dt` now")).toEqual([
      { type: "text", text: "use " },
      { type: "code", text: "dt" },
      { type: "text", text: " now" },
    ]);
    expect(parseInlines("**bold** tail")).toEqual([
      { type: "strong", children: [{ type: "text", text: "bold" }] },
      { type: "text", text: " tail" },
    ]);
    expect(parseInlines("**set `dt` first**")).toEqual([
      {
        type: "strong",
        children: [
          { type: "text", text: "set " },
          { type: "code", text: "dt" },
          { type: "text", text: " first" },
        ],
      },
    ]);
  });

  it("leaves emphasis, math, and backslashes as plain text", () => {
    expect(parseInlines("*emphasis* and \\(x_i\\) and \\[y\\]")).toEqual([
      { type: "text", text: "*emphasis* and \\(x_i\\) and \\[y\\]" },
    ]);
  });

  it("keeps an unclosed backtick or bold marker as text", () => {
    expect(parseInlines("half `open")).toEqual([{ type: "text", text: "half `open" }]);
    expect(parseInlines("**dangling")).toEqual([{ type: "text", text: "**dangling" }]);
  });
});

describe("parseMarkdownLite tables", () => {
  it("reads a header, pads a short row, and stops at a non-pipe line", () => {
    const blocks = parseMarkdownLite("| key | default |\n|-----|------|\n| a | 1 |\n| b |\nafter");
    expect(blocks.map((b) => b.type)).toEqual(["table", "paragraph"]);
    const table = blocks[0];
    if (table.type !== "table") throw new Error("expected a table block");
    expect(table.header.map(textOf)).toEqual(["key", "default"]);
    expect(table.rows).toHaveLength(2);
    expect(table.rows[0].map(textOf)).toEqual(["a", "1"]);
    expect(table.rows[1].map(textOf)).toEqual(["b", ""]);
  });

  it("truncates a row longer than the header", () => {
    const blocks = parseMarkdownLite("| a |\n| --- |\n| 1 | 2 |");
    const table = blocks[0];
    if (table.type !== "table") throw new Error("expected a table block");
    expect(table.header).toHaveLength(1);
    expect(table.rows[0].map(textOf)).toEqual(["1"]);
  });
});

describe("parseMarkdownLite rules", () => {
  it("recognizes dash, star, and underscore rules", () => {
    const blocks = parseMarkdownLite("a\n\n---\n\nb\n\n***\n\n___");
    expect(blocks.map((b) => b.type)).toEqual([
      "paragraph",
      "rule",
      "paragraph",
      "rule",
      "rule",
    ]);
  });
});

describe("parseMarkdownLite mixed document", () => {
  it("returns the blocks in source order", () => {
    const document = [
      "# Title",
      "",
      "Intro paragraph with `code`.",
      "",
      "- bullet one",
      "- bullet two",
      "",
      "```bash",
      "ninja -C build",
      "```",
      "",
      "| key | value |",
      "| --- | --- |",
      "| a | 1 |",
      "",
      "---",
      "",
      "Closing line.",
    ].join("\n");
    const types = parseMarkdownLite(document).map((block: MdBlock) => block.type);
    expect(types).toEqual([
      "heading",
      "paragraph",
      "list",
      "code",
      "table",
      "rule",
      "paragraph",
    ]);
  });
});
