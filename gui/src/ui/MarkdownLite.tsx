import type { CSSProperties, ReactNode } from "react";
import { parseMarkdownLite, type MdBlock, type MdInline } from "../core/markdownLite";

const INLINE_CODE_STYLE: CSSProperties = {
  fontFamily: "var(--mono)",
  background: "var(--bg-inset)",
  padding: "0 3px",
  borderRadius: "var(--radius-sm)",
};

const BLOCK_CODE_STYLE: CSSProperties = {
  fontFamily: "var(--mono)",
  background: "var(--bg-inset)",
  border: "1px solid var(--separator)",
  borderRadius: "var(--radius-sm)",
  padding: "6px 8px",
  overflowX: "auto",
  margin: "4px 0",
  whiteSpace: "pre",
};

const CELL_STYLE: CSSProperties = {
  border: "1px solid var(--separator)",
  padding: "2px 6px",
};

function renderInline(node: MdInline, key: number): ReactNode {
  if (node.type === "text") return node.text;
  if (node.type === "code") {
    return (
      <code key={key} style={INLINE_CODE_STYLE}>
        {node.text}
      </code>
    );
  }
  return <strong key={key}>{node.children.map((child, index) => renderInline(child, index))}</strong>;
}

function renderInlines(nodes: MdInline[]): ReactNode[] {
  return nodes.map((node, index) => renderInline(node, index));
}

function renderBlock(block: MdBlock, key: number): ReactNode {
  switch (block.type) {
    case "heading":
      return (
        <div
          key={key}
          style={{
            fontWeight: 600,
            fontSize: block.level === 1 ? 14 : block.level === 2 ? 13 : 12,
            marginTop: 6,
          }}
        >
          {renderInlines(block.children)}
        </div>
      );
    case "paragraph":
      return (
        <p key={key} style={{ whiteSpace: "pre-wrap", margin: "4px 0" }}>
          {renderInlines(block.children)}
        </p>
      );
    case "list": {
      const items = block.items.map((item, index) => <li key={index}>{renderInlines(item)}</li>);
      const style: CSSProperties = {
        paddingLeft: "1.2em",
        margin: "4px 0",
        listStyle: block.ordered ? "decimal" : "disc",
      };
      return block.ordered ? (
        <ol key={key} style={style}>
          {items}
        </ol>
      ) : (
        <ul key={key} style={style}>
          {items}
        </ul>
      );
    }
    case "code":
      return (
        <pre key={key} style={BLOCK_CODE_STYLE}>
          <code>{block.text}</code>
        </pre>
      );
    case "table":
      return (
        <div key={key} style={{ overflowX: "auto" }}>
          <table style={{ borderCollapse: "collapse" }}>
            <thead>
              <tr>
                {block.header.map((cell, index) => (
                  <th key={index} style={{ ...CELL_STYLE, fontWeight: 600 }}>
                    {renderInlines(cell)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {block.rows.map((row, rowIndex) => (
                <tr key={rowIndex}>
                  {row.map((cell, cellIndex) => (
                    <td key={cellIndex} style={CELL_STYLE}>
                      {renderInlines(cell)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    case "rule":
      return <hr key={key} style={{ borderColor: "var(--separator)", margin: "6px 0" }} />;
  }
}

/** Renders the Markdown subset produced by the assistant, without raw HTML. */
export function MarkdownLite({ text }: { text: string }) {
  return <>{parseMarkdownLite(text).map((block, index) => renderBlock(block, index))}</>;
}
