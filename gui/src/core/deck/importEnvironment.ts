/** Discover literal environment reads without executing Python. */
export function discoverImportEnvironment(source: string): string[] {
  // Keep strings as single tokens and discard comments, so examples inside
  // comments/docstrings cannot be mistaken for executable environment reads.
  const tokens = Array.from(source.matchAll(/#[^\n]*|(?:[rRuUbBfF]{0,2})(?:"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')|[A-Za-z_]\w*|[^\s]/g))
    .map(match=>match[0]).filter(token=>!token.startsWith("#"));
  const names = new Set<string>();
  let environImported = false;
  for (let i=0; i<tokens.length; i++) {
    if (tokens[i-1] === ".") continue;
    if (tokens[i] === "from" && tokens[i+1] === "os" && tokens[i+2] === "import") {
      let j = i+3;
      if (tokens[j] === "(") j++;
      while (/^[A-Za-z_]\w*$/.test(tokens[j] ?? "")) {
        const name = tokens[j++];
        const aliased = tokens[j] === "as";
        if (aliased) j+=2;
        if (name === "environ" && !aliased) environImported = true;
        if (tokens[j] !== ",") break;
        j++;
      }
    }
    let j = i;
    if (tokens[j] === "os" && tokens[j+1] === ".") {
      j+=2;
      if (tokens[j] === "getenv" && tokens[j+1] === "(") j+=2;
      else if (tokens[j++] === "environ") {
        if (tokens[j] === "[") j++;
        else if (tokens[j] === "." && tokens[j+1] === "get" && tokens[j+2] === "(") j+=3;
        else continue;
      } else continue;
    } else if (environImported && tokens[j] === "environ" && tokens[j-1] !== ".") {
      j++;
      if (tokens[j] === "[") j++;
      else if (tokens[j] === "." && tokens[j+1] === "get" && tokens[j+2] === "(") j+=3;
      else continue;
    } else continue;
    const match = /^(["'])([A-Za-z_][A-Za-z0-9_]*)\1$/.exec(tokens[j] ?? "");
    const assignment = tokens[j+1] === "]" && tokens[j+2] === "=" && tokens[j+3] !== "=";
    if (match && [")", "]", ","].includes(tokens[j+1]) && !assignment) names.add(match[2]);
  }
  return [...names];
}

export function appendImportEnvironment(text: string, names: string[]): string {
  const present = new Set(text.split(/\r?\n/).flatMap(line=>{
    const match = /^\s*([A-Za-z_][A-Za-z0-9_]*)=/.exec(line);
    return match ? [match[1]] : [];
  }));
  const missing = [...new Set(names)].filter(name=>!present.has(name));
  return missing.length ? text + (text && !text.endsWith("\n") ? "\n" : "") + missing.map(name=>`${name}=`).join("\n") : text;
}
