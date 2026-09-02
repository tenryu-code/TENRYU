export interface RemoteDirEntry {
  name: string;
  dir: boolean;
}

/** Parse `ls -1paL` output: strip ./ and ../, mark trailing-slash entries as dirs. */
export function parseLsOutput(stdout: string): RemoteDirEntry[] {
  const entries: RemoteDirEntry[] = [];
  for (const raw of stdout.split("\n")) {
    const line = raw.trim();
    if (line === "" || line === "./" || line === "../") continue;
    const dir = line.endsWith("/");
    const name = dir ? line.slice(0, -1) : line;
    if (name === "" || name === "." || name === "..") continue;
    entries.push({ name, dir });
  }
  entries.sort((a, b) => (a.dir === b.dir ? a.name.localeCompare(b.name) : a.dir ? -1 : 1));
  return entries;
}

/** Parent of an absolute remote path ("/a/b" -> "/a", "/a" -> "/", "/" -> "/"). */
export function remoteParentDir(p: string): string {
  if (p === "/" || !p.startsWith("/")) return "/";
  const trimmed = p.endsWith("/") ? p.slice(0, -1) : p;
  const i = trimmed.lastIndexOf("/");
  return i <= 0 ? "/" : trimmed.slice(0, i);
}

export function joinRemoteDir(dir: string, name: string): string {
  return dir.endsWith("/") ? dir + name : dir + "/" + name;
}
