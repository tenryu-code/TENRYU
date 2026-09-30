#!/usr/bin/env python3
"""Regenerate or check the docs/sections/ split files.

The canonical documents (docs/NUMERICS.md, docs/ARCHITECTURE.md,
docs/CUDA_KERNELS.md) are authoritative.  Each split file is a verbatim copy
of one contiguous heading range of its canonical document, preceded by a
one-line header comment.  Edit the canonical document, then run

    python3 tools/docs/sync_sections.py --write

to regenerate the splits.  `--check` exits with status 1 and lists the split
files that differ from their canonical ranges.
"""

import argparse
import os
import sys

# (split name, canonical name, first line prefix or None for the file start,
#  first line prefix of the next range or None for the file end)
SPLITS = [
    ("NUMERICS_00-02", "NUMERICS", None, "## 3. "),
    ("NUMERICS_03", "NUMERICS", "## 3. ", "## 4. "),
    ("NUMERICS_04-05", "NUMERICS", "## 4. ", "## 6. "),
    ("NUMERICS_06-07", "NUMERICS", "## 6. ", "## 8. "),
    ("NUMERICS_06_7-06_8", "NUMERICS", "### 6.7 ", "## 7. "),
    ("NUMERICS_08-11", "NUMERICS", "## 8. ", "## Appendix A. "),
    ("NUMERICS_AppA-12", "NUMERICS", "## Appendix A. ", "## 13. "),
    ("NUMERICS_13", "NUMERICS", "## 13. ", "## 14. "),
    ("NUMERICS_14-15", "NUMERICS", "## 14. ", None),
    ("ARCHITECTURE_01-04_1", "ARCHITECTURE", None, "### 4.2 "),
    ("ARCHITECTURE_04_2-04_9", "ARCHITECTURE", "### 4.2 ", "## 5. "),
    ("ARCHITECTURE_05", "ARCHITECTURE", "## 5. ", "## 6. "),
    ("ARCHITECTURE_06-10", "ARCHITECTURE", "## 6. ", None),
    ("CUDA_KERNELS_00-02", "CUDA_KERNELS", None, "## 3. "),
    ("CUDA_KERNELS_03-05", "CUDA_KERNELS", "## 3. ", "## 6. "),
    ("CUDA_KERNELS_06", "CUDA_KERNELS", "## 6. ", "## 7. "),
    ("CUDA_KERNELS_07-08", "CUDA_KERNELS", "## 7. ", "## 9. "),
    ("CUDA_KERNELS_09-13", "CUDA_KERNELS", "## 9. ", None),
]


def find_line(lines, prefix, start, canonical):
    for i in range(start, len(lines)):
        if lines[i].startswith(prefix):
            return i
    raise SystemExit(f"{canonical}: no line starts with {prefix!r}")


def render(docs_dir, name, canonical, first, nxt):
    with open(os.path.join(docs_dir, canonical + ".md"), encoding="utf-8") as f:
        lines = f.read().split("\n")
    begin = 0 if first is None else find_line(lines, first, 0, canonical)
    end = len(lines) if nxt is None else find_line(lines, nxt, begin + 1, canonical)
    header = (f"<!-- 分割元: docs/{canonical}.md | このファイルは参照用です。"
              f"原本（docs/{canonical}.md）が権威です。 -->")
    text = header + "\n" + "\n".join(lines[begin:end])
    return text if text.endswith("\n") else text + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    parser.add_argument("--docs", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "docs"))
    args = parser.parse_args()
    stale = []
    for name, canonical, first, nxt in SPLITS:
        path = os.path.join(args.docs, "sections", name + ".md")
        text = render(args.docs, name, canonical, first, nxt)
        current = None
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                current = f.read()
        if current == text:
            continue
        stale.append(name)
        if args.write:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
    verb = "rewrote" if args.write else "out of date"
    for name in stale:
        print(f"{verb}: docs/sections/{name}.md")
    return 1 if (args.check and stale) else 0


if __name__ == "__main__":
    sys.exit(main())
