"""Strict parser for the TOML subset used by the assistant configuration."""

import io
import re
from typing import Any, Dict, List


_COMPONENT_RE = r"[A-Za-z0-9_-]+"
_HEADER_RE = re.compile(
    r"^\[(" + _COMPONENT_RE + r"(?:\." + _COMPONENT_RE + r")*)\]$"
)
_ASSIGNMENT_RE = re.compile(r"^(" + _COMPONENT_RE + r")\s*=\s*(.*)$")
_INTEGER_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(
    r"^[+-]?(?:(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?|"
    r"\d+[eE][+-]?\d+)$"
)


class TomlSubsetError(ValueError):
    """Raised when input uses syntax outside the supported TOML subset."""


def _fail(line_number: int, message: str) -> None:
    raise TomlSubsetError("line {0}: {1}".format(line_number, message))


def _strip_comment(line: str) -> str:
    in_string = False
    escaped = False
    for index, character in enumerate(line):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character == "#":
            return line[:index]
    return line


def _parse_string(value: str, line_number: int) -> str:
    if not value.startswith('"'):
        _fail(line_number, "expected a double-quoted string")

    result = []
    index = 1
    while index < len(value):
        character = value[index]
        if character == '"':
            if index != len(value) - 1:
                _fail(line_number, "unexpected text after string")
            return "".join(result)
        if character == "\\":
            index += 1
            if index >= len(value):
                _fail(line_number, "unterminated string escape")
            escaped = value[index]
            replacements = {
                "\\": "\\",
                '"': '"',
                "n": "\n",
                "t": "\t",
            }
            if escaped not in replacements:
                _fail(line_number, "unknown string escape '\\{0}'".format(escaped))
            result.append(replacements[escaped])
        else:
            result.append(character)
        index += 1

    _fail(line_number, "unterminated double-quoted string")
    return ""


def _split_array(value: str, line_number: int) -> List[str]:
    contents = value[1:-1]
    if not contents.strip():
        return []

    items = []
    start = 0
    in_string = False
    escaped = False
    for index, character in enumerate(contents):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[]":
            _fail(line_number, "nested arrays are not supported")
        elif character == ",":
            item = contents[start:index].strip()
            if not item:
                _fail(line_number, "empty array element")
            items.append(item)
            start = index + 1

    if in_string:
        _fail(line_number, "unterminated double-quoted string")

    final = contents[start:].strip()
    if final:
        items.append(final)
    elif not items:
        _fail(line_number, "empty array element")
    return items


def _parse_value(value: str, line_number: int, allow_array: bool = True) -> Any:
    value = value.strip()
    if not value:
        _fail(line_number, "missing value")
    if value.startswith('"'):
        return _parse_string(value, line_number)
    if value == "true":
        return True
    if value == "false":
        return False
    if _INTEGER_RE.fullmatch(value):
        return int(value)
    if _FLOAT_RE.fullmatch(value):
        return float(value)
    if value.startswith("["):
        if not allow_array or not value.endswith("]"):
            _fail(line_number, "invalid array")
        return [
            _parse_value(item, line_number, allow_array=False)
            for item in _split_array(value, line_number)
        ]
    _fail(line_number, "unsupported value")
    return None


def parse_toml_subset(text: str) -> dict:
    """Parse the strict TOML subset accepted for assistant configuration."""
    root: Dict[str, Any] = {}
    current = root

    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = _strip_comment(raw_line).strip()
        if not line:
            continue

        header = _HEADER_RE.fullmatch(line)
        if header:
            components = header.group(1).split(".")
            table = root
            for component in components[:-1]:
                if component not in table:
                    table[component] = {}
                elif not isinstance(table[component], dict):
                    _fail(line_number, "table header collides with a value")
                table = table[component]
            final = components[-1]
            if final in table:
                if isinstance(table[final], dict):
                    _fail(line_number, "duplicate table header")
                _fail(line_number, "table header collides with a value")
            table[final] = {}
            current = table[final]
            continue

        if line.startswith("["):
            _fail(line_number, "unsupported table header")

        assignment = _ASSIGNMENT_RE.fullmatch(line)
        if not assignment:
            _fail(line_number, "expected key = value")
        key = assignment.group(1)
        if key in current:
            _fail(line_number, "duplicate key '{0}'".format(key))
        current[key] = _parse_value(assignment.group(2), line_number)

    return root


def load_toml(path: str) -> dict:
    """Load a file after enforcing the supported TOML subset."""
    with open(path, "rb") as stream:
        raw = stream.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        line_number = raw[:error.start].count(b"\n") + 1
        _fail(line_number, "file is not valid UTF-8")

    parsed = parse_toml_subset(text)
    try:
        import tomllib
    except ImportError:
        return parsed

    return tomllib.load(io.BytesIO(raw))
