"""Provider command abstraction for the experimental assistant."""

import os
import signal
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import List

from tools.assist.config import (
    AssistConfig,
    AssistConfigError,
    KNOWN_ROLES,
    ProviderSpec,
)
from tools.assist.journal import JournalWriter, sha256_hex


DRY_RUN_PROVIDER = ProviderSpec(
    name="dry_run", command_template="dry_run", model="none"
)
_ACTIVE_CHILDREN: set = set()
_CANCELLATION_HANDLERS_INSTALLED = False


class AssistDisabledError(RuntimeError):
    """Raised when an LLM invocation is attempted while disabled."""


def _terminate_group(child, sig=signal.SIGTERM):
    try:
        os.killpg(os.getpgid(child.pid), sig)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(child.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def install_cancellation_handlers():
    global _CANCELLATION_HANDLERS_INSTALLED
    if _CANCELLATION_HANDLERS_INSTALLED:
        return

    def handle_cancellation(_signum, _frame):
        for child in list(_ACTIVE_CHILDREN):
            _terminate_group(child)
        sys.exit(130)

    signal.signal(signal.SIGTERM, handle_cancellation)
    signal.signal(signal.SIGINT, handle_cancellation)
    _CANCELLATION_HANDLERS_INSTALLED = True


def resolve_role(cfg: AssistConfig, role: str) -> ProviderSpec:
    """Resolve a configured role to its provider specification."""
    if role not in KNOWN_ROLES:
        raise ValueError("unknown assistant role '{0}'".format(role))
    provider_name = cfg.roles.get(role)
    if provider_name == "dry_run":
        return DRY_RUN_PROVIDER
    if provider_name is None or provider_name not in cfg.providers:
        raise AssistConfigError("role '{0}' has no provider configured".format(role))
    return cfg.providers[provider_name]


def build_command(spec: ProviderSpec, prompt_file: str) -> List[str]:
    """Build an argv list from a provider command template."""
    if spec.name == "dry_run":
        return ["dry_run", prompt_file]
    values = {"model": spec.model, "prompt_file": prompt_file}
    try:
        rendered = spec.command_template.format(**values)
    except KeyError as error:
        placeholder = str(error.args[0])
        raise AssistConfigError(
            "unknown command placeholder '{0}'".format(placeholder)
        ) from error
    command = shlex.split(rendered)
    if "{prompt_file}" not in spec.command_template:
        command.append(prompt_file)
    return command


def invoke(
    cfg: AssistConfig,
    role: str,
    prompt_text: str,
    workdir: str,
    journal: "JournalWriter",
    timeout_s: int = 600,
    cwd=None,
) -> str:
    """Invoke a configured provider and journal the completed attempt."""
    if not cfg.enabled:
        raise AssistDisabledError(
            "assistant is disabled (enabled=false or TENRYU_ASSIST_DISABLE)"
        )

    started = time.monotonic()
    spec = None
    command = []
    response = ""
    exit_status = None
    error = None
    prompt_file = ""
    child = None
    prompt_bytes = prompt_text.encode("utf-8")
    invocation_cwd = cwd if cwd is not None else workdir

    try:
        spec = resolve_role(cfg, role)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        prompt_path = os.path.join(
            workdir, "prompt_{0}_{1}.txt".format(role, timestamp)
        )
        with open(prompt_path, "w", encoding="utf-8") as stream:
            stream.write(prompt_text)
        prompt_file = prompt_path

        if any(character.isspace() for character in prompt_path):
            safe_dir = tempfile.mkdtemp(prefix="tenryu-assist-", dir=None)
            if any(character.isspace() for character in safe_dir):
                raise AssistConfigError(
                    "temporary directory path contains whitespace: {0}".format(
                        safe_dir
                    )
                )
            safe_prompt = os.path.join(safe_dir, os.path.basename(prompt_path))
            with open(safe_prompt, "w", encoding="utf-8") as stream:
                stream.write(prompt_text)
            for filename in ("docmap.md", "namelist_keys.txt"):
                source = os.path.join(workdir, filename)
                if os.path.isfile(source):
                    shutil.copyfile(source, os.path.join(safe_dir, filename))
            prompt_file = safe_prompt

        if spec.name == "dry_run":
            command = ["dry_run"]
            response = "DRY-RUN RESPONSE\n" + sha256_hex(prompt_text)[:16]
            exit_status = 0
        else:
            command = build_command(spec, prompt_file)
            child = subprocess.Popen(
                command,
                cwd=invocation_cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            _ACTIVE_CHILDREN.add(child)
            try:
                try:
                    stdout, stderr = child.communicate(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    _terminate_group(child)
                    stdout, stderr = child.communicate()
                    response = stdout
                    raise
            finally:
                _ACTIVE_CHILDREN.discard(child)
            response = stdout
            exit_status = child.returncode
            if child.returncode != 0:
                raise RuntimeError(
                    "provider command exited with code {0}: {1}".format(
                        child.returncode, stderr[-2000:]
                    )
                )
    except Exception as caught:
        error = caught

    provider_name = spec.name if spec is not None else cfg.roles.get(role, "")
    model = spec.model if spec is not None else ""
    payload = {
        "role": role,
        "provider": provider_name,
        "model": model,
        "command": command,
        "child_pid": child.pid if child is not None else None,
        "cwd": invocation_cwd,
        "prompt_file": prompt_file,
        "prompt_sha256": sha256_hex(prompt_bytes),
        "prompt_bytes": len(prompt_bytes),
        "response_sha256": sha256_hex(response),
        "response_bytes": len(response.encode("utf-8")),
        "exit_status": exit_status,
        "duration_s": time.monotonic() - started,
    }
    if error is not None:
        payload["error"] = str(error)
    journal.append("llm_invocation", payload)

    if error is not None:
        raise error
    return response
