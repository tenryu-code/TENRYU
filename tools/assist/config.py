"""Configuration model and resolution for the experimental assistant."""

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

from tools.assist import tomlmini


KNOWN_ROLES = (
    "deck_design",
    "forensics",
    "intervention_proposer",
    "intervention_reviewer",
    "explanation",
    "question_answering",
)
ENV_KILL_SWITCH = "TENRYU_ASSIST_DISABLE"
ENV_CONFIG_PATH = "TENRYU_ASSIST_CONFIG"
DEFAULT_CONFIG_BASENAME = "assistant.toml"
USER_CONFIG_PATH = "~/.tenryu/assistant.toml"
APP_CONFIG_IDENTIFIER = "jp.osaka-u.ile.tenryu-studio"


@dataclass
class ProviderSpec:
    name: str
    command_template: str
    model: str


@dataclass
class Budget:
    max_interventions_per_run: int = 3
    max_tokens_per_decision: int = 30000


@dataclass
class AssistConfig:
    enabled: bool = False
    providers: Dict[str, ProviderSpec] = field(default_factory=dict)
    roles: Dict[str, str] = field(default_factory=dict)
    budget: Budget = field(default_factory=Budget)
    config_source: str = "builtin-defaults"
    disabled_by: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class AssistConfigError(ValueError):
    """Raised when assistant configuration is invalid."""


def app_config_path(
    env: Mapping[str, str], platform: Optional[str] = None
) -> Optional[str]:
    """Path of the Studio-managed assistant.toml in the app configuration folder."""
    platform = platform if platform is not None else sys.platform
    home = env.get("HOME") or os.path.expanduser("~")
    if platform.startswith("darwin"):
        return os.path.join(
            home,
            "Library",
            "Application Support",
            APP_CONFIG_IDENTIFIER,
            "assistant.toml",
        )
    if platform.startswith("win"):
        appdata = env.get("APPDATA")
        if not appdata:
            return None
        return os.path.join(appdata, APP_CONFIG_IDENTIFIER, "assistant.toml")
    base = env.get("XDG_CONFIG_HOME") or os.path.join(home, ".config")
    return os.path.join(base, APP_CONFIG_IDENTIFIER, "assistant.toml")


def find_config_path(
    cli_path: Optional[str],
    env: Mapping[str, str],
    cwd: str,
    platform: Optional[str] = None,
    optional_cli_path: Optional[str] = None,
) -> Optional[str]:
    """Resolve the assistant configuration path in precedence order."""
    if cli_path is not None and optional_cli_path is not None:
        raise AssistConfigError(
            "--config and --config-or-defaults are mutually exclusive"
        )
    if optional_cli_path is not None:
        return optional_cli_path if os.path.exists(optional_cli_path) else None

    if cli_path is not None:
        if not os.path.exists(cli_path):
            raise AssistConfigError("config file does not exist: {0}".format(cli_path))
        return cli_path

    env_path = env.get(ENV_CONFIG_PATH, "")
    if env_path:
        if not os.path.exists(env_path):
            raise AssistConfigError("config file does not exist: {0}".format(env_path))
        return env_path

    cwd_path = os.path.join(cwd, DEFAULT_CONFIG_BASENAME)
    if os.path.exists(cwd_path):
        return cwd_path

    home = env.get("HOME") or os.path.expanduser("~")
    user_path = os.path.join(home, ".tenryu", "assistant.toml")
    if os.path.exists(user_path):
        return user_path

    app_path = app_config_path(env, platform)
    if app_path is not None and os.path.exists(app_path):
        return app_path
    return None


def _unknown_keys(values: dict, allowed: set, where: str) -> None:
    for key in values:
        if key not in allowed:
            raise AssistConfigError("unknown key '{0}' at {1}".format(key, where))


def _parse_config(data: dict, source: str) -> AssistConfig:
    if not isinstance(data, dict):
        raise AssistConfigError("configuration root must be a table")
    _unknown_keys(data, {"enabled", "providers", "roles", "budget"}, "top level")

    enabled = data.get("enabled", False)
    if not isinstance(enabled, bool):
        raise AssistConfigError("enabled must be a boolean")

    provider_values = data.get("providers", {})
    if not isinstance(provider_values, dict):
        raise AssistConfigError("providers must be a table")
    providers = {}
    for name, values in provider_values.items():
        if name == "dry_run":
            raise AssistConfigError("provider name 'dry_run' is reserved")
        if not isinstance(values, dict):
            raise AssistConfigError("provider '{0}' must be a table".format(name))
        _unknown_keys(values, {"command", "model"}, "providers.{0}".format(name))
        if set(values) != {"command", "model"}:
            raise AssistConfigError(
                "provider '{0}' must define command and model".format(name)
            )
        command = values["command"]
        model = values["model"]
        if not isinstance(command, str) or not command:
            raise AssistConfigError(
                "provider '{0}' command must be a non-empty string".format(name)
            )
        if not isinstance(model, str) or not model:
            raise AssistConfigError(
                "provider '{0}' model must be a non-empty string".format(name)
            )
        providers[name] = ProviderSpec(
            name=name, command_template=command, model=model
        )

    role_values = data.get("roles", {})
    if not isinstance(role_values, dict):
        raise AssistConfigError("roles must be a table")
    roles = {}
    for role, provider_name in role_values.items():
        if role not in KNOWN_ROLES:
            raise AssistConfigError(
                "unknown role '{0}'; valid roles: {1}".format(
                    role, ", ".join(KNOWN_ROLES)
                )
            )
        if not isinstance(provider_name, str):
            raise AssistConfigError(
                "role '{0}' provider must be a string".format(role)
            )
        if provider_name != "dry_run" and provider_name not in providers:
            raise AssistConfigError(
                "role '{0}' references unknown provider '{1}'".format(
                    role, provider_name
                )
            )
        roles[role] = provider_name

    budget_values = data.get("budget", {})
    if not isinstance(budget_values, dict):
        raise AssistConfigError("budget must be a table")
    _unknown_keys(
        budget_values,
        {"max_interventions_per_run", "max_tokens_per_decision"},
        "budget",
    )
    budget = Budget()
    for key in ("max_interventions_per_run", "max_tokens_per_decision"):
        if key not in budget_values:
            continue
        value = budget_values[key]
        if type(value) is not int or value <= 0:
            raise AssistConfigError("budget.{0} must be an integer > 0".format(key))
        setattr(budget, key, value)

    warnings = []
    proposer = roles.get("intervention_proposer")
    reviewer = roles.get("intervention_reviewer")
    if proposer is not None and reviewer is not None and proposer == reviewer:
        warnings.append(
            "intervention_proposer and intervention_reviewer use the same "
            "provider '{0}' — cross-check independence is degraded".format(proposer)
        )

    return AssistConfig(
        enabled=enabled,
        providers=providers,
        roles=roles,
        budget=budget,
        config_source=source,
        warnings=warnings,
    )


def load_config(
    cli_path: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    cwd: Optional[str] = None,
    optional_cli_path: Optional[str] = None,
) -> AssistConfig:
    """Load and validate assistant configuration.

    ``optional_cli_path`` uses that file when present or built-in defaults when absent.
    """
    if env is None:
        env = os.environ
    if cwd is None:
        cwd = os.getcwd()

    disabled = env.get(ENV_KILL_SWITCH, "") in ("1", "true", "TRUE", "yes")
    path = find_config_path(
        cli_path, env, cwd, optional_cli_path=optional_cli_path
    )
    if path is None:
        config = AssistConfig()
    else:
        config = _parse_config(tomlmini.load_toml(path), os.path.abspath(path))

    if disabled:
        config.enabled = False
        config.disabled_by = "env:TENRYU_ASSIST_DISABLE"
    return config


def resolved_summary(cfg: AssistConfig) -> dict:
    """Return a JSON-safe summary of the resolved configuration."""
    return {
        "enabled": cfg.enabled,
        "disabled_by": cfg.disabled_by,
        "config_source": cfg.config_source,
        "providers": {
            name: {
                "model": spec.model,
                "command_template": spec.command_template,
            }
            for name, spec in cfg.providers.items()
        },
        "roles": dict(cfg.roles),
        "budget": {
            "max_interventions_per_run": cfg.budget.max_interventions_per_run,
            "max_tokens_per_decision": cfg.budget.max_tokens_per_decision,
        },
        "warnings": list(cfg.warnings),
    }
