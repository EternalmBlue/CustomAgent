from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Settings:
    db_path: Path = Path("data/agent.sqlite3")
    workspace_root: Path = Path(".")
    api_key: str = field(default="", repr=False)
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-flash"
    max_decisions: int = 8
    context_tokens: int = 16_384
    reply_tokens: int = 1024
    tool_result_tokens: int = 2048
    compression_ratio: float = 0.70
    recent_interactions: int = 6
    tool_timeout: float = 5.0
    search_delay: float = 0.25
    poll_interval: float = 0.02
    model_timeout: float = 60.0
    api_tokens: dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for name in ("db_path", "workspace_root"):
            if not isinstance(getattr(self, name), Path):
                raise ValueError(f"{name} must be a path")
        for name in ("api_key", "base_url", "model"):
            value = getattr(self, name)
            if not isinstance(value, str) or (name != "api_key" and not value.strip()):
                raise ValueError(f"{name} must be a nonempty string")
        try:
            url = urlsplit(self.base_url)
            valid_url = (
                url.scheme in {"http", "https"}
                and bool(url.hostname)
                and url.username is None
                and url.password is None
                and not url.query
                and not url.fragment
            )
            _ = url.port
        except ValueError:
            valid_url = False
        if not valid_url:
            raise ValueError("base_url must be an HTTP(S) URL without credentials")
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))
        for name in (
            "max_decisions",
            "context_tokens",
            "reply_tokens",
            "tool_result_tokens",
            "recent_interactions",
        ):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        for name in (
            "compression_ratio",
            "tool_timeout",
            "search_delay",
            "poll_interval",
            "model_timeout",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or (
                isinstance(value, float) and not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
        if not isinstance(self.api_tokens, dict) or any(
            not isinstance(token, str) or not token or not isinstance(user, str) or not user.strip()
            for token, user in self.api_tokens.items()
        ):
            raise ValueError("API tokens must map nonempty tokens to user IDs")
        if self.max_decisions < 1 or self.recent_interactions < 0:
            raise ValueError("Invalid execution or history limit")
        if self.reply_tokens < 1 or self.tool_result_tokens < 0:
            raise ValueError("Invalid reply or tool-result reserve")
        if not 0 < self.compression_ratio < 1:
            raise ValueError("Compression ratio must be between 0 and 1")
        if self.context_tokens <= self.reply_tokens + self.tool_result_tokens + 512:
            raise ValueError("Context budget must leave room for inputs and instructions")
        if min(self.tool_timeout, self.model_timeout, self.poll_interval) <= 0:
            raise ValueError("Timeouts and poll interval must be positive")
        if self.search_delay < 0:
            raise ValueError("Search delay must be nonnegative")

    @property
    def input_budget(self) -> int:
        return self.context_tokens - self.reply_tokens - self.tool_result_tokens

    @classmethod
    def from_file(cls, path: Path | str = Path("config.toml")) -> Settings:
        config_path = Path(path).resolve()
        try:
            content = config_path.read_text(encoding="utf-8-sig")
        except FileNotFoundError:
            raise FileNotFoundError("Configuration file not found") from None
        except UnicodeError:
            raise ValueError("Configuration must be UTF-8 TOML") from None
        except OSError:
            raise ValueError("Cannot read configuration file") from None
        try:
            values = tomllib.loads(content)
        except tomllib.TOMLDecodeError:
            raise ValueError("Invalid TOML configuration") from None
        if values.keys() - {item.name for item in fields(cls)}:
            raise ValueError("Unknown configuration keys")
        defaults = cls()
        for name in ("db_path", "workspace_root"):
            value = values.get(name, str(getattr(defaults, name)))
            if not isinstance(value, str) or not value.strip() or "\x00" in value:
                raise ValueError(f"{name} must be a nonempty path string")
            values[name] = (config_path.parent / value).resolve()
        return cls(**values)
