"""Minimal prompt registry for the vNext pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromptTemplate:
    """Simple named prompt with system and user templates."""

    name: str
    system: str
    user: str


class PromptManager:
    """Register and render prompt templates by name."""

    def __init__(self, templates: list[PromptTemplate] | None = None) -> None:
        self._templates: dict[str, PromptTemplate] = {}
        for template in templates or []:
            self.register(template)

    def register(self, template: PromptTemplate) -> None:
        self._templates[template.name] = template

    def get(self, name: str) -> PromptTemplate:
        try:
            return self._templates[name]
        except KeyError as exc:
            raise KeyError(f"Unknown prompt template: {name}") from exc

    def render(self, name: str, context: dict[str, Any]) -> tuple[str, str]:
        template = self.get(name)
        try:
            system_prompt = template.system.format(**context)
            user_prompt = template.user.format(**context)
        except KeyError as exc:
            missing = str(exc).strip("'")
            raise KeyError(f"Missing prompt context variable `{missing}` for template `{name}`.") from exc
        return system_prompt, user_prompt

