from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .dynamicity import DynamicFieldProfile, DynamicityAnalyzer


class JsonlProfiler:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: dict[str, Any]) -> None:
        if not self.path:
            return
        payload = {"ts": time.time(), **event}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")


class DynamicityProfiler:
    def __init__(self, jsonl_path: str | Path | None = None):
        self.fields: dict[str, DynamicFieldProfile] = {}
        self.events = JsonlProfiler(jsonl_path)

    def observe(
        self,
        field: str,
        value: Any,
        *,
        in_graph_key: bool = False,
        semantic: bool | None = None,
        component: str | None = None,
    ) -> None:
        profile = self.fields.setdefault(field, DynamicFieldProfile(field=field))
        profile.observe(value, in_graph_key=in_graph_key, semantic=semantic)
        self.events.record(
            {
                "kind": "dynamic_field",
                "component": component,
                "field": field,
                "value": value,
                "in_graph_key": in_graph_key,
                "semantic": semantic,
            }
        )

    def observe_many(
        self,
        values: dict[str, Any],
        *,
        in_graph_key: bool = False,
        semantic: bool | None = None,
        component: str | None = None,
    ) -> None:
        for field, value in values.items():
            self.observe(
                field,
                value,
                in_graph_key=in_graph_key,
                semantic=semantic,
                component=component,
            )

    def summary(self) -> dict[str, Any]:
        analyzer = DynamicityAnalyzer()
        profiles = list(self.fields.values())
        decisions = analyzer.analyze(profiles)
        return {
            "fields": [profile.to_json() for profile in profiles],
            "decisions": analyzer.decisions_to_json(decisions),
        }

    def write_summary(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.summary(), indent=2, ensure_ascii=False), encoding="utf-8")
