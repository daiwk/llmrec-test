"""Utility LLM implementations for the demo graph."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class LocalRecommenderLLM:
    """
    A small, deterministic stand-in for a real chat model.

    The class exposes a :meth:`complete` method that mirrors the interface we
    use throughout the graph. It is intentionally simple so that the demo can
    run without network access while still showing where an actual model
    invocation would live.
    """

    canned_preferences: str | None = None

    def complete(self, prompt: str) -> str:
        """Return a lightweight, rule-based completion for the given prompt."""
        if "Summarize the viewer profile" in prompt:
            return self._summarize_preferences(prompt)
        if "Blend the following signals" in prompt:
            return self._rank_recommendations(prompt)
        if "Context snippets" in prompt:
            return self._contextualize(prompt)
        return "I am a local LLM placeholder."

    def _summarize_preferences(self, prompt: str) -> str:
        if self.canned_preferences:
            return self.canned_preferences
        match = re.search(r"User request:\s*(.+)", prompt)
        request = match.group(1).strip() if match else "movies with strong themes"
        return (
            "The viewer is interested in films featuring "
            f"{request}. They value immersive worlds, emotional arcs, "
            "and clear narrative stakes."
        )

    def _rank_recommendations(self, prompt: str) -> str:
        candidates: List[tuple[str, float]] = []
        for title, score in re.findall(r"Title: ([^|]+)\| Score: ([0-9.]+)", prompt):
            candidates.append((title.strip(), float(score)))
        candidates.sort(key=lambda item: item[1], reverse=True)
        top = candidates[:3]
        lines = [
            f"- {title} (score {score:.2f}): blends the requested tone with strong viewer reception."
            for title, score in top
        ]
        return "Here are tailored picks based on the merged signals:\n" + "\n".join(lines)

    def _contextualize(self, prompt: str) -> str:
        snippets = re.findall(r"Snippet:\s*(.+)", prompt)
        bullets = [f"- {snippet.strip()}" for snippet in snippets[:3]]
        if not bullets:
            bullets = ["- No supporting snippets were provided."]
        return "Context integrated from tools and memory:\n" + "\n".join(bullets)


def make_llm(preferences: str | None = None) -> LocalRecommenderLLM:
    """Factory to create the local LLM used by the demo graph."""

    return LocalRecommenderLLM(canned_preferences=preferences)
