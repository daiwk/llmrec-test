"""Utility LLM implementations for the demo graph."""

from __future__ import annotations

import re
import logging
import os
from dataclasses import dataclass
from typing import Iterable, List

from openai import OpenAI


logger = logging.getLogger(__name__)


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
        logger.debug("本地LLM收到提示词，长度=%d，内容预览=%s", len(prompt), prompt[:120])
        if "总结观影偏好" in prompt:
            return self._summarize_preferences(prompt)
        if "融合各代理信号" in prompt:
            return self._rank_recommendations(prompt)
        if "上下文片段" in prompt:
            return self._contextualize(prompt)
        return "I am a local LLM placeholder."

    def _summarize_preferences(self, prompt: str) -> str:
        if self.canned_preferences:
            return self.canned_preferences
        match = re.search(r"用户需求：\s*(.+)", prompt)
        request = match.group(1).strip() if match else "movies with strong themes"
        return (
            "观众希望看到包含"
            f"{request}的作品，倾向于沉浸式世界、情感张力和清晰的叙事动机。"
        )

    def _rank_recommendations(self, prompt: str) -> str:
        candidates: List[tuple[str, float]] = []
        for title, score in re.findall(r"Title: ([^|]+)\| Score: ([0-9.]+)", prompt):
            candidates.append((title.strip(), float(score)))
        candidates.sort(key=lambda item: item[1], reverse=True)
        top = candidates[:3]
        lines = [
            f"- {title}（综合得分 {score:.2f}）：契合用户意图且广受好评。"
            for title, score in top
        ]
        return "基于融合信号的推荐：\n" + "\n".join(lines)

    def _contextualize(self, prompt: str) -> str:
        snippets = re.findall(r"Snippet:\s*(.+)", prompt)
        bullets = [f"- {snippet.strip()}" for snippet in snippets[:3]]
        if not bullets:
            bullets = ["- 没有可用的上下文片段。"]
        return "以下是整合的上下文片段：\n" + "\n".join(bullets)


def make_llm(preferences: str | None = None) -> LocalRecommenderLLM:
    """Factory to create the local LLM used by the demo graph."""

    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("LLMREC_MODEL", "qwen3:1.7b")

    if base_url:
        logger.debug("构建远端LLM，base_url=%s，model=%s", base_url, model)
        client = OpenAI(api_key=api_key or "EMPTY", base_url=base_url)
        return RemoteRecommenderLLM(client=client, model=model)

    logger.debug("未检测到OPENAI_BASE_URL，使用本地LLM占位实现")
    return LocalRecommenderLLM(canned_preferences=preferences)


class RemoteRecommenderLLM(LocalRecommenderLLM):
    """Real LLM client backed by the OpenAI-compatible API."""

    def __init__(self, client: OpenAI, model: str, canned_preferences: str | None = None):
        super().__init__(canned_preferences=canned_preferences)
        self.client = client
        self.model = model

    def complete(self, prompt: str) -> str:  # type: ignore[override]
        logger.debug("调用远端LLM，模型=%s，提示词长度=%d", self.model, len(prompt))
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        content = response.choices[0].message.content or ""
        logger.debug("远端LLM响应长度=%d", len(content))
        return content
