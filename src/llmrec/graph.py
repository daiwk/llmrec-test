"""LangGraph pipeline for LLM-based recommendations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

from .llm import LocalRecommenderLLM, make_llm


logger = logging.getLogger(__name__)


@dataclass
class Movie:
    movie_id: int
    title: str
    genres: List[str]
    tags: List[str]
    plot: str
    rating_count: int
    avg_rating: float

    @classmethod
    def from_row(cls, row: pd.Series) -> "Movie":
        return cls(
            movie_id=int(row["movie_id"]),
            title=str(row["title"]),
            genres=str(row["genres"]).split("|"),
            tags=str(row["tags"]).split(";"),
            plot=str(row["plot"]),
            rating_count=int(row["rating_count"]),
            avg_rating=float(row["avg_rating"]),
        )


class RecommendationState(TypedDict):
    user_query: str
    preferences_summary: str
    candidate_pool: List[Movie]
    snippets: List[str]
    recommendations: str
    pending_agents: List[str]
    next_agent: str | None
    route_reason: str


def load_catalog(path: str | Path) -> List[Movie]:
    df = pd.read_csv(path)
    logger.debug("从%s加载影片库，共%d条", path, len(df))
    return [Movie.from_row(row) for _, row in df.iterrows()]


def preference_agent(llm: LocalRecommenderLLM) -> Callable[[RecommendationState], RecommendationState]:
    def _inner(state: RecommendationState) -> RecommendationState:
        prompt = (
            "你是一名个性化策划专家，请根据用户诉求总结观影偏好。\n"
            f"用户需求：{state['user_query']}\n"
            "请关注风格、题材、节奏和情绪意图，用两句话概括。"
        )
        logger.debug("偏好代理收到用户需求：%s", state["user_query"])
        summary = llm.complete(prompt)
        logger.debug("偏好总结结果：%s", summary)
        return {"preferences_summary": summary}

    return _inner


def _score_movie(movie: Movie, keywords: Iterable[str]) -> float:
    score = movie.avg_rating + (movie.rating_count / 5000)
    lowered = {kw.lower() for kw in keywords}
    for tag in movie.tags + movie.genres:
        if tag.lower() in lowered:
            score += 0.6
    for word in movie.plot.lower().split():
        if word.strip(',.') in lowered:
            score += 0.05
    return score


def retrieval_agent(catalog: List[Movie]) -> Callable[[RecommendationState], RecommendationState]:
    def _inner(state: RecommendationState) -> RecommendationState:
        keywords = state["preferences_summary"].replace("|", " ").replace(";", " ").split()
        scored = [
            (movie, _score_movie(movie, keywords))
            for movie in catalog
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        top_movies = [movie for movie, _ in scored[:6]]
        logger.debug("召回阶段选出前%d部影片：%s", len(top_movies), [m.title for m in top_movies])
        return {"candidate_pool": top_movies}

    return _inner


def rag_agent(catalog: List[Movie]) -> Callable[[RecommendationState], RecommendationState]:
    def _inner(state: RecommendationState) -> RecommendationState:
        snippets: List[str] = []
        for movie in state["candidate_pool"][:3]:
            snippet = f"{movie.title}: {movie.plot}"
            snippets.append(snippet)
        logger.debug("提取到的上下文片段数量：%d", len(snippets))
        return {"snippets": snippets}

    return _inner


def ranking_agent(llm: LocalRecommenderLLM) -> Callable[[RecommendationState], RecommendationState]:
    def _inner(state: RecommendationState) -> RecommendationState:
        scored_candidates: List[str] = []
        for movie in state["candidate_pool"]:
            score = movie.avg_rating + movie.rating_count / 1000
            scored_candidates.append(
                f"Title: {movie.title} | Score: {score:.2f} | Genres: {', '.join(movie.genres)} | Tags: {', '.join(movie.tags)}"
            )

        prompt = (
            "你是负责融合各代理信号的主代理。请综合以下信息给出3条推荐。\n"
            f"观众偏好摘要：{state['preferences_summary']}\n"
            "上下文片段：\n"
            + "\n".join(f"Snippet: {snippet}" for snippet in state["snippets"])
            + "\n候选池及分数（分数越高越优）：\n"
            + "\n".join(scored_candidates)
            + "\n请用亲切的中文项目符号输出结果。"
        )
        logger.debug("排序阶段候选数=%d，生成提示词长度=%d", len(state["candidate_pool"]), len(prompt))
        recs = llm.complete(prompt)
        logger.debug("排序代理返回结果长度=%d", len(recs))
        return {"recommendations": recs}

    return _inner


def planner_agent(llm: LocalRecommenderLLM) -> Callable[[RecommendationState], RecommendationState]:
    def _inner(state: RecommendationState) -> RecommendationState:
        prompt = (
            "你是调度员，需要根据用户诉求决定调用哪些代理。\n"
            "可用代理列表：\n"
            "- preference_agent：总结观影偏好\n"
            "- retrieval_agent：基于偏好召回影片\n"
            "- rag_agent：抽取候选影片的剧情片段\n"
            "- ranking_agent：融合信号生成推荐（必须最后执行）\n"
            "请输出执行顺序，使用逗号分隔的代理名称列表。如果无需上下文片段可以跳过 rag_agent。\n"
            f"用户诉求：{state['user_query']}\n"
            "只返回列表。"
        )
        plan = llm.complete(prompt)
        parsed = [part.strip() for part in plan.split(',') if part.strip()]
        allowed = [
            "preference_agent",
            "retrieval_agent",
            "rag_agent",
            "ranking_agent",
        ]
        sanitized: List[str] = []
        for agent in parsed:
            if agent in allowed and agent not in sanitized:
                sanitized.append(agent)
        if not sanitized:
            sanitized = ["preference_agent", "retrieval_agent", "rag_agent"]
        if "preference_agent" not in sanitized:
            sanitized.insert(0, "preference_agent")
        if "retrieval_agent" not in sanitized:
            insert_at = sanitized.index("ranking_agent") if "ranking_agent" in sanitized else len(sanitized)
            sanitized.insert(insert_at, "retrieval_agent")
        if sanitized[-1] != "ranking_agent":
            sanitized.append("ranking_agent")
        logger.debug("规划代理选择的执行序列：%s", sanitized)
        return {"pending_agents": sanitized, "route_reason": plan}

    return _inner


def routing_node(state: RecommendationState) -> RecommendationState:
    pending = list(state.get("pending_agents", []))
    if not pending:
        logger.debug("没有待调度代理，结束工作流")
        return {"next_agent": None, "pending_agents": []}
    next_agent = pending.pop(0)
    logger.debug("路由到代理：%s，剩余队列：%s", next_agent, pending)
    return {"next_agent": next_agent, "pending_agents": pending}


def build_graph(catalog_path: str | Path, llm: LocalRecommenderLLM | None = None):
    llm = llm or make_llm()
    catalog = load_catalog(catalog_path)

    workflow = StateGraph(RecommendationState)
    workflow.add_node("planner_agent", planner_agent(llm))
    workflow.add_node("routing_node", routing_node)
    workflow.add_node("preference_agent", preference_agent(llm))
    workflow.add_node("retrieval_agent", retrieval_agent(catalog))
    workflow.add_node("rag_agent", rag_agent(catalog))
    workflow.add_node("ranking_agent", ranking_agent(llm))

    workflow.set_entry_point("planner_agent")
    workflow.add_edge("planner_agent", "routing_node")

    workflow.add_conditional_edges(
        "routing_node",
        lambda state: state["next_agent"] or "END",
        {
            "preference_agent": "preference_agent",
            "retrieval_agent": "retrieval_agent",
            "rag_agent": "rag_agent",
            "ranking_agent": "ranking_agent",
            "END": END,
        },
    )

    for agent in ["preference_agent", "retrieval_agent", "rag_agent", "ranking_agent"]:
        workflow.add_edge(agent, "routing_node")

    logger.debug("推荐工作流构建完成")
    return workflow.compile()


def run_recommendation(user_query: str, catalog_path: str | Path = "data/movies.csv") -> str:
    graph = build_graph(catalog_path)
    result = graph.invoke(
        {
            "user_query": user_query,
            "preferences_summary": "",
            "candidate_pool": [],
            "snippets": [],
            "recommendations": "",
            "pending_agents": [],
            "next_agent": None,
            "route_reason": "",
        }
    )
    return result["recommendations"]
