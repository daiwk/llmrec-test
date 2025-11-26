"""LangGraph pipeline for LLM-based recommendations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

from .llm import LocalRecommenderLLM, make_llm


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


def load_catalog(path: str | Path) -> List[Movie]:
    df = pd.read_csv(path)
    return [Movie.from_row(row) for _, row in df.iterrows()]


def preference_agent(llm: LocalRecommenderLLM) -> Callable[[RecommendationState], RecommendationState]:
    def _inner(state: RecommendationState) -> RecommendationState:
        prompt = (
            "You are a personalization strategist. Summarize the viewer profile based on their goal.\n"
            f"User request: {state['user_query']}\n"
            "Focus on tone, genres, pacing, and emotional intent. Keep it to two sentences."
        )
        summary = llm.complete(prompt)
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
        return {"candidate_pool": top_movies}

    return _inner


def rag_agent(catalog: List[Movie]) -> Callable[[RecommendationState], RecommendationState]:
    def _inner(state: RecommendationState) -> RecommendationState:
        snippets: List[str] = []
        for movie in state["candidate_pool"][:3]:
            snippet = f"{movie.title}: {movie.plot}"
            snippets.append(snippet)
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
            "You are the master agent blending signals from other agents.\n"
            "Blend the following signals to pick 3 recommendations.\n"
            f"Viewer preferences: {state['preferences_summary']}\n"
            "Context snippets:\n"
            + "\n".join(f"Snippet: {snippet}" for snippet in state["snippets"])
            + "\nCandidate pool with scores (higher is better):\n"
            + "\n".join(scored_candidates)
            + "\nReturn friendly bullet points."
        )
        recs = llm.complete(prompt)
        return {"recommendations": recs}

    return _inner


def build_graph(catalog_path: str | Path, llm: LocalRecommenderLLM | None = None):
    llm = llm or make_llm()
    catalog = load_catalog(catalog_path)

    workflow = StateGraph(RecommendationState)
    workflow.add_node("preference_agent", preference_agent(llm))
    workflow.add_node("retrieval_agent", retrieval_agent(catalog))
    workflow.add_node("rag_agent", rag_agent(catalog))
    workflow.add_node("ranking_agent", ranking_agent(llm))

    workflow.set_entry_point("preference_agent")
    workflow.add_edge("preference_agent", "retrieval_agent")
    workflow.add_edge("retrieval_agent", "rag_agent")
    workflow.add_edge("rag_agent", "ranking_agent")
    workflow.add_edge("ranking_agent", END)

    return workflow.compile()


def run_recommendation(user_query: str, catalog_path: str | Path = "data/movies.csv") -> str:
    graph = build_graph(catalog_path)
    result = graph.invoke({"user_query": user_query, "preferences_summary": "", "candidate_pool": [], "snippets": [], "recommendations": ""})
    return result["recommendations"]
