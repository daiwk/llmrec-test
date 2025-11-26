"""Run a quick end-to-end demonstration of the LangGraph recommender."""

import logging
import sys
import os
from pathlib import Path

# Allow running directly from the repository root without installation.
sys.path.append(str(Path(__file__).parent / "src"))

from llmrec.graph import run_recommendation


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    os.environ.setdefault("OPENAI_API_KEY", "")
    os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
    os.environ.setdefault("LLMREC_MODEL", "qwen3:1.7b")
    catalog_path = Path("data/movies.csv")
    user_query = "我喜欢太空和冒险题材，但也想看到角色之间有情感联系"
    print("=== 正在运行推荐图谱 ===")
    print(run_recommendation(user_query=user_query, catalog_path=catalog_path))
