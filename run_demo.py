"""Run a quick end-to-end demonstration of the LangGraph recommender."""

import sys
from pathlib import Path

# Allow running directly from the repository root without installation.
sys.path.append(str(Path(__file__).parent / "src"))

from llmrec.graph import run_recommendation


if __name__ == "__main__":
    catalog_path = Path("data/movies.csv")
    user_query = "我喜欢太空和冒险题材，但也想看到角色之间有情感联系"
    print("=== Running graph ===")
    print(run_recommendation(user_query=user_query, catalog_path=catalog_path))
