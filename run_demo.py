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
    os.environ.setdefault("OPENAI_BASE_URL", "")
    os.environ.setdefault("LLMREC_MODEL", "qwen3:1.7b")
    catalog_path = Path("data/ctr_samples.csv")
    user_query = "想看AI行业深度解读，最好有简明的要点和趋势分析"
    print("=== 正在运行推荐图谱 ===")
    print(run_recommendation(user_query=user_query, catalog_path=catalog_path))
