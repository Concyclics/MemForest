# MemForest

Official implementation of **MemForest: An Efficient Agent Memory System with Hierarchical Temporal Indexing**.

## Paper

- VLDB submission (12 pages): [paper/MemForest_VLDB_submit.pdf](paper/MemForest_VLDB_submit.pdf)
- Extended version with appendix as supplemental material: [paper/MemForest_with_Appendix.pdf](paper/MemForest_with_Appendix.pdf)

## Overview

MemForest is a long-term conversation memory system for LLM agents. It extracts atomic facts from dialogue, organizes them in hierarchical memory trees (session, entity, scene), and serves them through a recall → plan → browse → rerank query pipeline.

Core ideas:
- **Three-tree forest per user**: session, entity, and scene trees provide complementary views over the same facts.
- **Three-phase build pipeline**: structure (no LLM) → summarize (LLM) → index (embedding), so each phase is independently restartable and cacheable.
- **Hybrid retrieval**: top-down planner-guided browsing combined with bottom-up vector recall, with iterative evidence checking.
- **Atomic fact normalization**: `On <UTC time>, The user/assistant said <single-clause description>.`

## Requirements

- Python 3.10+
- See [requirements.txt](requirements.txt) for the pinned dependency set (FastAPI, FAISS, OpenAI SDK, etc.).

```bash
python -m pip install -r requirements.txt
```

## Configuration

Default configurations live in [src/config/](src/config/):
- [src/config/default.yaml](src/config/default.yaml) — Qwen3-30B model setup
- [src/config/default_4b.yaml](src/config/default_4b.yaml) — Qwen3-4B model setup

Each config defines an OpenAI-compatible LLM endpoint, an embedding endpoint, and per-step overrides (extraction, tree summarization, query). To target your own model server, edit the `url`, `model_name`, and `key` fields, or override them via environment variables consumed by [src/config/config.py](src/config/config.py).

## Running

The library exposes a Python API; there is no separate HTTP server in `src/`.

```python
from src.forest.memforest import MemForest
from src.config.config import load_default_config

config = load_default_config()                 # or load_default_config("default_4b")
forest = MemForest("data/memforest", config=config)

forest.register_user("alice")
forest.ingest_session("alice", "sess_001", turns)   # turns: list[{role, content, timestamp?, content_id?}]

result = forest.query("alice", "What did Alice say about travel?")
for fact in result.top_facts:
    print(fact.text)

forest.save("alice")
```

Parallel multi-user operations:

```python
forest.ingest_parallel([
    {"user_id": "alice", "session_id": "s1", "turns": turns_a},
    {"user_id": "bob",   "session_id": "s1", "turns": turns_b},
])

forest.query_parallel([
    {"user_id": "alice", "question": "..."},
    {"user_id": "bob",   "question": "..."},
])
```

## Repository layout

```
src/
├── api/         OpenAI-compatible chat & embedding clients
├── build/       Tree construction, routing, and indexing
├── config/      Config dataclasses and default YAMLs
├── extraction/  Turn → atomic-fact extraction pipeline
├── forest/      Multi-user MemForest coordinator (public API)
├── logger/      Structured per-call and per-step logging
├── prompt/      Prompt templates for extraction, tree summary, answer
├── query/       Recall → plan → browse → rerank pipeline
└── utils/       Shared types, time, and text helpers
```

### Module overview

- [src/api/](src/api/) — Thin OpenAI-compatible wrappers ([src/api/client.py](src/api/client.py)) for chat completion (with JSON-mode) and embedding, plus optional API call logging.
- [src/build/](src/build/) — Three-phase tree builder ([src/build/tree_builder.py](src/build/tree_builder.py)), session/entity/scene routers ([src/build/scene_router.py](src/build/scene_router.py), [src/build/entity_router.py](src/build/entity_router.py)), summary cache ([src/build/summary_manager.py](src/build/summary_manager.py)), B+-style tree primitives ([src/build/tree.py](src/build/tree.py)), per-tree storage ([src/build/tree_store.py](src/build/tree_store.py)), and FAISS node index ([src/build/node_index.py](src/build/node_index.py)).
- [src/config/](src/config/) — [src/config/config.py](src/config/config.py) loads YAML into typed dataclasses ([src/config/api.py](src/config/api.py), [src/config/extraction_config.py](src/config/extraction_config.py), [src/config/tree_config.py](src/config/tree_config.py), [src/config/query_config.py](src/config/query_config.py), [src/config/logger_config.py](src/config/logger_config.py)).
- [src/extraction/](src/extraction/) — Per-turn chunking ([src/extraction/chunker.py](src/extraction/chunker.py)), extraction step orchestrator ([src/extraction/pipeline.py](src/extraction/pipeline.py), [src/extraction/runner.py](src/extraction/runner.py)), embedding-based deduplication ([src/extraction/dedup.py](src/extraction/dedup.py)), and the [src/extraction/fact_manager.py](src/extraction/fact_manager.py) that owns the FAISS-backed fact store.
- [src/forest/](src/forest/) — [src/forest/memforest.py](src/forest/memforest.py) is the public coordinator, [src/forest/user_forest.py](src/forest/user_forest.py) holds per-user state (facts, three trees, query pipeline), and [src/forest/forest_merge.py](src/forest/forest_merge.py) merges multiple sub-forests.
- [src/logger/](src/logger/) — [src/logger/api_log.py](src/logger/api_log.py) records per-API-call latency and tokens; [src/logger/extraction_log.py](src/logger/extraction_log.py) records extraction stage outcomes.
- [src/prompt/](src/prompt/) — Prompt builders for extraction ([src/prompt/extraction_prompt.py](src/prompt/extraction_prompt.py)), deduplication ([src/prompt/dedup_prompt.py](src/prompt/dedup_prompt.py)), tree summarization ([src/prompt/tree_prompts.py](src/prompt/tree_prompts.py)), and answer generation ([src/prompt/answer_prompts.py](src/prompt/answer_prompts.py)).
- [src/query/](src/query/) — End-to-end query pipeline ([src/query/pipeline.py](src/query/pipeline.py)) composed of recall ([src/query/retriever.py](src/query/retriever.py)), planner ([src/query/planner.py](src/query/planner.py)), browser ([src/query/browser.py](src/query/browser.py)), and reranker ([src/query/reranker.py](src/query/reranker.py)).
- [src/utils/](src/utils/) — Shared dataclasses ([src/utils/types.py](src/utils/types.py)), time helpers ([src/utils/time.py](src/utils/time.py)), text utilities ([src/utils/text.py](src/utils/text.py)).

## Data layout

Runtime artifacts are written under the `snapshot_dir` passed to `MemForest` (one subdirectory per user):

```
<snapshot_dir>/<user_id>/
├── sqlite/        fact and turn store
├── index/         FAISS indexes + sidecar metadata
├── trees/         serialized session/entity/scene trees
└── logs/          retrieval traces and per-step API logs
```

## Benchmark results

Per-question accuracy tables backing the numbers reported in the paper live under [benchmark/](benchmark/). Each row contains the question, gold answer, and the model answer plus LLM-judge verdict (`judge_1` … `judge_8`) for eight independent runs.

| Benchmark | Model size | File |
|---|---|---|
| LoCoMo | Qwen3-30B | [benchmark/locomo_per_question_30b.csv](benchmark/locomo_per_question_30b.csv) |
| LoCoMo | Qwen3-4B | [benchmark/locomo_per_question_4b.csv](benchmark/locomo_per_question_4b.csv) |
| LongMemEval-S | Qwen3-30B | [benchmark/longmemeval_per_question_30b.csv](benchmark/longmemeval_per_question_30b.csv) |
| LongMemEval-S | Qwen3-4B | [benchmark/longmemeval_per_question_4b.csv](benchmark/longmemeval_per_question_4b.csv) |

Columns: `method, model_size, question_type, qid, question, gold_answer, answer_{1..8}, judge_{1..8}`.
