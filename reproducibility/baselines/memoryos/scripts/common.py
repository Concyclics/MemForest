"""Shared utilities for the mem0 (append-only) vs LightMem comparison."""

from __future__ import annotations

import functools
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI


# ---------- model discovery ----------

def normalize_base_url(base_url: str) -> str:
    s = base_url.rstrip("/")
    return s if s.endswith("/v1") else f"{s}/v1"


def discover_model(base_url: str) -> str:
    base_url = normalize_base_url(base_url)
    r = httpx.get(f"{base_url}/models", timeout=30.0)
    r.raise_for_status()
    data = r.json().get("data") or []
    if not data:
        raise RuntimeError(f"no models at {base_url}/models")
    return data[0]["id"]


# ---------- LongMemEval judge prompt ----------

def get_anscheck_prompt(task: str, question: str, answer: str, response: str, abstention: bool = False) -> str:
    if not abstention:
        if task in ("single-session-user", "single-session-assistant", "multi-session"):
            t = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if "
                 "the response contains the correct answer. Otherwise, answer no. If the response is equivalent to "
                 "the correct answer or contains all the intermediate steps to get the correct answer, you should "
                 "also answer yes. If the response only contains a subset of the information required by the answer, "
                 "answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response "
                 "correct? Answer yes or no only.")
        elif task == "temporal-reasoning":
            t = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
                 "response contains the correct answer. Otherwise, answer no. If the response is equivalent to the "
                 "correct answer or contains all the intermediate steps to get the correct answer, you should also "
                 "answer yes. If the response only contains a subset of the information required by the answer, "
                 "answer no. In addition, do not penalize off-by-one errors for the number of days. If the question "
                 "asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., "
                 "predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}"
                 "\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.")
        elif task == "knowledge-update":
            t = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
                 "response contains the correct answer. Otherwise, answer no. If the response contains some previous "
                 "information along with an updated answer, the response should be considered as correct as long as "
                 "the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response:"
                 " {}\n\nIs the model response correct? Answer yes or no only.")
        elif task == "single-session-preference":
            t = ("I will give you a question, a rubric for desired personalized response, and a response from a "
                 "model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The "
                 "model does not need to reflect all the points in the rubric. The response is correct as long as it "
                 "recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\n"
                 "Model Response: {}\n\nIs the model response correct? Answer yes or no only.")
        else:
            raise NotImplementedError(task)
        return t.format(question, answer, response)
    t = ("I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if "
         "the model correctly identifies the question as unanswerable. The model could say that the information is "
         "incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\n"
         "Explanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? "
         "Answer yes or no only.")
    return t.format(question, answer, response)


def true_or_false(response: Optional[str]) -> bool:
    if response is None:
        return False
    n = str(response).strip().lower()
    if not n:
        return False
    first = n.splitlines()[0].strip()
    tokens = first.replace(".", "").replace("!", "").replace(":", "").replace(";", "").split()
    if not tokens:
        return False
    head = tokens[0]
    if head in ("yes", "y"):
        return True
    if head in ("no", "n"):
        return False
    if "yes" in first:
        return True
    if "no" in first:
        return False
    return False


# ---------- DeepSeek judge ----------

class DeepSeekJudge:
    def __init__(self, model: str = "deepseek-v4-pro", api_key: Optional[str] = None,
                 base_url: str = "https://api.deepseek.com"):
        key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY not set")
        self.model = model
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.base_url = base_url

    def judge(self, prompt_text: str, max_retries: int = 3) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": "You are a strict grader. Answer with yes or no only."},
            {"role": "user", "content": prompt_text},
        ]
        last_err = None
        for attempt in range(max_retries):
            t0 = time.perf_counter()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=64,
                    temperature=0.0,
                    stream=False,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                dur = time.perf_counter() - t0
                u = getattr(resp, "usage", None)
                return {
                    "text": resp.choices[0].message.content,
                    "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
                    "duration_seconds": dur,
                    "messages": messages,
                }
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"judge failed after {max_retries} retries: {last_err}")


# ---------- OpenAI usage tracking via monkey-patch ----------

@dataclass
class UsageBucket:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    llm_calls: int = 0
    embedding_calls: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "embedding_tokens": self.embedding_tokens,
            "llm_calls": self.llm_calls,
            "embedding_calls": self.embedding_calls,
        }


class OpenAIUsageTracker:
    """Context manager that monkey-patches OpenAI SDK to accumulate usage.

    Usage counters are global within the patch lifetime and additive across nested 'with' uses.
    Use a fresh tracker per stage; bucket.to_dict() captures the totals for that stage.
    """
    _patch_lock = threading.Lock()
    _depth = 0
    _orig_chat_create = None
    _orig_embeddings_create = None
    _active_buckets: List[UsageBucket] = []

    def __init__(self):
        self.bucket = UsageBucket()

    def __enter__(self) -> UsageBucket:
        with OpenAIUsageTracker._patch_lock:
            if OpenAIUsageTracker._depth == 0:
                self._install_patches()
            OpenAIUsageTracker._depth += 1
            OpenAIUsageTracker._active_buckets.append(self.bucket)
        return self.bucket

    def __exit__(self, exc_type, exc, tb):
        with OpenAIUsageTracker._patch_lock:
            try:
                OpenAIUsageTracker._active_buckets.remove(self.bucket)
            except ValueError:
                pass
            OpenAIUsageTracker._depth -= 1
            if OpenAIUsageTracker._depth == 0:
                self._uninstall_patches()

    @classmethod
    def _install_patches(cls):
        from openai.resources.chat.completions import Completions as _Completions
        from openai.resources.embeddings import Embeddings as _Embeddings

        cls._orig_chat_create = _Completions.create
        cls._orig_embeddings_create = _Embeddings.create

        @functools.wraps(cls._orig_chat_create)
        def chat_create(self, *args, **kwargs):
            resp = cls._orig_chat_create(self, *args, **kwargs)
            try:
                u = getattr(resp, "usage", None)
                if u is not None:
                    pt = int(getattr(u, "prompt_tokens", 0) or 0)
                    ct = int(getattr(u, "completion_tokens", 0) or 0)
                    for b in cls._active_buckets:
                        b.prompt_tokens += pt
                        b.completion_tokens += ct
                        b.llm_calls += 1
            except Exception:
                pass
            return resp

        @functools.wraps(cls._orig_embeddings_create)
        def emb_create(self, *args, **kwargs):
            resp = cls._orig_embeddings_create(self, *args, **kwargs)
            try:
                u = getattr(resp, "usage", None)
                tok = 0
                if u is not None:
                    tok = int(getattr(u, "prompt_tokens", 0) or getattr(u, "total_tokens", 0) or 0)
                for b in cls._active_buckets:
                    b.embedding_tokens += tok
                    b.embedding_calls += 1
            except Exception:
                pass
            return resp

        _Completions.create = chat_create
        _Embeddings.create = emb_create

    @classmethod
    def _uninstall_patches(cls):
        from openai.resources.chat.completions import Completions as _Completions
        from openai.resources.embeddings import Embeddings as _Embeddings
        if cls._orig_chat_create is not None:
            _Completions.create = cls._orig_chat_create
        if cls._orig_embeddings_create is not None:
            _Embeddings.create = cls._orig_embeddings_create
        cls._orig_chat_create = None
        cls._orig_embeddings_create = None


# ---------- dataset loading ----------

def load_multisession_questions(data_path: Path) -> List[Dict[str, Any]]:
    with open(data_path, "r") as f:
        data = json.load(f)
    return [d for d in data if d.get("question_type") == "multi-session"]


# ---------- io helpers ----------

def save_json(path: Path, payload: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


# ---------- answer prompt ----------

ANSWER_SYSTEM_PROMPT = "You are a helpful assistant."


def build_answer_messages(question: str, question_date: str, formatted_memories: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question time:{question_date} and question:{question}\n"
                f"Please answer the question based on the following memories: {formatted_memories}"
            ),
        },
    ]


def call_chat(client: OpenAI, model: str, messages: List[Dict[str, str]], *, max_tokens: int = 1024,
              temperature: float = 0.0) -> Dict[str, Any]:
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False,
    )
    dur = time.perf_counter() - t0
    u = getattr(resp, "usage", None)
    return {
        "text": resp.choices[0].message.content,
        "prompt_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(u, "completion_tokens", 0) or 0),
        "duration_seconds": dur,
    }
