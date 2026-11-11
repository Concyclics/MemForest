# Mem0 adapter

Base checkouts:

```bash
git clone https://github.com/EverMind-AI/EverMemOS.git
git -C EverMemOS checkout 539db77d5cc804c875246e34611fd266bf8c1e5d
git clone https://github.com/mem0ai/mem0.git
git -C mem0 checkout 4642a1d6e372e985fff9683070b0127bf6d77621
```

Install `aiohttp`, copy `mem0_adapter.py` to
`EverMemOS/evaluation/src/adapters/mem0_adapter.py`, copy
`mem0_local_qwen3.yaml` to `EverMemOS/evaluation/config/systems/`, and apply:

```bash
git -C EverMemOS apply \
  /path/to/MemForest/reproducibility/baselines/mem0/evermemos_reference_date.patch
```

Configure the local REST server with the selected generation and embedding
endpoints, a fresh Qdrant collection, and graph memory disabled. Run each
conversation in only one shard.

Store profile by retrieval budget (reads frozen Qdrant stores, no LLM calls):

```bash
python reproducibility/scripts/profile_mem0_retrieval_budget.py \
  --stores-root /path/to/mem0_run/stores \
  --dataset /path/to/longmemeval_s_cleaned.json \
  --output reproducibility/results/mem0_corrected/retrieval_budget_store_profile.json
```

Top-50/top-200 fixed-store control:
`reproducibility/scripts/run_mem0_budget_judge_control.py` (reads
`DEEPSEEK_API_KEY`).
