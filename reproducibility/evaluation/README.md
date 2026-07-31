# Shared evaluation code

`unified_deepseek_judge.py` is the shared answer collector and DeepSeek judge
used for the retained strict-prompt and sensitivity diagnostics. It separates
answer generation from judging, accepts an explicit result root, supports
LongMemEval and LoCoMo, and writes per-question labels plus summaries.

```bash
export DEEPSEEK_API_KEY=...
python reproducibility/evaluation/unified_deepseek_judge.py \
  --root /path/to/completed/baseline/runs \
  --out-dir /path/to/judged-output \
  --judge-url https://api.deepseek.com/v1 \
  --judge-model deepseek-chat \
  --prompt-version appendix \
  --benchmarks longmemeval locomo \
  --workers 32
```

The judge key is read from `DEEPSEEK_API_KEY` unless `--judge-api-key` is
provided. Released judged records contain no credentials. The prompt version
used by each result family is recorded in its adjacent prompt manifest.

The revised main tables instead use one `deepseek-v4-flash` call per frozen
answer with the released Mem0 benchmark prompts. Their exact prompt commits,
hashes, counts, and compact labels are frozen in
`../results/public_judge_three_backbone/`; do not use the example strict command
above to regenerate the main-table scale.
