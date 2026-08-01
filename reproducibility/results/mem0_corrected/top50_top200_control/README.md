# Corrected Mem0 retrieval-budget x judge control

This directory publishes the aggregate and per-question majority-label outputs
for the fixed-store Qwen3-30B LongMemEval-S control at top-50 and top-200.

The same 500 corrected Mem0 stores are queried once to top-200 and sliced at
top-50. Answers use the same Mem0-specific answer prompt. Every answer receives
three `deepseek-chat` votes under the released strict prompt
(`prompt_version=appendix`) and the
public LongMemEval prompt from `mem0ai/memory-benchmarks` commit `7ba1bd3`.

The local control confirms judge sensitivity but does not reproduce the managed
v3 snapshot's positive top-50-to-top-200 delta. It must not be presented as a
same-protocol reproduction of the managed 90%+ score.
