# Independent semantic audit

This directory releases the evidence-governance audits used by the VLDB
revision response. The original audit covers:

- 79 provisional temporal evidence mappings;
- 200 stratified entity-routing records;
- all 40 strict/public majority-label disagreements in the corrected Mem0
  top-50/top-200 control.

The semantic first pass uses
`Qwen3-30B-A3B-Instruct-2507-FP8` at temperature zero. Codex independently
checks joins, IDs, policy disagreements, and reporting boundaries. These files
are model-assisted review records, not author-verified human gold.

The `expanded/` directory freezes a larger sample selected before independent
review: 249 temporal mappings, 300 entity-routing facts, and 120 judge
calibration pairs. The judge sample contains all 40 policy disagreements plus
80 cutoff/label-stratified agreement controls.

## Headline audit decisions

- Only 33/79 provisional temporal rows were fully upheld. The revision omits
  exact semantic fragmentation/precision derived from this mapping unless the
  46 flagged rows receive author adjudication.
- Among 86 records with active entity keys, 84 pass semantic precision and 84
  cover at least one salient entity, but only five cover every salient entity.
  Entity trees are therefore reported as a selective overlay with fallback
  retrieval paths, not as complete entity routing.
- Of 40 judge disagreements, independent review prefers strict on 33 and public
  on seven. Eight subjective boundary cases remain separated for optional
  author sign-off. Since the sample is conditioned on disagreement, these
  ratios are not an unconditional judge-accuracy estimate.
- In the expanded audit, 162/249 temporal mappings satisfy the complete
  high-confidence exact-reproduction criterion, 124/127 active entity
  assignments pass precision, and 108/120 judge adjudications are stable
  without further author sign-off. The corresponding 87, 3, and 12 exceptions
  remain in explicit sign-off queues.

## Layout

- `semantic_audit_summary.json`: machine-readable aggregate counts;
- `*_independent_review.csv`: joined review records;
- `*_author_signoff_required.csv`: unresolved or subjective rows;
- `source/`: frozen source sheets;
- `raw/`: raw model outputs.
- `expanded/source/`: larger frozen source sheets and selection manifest;
- `expanded/raw/`: larger raw model outputs;
- `expanded/*_review_*.csv`: joined expanded reviews and sign-off queues.

To regenerate the joined records without API access:

```bash
python reproducibility/scripts/summarize_independent_semantic_audit.py
python reproducibility/scripts/summarize_expanded_semantic_audit.py
```

To rerun the semantic model pass, start the pinned Qwen endpoint at
`http://127.0.0.1:8001/v1` and run:

```bash
python reproducibility/scripts/run_independent_semantic_audit.py
python reproducibility/scripts/run_expanded_semantic_audit.py
```
