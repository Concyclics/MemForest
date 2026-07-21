# VLDB revision reproducibility package

This directory contains the public evidence added for the VLDB revision. The
conference uses single-blind review, so the response links directly to this
repository rather than to a separate anonymized archive.

## Contents

- [`BASELINES.md`](BASELINES.md): pinned upstream versions, local deployment
  boundaries, timestamp policy, retrieval budget, and reproduction procedure.
- [`baselines/`](baselines/): pinned code overlays, configs, licenses, and
  portable runners for every main/revision baseline.
- [`evaluation/`](evaluation/): the frozen shared DeepSeek judge implementation.
- [`PROTOCOL.md`](PROTOCOL.md): datasets, model roles, answer generation, judge,
  pass@k, and reporting policy.
- [`RESULTS.md`](RESULTS.md): revision headline results and interpretation.
- [`manifests/`](manifests/): machine-readable source and release metadata.
- [`results/`](results/): compact summaries and sanitized per-question records.
- [`scripts/verify_release.py`](scripts/verify_release.py): offline completeness,
  count, and headline-result checks.
- [`SHA256SUMS`](SHA256SUMS): checksums for every released revision file.

The original submission's four large per-question CSV files remain in
[`../benchmark/`](../benchmark/). They are not silently overwritten. Corrected
or newly added revision cells are versioned in this directory.

## Quick verification

From the repository root:

```bash
python reproducibility/scripts/verify_release.py
```

The check is API-free. It verifies expected question counts, judge-error counts,
and the headline values cited by the response. Re-running answer generation or
LLM judging requires the external baseline repositories and model services
listed in [`BASELINES.md`](BASELINES.md).

Every baseline implementation is indexed in
[`baselines/README.md`](baselines/README.md). We publish complete patches
relative to pinned upstream commits instead of mutable copies of third-party
repositories.

## Data policy

Per-question release files retain the benchmark question, gold answer,
generated answer, model identifier, judge label, and prompt version. Local
absolute paths, service URLs, credentials, and database directories are
removed. No API key is stored in this repository.
