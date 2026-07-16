# ACE-RL Anonymous Research Artifact

This repository is the double-blind artifact for the current controlled
ABIDES experiment package. It contains the final method implementation,
frozen configurations, paper-facing result tables and figures, audit data,
and tests needed to inspect the reported findings.

Development archives, superseded result versions, local outputs, source
repository history, author metadata, and internal repository commit IDs are
intentionally excluded.

## Supported Scope

In the tested controlled N=10 ABIDES dealer markets, simulator-supervised
public-risk timing with pre-quote-announced, post-quote-applied,
demand-conserving soft-rank routing improves low-risk customer execution over
matched announced-signal random timing under official full-policy MAPPO and in
two pre-frozen compact-adapter ecology perturbations. The effect is local to
the tested operating region and includes a retained low-dose reversal.

The evidence supports **high-spread coordination**, not punishment-supported
or learned collusion. It does not establish universal learner robustness,
literature-baseline superiority, larger-N robustness, cross-engine validity,
or real-market effectiveness.

## Headline Results

| Evaluation | Random minus risk-aware low-risk half-spread | Decision |
| --- | ---: | --- |
| Adaptive-v5 fresh panel, 20 policies | 0.06166 [0.05786, 0.06547], 20/20 positive | Supported |
| Official full-policy MAPPO, 10 policies | 0.18182 [0.04935, 0.31428], 8/10 positive | Supported |
| Official IPPO, 10 policies | 0.09397 [-0.13722, 0.32516], 7/10 positive | Inconclusive |
| E1 faster risk transitions | 0.03328 [0.02776, 0.03880], 10/10 positive | Supported |
| E2 weaker informed flow | 0.03984 [0.03358, 0.04610], 10/10 positive | Supported |

Intervals in this table are policy-seed t intervals. Official MAPPO is also
supported by the paired two-way bootstrap interval [0.07947, 0.30310]; its
exact sign test is not individually significant because two policy seeds
reverse. IPPO remains inconclusive under both interval estimators.

See [docs/RESULTS.md](docs/RESULTS.md) for the complete result inventory and
[docs/EXPERIMENT_STATUS.md](docs/EXPERIMENT_STATUS.md) for the frozen decision
boundaries.

## Repository Layout

- `revision/src/`: current ABIDES learner, gate, routing, and audit code.
- `revision/experiments/`: current runners, exporters, and source verifiers.
- `revision/configs/`: frozen final configurations and passive calibrations.
- `revision/paper/data/`: compact paper-facing CSV exports.
- `revision/paper/artifacts/`: primitive and derived audit data with source
  repository provenance metadata removed for double-blind review.
- `revision/paper/figures/`: paper-facing figures.
- `revision/paper/latex/icaif26_assets/`: current LaTeX/TikZ/PGFPlots visual
  assets and an ACM-layout preview source.
- `docs/protocols/`: frozen protocols for the retained experiment families.
- `scripts/verify_release.py`: anonymous release integrity and identity scan.

## Verification

```bash
python scripts/verify_release.py
python -m pytest -q revision/tests
```

The first command checks the release allowlist, SHA-256 hashes, CSV and JSONL
row counts, identity patterns, internal-commit redaction, and anonymous Git
author metadata. The tests exercise the current ABIDES implementation without
rerunning the long training jobs.

## Environment

The ABIDES integration is pinned in `revision/infra/abides/`. The external
ABIDES and official MAPPO source commits are retained because they identify
third-party dependencies rather than the anonymous authors. Full training uses
the Dockerized ABIDES matching engine and CPU workers; MPS is not required for
the reported final package.

Internal source-repository provenance is withheld only during double-blind
review. Dataset hashes, protocol fingerprints, model/path digests, external
dependency commits, and statistical outputs remain available in this artifact.
