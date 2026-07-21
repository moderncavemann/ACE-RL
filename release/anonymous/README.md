# ACE-RL Anonymous Research Artifact

This repository is the double-blind artifact for the frozen controlled
ABIDES Contract Evaluation (ABCE) study. It contains the current method,
formal configurations, paper-facing tables and figures, primitive audit data,
and deterministic verifiers for the reported findings.

Development history, superseded result versions, local outputs, author
metadata, and internal source-repository commit IDs are intentionally
excluded.

## Supported Scope

In a controlled N=10 ABIDES dealer market, aligning a fixed announced-contract
multiset with lagged public market state improves low-risk customer execution
relative to exact-multiset random, circularly shifted, and periodic schedules
after dealer adaptation. Timely current-contract revelation outperforms
one-step-lagged revelation, and quote-aligned allocation outperforms
identity-permuted redistribution. The alignment effect also appears under the
tested hard-best allocation contract and has a positive policy-seed mean effect
under the tested exact risk-only official MAPPO implementation. IPPO remains
inconclusive.

The method deliberately does not minimize all-state raw spread. The evidence
supports **high-spread coordination**, not punishment-supported or learned
collusion. It does not establish universal learner or population robustness,
superiority over original literature mechanisms, cross-engine validity, or
real-market effectiveness.

## Headline Results

Positive contrasts mean that the comparator has worse low-risk customer
execution than the target.

| Claim | Paired contrast | Mean [95% policy-seed t-CI] | Positive |
| --- | --- | ---: | ---: |
| A: state alignment | exact random minus aligned | 0.03996 [0.03632, 0.04359] | 20/20 |
| A: displaced timing | circular minus aligned | 0.03855 [0.03557, 0.04153] | 20/20 |
| A: state-independent schedule | periodic minus aligned | 0.06001 [0.05686, 0.06317] | 20/20 |
| B: timely revelation | lagged minus current | 0.03389 [0.03073, 0.03705] | 20/20 |
| C1: quote-sensitive allocation | identity-permuted minus aligned | 0.22426 [0.21813, 0.23039] | 20/20 |
| C2: allocation-family transfer | hard-best random minus aligned | 0.04077 [0.03712, 0.04442] | 20/20 |

The exact risk-only official MAPPO holdout is supported: mean 0.11386,
95% t-CI [0.03351, 0.19420], 9/10 policy seeds positive. Official IPPO has a
positive mean of 0.12352 but remains inconclusive because its primary t-CI
[-0.01658, 0.26363] crosses zero. The earlier adaptive-v5 fresh-panel effect,
0.06166 [0.05786, 0.06547] with 20/20 positive populations, is retained as an
independent replication and is not pooled with ABCE.

See [docs/RESULTS.md](docs/RESULTS.md) for the complete result inventory and
[docs/EXPERIMENT_STATUS.md](docs/EXPERIMENT_STATUS.md) for the frozen decision
boundaries.

## Repository Layout

- `revision/src/`: current ABIDES learner, contract, routing, and audit code.
- `revision/experiments/`: current runners, exporters, and bundle verifiers.
- `revision/configs/`: frozen final configurations and passive calibrations.
- `revision/paper/data/`: compact paper-facing CSV exports.
- `revision/paper/artifacts/`: primitive and derived audit data with internal
  source-repository provenance removed for double-blind review.
- `revision/paper/latex/icaif26_assets/`: current LaTeX/TikZ/PGFPlots assets.
- `docs/protocols/`: frozen protocols for retained experiment families.
- `scripts/verify_release.py`: anonymous-release integrity and identity scan.

## Verification

The primary compressed trace is stored with Git LFS. Fetch it before running
the release verifier.

```bash
git lfs install
git lfs pull
python scripts/verify_release.py
python -m pytest -q revision/tests
```

The verifier checks the allowlist, SHA-256 hashes, CSV and JSONL row counts,
identity patterns, internal-commit redaction, and anonymous Git metadata. The
tests exercise the retained implementation without rerunning long training.

## Environment

The ABIDES integration is pinned in `revision/infra/abides/`. The external
ABIDES and official MAPPO pins remain visible because they identify third-party
dependencies, not the anonymous authors. Full training uses the Dockerized
ABIDES matching engine and CPU workers; MPS is not required.

Internal source-repository provenance is withheld only during double-blind
review. Protocol fingerprints, model/path digests, external dependency pins,
primitive rows, and statistical outputs remain in the artifact.
