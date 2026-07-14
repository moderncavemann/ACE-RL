# ABIDES Audit Bundle v1 Contract

Status: schema frozen before runner implementation  
Date: 2026-07-14

The committed audit bundle is the sole source for paper-facing domain
benchmark numbers. It contains `manifest.json`, `checks.json`, `per_path.csv`,
`per_seed.csv`, `step_records.jsonl.gz`, `README.md`, and `SHA256SUMS`.

Per-path rows contain protocol, policy/evaluation seeds, method, low/high/all
execution half-spread, mean and episode-unit dealer reward, low/all customer
cost, activation, high-risk activation, signal strength, routing TV, flow HHI,
inventory dispersion, volume, low/high step counts, model digest, and path
digest. Per-seed rows are derived only from this table.

Step records contain seed/method/step identity, pre-state digest, actions,
announced signal or mechanism parameters, threshold and eligible dealers,
routing weights or realized shares, post-hoc latent risk, execution spread,
rewards, inventory summary, and action-uniform digest. The exporter must not
include checkpoints or private host paths.

The verifier recomputes method means, paired seed differences, t-based 95%
CI, exact sign tests, Holm-adjusted secondary p-values, paired two-way
bootstrap intervals, budget matching, conservation, all integrity checks, and
paper table/figure inputs. Bootstrap uses 10,000 replicates and seed
`20260714`.

The manifest records protocol/fingerprint and freeze/execution/archive
commits, source fingerprints/model digests, ABIDES and Docker provenance,
software versions, all namespaces, UTC boundaries, completion/failures,
commands, git dirtiness, file hashes, and completed-resume verification.
Changing any covered file must fail checksum verification.

CI runs tests and verification/export checks against committed artifacts. It
does not launch formal ABIDES training.
