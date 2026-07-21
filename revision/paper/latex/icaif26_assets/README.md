# ICAIF 2026 LaTeX Visual Assets

This directory contains paper-ready LaTeX assets for the frozen ABCE results.
The main-text package is intentionally limited to two figures and two tables.

| Asset | Question answered | Placement |
| --- | --- | --- |
| `figure_framework.tex` | What is observed, announced, and applied at each stage? | Main Figure 1 |
| `figure_abce_trace.tex` | How do the aligned and causal-control contracts differ on the preregistered representative path? | Main Figure 2 |
| `table_main_results.tex` | Do alignment, current revelation, quote-sensitive allocation, and allocation-family transfer hold? | Main Table 1 |
| `table_transfer_results.tex` | Does the exact risk-only method transfer to official MAPPO/IPPO? | Main Table 2 |
| `figure_paired_seeds.tex` | Which exact learner policy seeds reverse? | Supplement |
| `figure_sensitivity.tex` | Where does the earlier operating-region effect hold or reverse? | Supplement |
| `table_behavioral_boundary.tex` | What behavioral interpretation is supported? | Supplement or concise main-text paragraph |

## Integration

Add the packages and color definitions from `preamble.tex` to the main paper
preamble. Place the framework figure with the method overview, the two tables
with the confirmatory results, and the representative trace immediately after
the main table. `figure_abce_trace.tex` defines `\ABCEArtifactRoot`; override
that macro if the artifact directory has a different relative path in the
paper checkout.

The files are snippets, not standalone documents. `preview.tex` compiles all
assets together for layout QA under the ACM two-column class.

The ABCE values come from the sealed 20-policy, 10-path common panel. Positive
paired contrasts mean that the comparator has higher (worse) low-risk
execution half-spread than the target. The earlier adaptive-v5 panel remains
an independent replication and is never pooled with ABCE or the exact learner
holdout.
