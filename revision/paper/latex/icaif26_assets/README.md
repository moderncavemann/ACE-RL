# ICAIF 2026 LaTeX Visual Assets

This directory contains paper-ready LaTeX assets for the current frozen ABIDES
results. The assets answer one scientific question each:

| Asset | Question answered | Main text? |
| --- | --- | --- |
| `figure_framework.tex` | What is observed, announced, and applied at each stage? | Yes, Figure 1 |
| `table_main_results.tex` | Does timing help, is it selective, and what does it cost? | Yes, Table 1 |
| `table_transfer_results.tex` | Does the effect transfer across learners/ecologies? | Yes, Table 2 |
| `figure_sensitivity.tex` | Where does the timing advantage hold or reverse? | Yes, Figure 2 |
| `figure_paired_seeds.tex` | Which MAPPO/IPPO policy seeds reverse? | Supplement |
| `table_behavioral_boundary.tex` | What behavioral interpretation is supported? | Main-text paragraph or supplement |

## Integration

Add the packages and color definitions from `preamble.tex` to the main paper
preamble, then place the relevant `\input{...}` at the intended location. The
framework figure should appear before Section 3.1 or immediately after its
opening paragraph. The two tables and sensitivity figure should be placed in
the corresponding Results subsections.

The files are snippets, not standalone documents. `preview.tex` compiles all
assets together for layout QA. It uses the ACM two-column class but is not a
replacement for the paper source.

All values are frozen to the committed paper CSV/JSON artifacts. Positive
paired contrasts mean that the comparator has higher (worse) low-risk
execution half-spread than risk-aware timing.
