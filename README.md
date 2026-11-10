# Replication package: *Beyond Prediction: Interpretive Reliability in Computational Social Science*

Code, stored results and figures for the manuscript's analysis of the *Social Media and Mental Health* survey (primary sample: social-media users, N = 478; primary
specification: seven behavioural predictors + age + gender). The survey is used only as an empirical case for three methodological questions: does model flexibility improve
out-of-sample prediction, is apparent nonlinearity statistically supported, and are model-derived rankings stable under resampling. Nothing here supports causal or
population-level claims.

## What is in the repository

```
analysis.py            core analyses, tables and figures (the script to run)
simulations.py         OPTIONAL expensive simulations (selection-aware nulls, calibration checks, power); never run by analysis.py
requirements.txt       pinned package versions (Python 3.13)
data/README.md         where to get the data, how to place it, SHA-256 checksum (the data file itself is not included)
results/               stored numerical results behind every table and figure (JSON / CSV / NPZ)
results/simulations/   final saved simulation outputs, incl. all simulation replicates, and simulation_parameters.json
figures/               final manuscript figures (PNG 300 dpi, PDF; SVG for the two coefficient/importance plots)
docs/PROTOCOL.md       analytical protocol and the deviations log
docs/VERIFICATION.md   what was re-run during verification and what is only a previously saved output
docs/manuscript_numbers.csv   every number quoted in the manuscript Results/Abstract, with the stored result it comes from
results/PROVENANCE.csv        origin of every stored result file
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python analysis.py --saved        # about 10 seconds: rebuilds all tables and figures from results/ (no data file needed)
python analysis.py --check        # compares the stored results with the numbers printed in the manuscript (88 checks)
```
With the data file in place (`data/smmh.csv`, see `data/README.md`):
```bash
python analysis.py --quick        # a few minutes: whole pipeline with reduced replicates into quick_output/ (smoke test; numbers are not the reported ones)
python analysis.py                # full core analysis, regenerates results/ and figures/ (about 25 minutes on 2 cores)
python analysis.py --sensitivity  # adds the cross-validation for the sensitivity specifications S1, S3, S4 and the age 15-45 sample (about +50 minutes)
python simulations.py --stage all # OPTIONAL: regenerates results/simulations/ (about 2-2.5 hours on 2 cores); use --out DIR to keep the stored outputs untouched
```
All paths are relative to the location of the scripts; run them from any directory. `python analysis.py -h` and `python simulations.py -h` list stages and options.
Every random draw is seeded, so results do not depend on the number of cores; seeds and grids are written to `results/analysis_parameters.json` and
`results/simulations/simulation_parameters.json`.

## From manuscript to files

| Manuscript item | Script / stage | Files |
|---|---|---|
| Sample description (3.1), Figure A1 (correlations) | `analysis.py` audit, descriptives, tables, figures | `results/audit.json`, `descriptives.json`, `correlations_P.csv`, `figures/figA1_correlations.*` |
| OLS, HC3, VIF (3.1, 4.1); Figure 1 | ols, tables, figures | `results/ols.json`, `ols_figure_table.csv/.md`, `figures/fig1_ols_coefficients.*` |
| Nested CV, Table 1 (3.2) | cv, compare | `results/cv_S2_P.npz`, `cv_S2_P_summary.json`, `cv_S2_P_hp.json`, `compare.json`, `table1_cv_performance.csv`, `table2_cv_paired_differences.csv` |
| F-tests, partial dependence, Figure 2 (3.3) | nonlin | `results/nonlin.json`, `fig2_partial_dependence_data.csv`, `figures/fig2_partial_dependence.*` |
| Selection-aware nulls, Table 2, Figure 3a (3.3) | `simulations.py` nulls (stored) | `results/simulations/null_{ord,cont,disc}_B2000.json/.npz`, `results/fig3a_null_summary_from_saved_simulations.csv` |
| Calibration checks and power, Figure 3b (3.3) | `simulations.py` retune, size, power, bracket (stored) | `retune_null_B300.json`, `size_check.json`, `power.json`, `power_raw.csv`, `power_N478_bracket.*`, `results/fig3b_power_curves_from_saved_simulations.csv`, `figures/fig3_null_power.*` |
| First-pass nulls cited in Methods 2.7 | `simulations.py` initial (stored) | `results/simulations/initial_nulls_B300.json` |
| Ranking stability, Table 3, Figure 4, Supplementary Figure S1 (3.4) | stability, tables, figures | `results/stability.json`, `importance_boot_B500.npz`, `rf_importance_table.csv`, `rf_impurity_table.csv`, `table3_stability.csv`, `figures/fig4_rf_permutation_importance.*`, `figures/figS1_rf_impurity_importance.*` |

Figures included: the four main-text figures (1-4), Appendix Figure A1 and Supplementary Figure S1, exactly as numbered in the current manuscript draft.

## Stored outputs versus recomputed outputs

* `analysis.py` recomputes everything in the table above except the simulation results, which it only **reads** from `results/simulations/`.
* The files in `results/simulations/` are the final outputs of `simulations.py` as run during the analysis. Re-running that script is not needed to inspect or
  reuse them; `docs/VERIFICATION.md` states precisely which of them were independently regenerated in full or in part when the repository was assembled
  and which are previously saved outputs.
* Because replicate *i* is seeded by *i* alone, a shorter run reproduces the first replicates of the stored run exactly (this is how the stored outputs were checked).

## Notes on scope

* The cross-validated R² values (0.368, 0.362, 0.371) are for the primary specification; sensitivity specifications (`cv_S1_P`, `cv_S3_P`, `cv_S4_P`, `cv_S2_A1`) are stored and are recomputed only with `--sensitivity`.
* Not included: development scripts and intermediate outputs, superseded figures, and analyses that are not reported in the manuscript (see the Stage Four entry of `docs/PROTOCOL.md`).
* No licence file is included yet; add one of your choice before making the repository public. The survey data keep their own licence (see `data/README.md`).
