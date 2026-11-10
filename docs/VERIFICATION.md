# Verification report

Verification was done on 10 November 2026 in a clean Linux container (Python 3.13, numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, scikit-learn 1.9.1, matplotlib 3.11.2, joblib 1.6.0; 2 cores).
Method: the repository was copied to an empty folder, the data file was copied in, and scripts were run with relative paths only. Because replicate *i* is seeded by *i*,
a short run must equal the first replicates of the stored full run; "identical" below means array-equal (bitwise) unless stated.

## 1. Numbers in the manuscript (V6 draft) against stored results
`python analysis.py --check` compares 88 numbers quoted in the Abstract, Methods 2.5-2.7, Results 3.1-3.4, Tables 1-3 and Section 4.1 with the stored results
(`docs/manuscript_numbers.csv`; `results/manuscript_check.csv` is written by the check). **88 of 88 reproduce at the precision printed** (e.g. R² 0.368/0.362/0.371; p = 0.089/0.080/0.048; 0.063 after retuning; 91%/78%/57% and 92%/56%/22% top-three recovery).
Not covered by the check: the Introduction, Discussion and the qualitative Table 4, which contain no new numbers beyond those above, and the bibliography (not yet updated).
One wording point (not a numerical error): the power ranges in 3.3 (11-22%, 49-63%, 82-86%) span the three nulls *and* the critical value from the power simulation's own linear scenario; see Stage Four (vi) of `PROTOCOL.md`.

## 2. Recomputed from the data and identical to the stored outputs
* Audit (481 rows, 478 users, SHA-256 verified), descriptives, OLS for all samples/specifications: identical (the stored `ols.json` additionally had three development-only keys, removed).
* Nested cross-validation, primary specification S2, **all 10 repeats**: predictions, tuned hyperparameters and summaries identical (12 min).
* Paired comparison (`compare.json`): identical.
* Nonlinearity stage with B = 300 bootstrap forests (F-tests, observed partial dependence, G/D statistics, 90% bands): identical (the new file only adds a `B` field).
* Ranking stability, **all 500 bootstrap resamples and 300 split-halves**: `stability.json` and the raw bootstrap arrays identical.
* First-pass B = 300 nulls (p = 0.106 continuous, 0.047 rounded-and-clipped): identical, all replicates.
* Tables, importance tables and all six figures rebuilt from stored results; CSV tables byte-identical to the originals; figures pixel-identical except Figure 1 (antialiasing, max 11/255).
* `python analysis.py --quick` (whole pipeline, one CV repeat, 20 bootstraps) ran end to end in about 3 minutes; its first-replicate outputs equal the stored ones.

## 3. Simulation outputs: previously saved, checked by partial regeneration
The full simulations were **not** rerun (about 2.5 hours). The files in `results/simulations/` are the outputs of the development scripts that `simulations.py` consolidates. Checked with `simulations.py --quick --stage all` (3 minutes):

| Output | Regenerated | Result |
|---|---|---|
| `null_{ord,disc,cont}_B2000` | replicates 1-20 of 2,000 per null | identical (D and G statistics) |
| `null_retuned_B300`, `retune_null_B300` | replicates 1-10 of 300; observed data re-tuned under 8 CV seeds (complete) | identical; matched fixed-hyperparameter replicates equal the development partial dependence curves |
| `size_check` | 3 of 400 replicates per generating process | identical; stored json re-aggregates exactly from the stored raw CSV |
| `power` | 3 of 120 replicates in each of 18 N x scenario cells (54 replicates) | identical; stored json re-aggregates from the raw CSV (to 1e-16) |
| `power_N478_bracket` | complete (aggregation only) | identical |

The remaining replicates were not regenerated; they come from the same code path and seeds, but this was not independently confirmed for them.

## 4. Sensitivity cross-validation
`cv_S1_P`, `cv_S3_P`, `cv_S4_P`, `cv_S2_A1`: repeat 1 of 10 regenerated and identical; repeats 2-10 are previously saved outputs (about 50 minutes to recompute with `--sensitivity`).

## 5. What could not be verified
* **Fresh installation.** The container has no network, so `pip install -r requirements.txt` could not be tested; runs used the pre-installed packages at exactly the pinned versions. Other versions may change the last decimals of tree-based results.
* **Operating systems and core counts.** Only Linux with 2 cores was used (results are seeded and independent of core count by construction, but this was not tested on other machines).
* **Data licence and source address.** The Kaggle page, its licence and the dataset's redistribution terms could not be checked from this environment, so the data file is not included; the checksum is that of the file analysed.
* **The full simulations and sensitivity CV repeats noted above.**
* **Deleted development material.** The development-only analyses listed in `PROTOCOL.md` (Stage Four iii) are no longer reproducible from this repository.
