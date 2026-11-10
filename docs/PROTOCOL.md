# Analytical protocol and deviations log

This is the protocol written before the new analyses were run (Stage One), followed by its deviations log. The protocol text is reproduced unchanged from the
development package; it refers to the development scripts (`scripts/01`–`17`, `common.py`), which are not shipped. The final entry of the log (Stage Four)
maps those scripts to `analysis.py` and `simulations.py` and records where the protocol as written differs from what was executed. Nothing in
Stage Four changes a number.

# Stage One protocol (fixed before the new analyses were run)

Date written: 2026-11-09. Dataset: `data/smmh.csv` (Kaggle "Social Media and Mental Health"), SHA-256 5a6ae636760314616754b83c6e4626ea55247f7586e6b5e17ddcdff80ca11fa3.
Software: Python 3.13, numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, scikit-learn 1.9.1. (No statsmodels: OLS, HC3, VIF, Cook's D, tests are implemented directly and cross-checked.)

Status of earlier work: my earlier runs in this project were exploratory. Nothing below is tuned to them. Where a choice below coincides with an earlier one, that is stated as a decision with a reason, not inherited.

## 1. Samples
- PRIMARY sample (P): respondents who answer "Yes" to Q6 (use social media). Reason: every engagement item describes social-media use; non-users' answers to those items have no stable meaning.
- Sensitivity samples: A0 = all rows; A1 = age 15-45 (the restriction the manuscript states); A2 = P minus implausible records, rule fixed now: age > 80, or "School Student" with age > 25.
- Exclusions are logged one by one in `results/audit_*.json`.

## 2. Variables
- Outcome y = Q18 "feel depressed or down", integer 1-5, analysed as numeric (ordinal-outcome sensitivity is listed as a limitation, not run).
- Behavioural predictors (the manuscript's seven): time (Q8 mapped to hours: <1h 0.5, 1-2 1.5, 2-3 2.5, 3-4 3.5, 4-5 4.5, >5h 6.0), distractibility (Q12 single item), restless (Q11), validation (Q17), compare (Q15), interest (Q19), sleep (Q20).
- Extended items for sensitivity: purpose (Q9), distract_busy (Q10), concentrate (Q14), worries (Q13).
- Covariates: age (years, as given), gender (Female ref; Male; Other = every other entry), relationship (Single ref; In a relationship; Married/Divorced), occupation (University student ref; School student; Other = Salaried + Retired). Collapsing is fixed now because the Divorced (n=7), Retired (n=8) and Other-gender (n=7) cells are too sparse to estimate.

## 3. Specifications
- S1 (reference, = manuscript set): seven behavioural predictors.
- S2 (PRIMARY): S1 + age + gender. Reason: age and gender are fixed before, and plausibly confound, both engagement and mood. Relationship and occupation are largely functions of age/life stage and add sparse dummies, so they are sensitivity only.
- S3: S2 + four extended items (construct-overlap sensitivity).
- S4: S2 + relationship + occupation.
- Core predictors for the nonlinearity and PDP questions: interest, compare, sleep (the manuscript's three). Primary multiplicity family = these three (Holm). Secondary family = all seven (Holm over seven).

## 4. Models (all preprocessing inside the training fold)
- OLS (no tuning).
- Additive spline regression: quadratic B-splines on the seven behavioural features (knots n in {3,4,5}, uniform), covariates linear, ridge penalty alpha in {0.1, 1, 10, 100}; standardisation and spline basis fitted on training data only.
- Random Forest: 300 trees, min_samples_leaf in {1,3,5,10,20}, max_features in {0.33, 0.6, 1.0}; tuned by inner 5-fold CV on training data only.

## 5. Predictive evaluation (question 1)
- Repeated nested CV: outer 10-fold x 10 repeats (shuffle seeds 1000+r), inner 5-fold grid search (MSE), tuning and preprocessing inside the training fold.
- Primary metric: pooled out-of-sample R2 = 1 - sum((y - yhat)^2) / sum((y - ybar_train)^2), reference = the training-fold mean. Also RMSE and per-fold R2.
- Uncertainty: SD across repeats = partition variance only. Model differences are tested with a respondent-level bootstrap (B=2000) of repeat-averaged squared-error differences, which captures sampling variation.
- Record selected hyperparameters in every outer fold.

## 6. Diagnostics (OLS, S2 primary; S1 reference)
HC3 standard errors; VIF from regressions with an intercept; condition number; leverage (> 2p/n), studentised residuals (|r| > 3), Cook's D (> 4/n); Breusch-Pagan; Shapiro-Wilk; Ramsey RESET. Influence sensitivity: refit without the 10 largest Cook's D.

## 7. Nonlinear evidence (question 2)
- T1: categorical-vs-linear F-test per predictor (other terms and covariates linear). T2: quadratic term (HC3). T3: RESET.
- T4: Random Forest partial dependence (grid = observed levels 1-5; averaging over the empirical distribution of all other features including covariates). Statistic D_j = RMS deviation of the 5-point PDP from its own best straight line. Forest tuned on the full sample by 5-fold CV (seed 20261109).
- Linear-null calibration: B=300 datasets from the fitted S2 OLS plus residuals resampled with replacement (N1, continuous) and the same rounded and clipped to 1-5 (N2, discretised, matches the observed outcome). Same forest settings, same statistic. p = (1 + #null >= obs)/(B + 1), Holm over the three core predictors.
- PDP uncertainty: 300 respondent bootstraps, fixed hyperparameters, 90% percentile band per level.

## 8. Interpretive stability (question 3)
- Importance measures, named explicitly: OLS absolute standardised coefficient; RF permutation importance (increase in MSE when a feature is permuted, computed on held-out respondents, mean of 3 permutations); RF impurity (MDI) importance, included only to reproduce the manuscript's Figure 5. Ranking is over the seven behavioural predictors.
- Bootstrap (B=500): Kendall tau_b of each resample's ranking vs the full-sample ranking by the same method; identity of the top three; 5-95% rank intervals. RF permutation importance uses the out-of-bag respondents of each resample.
- Split-half (B=300): importance fitted on half A and evaluated on half B, and the reverse; tau between the two rankings (OLS betas compared directly between halves).
- Separate sources of variation: (a) seed: 50 seeds, fixed data and hyperparameters; (b) CV partition: the repeats in section 5; (c) respondent sampling: the bootstrap; (d) hyperparameter selection: rankings under the five most frequently selected configurations; (e) specification: S1-S4 and time-coding variants. Report each as SD of R2 and as mean tau.

## 9. Positive controls
Truth = fitted S2 OLS (coefficients, residual SD); predictor rows resampled with replacement from observed rows (keeps the joint distribution including covariates); N in {250, 478, 1000}; scenarios: linear, hockey-stick (range 0.5 and 1.0), step (0.5 and 1.0) on interest, noise SD fixed. 120 replicates per cell. Critical value for D_j = 95th percentile of the linear scenario at the same N. Report power, false-positive rate, and P(forest > OLS) and P(spline > OLS) on a fresh 1,500-row test set.

## 10. Reporting rules
Every number comes from a script and is written to `results/`. A deviation from this protocol is appended below with its reason before the affected result is used.

## Deviations log
(none yet)
- 2026-11-09 (Stage Two additions, made after the Stage One CV results were seen but before nonlinearity/power results): (i) selection-aware statistics added to the PDP null: G = RMS SSE reduction of the best continuous two-segment fit (breakpoint searched over levels 2-4) vs a straight line, and its maximum over the three core predictors (selection over predictors AND breakpoints); D_max = max over predictors. Reason: Stage Two request B (null must reproduce the selection of an apparent threshold). (ii) Paired model comparison uses Nadeau-Bengio corrected resampled t over the 100 outer folds plus a respondent-level bootstrap that conditions on fitted procedures (so understates training-set variability). (iii) The Stage One B=300 null gave p_Gmax = 0.106 (continuous) and 0.047 (discretised), i.e. borderline; a B=2000 re-run of the discretised null with new seeds is added to reduce Monte Carlo error. Not a change of the decision rule.
- Known issue: 06_power.py HC3 quadratic test returns NaN when a resampled row has leverage 1 (N=250 with sparse gender_other); NaN counts as "not rejected", which slightly understates quadratic-test power at small N. The F-test and PDP statistics are unaffected. To be fixed (clip leverage) and re-run if time allows.
- 2026-11-09 (Stage Three, after the selection-aware null gave p = 0.048 under the rounded-and-clipped null and 0.106 under the continuous null at B = 300): (i) scripts/11 adds an ordinal null (ordered-logit fit with a linear latent predictor, integer 1-5 outcomes) and reruns the continuous and rounded-and-clipped nulls at B = 2,000, saving all replicates; (ii) scripts/12 re-tunes the forest inside 300 simulated datasets (same outcomes as scripts/04's rounded-and-clipped null); (iii) scripts/13 checks false-positive rates on 400 fresh datasets per generating process against critical values from each null; (iv) scripts/14 re-thresholds the N = 478 power replicates under each null; (v) HC3 variance guard for leverage-1 rows added to common.fit_ols and scripts/06 rerun (no NaN p-values remained; N = 250/478/1000 results otherwise unchanged); (vi) paths made relative to the package, SMMH_OUT / SMMH_FIG override output folders. Decision rule for the primary null, fixed before reading its p-value: the null whose simulated outcomes match the observed outcome's variance, category frequencies and OLS R2 (diagnostics in results/null_*_B2000.json). That is the ordinal null.
- 2026-11-10 (Stage Four: repository consolidation and reconciliation; no analytical change, no number changed). (i) Code: the development scripts were merged into `analysis.py` (core: audit, OLS, descriptives, nested CV, paired comparison, nonlinearity tests and partial dependence, ranking stability, tables, figures) and `simulations.py` (optional: initial B = 300 nulls, three B = 2,000 nulls, forest re-tuning, size check, power, bracket re-thresholding). Seeds, grids, forest settings and statistics were carried over unchanged; the new scripts were verified against the stored outputs replicate by replicate (docs/VERIFICATION.md). (ii) Output filenames were renamed to the manuscript's figure numbers (fig5_ols_coefficients -> fig1_ols_coefficients; fig2_pdp -> fig2_partial_dependence; fig6_rf_permutation_importance -> fig4_rf_permutation_importance). (iii) Development analyses that are not reported in the manuscript were not carried into the repository, and their outputs are not shipped: the seed-only, partition-only, hyperparameter-configuration and specification variants of the ranking-stability analysis (protocol section 8 a, b, d, e), the time-coding and uncentred-VIF variants of the OLS diagnostics, the earlier separate B = 2,000 rounded-and-clipped run (`07_null_more.py`; same p-value as the B = 2,000 rounded-and-clipped null kept), the OOB-R2/bootstrap-beta extras (`09_ledger_extra.py`) and the superseded Figures 1 and 4 of the first draft (`fig1_prediction`, `fig4_rank_stability`). (iv) The first-pass B = 300 nulls (p = 0.106 continuous, 0.047 rounded-and-clipped), which the Methods cite, were previously stored inside `nonlin.json`; they are now stored as `results/simulations/initial_nulls_B300.json`, unchanged, and are regenerated by `simulations.py --stage initial`. (v) Reconciliation of the protocol text with what was executed, found while assembling the repository: section 5 specifies a respondent bootstrap with B = 2,000, the executed comparison used B = 5,000 draws (seed 42), and the Nadeau-Bengio corrected t over the 100 outer folds, which section 5 does not mention, is the reported inference (it was added in Stage Two, item ii above); section 9 describes the hockey-stick and step scenarios as ranges 0.5/1.0, whereas the executed design added to the interest term 0.15, 0.30 or 0.45 x max(level - 3, 0) (an extra rise over levels 3-5 of 0.3, 0.6 or 0.9 scale points) or a step of 0.30 or 0.60 at level 4 and above; section 9 reports power to be judged against the 95th percentile of the linear scenario at the same N, which is what `power.json` contains, and Figure 3b / the manuscript additionally re-threshold the N = 478 replicates under each of the three B = 2,000 nulls (`power_N478_bracket.*`). The protocol's section 8 also lists three importance measures with split-half agreement; the reported split-half statistic is the proportion of splits in which the same three predictors lead. (vi) Wording point for the manuscript: the 11-22%, 49-63% and 82-86% power ranges in Section 3.3 run from the lowest of the three null-based thresholds to the larger of the three null-based thresholds and the threshold from the power simulation's own linear scenario (`Gmax_own_linear_crit`); the text attributes the range to the three nulls only.
