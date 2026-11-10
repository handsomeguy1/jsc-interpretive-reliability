#!/usr/bin/env python3
"""
Core analyses, tables and figures for the manuscript
"Beyond Prediction: Interpretive Reliability in Computational Social Science".

Primary specification S2 (seven behavioural predictors + age + gender), primary sample = respondents who use social media (N = 478).

Usage (run from the repository root; all paths are relative to this file)
    python analysis.py                    # full core analysis: recomputes everything below (about 35-45 min on 2 cores)
    python analysis.py --saved            # rebuild tables and figures from the stored raw outputs in results/ (about 1 min)
    python analysis.py --quick            # reduced replicates, written to quick_output/ (about 5 min); smoke test only
    python analysis.py --sensitivity      # additionally run the nested CV for S1, S3, S4 and the age 15-45 sample (about 1.5 h)
    python analysis.py --check            # compare stored results with the numbers printed in the manuscript
    python analysis.py --stages cv,compare,tables,figures      # choose stages (audit, ols, descriptives, cv, compare, nonlin,
                                                               # stability, tables, figures, check)

The selection-aware null simulations, power analysis and calibration checks are NOT run here. They live in simulations.py;
their final saved outputs are in results/simulations/ and are read (never recomputed) by the table and figure stages below.
Every random draw is seeded (see PARAMS); results do not depend on the number of processor cores.
"""
import argparse, collections, glob, hashlib, json, os, sys, time, warnings
import numpy as np, pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, 'data', 'smmh.csv')
DATA_SHA256 = '5a6ae636760314616754b83c6e4626ea55247f7586e6b5e17ddcdff80ca11fa3'
SAVED = os.path.join(ROOT, 'results')                 # stored final outputs shipped with the repository
OUT = ROOT; RES = SAVED; FIG = os.path.join(ROOT, 'figures'); N_JOBS = 2

# Design constants (identical to the analysed protocol; see docs/PROTOCOL.md) -------------------------------------------------
PARAMS = dict(
    primary_sample='users of social media (Q6 = Yes), N = 478', primary_specification='S2',
    cv=dict(outer_folds=10, repeats=10, outer_seed='1000 + repeat', inner_folds=5, inner_seed='5000 + 100*repeat + fold',
            rf_trees=300, rf_seed='100*repeat + fold', rf_grid={'min_samples_leaf': [1, 3, 5, 10, 20], 'max_features': [0.33, 0.6, 1.0]},
            spline_degree=2, spline_grid={'n_knots': [3, 4, 5], 'ridge_alpha': [0.1, 1, 10, 100]}),
    compare=dict(method='Nadeau-Bengio corrected resampled t over the 100 outer folds', bootstrap_seed=42, bootstrap_draws=5000),
    nonlin=dict(full_sample_rf_search=dict(trees=300, forest_seed=20261109, cv_seed=20261109, folds=5), observed_forest_seed=1,
                pdp_bootstrap_B=300, pdp_bootstrap_seed='90000 + i (forest seed i)', pdp_levels=[1, 2, 3, 4, 5]),
    stability=dict(bootstrap_B=500, bootstrap_seed='300000 + i (forest seed i)', split_half_B=300, split_seed='400000 + i (forest seeds i and i + 10000)',
                   permutations_per_predictor=3, reference_forest_seed=1, reference_cv_folds=5, reference_cv_seed=1),
)
COLS = ['ts', 'age', 'gender', 'rel', 'occ', 'org', 'use_sm', 'platforms', 'time', 'purpose', 'distract_busy', 'restless',
        'easily_distracted', 'worries', 'concentrate', 'compare', 'compare_feel', 'validation', 'depressed', 'interest', 'sleep']
MID = {'Less than an Hour': 0.5, 'Between 1 and 2 hours': 1.5, 'Between 2 and 3 hours': 2.5, 'Between 3 and 4 hours': 3.5,
       'Between 4 and 5 hours': 4.5, 'More than 5 hours': 6.0}
LIKERT = ['purpose', 'distract_busy', 'restless', 'easily_distracted', 'worries', 'concentrate', 'compare', 'compare_feel', 'validation',
          'depressed', 'interest', 'sleep']
S1 = ['time_h', 'easily_distracted', 'restless', 'validation', 'compare', 'interest', 'sleep']
EXT = ['purpose', 'distract_busy', 'concentrate', 'worries']
CORE = ['interest', 'compare', 'sleep']
Y = 'depressed'
LABEL = {'time_h': 'Daily time on social media (hours)', 'easily_distracted': 'Easily distracted by social media',
         'restless': 'Restless without social media', 'validation': 'Seeks validation from social media',
         'compare': 'Compares self with others', 'interest': 'Fluctuation in interest in daily activities',
         'sleep': 'Sleep problems', 'age': 'Age (years)', 'male': 'Male (vs female)', 'gender_other': 'Other gender (vs female)'}
SHORT = {'time_h': 'Daily time on social media', 'easily_distracted': 'Easily distracted by social media',
         'restless': 'Restless without social media', 'validation': 'Seeks validation from social media',
         'compare': 'Compares self with others', 'interest': 'Fluctuation in interest in daily activities', 'sleep': 'Sleep problems'}
COV = {'S1': [], 'S2': ['age', 'male', 'gender_other'], 'S3': ['age', 'male', 'gender_other'] + EXT,
       'S4': ['age', 'male', 'gender_other', 'rel_inrel', 'rel_other', 'occ_school', 'occ_other']}
def feats(spec): return S1 + COV[spec]


# ---------------------------------------------------------------------------------------------------------------- data ------
def data_available(): return os.path.exists(DATA)
def raw():
    if not data_available():
        sys.exit('data/smmh.csv not found. See data/README.md for the source, retrieval steps and checksum; '
                 'use --saved to rebuild tables and figures from the stored results without the data.')
    d = pd.read_csv(DATA); d.columns = COLS; d['row'] = np.arange(2, len(d) + 2); return d
def prep(d):
    d = d.copy(); d['time_h'] = d.time.map(MID)
    d['male'] = (d.gender == 'Male').astype(float)
    d['gender_other'] = (~d.gender.isin(['Male', 'Female'])).astype(float)
    d['rel_inrel'] = (d.rel == 'In a relationship').astype(float)
    d['rel_other'] = (~d.rel.isin(['Single', 'In a relationship'])).astype(float)
    d['occ_school'] = (d.occ == 'School Student').astype(float)
    d['occ_other'] = (~d.occ.isin(['University Student', 'School Student'])).astype(float)
    return d
def sample(d, name='P'):
    if name == 'A0': return d
    p = d[d.use_sm == 'Yes']
    if name == 'P': return p
    if name == 'A1': return d[(d.age >= 15) & (d.age <= 45)]
    if name == 'A2': return p[~((p.age > 80) | ((p.occ == 'School Student') & (p.age > 25)))]
    raise ValueError(name)
def sha256_of_data():
    with open(DATA, 'rb') as f: return hashlib.sha256(f.read()).hexdigest()
def save(name, obj):
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, name), 'w') as f: json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
def load(name):
    for base in (RES, SAVED):
        p = os.path.join(base, name)
        if os.path.exists(p): return json.load(open(p))
    raise FileNotFoundError(name)
def find(name):
    for base in (RES, SAVED):
        p = os.path.join(base, name)
        if os.path.exists(p): return p
    raise FileNotFoundError(name)
def sim_path(name):                                    # saved simulation outputs (never recomputed here)
    for base in (os.path.join(RES, 'simulations'), os.path.join(SAVED, 'simulations')):
        p = os.path.join(base, name)
        if os.path.exists(p): return p
    raise FileNotFoundError('simulations/' + name)


# ------------------------------------------------------------------------------ OLS toolkit (no statsmodels) ----------------
def design(df, cols): return np.column_stack([np.ones(len(df)), df[cols].to_numpy(float)])
def fit_ols(X, y):
    XtXi = np.linalg.pinv(X.T @ X); b = XtXi @ X.T @ y; e = y - X @ b
    h = np.einsum('ij,jk,ik->i', X, XtXi, X); n, p = X.shape
    s2 = e @ e / (n - p)
    adj = np.where(1 - h > 1e-8, e / np.maximum(1 - h, 1e-8), 0.0)      # leverage-1 rows carry no residual; avoid 0/0
    hc3 = XtXi @ (X.T * adj ** 2) @ X @ XtXi
    with np.errstate(invalid='ignore'):                                  # tiny negative diagonal entries of unused SEs -> NaN, silently
        se, se_hc3 = np.sqrt(np.diag(s2 * XtXi)), np.sqrt(np.diag(hc3))
    return dict(b=b, e=e, h=h, n=n, p=p, s2=s2, XtXi=XtXi, se=se, se_hc3=se_hc3,
                r2=1 - e @ e / ((y - y.mean()) @ (y - y.mean())), rss=e @ e, yhat=X @ b)
def vif_centered(Xn):                     # Xn without intercept; each column regressed on the others WITH intercept
    out = []
    for j in range(Xn.shape[1]):
        o = np.delete(Xn, j, 1); Z = np.column_stack([np.ones(len(o)), o]); b = np.linalg.lstsq(Z, Xn[:, j], rcond=None)[0]
        r = Xn[:, j] - Z @ b; out.append(((Xn[:, j] - Xn[:, j].mean()) @ (Xn[:, j] - Xn[:, j].mean())) / (r @ r))
    return np.array(out)


# ----------------------------------------------------------------------------- partial-dependence statistics (shared) -------
def pdp_levels(m, Xm, cols):
    out = {}
    for c in CORE:
        j = cols.index(c); v = []
        for lv in [1, 2, 3, 4, 5]:
            Z = Xm.copy(); Z[:, j] = lv; v.append(m.predict(Z).mean())
        out[c] = np.array(v)
    return out
def dev_stat(v):                           # RMS deviation of the 5-point PDP from its best straight line
    lv = np.arange(1, 6); b = np.polyfit(lv, v, 1); return float(np.sqrt(np.mean((v - np.polyval(b, lv)) ** 2)))
def hinge_stat(v):                         # best continuous two-segment fit (breakpoint over levels 2-4): RMS SSE reduction vs straight line
    lv = np.arange(1, 6.); sl = np.sum((v - np.polyval(np.polyfit(lv, v, 1), lv)) ** 2); best = sl
    for k in (2, 3, 4):
        A = np.column_stack([np.ones(5), lv, np.maximum(lv - k, 0)]); r = v - A @ np.linalg.lstsq(A, v, rcond=None)[0]; best = min(best, r @ r)
    return float(np.sqrt((sl - best) / 5))
def hinge_fit(v):                          # as hinge_stat, and also returns the winning breakpoint (None if the straight line wins)
    lv = np.arange(1, 6.); sl = np.sum((v - np.polyval(np.polyfit(lv, v, 1), lv)) ** 2); best = sl; kbest = None
    for k in (2, 3, 4):
        A = np.column_stack([np.ones(5), lv, np.maximum(lv - k, 0)]); r = v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
        if r @ r < best - 1e-15: best = r @ r; kbest = k
    return float(np.sqrt((sl - best) / 5)), kbest
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for i, j in enumerate(o): run = max(run, (m - i) * p[j]); adj[j] = min(1, run)
    return adj


# -------------------------------------------------------------------------------------------------------------- stages ------
def stage_audit():
    out = {}; out['sha256'] = sha256_of_data()
    nlines = sum(1 for _ in open(DATA, encoding='utf-8'))
    d = raw(); out['file_lines'] = nlines; out['rows'] = len(d)
    out['missing'] = {c: int(d[c].isna().sum()) for c in COLS if d[c].isna().sum()}
    ans = [c for c in COLS if c != 'ts']
    out['dup_timestamp_rows'] = int(d.ts.duplicated(keep=False).sum())
    out['exact_dup_excl_ts_rows'] = int(d.duplicated(subset=ans, keep=False).sum())
    out['exact_dup_excl_ts_extra'] = int(d.duplicated(subset=ans, keep='first').sum())
    out['exact_dup_incl_ts_extra'] = int(d.duplicated(subset=COLS, keep='first').sum())
    dd = d[d.duplicated(subset=ans, keep=False)].sort_values(ans)
    out['dup_groups'] = [{'rows': g.row.tolist(), 'ts': g.ts.tolist(), 'age': int(g.age.iloc[0]), 'depressed': int(g.depressed.iloc[0])} for _, g in dd.groupby(ans)]
    mv = ['age', 'gender', 'time', 'easily_distracted', 'restless', 'validation', 'compare', 'interest', 'sleep', 'depressed']
    out['dup_on_model_vars_extra'] = int(d.duplicated(subset=mv, keep='first').sum())
    out['use_sm'] = d.use_sm.value_counts().to_dict()
    nu = d[d.use_sm != 'Yes']; out['nonusers'] = nu[['row', 'age', 'gender', 'time', 'depressed']].to_dict('records')
    out['age_describe'] = d.age.describe().to_dict()
    out['age_values_outside_15_45'] = d[(d.age < 15) | (d.age > 45)][['row', 'age', 'occ', 'use_sm']].to_dict('records')
    out['age_n_lt15'] = int((d.age < 15).sum()); out['age_n_gt45'] = int((d.age > 45).sum())
    out['n_age15_45_all'] = int(((d.age >= 15) & (d.age <= 45)).sum())
    for c in ['gender', 'rel', 'occ', 'time']: out['levels_' + c] = d[c].value_counts().to_dict()
    out['likert_ranges'] = {c: [int(d[c].min()), int(d[c].max())] for c in LIKERT}
    out['likert_dist'] = {c: d[c].value_counts().sort_index().to_dict() for c in LIKERT}
    out['unmapped_time'] = int(d.time.map(MID).isna().sum())
    out['occ_by_age'] = {k: g.age.describe().round(1).to_dict() for k, g in d.groupby('occ')}
    out['school_age_gt25'] = d[(d.occ == 'School Student') & (d.age > 25)][['row', 'age']].to_dict('records')
    p = sample(d, 'P'); out['n'] = {k: len(sample(d, k)) for k in ['A0', 'P', 'A1', 'A2']}
    out['P_gender'] = p.gender.value_counts().to_dict(); out['P_age_sd'] = float(p.age.std()); out['P_age_mean'] = float(p.age.mean())
    a1 = sample(d, 'A1'); out['A1_gender'] = a1.gender.value_counts().to_dict(); out['A1_n_users'] = int((a1.use_sm == 'Yes').sum()); out['A1_age_sd'] = float(a1.age.std())
    pa = p[(p.age >= 15) & (p.age <= 45)]; out['P_and_age15_45'] = len(pa)
    out['n_after_dedupe_all'] = len(d) - out['exact_dup_excl_ts_extra']
    p2 = prep(p)
    out['corr_distract_items'] = p2[['easily_distracted', 'distract_busy', 'concentrate', 'worries', 'purpose']].corr().round(3).to_dict()
    out['corr_outcome'] = p2[LIKERT + ['time_h', 'age']].corrwith(p2[Y]).round(3).to_dict()
    out['P_levels'] = {c: p.groupby(c).size().to_dict() for c in ['gender', 'rel', 'occ']}
    out['P_dup_extra'] = int(p.duplicated(subset=ans, keep='first').sum())
    save('audit.json', out); print('audit: rows', out['rows'], 'users', out['n']['P'], 'sha256 ok' if out['sha256'] == DATA_SHA256 else 'SHA-256 DIFFERS FROM EXPECTED')


def stage_ols():
    out = {}
    def run(df, spec, tag):
        cols = feats(spec); X = design(df, cols); y = df[Y].to_numpy(float); f = fit_ols(X, y); n, p = f['n'], f['p']
        sd = df[cols].std().to_numpy(); zb = f['b'][1:] * sd / y.std(ddof=1)
        t = f['b'] / f['se_hc3']; pv = 2 * stats.t.sf(abs(t), n - p)
        Xn = X[:, 1:]; vif = vif_centered(Xn)
        sv = np.linalg.svd((Xn - Xn.mean(0)) / Xn.std(0), compute_uv=False); cond = sv[0] / sv[-1]
        lev = f['h']; stud = f['e'] / np.sqrt(f['s2'] * (1 - lev))
        ext = stud * np.sqrt((n - p - 1) / (n - p - stud ** 2))
        cook = stud ** 2 / p * lev / (1 - lev)
        e2 = f['e'] ** 2; Z = X; g = np.linalg.lstsq(Z, e2, rcond=None)[0]; r2bp = 1 - ((e2 - Z @ g) @ (e2 - Z @ g)) / ((e2 - e2.mean()) @ (e2 - e2.mean()))
        bp = n * r2bp; bpp = stats.chi2.sf(bp, p - 1)
        yh = f['yhat']; Xr = np.column_stack([X, yh ** 2, yh ** 3]); fr = fit_ols(Xr, y)
        Fr = ((f['rss'] - fr['rss']) / 2) / (fr['rss'] / (n - Xr.shape[1])); Rp = stats.f.sf(Fr, 2, n - Xr.shape[1])
        keep = np.argsort(-cook)[10:]; f2 = fit_ols(X[keep], y[keep]); sw = stats.shapiro(f['e'])
        res = dict(n=int(n), p=int(p), cols=['const'] + cols, coef=f['b'].round(4).tolist(), se_classic=f['se'].round(4).tolist(), se_hc3=f['se_hc3'].round(4).tolist(),
                   p_hc3=pv.round(4).tolist(), std_beta=dict(zip(cols, zb.round(4).tolist())), r2=round(f['r2'], 4), adj_r2=round(1 - (1 - f['r2']) * (n - 1) / (n - p), 4),
                   resid_sd=round(float(np.sqrt(f['s2'])), 4), vif=dict(zip(cols, vif.round(2).tolist())), max_vif=round(float(vif.max()), 2), cond_number=round(float(cond), 2),
                   leverage_gt_2p_n=int((lev > 2 * p / n).sum()), max_leverage=round(float(lev.max()), 4), abs_ext_stud_gt3=int((abs(ext) > 3).sum()),
                   cook_gt_4n=int((cook > 4 / n).sum()), max_cook=round(float(cook.max()), 4), top_cook_rows=df.row.to_numpy()[np.argsort(-cook)[:5]].tolist(),
                   bp_stat=round(float(bp), 3), bp_p=round(float(bpp), 4), reset_F=round(float(Fr), 3), reset_p=round(float(Rp), 4), shapiro_p=float(sw.pvalue),
                   resid_skew=round(float(stats.skew(f['e'])), 3), resid_kurt=round(float(stats.kurtosis(f['e'])), 3),
                   r2_drop_top10_cook=round(f2['r2'], 4), coef_drop_top10_cook=dict(zip(['const'] + cols, f2['b'].round(4).tolist())),
                   pred_range=[round(float(yh.min()), 2), round(float(yh.max()), 2)], pred_outside_1_5=int(((yh < 1) | (yh > 5)).sum()))
        out[tag] = res
    for s in ['A0', 'P', 'A1', 'A2']:
        for sp in ['S1', 'S2']: run(prep(sample(raw(), s)), sp, f'{s}_{sp}')
    for sp in ['S3', 'S4']: run(prep(sample(raw(), 'P')), sp, f'P_{sp}')
    save('ols.json', out)
    r = out['P_S2']; print('ols S2: n', r['n'], 'R2', r['r2'], 'adjR2', r['adj_r2'], 'max VIF', r['max_vif'])


def stage_descriptives():
    d = prep(sample(raw(), 'P')); out = {'n': len(d)}
    out['describe'] = {c: dict(mean=float(d[c].mean()), sd=float(d[c].std(ddof=1)), min=float(d[c].min()), max=float(d[c].max())) for c in S1 + ['depressed', 'age']}
    out['gender_counts'] = d.gender.map(lambda g: g if g in ('Male', 'Female') else 'Other').value_counts().to_dict()
    out['occupation_counts'] = d.occ.value_counts().to_dict(); out['relationship_counts'] = d.rel.value_counts().to_dict()
    out['outcome_distribution'] = {str(k): int(v) for k, v in d[Y].value_counts().sort_index().items()}
    out['outcome_category_share'] = {str(k): float(v) for k, v in d[Y].value_counts(normalize=True).sort_index().items()}
    save('descriptives.json', out)


def stage_cv(spec, samp, REP, NT):
    from sklearn.model_selection import KFold, GridSearchCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import SplineTransformer, StandardScaler
    from joblib import Parallel, delayed
    d = prep(sample(raw(), samp)); cols = feats(spec); X = d[cols].to_numpy(float); y = d[Y].to_numpy(float); n = len(y)
    nb = [cols.index(c) for c in S1]; cv_ = [i for i in range(len(cols)) if i not in nb]
    def spline_est():
        ct = ColumnTransformer([('s', SplineTransformer(degree=2, n_knots=4, knots='uniform', extrapolation='constant'), nb)] + ([('p', 'passthrough', cv_)] if cv_ else []))
        return make_pipeline(ct, StandardScaler(), Ridge())
    SPL_GRID = {'columntransformer__s__n_knots': [3, 4, 5], 'ridge__alpha': [0.1, 1, 10, 100]}
    RF_GRID = {'min_samples_leaf': [1, 3, 5, 10, 20], 'max_features': [0.33, 0.6, 1.0]}
    def one_repeat(r):
        kf = KFold(10, shuffle=True, random_state=1000 + r)
        pred = {m: np.zeros(n) for m in ['mean', 'ols', 'spline', 'rf']}; base = np.zeros(n); hp = []
        for k, (tr, te) in enumerate(kf.split(X)):
            base[te] = y[tr].mean(); pred['mean'][te] = y[tr].mean()
            pred['ols'][te] = LinearRegression().fit(X[tr], y[tr]).predict(X[te])
            inner = KFold(5, shuffle=True, random_state=5000 + 100 * r + k)
            gs = GridSearchCV(spline_est(), SPL_GRID, cv=inner, scoring='neg_mean_squared_error').fit(X[tr], y[tr]); pred['spline'][te] = gs.predict(X[te])
            rf = GridSearchCV(RandomForestRegressor(n_estimators=NT, random_state=r * 100 + k, n_jobs=1), RF_GRID, cv=inner, scoring='neg_mean_squared_error').fit(X[tr], y[tr]); pred['rf'][te] = rf.predict(X[te])
            hp.append(dict(r=r, k=k, spline=gs.best_params_, rf=rf.best_params_))
        return pred, base, hp
    t0 = time.time(); res = Parallel(n_jobs=N_JOBS)(delayed(one_repeat)(r) for r in range(REP))
    P = {m: np.array([r[0][m] for r in res]) for m in ['mean', 'ols', 'spline', 'rf']}; B = np.array([r[1] for r in res])
    os.makedirs(RES, exist_ok=True); np.savez(f'{RES}/cv_{spec}_{samp}.npz', y=y, base=B, rows=d.row.to_numpy(), **P)
    save(f'cv_{spec}_{samp}_hp.json', [h for r in res for h in r[2]])
    sst = ((y - B) ** 2).sum(1); summ = {}
    for m in ['mean', 'ols', 'spline', 'rf']:
        r2 = 1 - ((y - P[m]) ** 2).sum(1) / sst; rmse = np.sqrt(((y - P[m]) ** 2).mean(1))
        summ[m] = dict(R2_mean=float(r2.mean()), R2_sd_over_repeats=float(r2.std(ddof=1)) if REP > 1 else None, R2_min=float(r2.min()), R2_max=float(r2.max()), RMSE=float(rmse.mean()))
    save(f'cv_{spec}_{samp}_summary.json', summ)
    print(f'cv {spec} {samp} n={n} repeats={REP} trees={NT}: R2 ols {summ["ols"]["R2_mean"]:.4f} spline {summ["spline"]["R2_mean"]:.4f} rf {summ["rf"]["R2_mean"]:.4f} ({time.time()-t0:.0f}s)')


def stage_compare():
    from sklearn.model_selection import KFold
    out = {}
    files = sorted(glob.glob(RES + '/cv_*_*.npz')) or sorted(glob.glob(SAVED + '/cv_*_*.npz'))     # cross-validation arrays from this run, else the stored ones
    for f in files:
        tag = os.path.basename(f)[3:-4]; z = np.load(f); y = z['y']; n = len(y); R = z['ols'].shape[0]; base = z['base']
        pairs = [('rf', 'ols'), ('spline', 'ols'), ('rf', 'spline')]; res = {}
        sst = ((y - base) ** 2).sum(1)
        for m in ['ols', 'spline', 'rf']:
            r2 = 1 - ((y - z[m]) ** 2).sum(1) / sst; res['R2_' + m] = dict(mean=float(r2.mean()), sd_repeats=float(r2.std(ddof=1)) if R > 1 else None)
        for a, b in pairs:
            dif = []                                    # fold-level paired differences in test-fold MSE (A minus B); negative = A better
            for r in range(R):
                for k, (tr, te) in enumerate(KFold(10, shuffle=True, random_state=1000 + r).split(np.arange(n))):
                    dif.append(np.mean((y[te] - z[a][r][te]) ** 2) - np.mean((y[te] - z[b][r][te]) ** 2))
            dif = np.array(dif); J = len(dif); ntr = n * 0.9; nte = n * 0.1
            mean = dif.mean(); naive_t = mean / np.sqrt(dif.var(ddof=1) / J)
            nb_se = np.sqrt((1 / J + nte / ntr) * dif.var(ddof=1)); nb_t = mean / nb_se; p_nb = 2 * stats.t.sf(abs(nb_t), J - 1)
            ci = (mean - stats.t.ppf(.975, J - 1) * nb_se, mean + stats.t.ppf(.975, J - 1) * nb_se)
            d_i = np.mean((y - z[a]) ** 2 - (y - z[b]) ** 2, axis=0); rng = np.random.default_rng(42); bs = np.array([d_i[rng.integers(0, n, n)].mean() for _ in range(5000)])
            scale = 1 / np.mean(((y - base) ** 2).mean(1))      # MSE difference -> R2 difference
            res[f'{a}_minus_{b}'] = dict(mse_diff_mean=float(mean), R2_diff_mean=float(-mean * scale), NB_se_mse=float(nb_se), NB_t=float(nb_t), NB_p=float(p_nb),
                                         NB_CI95_R2diff=[float(-ci[1] * scale), float(-ci[0] * scale)], naive_t_overlapping_folds=float(naive_t),
                                         boot_resp_CI95_R2diff=[float(-np.quantile(bs, .975) * scale), float(-np.quantile(bs, .025) * scale)],
                                         prop_respondents_a_better=float(np.mean(d_i < 0)))
        out[tag] = res
    save('compare.json', out)
    if 'S2_P' in out: print('compare S2_P: rf-ols', round(out['S2_P']['rf_minus_ols']['R2_diff_mean'], 4), 'spline-ols', round(out['S2_P']['spline_minus_ols']['R2_diff_mean'], 4))


def stage_nonlin(B):
    """F-tests, quadratic terms, observed RF partial dependence (+ breakpoint statistics) and respondent-bootstrap bands.
    The linear-null simulations that give the selection-aware p-values are in simulations.py."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold, GridSearchCV
    from joblib import Parallel, delayed
    d = prep(sample(raw(), 'P')); cols = feats('S2'); X = d[cols].to_numpy(float); y = d[Y].to_numpy(float); n = len(y)
    out = {}
    base = fit_ols(design(d, cols), y); Ftab = {}
    for c in S1:
        lv = sorted(d[c].unique()); dums = np.column_stack([(d[c] == l).astype(float) for l in lv[1:]])
        others = [k for k in cols if k != c]
        Xu = np.column_stack([design(d, others), dums]); fr = base; fu = fit_ols(Xu, y)
        q = len(lv) - 2; F = ((fr['rss'] - fu['rss']) / q) / (fu['rss'] / (n - Xu.shape[1])); pF = stats.f.sf(F, q, n - Xu.shape[1])
        Xq = np.column_stack([design(d, cols), (d[c] - d[c].mean()) ** 2]); fq = fit_ols(Xq, y); t = fq['b'][-1] / fq['se_hc3'][-1]; pq = 2 * stats.t.sf(abs(t), n - Xq.shape[1])
        Ftab[c] = dict(levels=len(lv), F=float(F), df1=q, p_F=float(pF), quad_coef=float(fq['b'][-1]), quad_p_hc3=float(pq))
    for fam, name in [(CORE, 'holm_core3'), (S1, 'holm_all7')]:
        for key in ['p_F', 'quad_p_hc3']:
            adj = holm([Ftab[c][key] for c in fam])
            for c, a in zip(fam, adj): Ftab[c][f'{key}_{name}'] = float(a)
    out['F_quad'] = Ftab
    RF_GRID = {'min_samples_leaf': [1, 3, 5, 10, 20], 'max_features': [0.33, 0.6, 1.0]}
    gs = GridSearchCV(RandomForestRegressor(n_estimators=300, random_state=20261109, n_jobs=N_JOBS), RF_GRID, cv=KFold(5, shuffle=True, random_state=20261109), scoring='neg_mean_squared_error').fit(X, y)
    hp = gs.best_params_; out['full_sample_rf_hp'] = hp; out['full_sample_rf_cv_mse'] = float(-gs.best_score_)
    def mk(seed): return RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=1, **hp)
    obs = pdp_levels(mk(1).fit(X, y), X, cols); out['pdp_obs'] = {c: obs[c].tolist() for c in CORE}; out['D_obs'] = {c: dev_stat(obs[c]) for c in CORE}
    out['pdp_slope_obs'] = {c: float(np.polyfit(np.arange(1, 6), obs[c], 1)[0]) for c in CORE}
    Gobs = {c: hinge_fit(obs[c])[0] for c in CORE}; out['G_obs'] = Gobs; out['G_obs_breakpoint'] = {c: hinge_fit(obs[c])[1] for c in CORE}
    out['Dmax_obs'] = max(out['D_obs'].values()); out['Gmax_obs'] = max(Gobs.values())
    def boot_one(i):
        rng = np.random.default_rng(90_000 + i); ix = rng.integers(0, n, n); m = mk(i).fit(X[ix], y[ix]); p = pdp_levels(m, X[ix], cols); return {c: p[c].tolist() for c in CORE}
    r = Parallel(n_jobs=N_JOBS)(delayed(boot_one)(i) for i in range(B))
    out['level_counts'] = {c: d[c].value_counts().sort_index().to_dict() for c in CORE}; out['pdp_boot'] = {'B': B}
    for c in CORE:
        A = np.array([x[c] for x in r]); Dv = np.array([dev_stat(v) for v in A]); H = [hinge_fit(v) for v in A]; sl = np.array([np.polyfit(np.arange(1, 6), v, 1)[0] for v in A])
        out['pdp_boot'][c] = dict(lo=np.quantile(A, .05, axis=0).tolist(), hi=np.quantile(A, .95, axis=0).tolist(), D_q05=float(np.quantile(Dv, .05)), D_q50=float(np.quantile(Dv, .5)), D_q95=float(np.quantile(Dv, .95)),
                                  slope_q05=float(np.quantile(sl, .05)), slope_q95=float(np.quantile(sl, .95)), G_q50=float(np.quantile([h[0] for h in H], .5)),
                                  breakpoint_freq={str(k): float(np.mean([h[1] == k for h in H])) for k in (None, 2, 3, 4)}, prop_boot_D_ge_obs=float(np.mean(Dv >= out['D_obs'][c])))
    save('nonlin.json', out)
    print('nonlin: F-test p', {c: round(Ftab[c]['p_F'], 3) for c in CORE}, 'RF hyperparameters', hp, 'Gmax_obs', round(out['Gmax_obs'], 4))


def stage_stability(NB, NS):
    """Respondent-bootstrap and split-half stability of three importance measures (seven behavioural predictors)."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold
    from joblib import Parallel, delayed
    HP = load('nonlin.json')['full_sample_rf_hp']
    d = prep(sample(raw(), 'P')); cols = feats('S2'); X = d[cols].to_numpy(float); y = d[Y].to_numpy(float); n = len(y)
    J = [cols.index(c) for c in S1]
    def rf(seed, hp=HP): return RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=1, **hp)
    def rank(v): return stats.rankdata(-np.asarray(v), method='average')              # 1 = most important
    def tau(a, b): return float(stats.kendalltau(a, b)[0])
    def top3(v): return frozenset(np.argsort(-np.asarray(v))[:3].tolist())
    def ols_imp(Xs, ys):
        f = fit_ols(np.column_stack([np.ones(len(ys)), Xs]), ys); sd = Xs.std(0, ddof=1); return np.abs(f['b'][1:] * sd / ys.std(ddof=1))[J]
    def perm_imp(m, Xh, yh, rng, rep=3):
        b = np.mean((yh - m.predict(Xh)) ** 2); out = []
        for j in J:
            s = 0
            for _ in range(rep):
                Z = Xh.copy(); Z[:, j] = rng.permutation(Z[:, j]); s += np.mean((yh - m.predict(Z)) ** 2) - b
            out.append(s / rep)
        return np.array(out)
    def mdi(m): return m.feature_importances_[J]
    names = S1
    def cv_perm(Xd, yd, seed_forest, seed_folds, hp=HP):
        rng = np.random.default_rng(seed_folds); acc = np.zeros(len(J))
        for tr, te in KFold(5, shuffle=True, random_state=seed_folds).split(Xd):
            m = rf(seed_forest, hp).fit(Xd[tr], yd[tr]); acc += perm_imp(m, Xd[te], yd[te], rng) * len(te)
        return acc / len(yd)
    ref_ols = ols_imp(X, y); mfull = rf(1).fit(X, y); ref_mdi = mdi(mfull); ref_perm = cv_perm(X, y, 1, 1)
    out = {'names': names, 'full': {'ols_abs_std_beta': ref_ols.tolist(), 'rf_mdi': ref_mdi.tolist(), 'rf_perm_cv': ref_perm.tolist()},
           'full_rank': {'ols': rank(ref_ols).tolist(), 'rf_mdi': rank(ref_mdi).tolist(), 'rf_perm': rank(ref_perm).tolist()}}
    refs = {'ols': ref_ols, 'mdi': ref_mdi, 'perm': ref_perm}
    def boot(i):
        rng = np.random.default_rng(300_000 + i); ix = rng.integers(0, n, n); oob = np.setdiff1d(np.arange(n), ix)
        m = rf(i).fit(X[ix], y[ix]); return dict(ols=ols_imp(X[ix], y[ix]), mdi=mdi(m), perm=perm_imp(m, X[oob], y[oob], rng))
    BR = Parallel(n_jobs=N_JOBS)(delayed(boot)(i) for i in range(NB))
    np.savez(f'{RES}/importance_boot_B{NB}.npz', perm=np.array([b['perm'] for b in BR]), mdi=np.array([b['mdi'] for b in BR]), ols=np.array([b['ols'] for b in BR]))
    out['bootstrap'] = {}
    for k in ['ols', 'mdi', 'perm']:
        A = np.array([b[k] for b in BR]); R = np.array([rank(a) for a in A]); t = np.array([tau(a, refs[k]) for a in A])
        t3 = np.array([top3(a) == top3(refs[k]) for a in A])
        out['bootstrap'][k] = dict(B=NB, tau_mean=float(t.mean()), tau_q05=float(np.quantile(t, .05)), tau_q95=float(np.quantile(t, .95)), top3_exact_recovery=float(t3.mean()),
                                   rank_q05=np.quantile(R, .05, axis=0).tolist(), rank_q50=np.median(R, axis=0).tolist(), rank_q95=np.quantile(R, .95, axis=0).tolist(),
                                   prop_rank1=dict(zip(names, [float((R[:, j] == 1).mean()) for j in range(7)])),
                                   prop_in_top3=dict(zip(names, [float((R[:, j] <= 3).mean()) for j in range(7)])),
                                   top3_sets={'|'.join(names[j] for j in sorted(s)): float(c / NB) for s, c in collections.Counter(top3(a) for a in A).most_common(5)})
    def split(i):
        rng = np.random.default_rng(400_000 + i); p = rng.permutation(n); a, b = p[:n // 2], p[n // 2:]
        ma = rf(i).fit(X[a], y[a]); mb = rf(i + 10_000).fit(X[b], y[b])
        return dict(ols=(ols_imp(X[a], y[a]), ols_imp(X[b], y[b])), mdi=(mdi(ma), mdi(mb)), perm=(perm_imp(ma, X[b], y[b], rng), perm_imp(mb, X[a], y[a], rng)))
    SR = Parallel(n_jobs=N_JOBS)(delayed(split)(i) for i in range(NS)); out['split_half'] = {}
    for k in ['ols', 'mdi', 'perm']:
        t = np.array([tau(r[k][0], r[k][1]) for r in SR]); t3 = np.array([top3(r[k][0]) == top3(r[k][1]) for r in SR])
        out['split_half'][k] = dict(B=NS, tau_mean=float(t.mean()), tau_q05=float(np.quantile(t, .05)), tau_q95=float(np.quantile(t, .95)), top3_agree=float(t3.mean()))
    save('stability.json', out)
    print('stability: top-3 recovery (bootstrap) perm/ols/mdi =', *(f"{out['bootstrap'][k]['top3_exact_recovery']:.3f}" for k in ('perm', 'ols', 'mdi')))


# ------------------------------------------------------------------------------------------------------------- tables -------
NMI = {'time_h': 'Daily time on social media', 'easily_distracted': 'Easily distracted by social media', 'restless': 'Restless without social media',
       'validation': 'Seeks validation from social media', 'compare': 'Compares self with others', 'interest': 'Fluctuation in interest in daily activities',
       'sleep': 'Sleep problems'}
def ols_table():
    """Standardised OLS coefficients with HC3 95% CIs (primary specification S2, N = 478). Needs the data; otherwise the stored CSV is used."""
    if not data_available(): return pd.read_csv(find('ols_figure_table.csv'))
    df = prep(sample(raw(), 'P')); cols = feats('S2'); X = design(df, cols); y = df[Y].to_numpy(float)
    f = fit_ols(X, y); n, p = f['n'], f['p']; dfree = n - p; tc = stats.t.ppf(.975, dfree)
    sd = df[cols].std().to_numpy(); sy = y.std(ddof=1); b = f['b'][1:]; se = f['se_hc3'][1:]; pv = 2 * stats.t.sf(abs(f['b'] / f['se_hc3']), dfree)[1:]
    beh = S1; rows = []
    for i, c in enumerate(cols):
        k = sd[i] / sy; std = c not in ('male', 'gender_other')
        rows.append(dict(group='Behavioural predictor' if c in beh else 'Demographic covariate', var=LABEL[c], key=c,
                         scaling='per 1 SD of predictor, in SDs of outcome' if std else 'raw: score-point difference vs female (0/1 indicator, not standardised)',
                         est=b[i] * k if std else b[i], lo=(b[i] - tc * se[i]) * k if std else b[i] - tc * se[i], hi=(b[i] + tc * se[i]) * k if std else b[i] + tc * se[i],
                         b_raw=b[i], se_hc3=se[i], p=pv[i]))
    T = pd.DataFrame(rows)
    try:                                                     # consistency with the OLS stage
        J = load('ols.json')['P_S2']; flags = []
        for i, c in enumerate(cols):
            for nm, new, old in [('coef', b[i], J['coef'][i + 1]), ('se_hc3', se[i], J['se_hc3'][i + 1]), ('p', pv[i], J['p_hc3'][i + 1]), ('std_beta', b[i] * sd[i] / sy, J['std_beta'][c])]:
                if abs(new - old) > 6e-5: flags.append(f'{c} {nm}: {new:.4f} vs ols.json {old:.4f}')
        print('ols table vs ols.json:', flags if flags else 'consistent')
    except FileNotFoundError: pass
    return T
def write_ols_table(T):
    T.to_csv(os.path.join(RES, 'ols_figure_table.csv'), index=False, float_format='%.4f')
    with open(os.path.join(RES, 'ols_figure_table.md'), 'w') as fh:
        fh.write('| Group | Predictor | Estimate | 95% CI (HC3) | p (HC3) | Scale |\n|---|---|---|---|---|---|\n')
        for _, r in T.iterrows():
            fh.write(f"| {r.group} | {r['var']} | {r.est:.3f} | [{r.lo:.3f}, {r.hi:.3f}] | {('<0.001' if r.p < 0.001 else format(r.p, '.3f'))} | "
                     f"{'standardised' if r.key not in ('male', 'gender_other') else 'raw (score points)'} |\n")
def rank_mat(a): return np.array([stats.rankdata(-r) for r in a])
def importance_tables(NB):
    S = load('stability.json'); names = S['names']; z = np.load(find(f'importance_boot_B{NB}.npz')); ref = {'perm': np.array(S['full']['rf_perm_cv']), 'mdi': np.array(S['full']['rf_mdi'])}
    out = {}
    for k in ('perm', 'mdi'):
        A = z[k]; lo, hi = np.quantile(A, [.025, .975], axis=0); Rr = rank_mat(A)
        out[k] = pd.DataFrame(dict(predictor=[NMI[c] for c in names], key=names, estimate=ref[k], boot_median=np.median(A, 0), ci_lo=lo, ci_hi=hi, rank_full=stats.rankdata(-ref[k]),
                                   rank_median=np.median(Rr, 0), rank_q05=np.quantile(Rr, .05, 0), rank_q95=np.quantile(Rr, .95, 0), p_top3=(Rr <= 3).mean(0), p_rank1=(Rr == 1).mean(0)))
    out['perm'].to_csv(os.path.join(RES, 'rf_importance_table.csv'), index=False, float_format='%.4f'); out['mdi'].to_csv(os.path.join(RES, 'rf_impurity_table.csv'), index=False, float_format='%.4f')
    return out
CORR_ORDER = ['depressed', 'interest', 'compare', 'sleep', 'easily_distracted', 'restless', 'validation', 'time_h']
def corr_matrix():
    if not data_available(): return pd.read_csv(find('correlations_P.csv'), index_col=0)
    d = prep(sample(raw(), 'P')); C = d[CORR_ORDER].corr().round(4); C.to_csv(os.path.join(RES, 'correlations_P.csv')); return C
def stage_tables(NB):
    os.makedirs(RES, exist_ok=True)
    T = ols_table(); write_ols_table(T); importance_tables(NB); corr_matrix()
    cmp_ = load('compare.json'); rows = []
    for tag, r in cmp_.items():
        if tag not in ('S1_P', 'S2_P', 'S3_P', 'S4_P', 'S2_A1'): continue
        for m in ('ols', 'spline', 'rf'): rows.append(dict(specification=tag, model=m, pooled_R2_mean=r['R2_' + m]['mean'], R2_sd_over_repeats=r['R2_' + m]['sd_repeats']))
    pd.DataFrame(rows).to_csv(os.path.join(RES, 'table1_cv_performance.csv'), index=False, float_format='%.4f')
    rows = []
    for tag, r in cmp_.items():
        if tag not in ('S1_P', 'S2_P', 'S3_P', 'S4_P', 'S2_A1'): continue
        for k in ('rf_minus_ols', 'spline_minus_ols', 'rf_minus_spline'):
            v = r[k]; rows.append(dict(specification=tag, contrast=k, R2_diff=v['R2_diff_mean'], NB_CI95_lo=v['NB_CI95_R2diff'][0], NB_CI95_hi=v['NB_CI95_R2diff'][1], NB_t=v['NB_t'], NB_p=v['NB_p'],
                                       boot_CI95_lo=v['boot_resp_CI95_R2diff'][0], boot_CI95_hi=v['boot_resp_CI95_R2diff'][1], prop_respondents_first_better=v['prop_respondents_a_better']))
    pd.DataFrame(rows).to_csv(os.path.join(RES, 'table2_cv_paired_differences.csv'), index=False, float_format='%.4f')
    S = load('stability.json'); rows = []
    for part in ('bootstrap', 'split_half'):
        for k, lab in (('ols', 'OLS |std. beta|'), ('perm', 'RF permutation'), ('mdi', 'RF impurity')):
            v = S[part][k]; rows.append(dict(design=part, measure=lab, B=v['B'], tau_mean=v['tau_mean'], tau_q05=v['tau_q05'], tau_q95=v['tau_q95'], top3=v.get('top3_exact_recovery', v.get('top3_agree'))))
    pd.DataFrame(rows).to_csv(os.path.join(RES, 'table3_stability.csv'), index=False, float_format='%.4f')
    nl = load('nonlin.json'); rows = []
    for c in CORE:
        b = nl['pdp_boot'][c]
        for i, lv in enumerate([1, 2, 3, 4, 5]): rows.append(dict(predictor=c, level=lv, n=nl['level_counts'][c][str(lv)], pdp=nl['pdp_obs'][c][i], boot_lo90=b['lo'][i], boot_hi90=b['hi'][i]))
    pd.DataFrame(rows).to_csv(os.path.join(RES, 'fig2_partial_dependence_data.csv'), index=False, float_format='%.4f')
    nr = []                                                  # selection-aware null results (read from the stored simulation outputs; not recomputed here)
    for k, lab in (('ord', 'ordered-logit (primary)'), ('cont', 'continuous'), ('disc', 'rounded and clipped')):
        j = json.load(open(sim_path(f'null_{k}_B2000.json')))
        nr.append(dict(null=lab, B=j['B'], p_Gmax=j['p_Gmax'], p_Dmax=j['p_Dmax'], p_G_interest=j['p_G']['interest'], p_G_compare=j['p_G']['compare'], p_G_sleep=j['p_G']['sleep'],
                       q95_Gmax=j['q95_Gmax'], observed_Gmax=nl['Gmax_obs']))
    pd.DataFrame(nr).to_csv(os.path.join(RES, 'fig3a_null_summary_from_saved_simulations.csv'), index=False, float_format='%.4f')
    pw = json.load(open(sim_path('power.json'))); rows = []
    for N in (250, 478, 1000):
        for q in (0, .15, .30, .45):
            r = pw[f'N{N}_hinge_{q}'] if q > 0 else pw[f'N{N}_lin_0']
            rows.append(dict(N=N, extra_rise_levels3to5=2 * q, rej_Gmax=r['rej_Gmax'] if q > 0 else .05, rej_Ftest=r['rej_Ftest'], reps=r['reps']))
    pd.DataFrame(rows).to_csv(os.path.join(RES, 'fig3b_power_curves_from_saved_simulations.csv'), index=False, float_format='%.4f')
    print('tables written to', RES)


# ------------------------------------------------------------------------------------------------------------- figures ------
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
_INK, _INK2 = '#0b0b0b', '#52514e'
def _rc(face='white', grid='#d9d8d4', left=False):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': _INK2, 'axes.labelcolor': _INK if face == 'white' else _INK2,
                         'xtick.color': _INK2, 'ytick.color': _INK if face == 'white' else _INK2, 'text.color': _INK, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': left, 'axes.grid': True, 'grid.color': grid, 'grid.linewidth': .5 if face == 'white' else .6, 'axes.axisbelow': True,
                         'figure.facecolor': face, 'axes.facecolor': face, 'savefig.facecolor': face})
def _save(fig, name, exts=('png', 'pdf')):
    os.makedirs(FIG, exist_ok=True)
    for e in exts: fig.savefig(f'{FIG}/{name}.{e}', dpi=300, bbox_inches='tight')
    plt.close(fig)
def fig1_ols(T):
    BLUE, GREY, GRID = '#1f5fae', '#6b6a66', '#d9d8d4'; _rc(left=False)
    beh = S1; std_rows = T[T.key.isin(beh + ['age'])].reset_index(drop=True); g_rows = T[~T.key.isin(beh + ['age'])].reset_index(drop=True)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.6, 5.2), gridspec_kw={'height_ratios': [8.6, 2.6], 'hspace': .55})
    ypos = [len(std_rows) - 1 - i for i in range(7)] + [0]
    for (i, r), y0 in zip(std_rows.iterrows(), ypos):
        col = BLUE if r.group.startswith('Beh') else GREY
        a1.plot([r.lo, r.hi], [y0, y0], color=col, lw=1.8, solid_capstyle='butt'); a1.scatter([r.est], [y0], s=30, color=col, zorder=3, edgecolor='white', linewidth=.8)
        a1.text(1.02, y0, f'{r.est:.3f} [{r.lo:.3f}, {r.hi:.3f}]', va='center', fontsize=7.5, color=_INK2, transform=a1.get_yaxis_transform())
    a1.axvline(0, color=_INK, lw=1); a1.set_yticks(ypos); a1.set_yticklabels(std_rows['var'], fontsize=8.5); a1.set_ylim(-.8, 8.3); a1.set_xlim(-.30, .40)
    a1.axhline(.5, color=GRID, lw=1, ls='-'); a1.grid(axis='y', visible=False)
    a1.text(-.30, 7.75, 'Behavioural predictors', fontsize=8.5, fontweight='bold', color=BLUE, ha='left'); a1.text(-.30, .7, 'Demographic covariate', fontsize=8.5, fontweight='bold', color=GREY, ha='left')
    a1.set_xlabel('Standardised coefficient (SDs of depressed-mood score per SD of predictor)')
    a1.text(1.02, 7.75, 'β  [95% CI]', fontsize=7.5, color=_INK2, transform=a1.get_yaxis_transform(), va='center'); a1.set_title('a', loc='left', fontsize=10, fontweight='bold', x=-.62, y=1.0)
    for (i, r) in g_rows.iterrows():
        y0 = 1 - i; a2.plot([r.lo, r.hi], [y0, y0], color=GREY, lw=1.8); a2.scatter([r.est], [y0], s=30, color=GREY, zorder=3, edgecolor='white', linewidth=.8)
        a2.text(1.02, y0, f'{r.est:.3f} [{r.lo:.3f}, {r.hi:.3f}]', va='center', fontsize=7.5, color=_INK2, transform=a2.get_yaxis_transform())
    a2.axvline(0, color=_INK, lw=1); a2.set_yticks([1, 0]); a2.set_yticklabels(g_rows['var'], fontsize=8.5); a2.set_ylim(-.7, 1.7); a2.grid(axis='y', visible=False)
    a2.set_xlabel('Unstandardised difference in score points (reference category: female)'); a2.set_title('b', loc='left', fontsize=10, fontweight='bold', x=-.62, y=1.0)
    _save(fig, 'fig1_ols_coefficients', ('png', 'pdf', 'svg'))
def fig2_pdp(nl):
    ORANGE = '#eb6834'; _rc('#fcfcfb', '#e4e3df', left=True)
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7), sharey=True); nm = {'interest': 'Interest fluctuation', 'compare': 'Social comparison', 'sleep': 'Sleep problems'}; lv = np.arange(1, 6)
    for a, c in zip(ax, CORE):
        b = nl['pdp_boot'][c]; a.fill_between(lv, b['lo'], b['hi'], color=ORANGE, alpha=.2, lw=0); a.plot(lv, nl['pdp_obs'][c], color=ORANGE, lw=2, marker='o', ms=4)
        sl = np.polyfit(lv, nl['pdp_obs'][c], 1); a.plot(lv, np.polyval(sl, lv), color=_INK2, lw=1.2, ls='--'); cnt = nl['level_counts'][c]
        a.set_title(nm[c], loc='left', fontsize=9); a.set_xticks(lv); a.set_xticklabels([f"{l}\nn={cnt[str(l)]}" for l in lv], fontsize=7); a.set_xlabel('Response level (1–5)'); a.set_ylim(2.7, 4.05)
    ax[0].set_ylabel('Predicted depressed/down (1–5)')
    ax[0].text(1.1, 3.9, 'RF partial dependence\n(90% bootstrap band)', color=ORANGE, fontsize=7, va='top'); ax[0].text(3.2, 3.1, 'best straight line', color=_INK2, fontsize=7)
    _save(fig, 'fig2_partial_dependence')
def fig3_null_power(nl):
    BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'; _rc('#fcfcfb', '#e4e3df', left=True)
    pw = json.load(open(sim_path('power.json'))); br = json.load(open(sim_path('power_bracket_N478.json')))['rows']
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.1)); Bn = 2000
    NUL = [('ord', 'Ordinal', BLUE, '-'), ('cont', 'Continuous', AQUA, '-'), ('disc', 'Clipped', '#8a8a85', '--')]; bins = np.linspace(0.015, 0.14, 34); txt = []
    for k, lab_, c, ls_ in NUL:
        G = np.load(sim_path(f'null_{k}_B{Bn}.npz'))['G'].max(1); js = json.load(open(sim_path(f'null_{k}_B{Bn}.json')))
        h, _ = np.histogram(G, bins=bins, density=True); ax[0].step(bins[:-1], h, where='post', color=c, lw=1.8, ls=ls_); txt.append((lab_, js['p_Gmax'], c))
    ax[0].axvline(nl['Gmax_obs'], color=ORANGE, lw=2); ax[0].text(nl['Gmax_obs'] - .002, ax[0].get_ylim()[1] * .97, 'observed', color=ORANGE, fontsize=7.5, va='top', ha='right')
    for i, (l, p, c) in enumerate(txt): ax[0].text(.141, ax[0].get_ylim()[1] * (.88 - .09 * i), f'{l}: p = {p:.3f}', color=c, fontsize=7.5, ha='right', va='top')
    ax[0].set_xlim(.015, .141); ax[0].set_xlabel('Max best-breakpoint statistic (3 predictors)'); ax[0].set_ylabel('Density of null statistic'); ax[0].set_title('a  Three linear nulls', loc='left', fontsize=9)
    sizes = [0, .15, .30, .45]; xs_ = [2 * q for q in sizes]
    for N, ls_, dy in [(250, ':', .035), (478, '-', 0), (1000, '--', 0)]:
        ys = [pw[f'N{N}_hinge_{q}']['rej_Gmax'] if q > 0 else .05 for q in sizes]; ax[1].plot(xs_, ys, color=ORANGE, ls=ls_, lw=2, marker='o', ms=3.5); ax[1].text(.93, ys[-1] + dy, f'N = {N}', color=ORANGE, fontsize=7.5, va='center')
    for i, q in enumerate(sizes[1:], start=1):
        r = br[i]; v = [r['Gmax_disc'], r['Gmax_cont'], r['Gmax_ord'], r['Gmax_own_linear_crit']]; ax[1].plot([xs_[i]] * 2, [min(v), max(v)], color=ORANGE, lw=5, alpha=.25, solid_capstyle='butt')
    ys = [pw['N478_hinge_%s' % q]['rej_Ftest'] if q > 0 else pw['N478_lin_0']['rej_Ftest'] for q in sizes]; ax[1].plot(xs_, ys, color=BLUE, lw=2, marker='o', ms=3.5); ax[1].text(.93, ys[-1] - .035, 'F-test, N = 478', color=BLUE, fontsize=7.5, va='center')
    ax[1].axhline(.8, color=_INK2, lw=.8, ls='--'); ax[1].text(.02, .82, '80% power', fontsize=7, color=_INK2)
    ax[1].set_xlabel('Extra rise, levels 3→5 (scale points)'); ax[1].set_ylabel('Power (α = 0.05)'); ax[1].set_ylim(0, 1.04); ax[1].set_xlim(-.04, 1.4); ax[1].set_title('b  Detection of a hinge at level 3', loc='left', fontsize=9)
    _save(fig, 'fig3_null_power')
def _imp_fig(T, xlabel, fn, color):
    _rc(left=False); o = T.sort_values('estimate').reset_index(drop=True); fig, ax = plt.subplots(figsize=(6.8, 3.7))
    for i, r in o.iterrows():
        ax.plot([r.ci_lo, r.ci_hi], [i, i], color=color, lw=1.8, solid_capstyle='butt'); ax.scatter([r.estimate], [i], s=34, color=color, zorder=3, edgecolor='white', linewidth=.8)
        ax.text(1.02, i, f'{r.estimate:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]   {int(r.rank_full)} ({round(r.rank_q05):d}–{round(r.rank_q95):d})', transform=ax.get_yaxis_transform(), va='center', fontsize=7.5, color=_INK2)
    ax.text(1.02, len(o) - .35, 'Importance [95% CI]   Rank (90% interval)', transform=ax.get_yaxis_transform(), fontsize=7.5, color=_INK2, va='center')
    ax.set_yticks(range(len(o))); ax.set_yticklabels(o.predictor, fontsize=8.5); ax.set_ylim(-.6, len(o) - .1); ax.grid(axis='y', visible=False); ax.axvline(0, color=_INK, lw=1)
    ax.set_xlim(left=min(0, o.ci_lo.min() - .005)); ax.set_xlabel(xlabel); _save(fig, fn, ('png', 'pdf', 'svg'))
def figA1_corr(C):
    plt.rcdefaults(); NM = ['Depressive affect', 'Interest fluctuation', 'Social comparison', 'Sleep problems', 'Distractibility', 'Restlessness', 'Validation-seeking', 'Daily time on\nsocial media']
    M = C.to_numpy(); fig, ax = plt.subplots(figsize=(6.2, 5.2)); im = ax.imshow(M, cmap='Blues', vmin=0, vmax=1)
    for i in range(8):
        for j in range(8): ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=8, color='white' if M[i, j] > .6 else '#0b0b0b')
    ax.set_xticks(range(8)); ax.set_xticklabels(NM, rotation=40, ha='right', fontsize=8); ax.set_yticks(range(8)); ax.set_yticklabels([n.replace('\n', ' ') for n in NM], fontsize=8)
    for s in ax.spines.values(): s.set_visible(False)
    cb = fig.colorbar(im, fraction=.046, pad=.03); cb.ax.tick_params(labelsize=8); cb.set_label('Pearson r', fontsize=8); _save(fig, 'figA1_correlations')
def stage_figures(NB):
    T = ols_table(); nl = load('nonlin.json'); I = importance_tables(NB); C = corr_matrix()
    fig1_ols(T); fig2_pdp(nl); fig3_null_power(nl)
    _imp_fig(I['perm'], 'Permutation importance (increase in out-of-sample MSE when the predictor is shuffled)', 'fig4_rf_permutation_importance', '#1f5fae')
    figA1_corr(C)
    _imp_fig(I['mdi'], 'Impurity-based importance (mean decrease in impurity; fraction of total)', 'figS1_rf_impurity_importance', '#6b6a66')
    print('figures written to', FIG)


# --------------------------------------------------------------------------------------- check against manuscript -----------
def _resolve(src):
    kind, _, rest = src.partition(':'); path, _, expr = rest.partition('::')
    if kind == 'json': obj = load(path)
    elif kind == 'csv': obj = pd.read_csv(find(path))
    elif kind == 'simjson': obj = json.load(open(sim_path(path)))
    else: raise ValueError(src)
    return obj, expr
def stage_check(tol_extra=0.0):
    p = os.path.join(ROOT, 'docs', 'manuscript_numbers.csv')
    if not os.path.exists(p): sys.exit('docs/manuscript_numbers.csv missing')
    M = pd.read_csv(p); bad = 0; rows = []
    for _, r in M.iterrows():
        obj, expr = _resolve(r.source)
        v = eval(expr, {'np': np, 'obj': obj, 'abs': abs, 'max': max, 'min': min, 'len': len, 'round': round}) if expr else obj
        val = float(v); dec = int(r.decimals); ok = abs(val - float(r.manuscript_value)) <= 10 ** (-dec) * 0.5 + 1e-9      # value rounds to the printed number
        rows.append(dict(id=r.id, manuscript_text=r.manuscript_text, manuscript_value=r.manuscript_value, recomputed=round(val, dec + 2), status='OK' if ok else 'MISMATCH'))
        bad += (not ok)
    D = pd.DataFrame(rows); print(D.to_string(index=False)); D.to_csv(os.path.join(RES, 'manuscript_check.csv'), index=False)
    print(f'\n{len(D) - bad}/{len(D)} manuscript numbers reproduced from stored results'); return bad


# ------------------------------------------------------------------------------------------------------------------ main ----
def main():
    global RES, FIG, OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--saved', action='store_true', help='skip all heavy stages; rebuild tables and figures from stored results')
    ap.add_argument('--quick', action='store_true', help='reduced replicates, smoke test only; writes to quick_output/')
    ap.add_argument('--sensitivity', action='store_true', help='also run nested CV for S1, S3, S4 (sample P) and S2 on the age 15-45 sample')
    ap.add_argument('--check', action='store_true', help='compare stored results with the numbers printed in the manuscript (docs/manuscript_numbers.csv)')
    ap.add_argument('--stages', default=None, help='comma list of: audit,ols,descriptives,cv,compare,nonlin,stability,tables,figures,check')
    ap.add_argument('--out', default=None, help='output directory (default: results/ and figures/ in the repository)')
    ap.add_argument('--jobs', type=int, default=2)
    a = ap.parse_args(); global N_JOBS; N_JOBS = a.jobs
    allst = ['audit', 'ols', 'descriptives', 'cv', 'compare', 'nonlin', 'stability', 'tables', 'figures']
    stages = a.stages.split(',') if a.stages else (['check'] if a.check else (['tables', 'figures'] if a.saved else allst))
    if a.quick:
        OUT = os.path.join(ROOT, 'quick_output'); RES = os.path.join(OUT, 'results'); FIG = os.path.join(OUT, 'figures'); REP, NT, BP, NB, NS = 1, 300, 20, 20, 10
    else:
        if a.out: OUT = os.path.abspath(a.out); RES = os.path.join(OUT, 'results'); FIG = os.path.join(OUT, 'figures')
        REP, NT, BP, NB, NS = 10, 300, 300, 500, 300
    os.makedirs(RES, exist_ok=True)
    if a.quick or a.out:                       # keep stored simulation outputs and any saved raw files visible to downstream stages
        pass
    needs_data = [s for s in stages if s in ('audit', 'ols', 'descriptives', 'cv', 'nonlin', 'stability')]
    if needs_data:
        if not data_available(): raw()
        h = sha256_of_data()
        if h != DATA_SHA256: print(f'WARNING: data/smmh.csv SHA-256 {h} differs from the analysed file {DATA_SHA256}; results will not match the manuscript.')
    warnings.filterwarnings('ignore', category=FutureWarning)
    t0 = time.time(); stat = 0
    for s in stages:
        print(f'\n=== {s} ===', flush=True)
        if s == 'audit': stage_audit()
        elif s == 'ols': stage_ols()
        elif s == 'descriptives': stage_descriptives()
        elif s == 'cv':
            stage_cv('S2', 'P', REP, NT)
            if a.sensitivity:
                for sp, sm in (('S1', 'P'), ('S3', 'P'), ('S4', 'P'), ('S2', 'A1')): stage_cv(sp, sm, REP, NT)
        elif s == 'compare': stage_compare()
        elif s == 'nonlin': stage_nonlin(BP)
        elif s == 'stability': stage_stability(NB, NS)
        elif s == 'tables': stage_tables(NB)
        elif s == 'figures': stage_figures(NB)
        elif s == 'check': stat = stage_check()
        else: sys.exit('unknown stage ' + s)
    json.dump(PARAMS, open(os.path.join(RES, 'analysis_parameters.json'), 'w'), indent=1)
    print(f'\ndone in {time.time() - t0:.0f}s'); sys.exit(1 if stat else 0)

if __name__ == '__main__':
    main()
