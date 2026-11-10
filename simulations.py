#!/usr/bin/env python3
"""
OPTIONAL, computationally expensive simulations for the manuscript
"Beyond Prediction: Interpretive Reliability in Computational Social Science".

The final saved outputs of every stage below are shipped in results/simulations/ and are what analysis.py reads (to build Table 2 and Figure 3).
Nothing here is run by analysis.py. Re-running is only needed to regenerate those outputs.

    python simulations.py --stage nulls            # selection-aware linear nulls (ordinal, continuous, rounded-and-clipped), B = 2000 each (about 35 min)
    python simulations.py --stage initial          # first-pass nulls with B = 300 (continuous, rounded-and-clipped) cited in Methods 2.7 (about 8 min)
    python simulations.py --stage retune           # calibration check: forest hyperparameters re-tuned inside each simulated dataset, B = 300 (about 40 min)
    python simulations.py --stage size             # independent false-positive check, 400 fresh datasets per generating process (about 25 min)
    python simulations.py --stage power            # power simulations, N = 250/478/1000, 6 scenarios, 120 replicates each (about 60 min)
    python simulations.py --stage bracket          # re-thresholds the N = 478 power replicates at the critical value of each null (seconds)
    python simulations.py --stage all              # everything, in dependency order (about 2.5 h on 2 cores)
    python simulations.py --quick --stage all      # a few replicates per stage into quick_output/simulations/ (about 5 min); smoke test
    python simulations.py --out some/dir ...       # write somewhere other than results/simulations/ (so the shipped outputs are never overwritten)

Replicate i is seeded by i alone (see simulation_parameters.json), so a run with fewer replicates reproduces the first replicates of the full run exactly.
Requires results/nonlin.json (full-sample Random Forest hyperparameters and the observed statistics), produced by analysis.py.
"""
import argparse, collections, json, os, sys, time, warnings
import numpy as np, pandas as pd
from scipy import stats
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import analysis as A
from analysis import (prep, sample, raw, feats, design, fit_ols, pdp_levels, dev_stat, hinge_stat, holm, CORE, S1, Y)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GridSearchCV
from joblib import Parallel, delayed

SIM = os.path.join(ROOT, 'results', 'simulations'); OUT = SIM; NJ = 2
PARAMS = dict(
    forest='RandomForestRegressor, 300 trees, hyperparameters fixed at the full-sample cross-validated choice stored in results/nonlin.json (full_sample_rf_hp)',
    pdp_levels=[1, 2, 3, 4, 5], statistics='D = RMS deviation of the 5-point partial dependence from its best straight line; G = best continuous hinge (breakpoint over levels 2-4), RMS reduction in squared error; Gmax / Dmax = maximum over interest, compare, sleep',
    p_value='(1 + #{null >= observed}) / (B + 1)',
    initial=dict(B=300, continuous=dict(resample_seed='i', forest_seed='i'), rounded_clipped=dict(resample_seed='10000 + i', forest_seed='i'),
                 generation='OLS fitted values + residuals resampled with replacement (rounded to integers and clipped to 1-5 for the rounded-and-clipped null)'),
    nulls=dict(B=2000, rounded_clipped=dict(resample_seed='50000 + i', forest_seed='70000 + i'), continuous=dict(resample_seed='60000 + i', forest_seed='75000 + i'),
               ordinal=dict(resample_seed='80000 + i', forest_seed='85000 + i', generation='ordered-logit fit (BFGS) on standardised predictors; outcomes sampled from the fitted category probabilities')),
    retune=dict(B=300, data='same simulated outcomes as the rounded-and-clipped initial null (resample seed 10000 + i)', inner_cv='5-fold, seed i, 100 trees, grid min_samples_leaf {1,3,5,10,20} x max_features {0.33,0.6,1.0}', final_forest='300 trees, seed i',
                observed_retuned_cv_seeds=[1, 2, 3, 4, 5, 6, 7, 8]),
    size=dict(R=400, seed='7000000 + 100000*dgp + rep with dgp 0 = rounded-and-clipped, 1 = continuous, 2 = ordinal', forest_seed='rep', rows='resampled from the observed data', critical_values='95th percentiles of the saved B = 2000 nulls'),
    power=dict(REP=120, N=[250, 478, 1000], scenarios=['linear', 'hinge 0.15', 'hinge 0.30', 'hinge 0.45', 'step 0.30', 'step 0.60'], hinge_form='0.15/0.30/0.45 * max(interest - 3, 0) (extra rise over levels 3-5 = 2 x slope)', step_form='0.30/0.60 * 1(interest >= 4)',
               seed='1000000*N + 10000*scenario_index + rep', forest_seed='rep', test_set='1500 fresh rows, seed 999 + rep', spline='quadratic B-spline, 3 knots, ridge alpha 100 (fixed)', critical_values='95th percentile of the linear scenario at the same N'))

def wr(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), 'w') as f: json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
def stored(name):                                    # a file from OUT, else the shipped copy
    for b in (OUT, SIM):
        p = os.path.join(b, name)
        if os.path.exists(p): return p
    raise FileNotFoundError(name)
def ordered_logit_fit(Z, y, K=5):
    from scipy.optimize import minimize
    n, p = Z.shape; cum = np.array([(y <= k).mean() for k in range(1, K)]); th0 = np.log(cum / (1 - cum))
    def unpack(w): b = w[:p]; th = np.cumsum(np.concatenate([[w[p]], np.exp(w[p + 1:])])); return b, th
    def nll(w):
        b, th = unpack(w); eta = Z @ b; ub = np.concatenate([th, [np.inf]]); lb = np.concatenate([[-np.inf], th])
        pr = 1 / (1 + np.exp(-(ub[y.astype(int) - 1] - eta))) - 1 / (1 + np.exp(-(lb[y.astype(int) - 1] - eta))); return -np.sum(np.log(np.clip(pr, 1e-12, 1)))
    w0 = np.concatenate([np.zeros(p), [th0[0]], np.log(np.maximum(np.diff(th0), 1e-3))]); r = minimize(nll, w0, method='BFGS'); b, th = unpack(r.x); return b, th, r
def ordered_logit_sample(Z, b, th, rng):
    eta = Z @ b; cp = 1 / (1 + np.exp(-(th[None, :] - eta[:, None]))); u = rng.random(len(eta)); return 1 + (u[:, None] > cp).sum(1).astype(float)
def setup():
    nl = A.load('nonlin.json'); hp = nl['full_sample_rf_hp']
    d = prep(sample(raw(), 'P')); cols = feats('S2'); X = d[cols].to_numpy(float); y = d[Y].to_numpy(float); n = len(y)
    base = fit_ols(design(d, cols), y); return dict(nl=nl, hp=hp, d=d, cols=cols, X=X, y=y, n=n, base=base, mu=base['yhat'], res=base['e'])
pv = lambda a, o: float((1 + (a >= o).sum()) / (len(a) + 1))


def stage_initial(S, B):
    X, n, hp, cols, mu, res_e, nl = S['X'], S['n'], S['hp'], S['cols'], S['mu'], S['res'], S['nl']
    def one(i, disc):
        rng = np.random.default_rng(10_000 * (1 if disc else 0) + i); ys = mu + rng.choice(res_e, n, replace=True)
        if disc: ys = np.clip(np.rint(ys), 1, 5)
        m = RandomForestRegressor(n_estimators=300, random_state=i, n_jobs=1, **hp).fit(X, ys); p = pdp_levels(m, X, cols); return {c: p[c].tolist() for c in CORE}
    out = {'description': 'Initial selection-aware null tests (B simulated datasets) from the first analysis pass; cited in the manuscript Methods (2.7).', 'B': B}
    Gobs, Dobs = nl['G_obs'], nl['D_obs']
    for disc, nm in [(False, 'N1_continuous'), (True, 'N2_discretised')]:
        r = Parallel(n_jobs=NJ)(delayed(one)(i, disc) for i in range(B)); P = {c: np.array([x[c] for x in r]) for c in CORE}
        arr = {c: np.array([dev_stat(v) for v in P[c]]) for c in CORE}; garr = {c: np.array([hinge_stat(v) for v in P[c]]) for c in CORE}
        dmax = np.max([arr[c] for c in CORE], 0); gmax = np.max([garr[c] for c in CORE], 0)
        pvv = {c: pv(arr[c], Dobs[c]) for c in CORE}; ad = holm([pvv[c] for c in CORE])
        out[nm] = dict(B=B, p_D_fixed_predictor=pvv, p_D_holm=dict(zip(CORE, map(float, ad))), p_G_best_breakpoint_per_predictor={c: pv(garr[c], Gobs[c]) for c in CORE},
                       p_Dmax_selection_over_predictors=pv(dmax, nl['Dmax_obs']), p_Gmax_selection_over_predictors_and_breakpoints=pv(gmax, nl['Gmax_obs']),
                       null_q50_D={c: float(np.quantile(arr[c], .5)) for c in CORE}, null_q95_D={c: float(np.quantile(arr[c], .95)) for c in CORE},
                       null_q95_Dmax=float(np.quantile(dmax, .95)), null_q95_Gmax=float(np.quantile(gmax, .95)))
        print(nm, 'p_Gmax', round(out[nm]['p_Gmax_selection_over_predictors_and_breakpoints'], 4), flush=True)
    wr('initial_nulls_B300.json' if B == 300 else f'initial_nulls_B{B}.json', out)


def stage_nulls(S, B, which=('ord', 'disc', 'cont')):
    X, n, hp, cols, mu, res_e, nl, d = S['X'], S['n'], S['hp'], S['cols'], S['mu'], S['res'], S['nl'], S['d']
    Zs = (X - X.mean(0)) / X.std(0); ob, oth, _ = ordered_logit_fit(Zs, S['y'])
    for w in which:
        SEED = {'disc': (50_000, 70_000), 'cont': (60_000, 75_000), 'ord': (80_000, 85_000)}[w]
        def gen(i):
            rng = np.random.default_rng(SEED[0] + i)
            if w == 'disc': return np.clip(np.rint(mu + rng.choice(res_e, n)), 1, 5)
            if w == 'cont': return mu + rng.choice(res_e, n)
            return ordered_logit_sample(Zs, ob, oth, rng)
        def one(i):
            ys = gen(i); m = RandomForestRegressor(n_estimators=300, random_state=SEED[1] + i, n_jobs=1, **hp).fit(X, ys); p = pdp_levels(m, X, cols); f = fit_ols(design(d, cols), ys)
            return dict(D=[dev_stat(p[c]) for c in CORE], G=[hinge_stat(p[c]) for c in CORE], var=float(ys.var(ddof=1)), res_sd=float(np.sqrt(f['s2'])), r2=float(f['r2']),
                        out=float(np.mean((ys < 1) | (ys > 5))), freq=[float(np.mean(np.rint(ys) == k)) for k in range(1, 6)], mean=float(ys.mean()))
        t0 = time.time(); R = Parallel(n_jobs=NJ)(delayed(one)(i) for i in range(B)); print(w, 'time', round(time.time() - t0), flush=True)
        D = np.array([r['D'] for r in R]); G = np.array([r['G'] for r in R]); os.makedirs(OUT, exist_ok=True); np.savez(os.path.join(OUT, f'null_{w}_B{B}.npz'), D=D, G=G)
        Do = np.array([nl['D_obs'][c] for c in CORE]); Go = np.array([nl['G_obs'][c] for c in CORE])
        res = dict(null=w, B=B, mc_se_at_0p05=float(np.sqrt(.05 * .95 / B)), p_D=dict(zip(CORE, [pv(D[:, j], Do[j]) for j in range(3)])), p_G=dict(zip(CORE, [pv(G[:, j], Go[j]) for j in range(3)])),
                   p_Dmax=pv(D.max(1), Do.max()), p_Gmax=pv(G.max(1), Go.max()), q95_Dmax=float(np.quantile(D.max(1), .95)), q95_Gmax=float(np.quantile(G.max(1), .95)),
                   q95_G=dict(zip(CORE, [float(np.quantile(G[:, j], .95)) for j in range(3)])), q95_D=dict(zip(CORE, [float(np.quantile(D[:, j], .95)) for j in range(3)])), median_Gmax=float(np.median(G.max(1))),
                   diag=dict(outcome_var=float(np.mean([r['var'] for r in R])), mean=float(np.mean([r['mean'] for r in R])), ols_resid_sd=float(np.mean([r['res_sd'] for r in R])), ols_r2=float(np.mean([r['r2'] for r in R])),
                             share_outside_1_5=float(np.mean([r['out'] for r in R])), category_freq=np.mean([r['freq'] for r in R], axis=0).round(4).tolist()))
        wr(f'null_{w}_B{B}.json', res); print(w, 'p_Gmax', round(res['p_Gmax'], 4), flush=True)


def stage_retune(S, B):
    X, n, hp0, cols, mu, res_e, nl = S['X'], S['n'], S['hp'], S['cols'], S['mu'], S['res'], S['nl']; y = S['y']
    GRID = {'min_samples_leaf': [1, 3, 5, 10, 20], 'max_features': [0.33, 0.6, 1.0]}
    def tune_and_stat(ys, seed, cvseed):
        gs = GridSearchCV(RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1), GRID, cv=KFold(5, shuffle=True, random_state=cvseed), scoring='neg_mean_squared_error').fit(X, ys)
        hp = gs.best_params_; m = RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=1, **hp).fit(X, ys); p = pdp_levels(m, X, cols)
        return hp, [dev_stat(p[c]) for c in CORE], [hinge_stat(p[c]) for c in CORE]
    def one(i):
        rng = np.random.default_rng(10_000 + i); ys = np.clip(np.rint(mu + rng.choice(res_e, n)), 1, 5); hp, D, G = tune_and_stat(ys, i, i)
        pf = pdp_levels(RandomForestRegressor(n_estimators=300, random_state=i, n_jobs=1, **hp0).fit(X, ys), X, cols)       # fixed-hyperparameter forest on the same simulated outcome
        return dict(hp=hp, D=D, G=G, Df=[dev_stat(pf[c]) for c in CORE], Gf=[hinge_stat(pf[c]) for c in CORE])
    t0 = time.time(); R = Parallel(n_jobs=NJ)(delayed(one)(i) for i in range(B)); print('retune time', round(time.time() - t0), flush=True)
    obs = Parallel(n_jobs=NJ)(delayed(tune_and_stat)(y, 1, s) for s in range(1, 9))
    D = np.array([r['D'] for r in R]); G = np.array([r['G'] for r in R]); Df = np.array([r['Df'] for r in R]); Gf = np.array([r['Gf'] for r in R])
    os.makedirs(OUT, exist_ok=True); np.savez(os.path.join(OUT, f'null_retuned_B{B}.npz'), D=D, G=G); np.savez(os.path.join(OUT, f'null_fixed_matched_B{B}.npz'), D=Df, G=Gf)
    Go = np.array([nl['G_obs'][c] for c in CORE]); Do = np.array([nl['D_obs'][c] for c in CORE])
    cnt = collections.Counter((r['hp']['min_samples_leaf'], r['hp']['max_features']) for r in R)
    res = dict(B=B, hp_selected_in_null={f'leaf{k[0]}_mtry{k[1]}': v / B for k, v in cnt.most_common()}, hp_fixed_used=hp0,
               observed_retuned=[dict(cv_seed=s + 1, hp=o[0], D=dict(zip(CORE, o[1])), G=dict(zip(CORE, o[2]))) for s, o in enumerate(obs)],
               retuned=dict(p_Gmax_vs_fixedobs=pv(G.max(1), Go.max()), p_Dmax_vs_fixedobs=pv(D.max(1), Do.max()), q95_Gmax=float(np.quantile(G.max(1), .95)), median_Gmax=float(np.median(G.max(1))),
                            p_G=dict(zip(CORE, [pv(G[:, j], Go[j]) for j in range(3)]))),
               fixed_same_datasets=dict(p_Gmax=pv(Gf.max(1), Go.max()), p_Dmax=pv(Df.max(1), Do.max()), q95_Gmax=float(np.quantile(Gf.max(1), .95)), median_Gmax=float(np.median(Gf.max(1))),
                                        p_G=dict(zip(CORE, [pv(Gf[:, j], Go[j]) for j in range(3)]))),
               paired_corr_Gmax=float(np.corrcoef(G.max(1), Gf.max(1))[0, 1]), mean_diff_Gmax_retuned_minus_fixed=float(np.mean(G.max(1) - Gf.max(1))))
    wr(f'retune_null_B{B}.json', res); print('retune p_Gmax (retuned, fixed):', round(res['retuned']['p_Gmax_vs_fixedobs'], 4), round(res['fixed_same_datasets']['p_Gmax'], 4))


def stage_size(S, R_, Bn):
    hp, cols, d = S['hp'], S['cols'], S['d']; Xall, yall, base = S['X'], S['y'], S['base']; b = base['b']; res_e = base['e']; N = len(yall); ji = cols.index('interest')
    Zs = (Xall - Xall.mean(0)) / Xall.std(0); ob, oth, _ = ordered_logit_fit(Zs, yall)
    def one(dgp, rep):
        rng = np.random.default_rng(7_000_000 + (0, 1, 2)[['disc', 'cont', 'ord'].index(dgp)] * 100_000 + rep); ix = rng.integers(0, N, N); X = Xall[ix]; mu = np.column_stack([np.ones(N), X]) @ b
        if dgp == 'disc': ys = np.clip(np.rint(mu + rng.choice(res_e, N)), 1, 5)
        elif dgp == 'cont': ys = mu + rng.choice(res_e, N)
        else: ys = ordered_logit_sample((X - Xall.mean(0)) / Xall.std(0), ob, oth, rng)
        m = RandomForestRegressor(n_estimators=300, random_state=rep, n_jobs=1, **hp).fit(X, ys); p = pdp_levels(m, X, cols)
        Xb = np.column_stack([np.ones(N), X]); f0 = fit_ols(Xb, ys); dums = np.column_stack([(X[:, ji] == l).astype(float) for l in (2, 3, 4, 5)]); Xo = np.delete(Xb, ji + 1, 1); fu = fit_ols(np.column_stack([Xo, dums]), ys)
        F = ((f0['rss'] - fu['rss']) / 3) / (fu['rss'] / (N - Xo.shape[1] - 4)); pF = float(stats.f.sf(F, 3, N - Xo.shape[1] - 4))
        xc = X[:, ji] - X[:, ji].mean(); fq = fit_ols(np.column_stack([Xb, xc ** 2]), ys); pq = float(2 * stats.t.sf(abs(fq['b'][-1] / fq['se_hc3'][-1]), N - Xb.shape[1] - 1))
        D = [dev_stat(p[c]) for c in CORE]; G = [hinge_stat(p[c]) for c in CORE]; return dict(dgp=dgp, rep=rep, Dmax=max(D), Gmax=max(G), Di=D[0], Gi=G[0], pF=pF, pq=pq)
    t0 = time.time(); rows = Parallel(n_jobs=NJ)(delayed(one)(g, r) for g in ('disc', 'cont', 'ord') for r in range(R_)); print('size time', round(time.time() - t0))
    df = pd.DataFrame(rows); os.makedirs(OUT, exist_ok=True); df.to_csv(os.path.join(OUT, 'size_check_raw.csv'), index=False)
    crit = {k: json.load(open(stored(f'null_{k}_B{Bn}.json'))) for k in ('disc', 'cont', 'ord')}; out = {}
    for g in ('disc', 'cont', 'ord'):
        s = df[df.dgp == g]; out[g] = dict(reps=len(s), F_test_size=float((s.pF < .05).mean()), quad_test_size=float((s.pq < .05).mean()))
        for k in ('disc', 'cont', 'ord'):
            out[g][f'Gmax_size_with_{k}_null'] = float((s.Gmax > crit[k]['q95_Gmax']).mean()); out[g][f'Dmax_size_with_{k}_null'] = float((s.Dmax > crit[k]['q95_Dmax']).mean())
        out[g]['binomial_se_at_0p05'] = float(np.sqrt(.05 * .95 / len(s)))
    wr('size_check.json', out); print(json.dumps(out, indent=1))


def stage_power(S, REP):
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import SplineTransformer, StandardScaler
    hp, cols, Xall, yall, base = S['hp'], S['cols'], S['X'], S['y'], S['base']; b = base['b']; res_e = base['e']; ji = cols.index('interest')
    idx = {c: cols.index(c) for c in CORE}; nb = [cols.index(c) for c in S1]; cv_ = [i for i in range(len(cols)) if i not in nb]
    def g(x, sc):
        kind, a = sc
        if kind == 'lin': return 0 * x
        if kind == 'hinge': return a * np.maximum(x - 3, 0)
        if kind == 'step': return a * (x >= 4)
    SCEN = [('lin', 0), ('hinge', 0.15), ('hinge', 0.30), ('hinge', 0.45), ('step', 0.30), ('step', 0.60)]
    def gen(N, sc, rng):
        ix = rng.integers(0, len(yall), N); X = Xall[ix]; mu = np.column_stack([np.ones(N), X]) @ b + g(X[:, ji], sc); yy = np.clip(np.rint(mu + rng.choice(res_e, N)), 1, 5); return X, yy, mu
    def one(N, si, rep):
        sc = SCEN[si]; rng = np.random.default_rng(1_000_000 * N + 10_000 * si + rep); X, yy, _ = gen(N, sc, rng)
        rf = RandomForestRegressor(n_estimators=300, random_state=rep, n_jobs=1, **hp).fit(X, yy); D = {}; G = {}
        for c, j in idx.items():
            v = []
            for lv in [1, 2, 3, 4, 5]:
                Z = X.copy(); Z[:, j] = lv; v.append(rf.predict(Z).mean())
            v = np.array(v); D[c] = dev_stat(v); G[c] = hinge_stat(v)
        n = N; Xb = np.column_stack([np.ones(n), X]); f0 = fit_ols(Xb, yy)
        dums = np.column_stack([(X[:, ji] == l).astype(float) for l in (2, 3, 4, 5)]); Xo = np.delete(Xb, ji + 1, 1); fu = fit_ols(np.column_stack([Xo, dums]), yy)
        F = ((f0['rss'] - fu['rss']) / 3) / (fu['rss'] / (n - Xo.shape[1] - 4)); pF = stats.f.sf(F, 3, n - Xo.shape[1] - 4)
        xc = X[:, ji] - X[:, ji].mean(); fq = fit_ols(np.column_stack([Xb, xc ** 2]), yy); pq = 2 * stats.t.sf(abs(fq['b'][-1] / fq['se_hc3'][-1]), n - Xb.shape[1] - 1)
        Xt, yt, mut = gen(1500, sc, np.random.default_rng(999 + rep)); ols = LinearRegression().fit(X, yy).predict(Xt)
        spl = make_pipeline(ColumnTransformer([('s', SplineTransformer(degree=2, n_knots=3, knots='uniform', extrapolation='constant'), nb)] + ([('p', 'passthrough', cv_)] if cv_ else [])), StandardScaler(), Ridge(alpha=100)).fit(X, yy).predict(Xt)
        rfp = rf.predict(Xt); mse = lambda p: float(np.mean((yt - p) ** 2))
        return dict(N=N, si=si, rep=rep, Di=D['interest'], Gi=G['interest'], Dmax=max(D.values()), Gmax=max(G.values()), pF=float(pF), pq=float(pq), m_ols=mse(ols), m_spl=mse(spl), m_rf=mse(rfp))
    jobs = [(N, si, r) for N in (250, 478, 1000) for si in range(len(SCEN)) for r in range(REP)]
    t0 = time.time(); R = Parallel(n_jobs=NJ)(delayed(one)(*j) for j in jobs); print('power time', round(time.time() - t0))
    df = pd.DataFrame(R); os.makedirs(OUT, exist_ok=True); df.to_csv(os.path.join(OUT, 'power_raw.csv'), index=False); out = {}
    for N in (250, 478, 1000):
        ref = df[(df.N == N) & (df.si == 0)]; crit = {k: float(ref[k].quantile(.95)) for k in ['Di', 'Gi', 'Dmax', 'Gmax']}
        for si, sc in enumerate(SCEN):
            s = df[(df.N == N) & (df.si == si)]
            out[f'N{N}_{sc[0]}_{sc[1]}'] = dict(N=N, scenario=sc[0], size=sc[1], reps=len(s), rej_Di=float((s.Di > crit['Di']).mean()), rej_Gi=float((s.Gi > crit['Gi']).mean()), rej_Dmax=float((s.Dmax > crit['Dmax']).mean()),
                                                rej_Gmax=float((s.Gmax > crit['Gmax']).mean()), rej_Ftest=float((s.pF < .05).mean()), rej_quad=float((s.pq < .05).mean()), p_rf_beats_ols=float((s.m_rf < s.m_ols).mean()),
                                                p_spl_beats_ols=float((s.m_spl < s.m_ols).mean()), test_mse_ols=float(s.m_ols.mean()), test_mse_rf=float(s.m_rf.mean()), test_mse_spl=float(s.m_spl.mean()))
        out[f'N{N}_crit'] = crit
    wr('power.json', out)


def stage_bracket(Bn):
    df = pd.read_csv(stored('power_raw.csv')); pw = json.load(open(stored('power.json'))); nulls = {k: json.load(open(stored(f'null_{k}_B{Bn}.json'))) for k in ('disc', 'cont', 'ord')}
    SC = [('lin', 0), ('hinge', .15), ('hinge', .30), ('hinge', .45), ('step', .30), ('step', .60)]; rows = []
    for si, sc in enumerate(SC):
        s = df[(df.N == 478) & (df.si == si)]; r = dict(scenario=sc[0], size=sc[1], reps=len(s))
        for k, v in nulls.items(): r[f'Gmax_{k}'] = float((s.Gmax > v['q95_Gmax']).mean()); r[f'Dmax_{k}'] = float((s.Dmax > v['q95_Dmax']).mean())
        key = f'N478_{sc[0]}_{sc[1]}'; r['Gmax_own_linear_crit'] = pw[key]['rej_Gmax']; r['F'] = pw[key]['rej_Ftest']; r['quad'] = pw[key]['rej_quad']; rows.append(r)
    os.makedirs(OUT, exist_ok=True); pd.DataFrame(rows).to_csv(os.path.join(OUT, 'power_N478_bracket.csv'), index=False); wr('power_bracket_N478.json', {'null_B': Bn, 'rows': rows}); print(pd.DataFrame(rows).round(2).to_string())


def main():
    global OUT, NJ
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', default='all', choices=['initial', 'nulls', 'retune', 'size', 'power', 'bracket', 'all']); ap.add_argument('--quick', action='store_true')
    ap.add_argument('--out', default=None); ap.add_argument('--jobs', type=int, default=2); a = ap.parse_args(); NJ = a.jobs
    if a.quick: OUT = os.path.join(ROOT, 'quick_output', 'simulations'); cfg = dict(initial=20, nulls=20, retune=10, size=3, power=3, Bn=2000)
    else:
        OUT = os.path.abspath(a.out) if a.out else SIM; cfg = dict(initial=300, nulls=2000, retune=300, size=400, power=120, Bn=2000)
    A.RES = os.path.join(ROOT, 'quick_output', 'results') if a.quick else A.RES            # observed statistics come from nonlin.json (quick output if present, else shipped)
    warnings.filterwarnings('ignore'); os.makedirs(OUT, exist_ok=True)
    wr('simulation_parameters.json', PARAMS); S = setup(); order = ['initial', 'nulls', 'retune', 'size', 'power', 'bracket'] if a.stage == 'all' else [a.stage]; t0 = time.time()
    for st in order:
        print(f'\n=== {st} ===', flush=True)
        if st == 'initial': stage_initial(S, cfg['initial'])
        elif st == 'nulls': stage_nulls(S, cfg['nulls'])
        elif st == 'retune': stage_retune(S, cfg['retune'])
        elif st == 'size': stage_size(S, cfg['size'], cfg['Bn'])
        elif st == 'power': stage_power(S, cfg['power'])
        elif st == 'bracket': stage_bracket(cfg['Bn'])
    print(f'\ndone in {time.time() - t0:.0f}s')
if __name__ == '__main__': main()
