import pickle, itertools, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
warnings.filterwarnings("ignore")

BASE = Path.cwd()
if not (BASE / "data").is_dir(): raise SystemExit(f"`data/` folder not found in {BASE}. Put this file next to the `data/` folder.")
IN1 = BASE / "result" / "1"; RES = BASE / "result" / "4"; RES.mkdir(parents=True, exist_ok=True)

EDITIONS = ["2026", "2025", "2024", "2023", "2022"]
REQUIRED = 0.95
SCORED_MIN = 0.95
VARIANCE_TARGET = 0.90
POWER_MEANS = [1.0, 0.5, 0.0, -1.0, "min"]
OFFSET = 1.0
MIN_EXTRA_ROWS = 5
EPS = 1e-9
TOL = pd.read_csv(IN1 / "tolerances.csv", encoding="utf-8-sig").set_index("result").tolerance
POINTS, PLACES = float(TOL["weakest indicator"]), int(TOL["rank within goal"])
print(f"tolerances within goals: weakest indicator within {POINTS:g} points; rank within {PLACES} places")


def save(name, table):
    pd.DataFrame(table).to_csv(RES / name, index=False, encoding="utf-8-sig")


def goal_score(Z):
    with np.errstate(all="ignore"):
        return np.nanmean(Z, axis=1)


def power_mean(Z, p):
    """Power mean of the observed scores of each country; a lower p gives more weight to low scores."""
    obs = np.isfinite(Z)
    with np.errstate(all="ignore"):
        if p == "min":
            m = np.where(obs, Z, np.inf).min(1); return np.where(np.isfinite(m), m, np.nan)
        if p == 1.0: return np.nanmean(Z, axis=1)
        X = Z + OFFSET
        if p == 0.0: return np.exp(np.nanmean(np.log(X), axis=1)) - OFFSET
        return np.nanmean(X ** p, axis=1) ** (1.0 / p) - OFFSET


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 8:
        return np.nan
    return float(np.corrcoef(rankdata(a[m]), rankdata(b[m]))[0, 1])


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z); B = np.column_stack([np.ones(len(rz)), rz])
    res = lambda a: a - B @ np.linalg.lstsq(B, a, rcond=None)[0]
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def all_subsets(Z, codes):
    """Evaluate every subset of a goal's indicators against the goal score from all indicators."""
    K = Z.shape[1]; full = goal_score(Z); has = np.isfinite(full); obs = np.isfinite(Z); ok = obs.sum(1) >= 2; fmin = np.where(obs, Z, np.inf).min(1); rows = []
    for k in range(1, K + 1):
        for sub in itertools.combinations(range(K), k):
            sub = list(sub); sc = goal_score(Z[:, sub]); ev = has & np.isfinite(sc); scored = float(ev.sum() / has.sum()) if has.any() else np.nan
            rho = spearman(sc[ev], full[ev])
            within = float((np.abs(rankdata(sc[ev]) - rankdata(full[ev])) <= PLACES + EPS).mean()) if ev.sum() >= 8 else np.nan
            gap = np.where(obs[:, sub], Z[:, sub], np.inf).min(1) - fmin
            rows.append(dict(indicators=" ".join(codes[j] for j in sub), columns=tuple(sub), k=k, rho=rho, scored=scored, rho_valid=bool(np.isfinite(rho) and scored >= SCORED_MIN),
                             within_places=within, weakest_found=float(np.mean((gap <= POINTS + EPS)[ok]))))
    return pd.DataFrame(rows)


def distinct_weakest(Z):
    """Effective number of distinct weakest indicators across countries (inverse Simpson index)."""
    obs = np.isfinite(Z); p = np.zeros(Z.shape[1])
    for i in np.where(obs.sum(1) >= 2)[0]:
        w = np.where(obs[i] & (Z[i] <= Z[i][obs[i]].min() + EPS))[0]; p[w] += 1.0 / len(w)
    p = p / p.sum() if p.sum() else p
    return float(1.0 / np.sum(p ** 2)) if p.sum() else np.nan


def country_specific_variation(Z, iters=500):
    """Fit an additive model to a goal's scores and return the variation of the country-specific part relative to that of the country levels."""
    rows = np.isfinite(Z).sum(1) >= 2; Y = Z[rows]; obs = np.isfinite(Y)
    mu = np.nanmean(Y); a = np.zeros(Y.shape[0]); b = np.zeros(Y.shape[1])
    with np.errstate(all="ignore"):
        for _ in range(iters):
            bn = np.nanmean(Y - mu - a[:, None], axis=0); bn = np.where(np.isfinite(bn), bn, 0.0); bn -= bn.mean()
            an = np.nanmean(Y - mu - bn[None, :], axis=1); an -= an.mean()
            done = max(np.abs(an - a).max(), np.abs(bn - b).max()) < 1e-10; a, b = an, bn
            if done: break
    R = np.where(obs, Y - mu - a[:, None] - b[None, :], 0.0)
    ss_country, ss_specific = float(np.sum(obs * a[:, None] ** 2)), float(np.sum(R ** 2))
    return ss_specific / ss_country if ss_country > 0 else np.nan


def results_of_matrix(Zs, Z):
    full = goal_score(Z); sc = goal_score(Zs); has = np.isfinite(full); ev = has & np.isfinite(sc); ok = np.isfinite(Z).sum(1) >= 2
    d = np.abs(rankdata(sc[ev]) - rankdata(full[ev])); fmin = np.where(np.isfinite(Z), Z, np.inf).min(1)
    obs = np.isfinite(Zs); rmin = np.where(obs, Zs, np.inf).min(1); L = obs & (Zs <= rmin[:, None] + EPS)
    found = np.where(L, Z, np.inf).min(1) <= fmin + POINTS + EPS
    return dict(rho=spearman(sc[ev], full[ev]), within_places=float((d <= PLACES + EPS).mean()), weakest_found=float(found[ok].mean())), found


def predict_dropped(Z, keep):
    """Predict each dropped indicator from the kept ones by leave-one-out linear regression."""
    X = Z[:, keep]; Zr = np.full(Z.shape, np.nan); Zr[:, keep] = Z[:, keep]; xok = np.isfinite(X).all(1)
    for j in range(Z.shape[1]):
        if j in keep: continue
        T = xok & np.isfinite(Z[:, j])
        if T.sum() < X.shape[1] + 1 + MIN_EXTRA_ROWS: continue
        A = np.column_stack([np.ones(T.sum()), X[T]]); y = Z[T, j]
        inv = np.linalg.pinv(A.T @ A); beta = inv @ A.T @ y; e = y - A @ beta
        h = np.einsum("ij,jk,ik->i", A, inv, A); Zr[np.where(T)[0], j] = np.clip(y - e / np.clip(1 - h, 1e-6, None), 0, 100)
    return Zr


NEED, VAR, PRED, AVG = [], [], [], []
for label in EDITIONS:
    with open(IN1 / f"edition_{label}.pkl", "rb") as f: E = pickle.load(f)
    goals = E["goals"]; t0 = time.time()
    for g in sorted(goals):
        Z, codes = goals[g]["Z"], goals[g]["codes"]; K = len(codes); T = all_subsets(Z, codes)
        ranking = T.rho_valid & (T.rho >= REQUIRED); rank = (T.scored >= SCORED_MIN - 1e-12) & (T.within_places >= REQUIRED); weakest = T.weakest_found >= REQUIRED
        kmin = lambda m: int(T[m].k.min())
        row = dict(edition=label, goal=g, K=K, k_ranking=kmin(ranking), k_rank=kmin(rank), k_weakest=kmin(weakest), k_rank_and_weakest=kmin(rank & weakest),
                   k_ranking_and_weakest=kmin(ranking & weakest), distinct_weakest_indicators=distinct_weakest(Z), country_specific_variation=country_specific_variation(Z))
        R = T[(T.k == row["k_ranking"]) & T.rho_valid].sort_values(["rho", "indicators"], ascending=[False, True]).iloc[0]
        row.update(ranking_list=R.indicators, ranking_list_rho=R.rho, ranking_list_within_places=R.within_places, ranking_list_weakest_found=R.weakest_found)
        NEED.append(row)
        if K >= 3:
            cc = np.isfinite(Z).all(1); rec = dict(edition=label, goal=g, K=K, complete_cases=int(cc.sum()))
            if cc.sum() < K + 10: VAR.append({**rec, "status": "too few complete cases"})
            else:
                C = np.corrcoef(Z[cc].T); best = {}
                for s, S in zip(T.indicators, T["columns"]):
                    S = list(S)
                    if len(S) == K: continue
                    sh = float(np.trace(C[:, S] @ np.linalg.pinv(C[np.ix_(S, S)]) @ C[S, :]) / K)
                    if len(S) not in best or sh > best[len(S)][0] + 1e-12: best[len(S)] = (sh, s)
                ks = [k for k in sorted(best) if best[k][0] >= VARIANCE_TARGET]
                if not ks: VAR.append({**rec, "status": "needs all indicators"})
                else:
                    sh, s = best[ks[0]]; t = T[T.indicators == s].iloc[0]
                    VAR.append({**rec, "status": "evaluated", "k": ks[0], "variance_explained": sh, "indicators": s, "rho": t.rho, "scored": t.scored,
                                "within_places": t.within_places, "weakest_found": t.weakest_found})
        S = list(R["columns"])
        if 1 <= len(S) < K:
            Zb = np.full(Z.shape, np.nan); Zb[:, S] = Z[:, S]; base, found_b = results_of_matrix(Zb, Z)
            if abs(base["weakest_found"] - R.weakest_found) > 1e-9 or abs(base["within_places"] - R.within_places) > 1e-9: raise SystemExit(f"{label} SDG {g}: check failed, two evaluations of the same list differ")
            pred, found_p = results_of_matrix(predict_dropped(Z, S), Z)
            obs = np.isfinite(Z); ok = obs.sum(1) >= 2; W = obs & (Z <= np.where(obs, Z, np.inf).min(1)[:, None] + EPS); W[~ok] = False
            top = int(np.argmax((W / np.maximum(W.sum(1, keepdims=True), 1))[ok].sum(0)))
            missed = ok & ~found_b; shared = missed & W[:, top]; specific = missed & ~W[:, top]
            PRED.append(dict(edition=label, goal=g, K=K, kept=len(S), most_common_weakest=codes[top], weakest_found_before=base["weakest_found"], weakest_found_after=pred["weakest_found"],
                             within_places_before=base["within_places"], within_places_after=pred["within_places"], missed=int(missed.sum()), missed_shared=int(shared.sum()),
                             missed_specific=int(specific.sum()), recovered_shared=float(found_p[shared].mean()) if shared.any() else np.nan,
                             recovered_specific=float(found_p[specific].mean()) if specific.any() else np.nan))
        subs = [list(c) for c in T["columns"]]; k_all = T.k.to_numpy(); names = T.indicators.to_numpy()
        for p in POWER_MEANS:
            fp = power_mean(Z, p); hasp = np.isfinite(fp); rho = np.empty(len(subs)); scored = np.empty(len(subs))
            for i, sub in enumerate(subs):
                sc = power_mean(Z[:, sub], p); ev = hasp & np.isfinite(sc); scored[i] = ev.sum() / hasp.sum(); rho[i] = spearman(sc[ev], fp[ev])
            okr = np.isfinite(rho) & (rho >= REQUIRED) & (scored >= SCORED_MIN - 1e-12); k = int(k_all[okr].min()); cand = np.where(okr & (k_all == k))[0]
            best = cand[np.lexsort((names[cand], -rho[cand]))[0]]
            if p == 1.0 and (k != row["k_ranking"] or names[best] != R.indicators): raise SystemExit(f"{label} SDG {g}: check failed, the arithmetic mean does not reproduce the ranking list")
            AVG.append(dict(edition=label, goal=g, K=K, rule="minimum" if p == "min" else f"p={p:g}", k_ranking=k, ranking_list=names[best], ranking_list_rho=rho[best],
                            ranking_list_weakest_found=float(T.weakest_found.iloc[best])))
    x = pd.DataFrame([r for r in NEED if r["edition"] == label]); tot = x.K.sum()
    print(f"{label} ({tot} indicators): summed over goals, ranking {x.k_ranking.sum()} ({x.k_ranking.sum() / tot:.0%}) | weakest indicator {x.k_weakest.sum()} ({x.k_weakest.sum() / tot:.0%}) | "
          f"each country's rank {x.k_rank.sum()} ({x.k_rank.sum() / tot:.0%}) | rank and weakest {x.k_rank_and_weakest.sum()} | ranking and weakest {x.k_ranking_and_weakest.sum()}  [{time.time() - t0:.0f} s]", flush=True)
NEED, VAR, PRED, AVG = pd.DataFrame(NEED), pd.DataFrame(VAR), pd.DataFrame(PRED), pd.DataFrame(AVG)
save("within_goal_indicators_needed.csv", NEED); save("variance_based_lists.csv", VAR); save("prediction_within_goals.csv", PRED); save("averaging_rules.csv", AVG)


print(f"\nall {len(NEED)} goal and edition pairs: ranking {NEED.k_ranking.sum()} selections, weakest indicator {NEED.k_weakest.sum()}, ranking and weakest indicator {NEED.k_ranking_and_weakest.sum()} "
      f"({NEED.k_ranking_and_weakest.sum() / NEED.k_weakest.sum() - 1:.0%} more than weakest alone, {NEED.k_ranking_and_weakest.sum() / NEED.k_ranking.sum() - 1:.0%} more than ranking alone)")
print(f"weakest indicator needs more indicators than the ranking in {int((NEED.k_weakest > NEED.k_ranking).sum())} pairs; more than each country's rank in {int((NEED.k_weakest > NEED.k_rank).sum())} pairs")

V = VAR[VAR.status == "evaluated"]
VS = V.groupby("edition", sort=False).agg(goals_evaluated=("goal", "size"), share_of_indicators=("k", lambda k: float((k / V.loc[k.index, "K"]).mean())), rho=("rho", "mean"),
                                           within_places=("within_places", "mean"), weakest_found=("weakest_found", "mean")).reset_index()
save("variance_based_lists_by_edition.csv", VS)
print("\nlists that explain 90% of the variance (means over the goals evaluated):"); print(VS.round(3).to_string(index=False))

D = NEED[NEED.K > 2]; A = []
for pred, pname in [("country_specific_variation", "country-specific variation"), ("distinct_weakest_indicators", "distinct weakest indicators")]:
    for col, rname in [("k_ranking", "ranking"), ("k_rank", "each country's rank"), ("k_weakest", "weakest indicator")]:
        y = D[col] / D.K
        A.append(dict(property=pname, result=rname, pairs=len(D), spearman=float(spearmanr(D[pred], y)[0]), partial_spearman=partial_spearman(D[pred].to_numpy(), y.to_numpy(), D.K.to_numpy())))
A = pd.DataFrame(A); save("what_requirements_depend_on.csv", A)
print("\nshare of indicators needed against two properties of the goal:"); print(A.round(2).to_string(index=False))

rows = []
for e, x in PRED.groupby("edition", sort=False):
    fs, fo = x.missed_shared.sum(), x.missed_specific.sum()
    rows.append(dict(edition=e, goals=len(x), weakest_found_before=x.weakest_found_before.mean(), weakest_found_after=x.weakest_found_after.mean(),
                     missed_shared=int(fs), recovered_shared=float((x.missed_shared * x.recovered_shared.fillna(0)).sum() / fs) if fs else np.nan,
                     missed_specific=int(fo), recovered_specific=float((x.missed_specific * x.recovered_specific.fillna(0)).sum() / fo) if fo else np.nan,
                     specific_share_of_missed=float(fo / (fs + fo)) if fs + fo else np.nan))
PS = pd.DataFrame(rows); save("prediction_by_edition.csv", PS)
print("\nprediction of dropped indicators (goals whose ranking list drops at least one indicator):"); print(PS.round(3).to_string(index=False))

AS = AVG.groupby(["edition", "rule"], sort=False).agg(k_ranking=("k_ranking", "sum"), ranking_list_weakest_found=("ranking_list_weakest_found", "mean")).reset_index()
save("averaging_rules_by_edition.csv", AS)
print("\naveraging rules (indicators needed for the ranking, summed over goals; weakest indicator kept by the ranking lists, mean over goals):")
print(AS.pivot(index="rule", columns="edition", values="ranking_list_weakest_found").reindex(["p=1", "p=0.5", "p=0", "p=-1", "minimum"]).round(3).to_string())
print(AS.pivot(index="rule", columns="edition", values="k_ranking").reindex(["p=1", "p=0.5", "p=0", "p=-1", "minimum"]).to_string())
print("\ndone ->", RES)
