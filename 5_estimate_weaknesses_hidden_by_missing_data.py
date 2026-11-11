import pickle, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

BASE = Path.cwd(); DATA = BASE / "data"
if not DATA.is_dir(): raise SystemExit(f"`data/` folder not found in {BASE}. Put this file next to the `data/` folder.")
IN1 = BASE / "result" / "1"; RES = BASE / "result" / "5"; RES.mkdir(parents=True, exist_ok=True)

EDITIONS = ["2026", "2025", "2024", "2023", "2022"]
NOT_APPLICABLE_SHARE = 0.10
SMALL_POPULATION = 1e6
TEST_REPEATS, TEST_SHARE, SEED = 20, 0.20, 20261002
EPS = 1e-9
TOL = pd.read_csv(IN1 / "tolerances.csv", encoding="utf-8-sig").set_index("result").tolerance
POINTS = float(TOL["weakest indicator"])
if not (DATA / "wdi_population.csv").exists():
    raise SystemExit("data/wdi_population.csv not found: total population by ISO3 code (World Bank WDI, SP.POP.TOTL), columns iso, year, population.")
POPULATION = pd.read_csv(DATA / "wdi_population.csv"); POPULATION = dict(zip(POPULATION.iso, POPULATION.population))
print(f"tolerance: weakest indicator within {POINTS:g} points")


def save(name, table):
    pd.DataFrame(table).to_csv(RES / name, index=False, encoding="utf-8-sig")


def additive_model(Z, iters=500, tol=1e-10):
    """Fit score = overall mean + country level + indicator level to the observed scores of a goal."""
    obs = np.isfinite(Z); fit = obs.sum(1) >= 2; Y = Z[fit]
    mu = np.nanmean(Y); a = np.zeros(Y.shape[0]); b = np.zeros(Z.shape[1])
    with np.errstate(all="ignore"):
        for _ in range(iters):
            bn = np.nanmean(Y - mu - a[:, None], axis=0); bn = np.where(np.isfinite(bn), bn, 0.0); bn -= bn.mean()
            an = np.nanmean(Y - mu - bn[None, :], axis=1); an -= an.mean()
            done = max(np.abs(an - a).max(), np.abs(bn - b).max()) < tol; a, b = an, bn
            if done: break
    A = np.full(Z.shape[0], np.nan); A[fit] = a; one = obs.sum(1) == 1
    if one.any():
        j1 = np.argmax(obs[one], axis=1); A[np.where(one)[0]] = Z[one, j1] - mu - b[j1]
    R = np.full(Z.shape, np.nan); R[fit] = Y - mu - a[:, None] - b[None, :]
    return mu, A, b, R


def probability_hidden(Z, cells):
    """Probability that each missing value lies more than the tolerance below the country's lowest observed score."""
    mu, A, b, R = additive_model(Z); obs = np.isfinite(Z); t = np.where(obs, Z, np.inf).min(1) - POINTS
    pooled = R[np.isfinite(R)]; P = np.full(Z.shape, np.nan)
    for j in range(Z.shape[1]):
        rows = np.where(cells[:, j] & np.isfinite(A))[0]
        if not len(rows): continue
        r = R[:, j]; r = r[np.isfinite(r)]; r = r if len(r) >= 10 else pooled
        pred = mu + A[rows] + b[j]
        P[rows, j] = np.where(t[rows] > 0, (pred[:, None] + r[None, :] < t[rows][:, None]).mean(1), 0.0)
    return P


def hide_and_predict(Z, rng):
    """Hide observed values, predict them, and record each predicted probability with the observed outcome."""
    obs = np.isfinite(Z); rows3 = np.where(obs.sum(1) >= 3)[0]; out = []
    if len(rows3) < 10: return out
    for _ in range(TEST_REPEATS):
        pick = rng.choice(rows3, size=max(1, int(TEST_SHARE * len(rows3))), replace=False)
        Zm = Z.copy(); M = np.zeros(Z.shape, bool)
        for i in pick:
            j = rng.choice(np.where(obs[i])[0]); Zm[i, j] = np.nan; M[i, j] = True
        P = probability_hidden(Zm, M); low = np.where(np.isfinite(Zm), Zm, np.inf).min(1)
        for i, j in zip(*np.where(M)): out.append((P[i, j], float(Z[i, j] < low[i] - POINTS - EPS)))
    return out


def test_statistics(pairs, label):
    """Area under the ROC curve, Brier skill score and calibration groups for the predicted probabilities."""
    p = np.array([a for a, _ in pairs]); y = np.array([b for _, b in pairs]); ok = np.isfinite(p); p, y = p[ok], y[ok]
    brier = float(np.mean((p - y) ** 2)); base = float(np.mean((y.mean() - y) ** 2)); pos, neg = p[y == 1], p[y == 0]
    auc = float((np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :])) / (len(pos) * len(neg))) if len(pos) and len(neg) else np.nan
    rows = [dict(edition=label, group="all", hidden_values=len(p), predicted=float(p.mean()), observed=float(y.mean()), brier_skill=1 - brier / base if base > 0 else np.nan, auc=auc)]
    for lo, hi in [(0, .05), (.05, .2), (.2, .5), (.5, 1.01)]:
        m = (p >= lo) & (p < hi)
        if m.any(): rows.append(dict(edition=label, group=f"{lo:g}-{min(hi, 1):g}", hidden_values=int(m.sum()), predicted=float(p[m].mean()), observed=float(y[m].mean())))
    return rows


NA_T, IND, CTRY, EDT, CAL = [], [], [], [], []
for label in EDITIONS:
    with open(IN1 / f"edition_{label}.pkl", "rb") as f: E = pickle.load(f)
    goals, meta = E["goals"], E["meta"].reset_index(drop=True); order = sorted(goals); n = len(meta); rng = np.random.default_rng(SEED)
    groups = {"landlocked countries": meta.landlocked.to_numpy(bool), **{f"income group {k}": (meta.income == k).to_numpy() for k in ["HIC", "UMIC", "LMIC", "LIC"]}}
    hidden_goals = np.zeros(n); goals_missing = np.zeros(n, int); pairs_with_data = 0; test_pairs = []
    for g in order:
        Z = goals[g]["Z"]; codes = goals[g]["codes"]; obs = np.isfinite(Z); has = obs.any(1); pairs_with_data += int(has.sum())
        na = np.zeros(Z.shape, bool)
        for j, c in enumerate(codes):
            for name, m in groups.items():
                share = float(obs[m, j].mean())
                if share < NOT_APPLICABLE_SHARE: na[m, j] = True; NA_T.append(dict(edition=label, goal=g, code=c, group=name, countries=int(m.sum()), observed_share=share))
        na &= ~obs
        cells = ~obs & ~na & has[:, None]; P = np.where(cells, np.nan_to_num(probability_hidden(Z, cells)), 0.0)
        hidden_goals += 1 - np.prod(1 - P, axis=1); goals_missing += cells.any(1)
        for j, c in enumerate(codes):
            IND.append(dict(edition=label, goal=g, code=c, countries_applicable=int((~na[:, j]).sum()), countries_observed=int(obs[:, j].sum()),
                            missing_values=int(cells[:, j].sum()), expected_hidden_low_scores=float(P[:, j].sum())))
        if Z.shape[1] >= 3: test_pairs += hide_and_predict(Z, rng)
    CAL += test_statistics(test_pairs, label)
    pop = meta.iso.map(POPULATION).to_numpy(float)
    CTRY.append(pd.DataFrame(dict(edition=label, iso=meta.iso, region=meta.region, income=meta.income, sids=meta.sids.astype(bool), landlocked=meta.landlocked.astype(bool),
                                  population=pop, under_one_million=pop < SMALL_POPULATION, goals_with_missing_values=goals_missing, expected_goals_hidden=hidden_goals)))
    I = pd.DataFrame([r for r in IND if r["edition"] == label]); c = CAL[[i for i, r in enumerate(CAL) if r["edition"] == label and r["group"] == "all"][0]]
    EDT.append(dict(edition=label, missing_values=int(I.missing_values.sum()), expected_hidden_low_scores=float(I.expected_hidden_low_scores.sum()),
                    country_goal_pairs=pairs_with_data, expected_pairs_hidden=float(hidden_goals.sum()), share_of_pairs_hidden=float(hidden_goals.sum() / pairs_with_data)))
    print(f"{label}: {EDT[-1]['missing_values']} missing applicable values | expected country and goal pairs with a hidden weakest indicator {hidden_goals.sum():.0f} of {pairs_with_data} "
          f"({hidden_goals.sum() / pairs_with_data:.1%}) | test: AUC {c['auc']:.2f}, Brier skill {c['brier_skill']:.2f}, predicted {c['predicted']:.3f} against observed {c['observed']:.3f}", flush=True)
CTRY = pd.concat(CTRY, ignore_index=True)
save("indicators_not_applicable.csv", NA_T); save("hidden_by_indicator.csv", IND); save("hidden_by_country.csv", CTRY); save("hidden_by_edition.csv", EDT); save("calibration.csv", CAL)


GR = []
for col, names in [("sids", {True: "small island developing states", False: "other countries"}), ("under_one_million", {True: "fewer than one million people", False: "one million people or more"})]:
    for (e, v), x in CTRY.groupby(["edition", col], sort=False):
        GR.append(dict(edition=e, grouping=col, group=names[bool(v)], countries=len(x), mean_expected_goals_hidden=float(x.expected_goals_hidden.mean())))
GR = pd.DataFrame(GR); save("hidden_by_group.csv", GR)
one = CTRY.drop_duplicates("iso")
print(f"\ncountries with fewer than one million people: {int(one.under_one_million.sum())}, of which small island developing states {int((one.under_one_million & one.sids).sum())}; "
      f"countries without a population value: {sorted(one[one.population.isna()].iso)}")
print("expected number of goals with a hidden weakest indicator, mean per country (range across editions):")
for g, x in GR.groupby("group", sort=False): print(f"  {g:32s} {x.mean_expected_goals_hidden.min():.1f}-{x.mean_expected_goals_hidden.max():.1f}  ({x.countries.min()}-{x.countries.max()} countries)")
print("\ndone ->", RES)
