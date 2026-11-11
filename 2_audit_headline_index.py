import pickle, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
warnings.filterwarnings("ignore")

BASE = Path.cwd()
if not (BASE / "data").is_dir(): raise SystemExit(f"`data/` folder not found in {BASE}. Put this file next to the `data/` folder.")
IN1 = BASE / "result" / "1"; RES = BASE / "result" / "2"; RES.mkdir(parents=True, exist_ok=True)

EDITIONS = ["2026", "2025", "2024", "2023", "2022"]
HEADLINE = ["sdg1_wpc", "sdg2_undernsh", "sdg3_lifee", "sdg4_second", "sdg5_parl", "sdg6_sanita", "sdg7_elecac", "sdg8_accounts", "sdg9_intuse",
            "sdg10_gini", "sdg11_pm25", "sdg12_nprod", "sdg13_co2gcp", "sdg14_cpma", "sdg15_redlist", "sdg16_cpi", "sdg17_statperf"]
MAX_MISSING = 2
PERMUTATIONS = 20000; SEED = 111
EPS = 1e-9
TOL = pd.read_csv(IN1 / "tolerances.csv", encoding="utf-8-sig").set_index("result").tolerance
POINTS, PLACES = float(TOL["weakest goal"]), int(TOL["overall rank"])
print(f"tolerances: weakest goal within {POINTS:g} points; overall rank within {PLACES} places")


def save(name, table):
    pd.DataFrame(table).to_csv(RES / name, index=False, encoding="utf-8-sig")


def goal_score(Z):
    with np.errstate(all="ignore"):
        return np.nanmean(Z, axis=1)


def sdg_index(goal_scores, has_score, region, ranked):
    """Rebuild the SDG Index as the report does: a goal without a score gets the regional average, then the goal scores are averaged."""
    filled = np.where(has_score, goal_scores, np.nan)
    for r in np.unique(region):
        m = region == r; base = has_score & (m & ranked)[:, None]
        with np.errstate(all="ignore"):
            mean_r = np.where(base, goal_scores, 0.0).sum(0) / base.sum(0)
        filled = np.where(m[:, None] & ~has_score, mean_r[None, :], filled)
    with np.errstate(all="ignore"):
        return np.nanmean(filled, axis=1)


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(rankdata(a[m]), rankdata(b[m]))[0, 1]) if m.sum() >= 8 else np.nan


def full_index(goals, meta):
    """Score the full indicator set of an edition: goal scores, SDG Index, ranked countries and reference countries."""
    order = sorted(goals); codes = [c for g in order for c in goals[g]["codes"]]
    goal_of = np.concatenate([[i] * len(goals[g]["codes"]) for i, g in enumerate(order)]); Z = np.column_stack([goals[g]["Z"] for g in order])
    gm = np.column_stack([goal_score(goals[g]["Z"]) for g in order]); applicable = np.isfinite(gm)
    region = meta.region.astype(str).to_numpy(); ranked = meta.ranked.to_numpy(bool)
    index = sdg_index(gm, applicable, region, ranked)
    return dict(order=order, codes=codes, goal_of=goal_of, Z=Z, goal_scores=gm, applicable=applicable, index=index, region=region, ranked=ranked,
                rank_sample=ranked & np.isfinite(index), reference=applicable.sum(1) >= meta.n_goals_applicable.to_numpy() - 1)


def evaluate_list(F, columns, max_missing=MAX_MISSING):
    """Evaluate an indicator list against the full index: each country's rank shift and weakest goal, and their summary."""
    Zs = np.full(F["Z"].shape, np.nan); Zs[:, columns] = F["Z"][:, columns]; order, app, gm_full = F["order"], F["applicable"], F["goal_scores"]
    gm = np.column_stack([goal_score(Zs[:, F["goal_of"] == i]) for i in range(len(order))]); n_missing = (app & ~np.isfinite(gm)).sum(1)
    scored = n_missing <= max_missing
    index = np.where(scored, sdg_index(gm, np.isfinite(gm), F["region"], F["ranked"]), np.nan); e1 = F["rank_sample"] & scored; e3 = F["reference"] & scored
    shift = np.full(len(index), np.nan); shift[e1] = np.abs(rankdata(-index[e1]) - rankdata(-F["index"][e1]))
    rank_list = np.full(len(index), np.nan); rank_full = np.full(len(index), np.nan)
    rank_list[e1] = rankdata(-index[e1]); rank_full[e1] = rankdata(-F["index"][e1])
    full_w = np.where(app, gm_full, np.inf); fmin = full_w.min(1); truth = (full_w <= fmin[:, None] + EPS) & app
    list_w = np.where(app & np.isfinite(gm), gm, np.inf); named = (list_w <= list_w.min(1)[:, None] + EPS) & np.isfinite(list_w)
    gap = np.where(named, gm_full, np.inf).min(1) - fmin; found = gap <= POINTS + EPS
    status = np.where(~F["reference"], "no reference", np.where(~scored, "not assessed", np.where(found, "found", "missed")))
    lab = lambda M, i: "|".join(str(order[k]) for k in np.where(M[i])[0])
    per_country = pd.DataFrame(dict(status=status, goals_without_data=n_missing, true_weakest=[lab(truth, i) if F["reference"][i] else "" for i in range(len(index))],
                                    named_weakest=[lab(named, i) if e3[i] else "" for i in range(len(index))], gap_points=np.where(e3, gap, np.nan),
                                    rank_shift=shift, rank_list=rank_list, rank_full_same_countries=rank_full,
                                    within_places=np.where(e1, (shift <= PLACES + EPS).astype(float), np.nan), in_weakest_goal_sample=e3))
    summary = dict(rho=spearman(index[e1], F["index"][e1]), ranked_countries=int(F["rank_sample"].sum()), ranked_scored=int(e1.sum()),
                   share_ranked_scored=e1.sum() / F["rank_sample"].sum(), share_within_places=float((shift[e1] <= PLACES + EPS).mean()),
                   reference_countries=int(F["reference"].sum()), reference_scored=int(e3.sum()), share_reference_scored=e3.sum() / F["reference"].sum(),
                   share_weakest_goal_found=float(found[e3].mean()), gap_when_missed_mean=float(gap[e3 & ~found].mean()),
                   gap_when_missed_median=float(np.median(gap[e3 & ~found])) if (e3 & ~found).any() else np.nan)
    return per_country, summary


OUTCOMES, BY_EDITION, NAMED, LEVEL = [], [], [], []
for label in EDITIONS:
    with open(IN1 / f"edition_{label}.pkl", "rb") as f: E = pickle.load(f)
    goals, meta = E["goals"], E["meta"].reset_index(drop=True); F = full_index(goals, meta)
    if not all(c in F["codes"] for c in HEADLINE): raise SystemExit(f"{label}: a headline indicator is missing from the edition")
    H = [F["codes"].index(c) for c in HEADLINE]
    rank_full = np.full(len(meta), np.nan); rank_full[F["rank_sample"]] = rankdata(-F["index"][F["rank_sample"]], method="min")
    out = pd.DataFrame(dict(edition=label, iso=meta.iso, region=meta.region, income=meta.income, sids=meta.sids.astype(bool), landlocked=meta.landlocked.astype(bool),
                            rank_full=rank_full, index_full=F["index"], weakest_goal_score=np.where(F["applicable"], F["goal_scores"], np.inf).min(1)))
    per_country, s = evaluate_list(F, H)
    out = pd.concat([out, per_country], axis=1); OUTCOMES.append(out)
    BY_EDITION.append(dict(edition=label, version="report's rule", **s))
    print(f"{label}: rho {s['rho']:.3f} | within {PLACES} places {s['share_within_places']:.3f} ({s['ranked_scored']} of {s['ranked_countries']} ranked countries scored) | "
          f"weakest goal found {s['share_weakest_goal_found']:.3f} ({s['reference_scored']} of {s['reference_countries']} reference countries scored) | "
          f"when missed, the named goal scores {s['gap_when_missed_mean']:.1f} points above the weakest goal")
    _, s0 = evaluate_list(F, H, max_missing=0); BY_EDITION.append(dict(edition=label, version="countries with data on all headline indicators", **s0))
    e3 = per_country.in_weakest_goal_sample.to_numpy(); tab = pd.DataFrame(0.0, index=F["order"], columns=F["order"])
    for ts, ns in zip(per_country.true_weakest[e3].str.split("|"), per_country.named_weakest[e3].str.split("|")):
        ts = [int(v) for v in ts]; ns = [int(v) for v in ns]; w = 1.0 / (len(ts) * len(ns))
        for a in ts:
            for b in ns: tab.loc[a, b] += w
    for g in F["order"]:
        NAMED.append(dict(edition=label, goal=g, truly_weakest=float(tab.loc[g].sum()), named_weakest=float(tab[g].sum()), named_and_true=float(tab.loc[g, g]),
                          named_minus_true=float(tab[g].sum() - tab.loc[g].sum())))
    net = {r["goal"]: r["named_minus_true"] for r in NAMED if r["edition"] == label}
    for gi, g in enumerate(F["order"]):
        code = next(c for c in HEADLINE if c in goals[g]["codes"])
        x = goals[g]["Z"][:, list(goals[g]["codes"]).index(code)]; m = np.isfinite(x) & np.isfinite(F["goal_scores"][:, gi]); d = x[m] - F["goal_scores"][m, gi]
        LEVEL.append(dict(edition=label, goal=g, code=code, countries=int(m.sum()), level_bias=float(d.mean()), share_below_goal_score=float((d < 0).mean()), named_minus_true=net[g]))
    lv = pd.DataFrame([r for r in LEVEL if r["edition"] == label])
    print(f"      level bias of the headline indicators against net over-reporting of their goals: Spearman {spearmanr(lv.level_bias, lv.named_minus_true)[0]:+.2f}")
OUTCOMES = pd.concat(OUTCOMES, ignore_index=True)
save("headline_country_outcomes.csv", OUTCOMES.drop(columns="in_weakest_goal_sample")); save("headline_by_edition.csv", BY_EDITION)
save("headline_named_vs_true_goal.csv", NAMED); save("headline_indicator_level.csv", LEVEL)


d = OUTCOMES[OUTCOMES.status.ne("no reference")].copy()
d["scored"] = d.status.ne("not assessed"); d["found"] = d.status.eq("found")
d["rank_kept_goal_missed"] = d.status.eq("missed") & d.within_places.eq(1)
d["income_group"] = d.income.map({"LIC": "LIC+LMIC", "LMIC": "LIC+LMIC", "UMIC": "UMIC", "HIC": "HIC"}); d["all"] = "all"
ORDER = {"income_group": ["LIC+LMIC", "UMIC", "HIC"]}
ev = d[d.scored]; missed = ev[~ev.found]; pooled, by_edition = [], []
for col, name in [("income_group", "income group"), ("region", "SDR region"), ("all", "all")]:
    s = pd.DataFrame({"reference_countries": d.groupby(col).size(), "scored": d.groupby(col).scored.sum(), "found": ev.groupby(col).found.mean(),
                      "rank_kept_goal_missed": ev.groupby(col).rank_kept_goal_missed.mean(), "gap_when_missed": missed.groupby(col).gap_points.mean()})
    s = (s.reindex(ORDER[col]) if col in ORDER else s.sort_index()).rename_axis("group").reset_index(); s.insert(0, "group_type", name); pooled.append(s)
    e = pd.DataFrame({"reference_countries": d.groupby([col, "edition"]).size(), "scored": ev.groupby([col, "edition"]).size(),
                      "found": ev.groupby([col, "edition"]).found.mean(), "rank_kept_goal_missed": ev.groupby([col, "edition"]).rank_kept_goal_missed.mean()}).reset_index(level="edition")
    e = e.loc[ORDER[col] if col in ORDER else sorted(e.index.unique())].rename_axis("group").reset_index(); e.insert(0, "group_type", name); by_edition.append(e)
pooled = pd.concat(pooled, ignore_index=True); by_edition = pd.concat(by_edition, ignore_index=True)
save("headline_by_group.csv", pooled); save("headline_by_group_and_edition.csv", by_edition)

x = ev[ev.income_group.notna()]
c = x.assign(k=x.found.astype(int), n=1, hic=x.income_group.eq("HIC")).groupby("iso").agg(k=("k", "sum"), n=("n", "sum"), hic=("hic", "first")).reset_index()
K, N, Hl = c.k.to_numpy(), c.n.to_numpy(), c.hic.to_numpy(bool)
stat = lambda lab: K[lab].sum() / N[lab].sum() - K[~lab].sum() / N[~lab].sum()
rng = np.random.default_rng(SEED); obs = stat(Hl); perm = np.array([stat(rng.permutation(Hl)) for _ in range(PERMUTATIONS)])
TEST = pd.DataFrame([dict(comparison="high-income countries vs all other countries", country_editions_high_income=int(N[Hl].sum()), country_editions_other=int(N[~Hl].sum()),
                          found_high_income=K[Hl].sum() / N[Hl].sum(), found_other=K[~Hl].sum() / N[~Hl].sum(), difference=obs,
                          p_value=float((np.abs(perm) >= abs(obs) - 1e-12).mean()), test=f"permutation of income groups across {len(c)} countries ({PERMUTATIONS} draws), two-sided")])
save("headline_income_test.csv", TEST)

pd.set_option("display.width", 200)
print(pooled.round(3).to_string(index=False))
print("\nweakest goal found, by income group and edition:")
print(by_edition[by_edition.group_type == "income group"].pivot(index="group", columns="edition", values="found").reindex(ORDER["income_group"]).round(3).to_string())
print(f"\nhigh-income countries {TEST.found_high_income.iloc[0]:.3f} against all other countries {TEST.found_other.iloc[0]:.3f}: P = {TEST.p_value.iloc[0]:.3f} (permutation test)")
print("\ndone ->", RES)
