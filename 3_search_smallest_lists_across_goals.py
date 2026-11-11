import io, time, pickle, itertools, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
warnings.filterwarnings("ignore")

BASE = Path.cwd()
if not (BASE / "data").is_dir(): raise SystemExit(f"`data/` folder not found in {BASE}. Put this file next to the `data/` folder.")
IN1 = BASE / "result" / "1"; IN2 = BASE / "result" / "2"; RES = BASE / "result" / "3"; RES.mkdir(parents=True, exist_ok=True)

EDITIONS = ["2026", "2025", "2024", "2023", "2022"]
HEADLINE = ["sdg1_wpc", "sdg2_undernsh", "sdg3_lifee", "sdg4_second", "sdg5_parl", "sdg6_sanita", "sdg7_elecac", "sdg8_accounts", "sdg9_intuse",
            "sdg10_gini", "sdg11_pm25", "sdg12_nprod", "sdg13_co2gcp", "sdg14_cpma", "sdg15_redlist", "sdg16_cpi", "sdg17_statperf"]
REQUIRED = 0.95
SCORED_MIN = 0.95
MAX_MISSING = 2
RESULTS = ["overall ranking", "each country's rank", "weakest goal", "rank and weakest goal"]
COUNTRY_LEVEL = RESULTS[1:]
STARTS = 300; SEED = 81
SIZES_ONE_RESULT = [17, 20, 25, 30, 40, 50, 60]; FIXED_SIZE_STARTS = 30
MORE_STARTS = 2000; LOCAL_ROUNDS = 60; LOCAL_DROP = 4
TWO_FOR_ONE = True; MAX_PAIRS = 1500; SEED_MORE = 8283
SIZES_BOTH_RESULTS = [17, 20, 25, 30, 35, 40, 45, 50, 55, 60]
RESUME = False
EPS = 1e-9
TOL = pd.read_csv(IN1 / "tolerances.csv", encoding="utf-8-sig").set_index("result").tolerance
POINTS, PLACES = float(TOL["weakest goal"]), int(TOL["overall rank"])

LOG = io.StringIO()
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n")


def save(name, table):
    pd.DataFrame(table).to_csv(RES / name, index=False, encoding="utf-8-sig")


def goal_score(Z):
    with np.errstate(all="ignore"):
        return np.nanmean(Z, axis=1)


def sdg_index(goal_scores, has_score, region, ranked):
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


class FastList:
    """Evaluate an indicator list with fast updates when one indicator is added or removed."""
    def __init__(self, goals, meta):
        self.order = sorted(goals); self.G = len(self.order)
        self.goal_of = np.concatenate([[gi] * len(goals[g]["codes"]) for gi, g in enumerate(self.order)])
        self.codes = [c for g in self.order for c in goals[g]["codes"]]
        Z = np.column_stack([goals[g]["Z"] for g in self.order]); self.obs = np.isfinite(Z); self.Zf = np.where(self.obs, Z, 0.0)
        self.n, self.N = Z.shape; self.avail = self.obs.mean(0)
        self.cols_by_goal = [np.where(self.goal_of == gi)[0] for gi in range(self.G)]
        gmf = np.column_stack([goal_score(goals[g]["Z"]) for g in self.order]); self.app = np.isfinite(gmf); self.napp = np.maximum(self.app.sum(1), 1)
        self.ranked = meta.ranked.to_numpy(bool); self.reg = pd.factorize(meta.region.astype(str))[0]
        self.Rm = np.zeros((self.n, self.reg.max() + 1)); self.Rm[np.arange(self.n), self.reg] = 1.0
        self.index_full = sdg_index(gmf, self.app, meta.region.astype(str).to_numpy(), self.ranked)
        self.rank_sample = self.ranked & np.isfinite(self.index_full)
        self.reference = self.app.sum(1) >= meta.n_goals_applicable.to_numpy() - 1
        self.n1, self.n3 = int(self.rank_sample.sum()), int(self.reference.sum())
        full_min = np.where(self.app, gmf, np.inf).min(1)
        self.good = self.app & (np.where(self.app, gmf, np.inf) <= full_min[:, None] + POINTS + EPS)
        self._rank_cache = {}

    def full_ranks(self, rows):
        key = rows.tobytes()
        if key not in self._rank_cache: self._rank_cache[key] = rankdata(self.index_full[rows])
        return self._rank_cache[key]

    def _fill(self, gm, has):
        w = (has & self.ranked[:, None]).astype(float)
        with np.errstate(all="ignore"):
            rm = (self.Rm.T @ np.where(w > 0, gm, 0.0)) / (self.Rm.T @ w)
        fv = np.where(~has, rm[self.reg], np.nan); isf = np.isfinite(fv)
        return np.where(isf, fv, 0.0), isf

    def state(self, sel):
        sel = sorted(sel); S = np.zeros((self.n, self.G)); C = np.zeros((self.n, self.G))
        for j in sel: S[:, self.goal_of[j]] += self.Zf[:, j]; C[:, self.goal_of[j]] += self.obs[:, j]
        with np.errstate(all="ignore"): gm = S / C
        has = C > 0; ok = self.app & has
        gmw = np.where(ok, gm, np.inf)
        min_excl = np.empty((self.n, self.G)); succ_excl = np.empty((self.n, self.G), bool)
        for g in range(self.G):
            w = gmw.copy(); w[:, g] = np.inf; mn = w.min(1)
            min_excl[:, g] = mn; succ_excl[:, g] = ((w <= mn[:, None] + EPS) & np.isfinite(w) & self.good).any(1)
        mn = gmw.min(1); succ = ((gmw <= mn[:, None] + EPS) & np.isfinite(gmw) & self.good).any(1)
        counts = np.bincount(self.goal_of[sel], minlength=self.G) if sel else np.zeros(self.G, int)
        return dict(sel=set(sel), S=S, C=C, gm=gm, has=has, miss=(self.app & ~has).sum(1), gsum=np.where(ok, gm, 0.0).sum(1),
                    min_excl=min_excl, succ_excl=succ_excl, succ=succ, counts=counts,
                    **dict(zip(["fillv", "isfill"], self._fill(gm, has))))

    def _rank_outputs(self, idx, e1):
        rho = np.full(idx.shape[1], np.nan); pos = np.full(idx.shape[1], np.nan); groups = {}
        for c in range(idx.shape[1]): groups.setdefault(e1[:, c].tobytes(), []).append(c)
        for cols in groups.values():
            rows = e1[:, cols[0]]
            if rows.sum() < 8: continue
            ra = rankdata(idx[rows][:, cols], axis=0); rb = self.full_ranks(rows)
            ac, bc = ra - ra.mean(0), rb - rb.mean()
            rho[cols] = (ac * bc[:, None]).sum(0) / np.sqrt((ac ** 2).sum(0) * (bc ** 2).sum())
            pos[cols] = (np.abs(ra - rb[:, None]) <= PLACES + EPS).mean(0)
        return rho, pos

    def current(self, st):
        scored = st["miss"] <= MAX_MISSING
        idx = np.where(scored, (st["gsum"] + st["fillv"].sum(1)) / (self.napp - st["miss"] + st["isfill"].sum(1)), np.nan)[:, None]
        e1 = (self.rank_sample & scored)[:, None]; e3 = self.reference & scored
        rho, pos = self._rank_outputs(idx, e1)
        low = float(st["succ"][e3].mean()) if e3.any() else np.nan
        return np.array([rho[0]]), np.array([e1.sum() / self.n1]), np.array([pos[0]]), np.array([low]), np.array([e3.sum() / self.n3])

    def change(self, st, cands, sign):
        cands = np.asarray(cands); g = self.goal_of[cands]
        Cg = st["C"][:, g] + sign * self.obs[:, cands]; Sg = st["S"][:, g] + sign * self.Zf[:, cands]
        with np.errstate(all="ignore"): new = Sg / Cg
        newhas = Cg > 0; oldhas = st["has"][:, g]; appg = self.app[:, g]
        miss = st["miss"][:, None] - (appg & ~oldhas) + (appg & ~newhas)
        gsum = st["gsum"][:, None] - np.where(appg & oldhas, st["gm"][:, g], 0.0) + np.where(appg & newhas, new, 0.0)
        fvg, isfg = self._fill(new, newhas)
        fillsum = st["fillv"].sum(1)[:, None] - st["fillv"][:, g] + fvg
        nfill = st["isfill"].sum(1)[:, None] - st["isfill"][:, g] + isfg
        scored = miss <= MAX_MISSING; idx = np.where(scored, (gsum + fillsum) / (self.napp[:, None] - miss + nfill), np.nan)
        e1 = self.rank_sample[:, None] & scored; e3 = self.reference[:, None] & scored
        mex, sex, gg = st["min_excl"][:, g], st["succ_excl"][:, g], self.good[:, g]
        succ = np.where(appg & newhas, np.where(new < mex - EPS, gg, np.where(new > mex + EPS, sex, sex | gg)), sex)
        n3 = e3.sum(0); low = np.where(n3 > 0, (succ & e3).sum(0) / np.maximum(n3, 1), np.nan)
        rho, pos = self._rank_outputs(idx, e1)
        return rho, e1.sum(0) / self.n1, pos, low, n3 / self.n3


def evaluate_directly(fs, Z, columns):
    """Evaluate one list from scratch; used to check the fast evaluation and for the final tables."""
    Zs = np.full(Z.shape, np.nan); Zs[:, columns] = Z[:, columns]; app = fs.app
    gm = np.column_stack([goal_score(Zs[:, fs.goal_of == i]) for i in range(fs.G)]); gm_full = np.column_stack([goal_score(Z[:, fs.goal_of == i]) for i in range(fs.G)])
    scored = (app & ~np.isfinite(gm)).sum(1) <= MAX_MISSING
    idx = np.where(scored, sdg_index(gm, np.isfinite(gm), fs.region_names, fs.ranked), np.nan); e1 = fs.rank_sample & scored; e3 = fs.reference & scored
    shift = np.abs(rankdata(-idx[e1]) - rankdata(-fs.index_full[e1]))
    full_w = np.where(app, gm_full, np.inf); fmin = full_w.min(1)
    list_w = np.where(app & np.isfinite(gm), gm, np.inf); named = (list_w <= list_w.min(1)[:, None] + EPS) & np.isfinite(list_w)
    found = (np.where(named, gm_full, np.inf).min(1) - fmin) <= POINTS + EPS
    return [spearman(idx[e1], fs.index_full[e1]), float(e1.sum() / fs.rank_sample.sum()), float((shift <= PLACES + EPS).mean()),
            float(found[e3].mean()), float(e3.sum() / fs.reference.sum())]


OUTPUT_NAMES = ["rho", "share_ranked_scored", "share_within_places", "share_weakest_goal_found", "share_reference_scored"]


def largest_gap(a, b):
    return max([0.0] + [abs(x - y) if np.isfinite(x) and np.isfinite(y) else np.inf for x, y in zip(a, b) if not (np.isnan(x) and np.isnan(y))])


def check_fast_evaluation(fs, Z, n_lists=200, seed=1):
    rng = np.random.default_rng(seed); worst = 0.0
    for t in range(n_lists):
        sel = start_list(fs, rng); k = int(rng.integers(len(sel), min(fs.N, 70) + 1))
        rest = [j for j in range(fs.N) if j not in sel]
        sel |= {int(j) for j in rng.choice(rest, size=k - len(sel), replace=False)}
        st = fs.state(sel); worst = max(worst, largest_gap([float(v[0]) for v in fs.current(st)], evaluate_directly(fs, Z, sorted(sel))))
        if t < 20:
            cands = [j for j in range(fs.N) if j not in sel][:10]; out = fs.change(st, cands, +1)
            for ci, c in enumerate(cands): worst = max(worst, largest_gap([float(v[ci]) for v in out], [float(v[0]) for v in fs.current(fs.state(sel | {c}))]))
    if worst > 1e-9: raise SystemExit(f"check failed: fast and direct evaluation differ by {worst:.3g}")
    return worst


def value_of(out, result):
    rho, cov1, pos, low, cov3 = out
    if result == "overall ranking": f, cov = rho, cov1
    elif result == "each country's rank": f, cov = pos, cov1
    elif result == "weakest goal": f, cov = low, cov3
    else: f, cov = np.fmin(pos, low), np.fmin(cov1, cov3)
    f = np.where(np.isfinite(f), f, -1.0)
    return f - 2.0 * np.maximum(0.0, SCORED_MIN - cov), np.where(cov >= SCORED_MIN - 1e-12, f, -1.0)


def start_list(fs, rng):
    return {int(rng.choice(c, p=(fs.avail[c] + 1e-6) / (fs.avail[c] + 1e-6).sum())) for c in fs.cols_by_goal}


def swap_once(fs, sel, st, val, result, rng, steer=False):
    for i in rng.permutation(sorted(sel)):
        i = int(i); gi = fs.goal_of[i]; rest = sel - {i}
        cands = [j for j in (fs.cols_by_goal[gi] if st["counts"][gi] == 1 else range(fs.N)) if j not in sel]
        if not cands: continue
        ob, ad = value_of(fs.change(fs.state(rest), cands, +1), result); v = ob if steer else ad; b = int(np.argmax(v))
        if v[b] > val + 1e-6:
            sel = rest | {int(cands[b])}; return sel, fs.state(sel), float(v[b]), True
    return sel, st, val, False


def shrink(fs, result, sel, rng):
    """Add indicators until the requirement holds, then remove and swap indicators while it still holds."""
    sel = set(sel); st = fs.state(sel); adm = float(value_of(fs.current(st), result)[1][0])
    while adm < REQUIRED:
        cands = [j for j in range(fs.N) if j not in sel]
        ob, ad = value_of(fs.change(st, cands, +1), result); b = int(np.argmax(ob))
        sel = sel | {cands[b]}; st = fs.state(sel); adm = float(ad[b])
    improved = True
    while improved:
        while True:
            cands = [j for j in sel if st["counts"][fs.goal_of[j]] > 1]
            if not cands: break
            _, ad = value_of(fs.change(st, cands, -1), result); okm = ad >= REQUIRED - 1e-12
            if not okm.any(): break
            b = int(np.argmax(np.where(okm, ad, -np.inf))); sel = sel - {cands[b]}; st = fs.state(sel); adm = float(ad[b])
        sel, st, adm, improved = swap_once(fs, sel, st, adm, result, rng)
    return sorted(sel), adm


def best_at_fixed_sizes(fs, result, rng, sizes):
    """Best value found for lists of each fixed size, by greedy additions and swaps from random starts."""
    best = {k: (-1.0, None) for k in sizes}
    for _ in range(FIXED_SIZE_STARTS):
        sel = start_list(fs, rng); st = fs.state(sel); path = {}
        while True:
            if len(sel) in best: path[len(sel)] = set(sel)
            if len(sel) >= max(sizes): break
            cands = [j for j in range(fs.N) if j not in sel]
            ob, _ = value_of(fs.change(st, cands, +1), result); sel = sel | {cands[int(np.argmax(ob))]}; st = fs.state(sel)
        for k, s0 in path.items():
            cur = set(s0); stc = fs.state(cur); obj = float(value_of(fs.current(stc), result)[0][0]); improved = True
            while improved: cur, stc, obj, improved = swap_once(fs, cur, stc, obj, result, rng, steer=True)
            adm = float(value_of(fs.current(stc), result)[1][0])
            if adm > best[k][0]: best[k] = (adm, sorted(cur))
    return best


def local_round(fs, result, best, rng):
    sel = set(best); st = fs.state(sel)
    removable = [j for j in sel if st["counts"][fs.goal_of[j]] > 1]
    if not removable: return sorted(sel), float(value_of(fs.current(st), result)[1][0])
    drop = rng.choice(removable, size=min(LOCAL_DROP, len(removable)), replace=False)
    sel = sel - {int(j) for j in drop}
    counts = np.bincount(fs.goal_of[sorted(sel)], minlength=fs.G)
    for gi in np.where(counts == 0)[0]: sel.add(int(rng.choice(fs.cols_by_goal[gi])))
    return shrink(fs, result, sel, rng)


def two_for_one(fs, result, best, rng):
    """Replace two indicators with one while the requirement still holds, then shrink again."""
    sel = set(best); improved = True
    while improved:
        improved = False; st = fs.state(sel)
        rem = [j for j in sorted(sel) if st["counts"][fs.goal_of[j]] > 1]
        pairs = list(itertools.combinations(rem, 2)); rng.shuffle(pairs)
        if MAX_PAIRS: pairs = pairs[:MAX_PAIRS]
        for a, b in pairs:
            if fs.goal_of[a] == fs.goal_of[b] and st["counts"][fs.goal_of[a]] < 3: continue
            rest = sel - {a, b}; st2 = fs.state(rest)
            cands = [j for j in range(fs.N) if j not in rest]
            _, ad = value_of(fs.change(st2, cands, +1), result)
            if ad.max() >= REQUIRED - 1e-12:
                sel = rest | {int(cands[int(np.argmax(ad))])}
                sel = set(shrink(fs, result, sel, rng)[0]); improved = True; break
    st = fs.state(sel)
    return sorted(sel), float(value_of(fs.current(st), result)[1][0])


def search_edition(label):
    """Run the searches for one edition and return the smallest lists found for each result."""
    with open(IN1 / f"edition_{label}.pkl", "rb") as f: E = pickle.load(f)
    goals, meta = E["goals"], E["meta"].reset_index(drop=True)
    fs = FastList(goals, meta); fs.region_names = meta.region.astype(str).to_numpy(); Z = np.column_stack([goals[g]["Z"] for g in fs.order]); t0 = time.time()
    name = lambda s: " ".join(fs.codes[j] for j in s); cols = lambda s: [fs.codes.index(c) for c in s.split()]
    outputs = lambda s: dict(zip(OUTPUT_NAMES, evaluate_directly(fs, Z, sorted(s))))
    say(f"\n=== {label} ({fs.N} indicators)")
    say(f"  check passed: fast and direct evaluation agree on 200 random lists (largest difference {check_fast_evaluation(fs, Z):.1e})")
    H2 = pd.read_csv(IN2 / "headline_by_edition.csv", encoding="utf-8-sig"); h2 = H2[(H2.edition.astype(str) == label) & (H2.version == "report's rule")].iloc[0]
    mine = [float(v[0]) for v in fs.current(fs.state([fs.codes.index(c) for c in HEADLINE]))]
    gap = largest_gap(mine, [h2[k] for k in OUTPUT_NAMES])
    if gap > 1e-9: raise SystemExit(f"check failed: the headline index evaluated here differs from file 2 by {gap:.3g}")
    say(f"  check passed: headline index as in file 2 (within {PLACES} places {mine[2]:.3f}, weakest goal found {mine[3]:.3f})")
    STARTS_T, SIZES_T, LISTS_T, ONE_T, BOTH_T = [], [], [], [], []
    rng = np.random.default_rng(SEED); first = {}
    for result in RESULTS:
        res = [shrink(fs, result, start_list(fs, rng), rng) for _ in range(STARTS)]
        STARTS_T += [dict(edition=label, result=result, search="first", start=r, size=len(s), value=v, indicators=name(s)) for r, (s, v) in enumerate(res)]
        sizes = np.array([len(s) for s, _ in res]); kmin = int(sizes.min())
        best = max((x for x in res if len(x[0]) == kmin), key=lambda x: x[1])[0]
        first[result] = dict(res=res, kmin=kmin, best=best, at_smallest=int((sizes == kmin).sum()), median=float(np.median(sizes)))
        say(f"  first search, {result:22s}: smallest {kmin:3d} ({kmin / fs.N:.0%}), reached by {int((sizes == kmin).sum())} of {STARTS} starts, median {np.median(sizes):.0f}")
    fixed = {}
    for result in ["each country's rank", "weakest goal", "overall ranking"]:
        fixed[result] = best_at_fixed_sizes(fs, result, rng, SIZES_ONE_RESULT)
    final = {"overall ranking": dict(k=first["overall ranking"]["kmin"], best=first["overall ranking"]["best"], found_by="first search",
                                     lists={frozenset(s) for s, _ in first["overall ranking"]["res"] if len(s) == first["overall ranking"]["kmin"]})}
    for result in COUNTRY_LEVEL:
        rng = np.random.default_rng(SEED_MORE + 10 * EDITIONS.index(label) + COUNTRY_LEVEL.index(result)); t1 = time.time()
        best = sorted(first[result]["best"]); best_k = len(best); found_by = "first search"; at_size = {}
        def record(s): at_size.setdefault(len(s), set()).add(frozenset(s))
        for s, _ in first[result]["res"]: record(s)
        sizes = []
        for r in range(MORE_STARTS):
            s, v = shrink(fs, result, start_list(fs, rng), rng); sizes.append(len(s)); record(s)
            STARTS_T.append(dict(edition=label, result=result, search="further", start=r, size=len(s), value=v, indicators=name(s)))
            if len(s) < best_k: best, best_k, found_by = s, len(s), "further random start"
        hits = 0
        for r in range(LOCAL_ROUNDS):
            s, v = local_round(fs, result, best, rng); record(s)
            if len(s) < best_k: best, best_k, found_by = s, len(s), "local search"; hits += 1
            elif len(s) == best_k and frozenset(s) != frozenset(best): best = s
        if TWO_FOR_ONE:
            s, v = two_for_one(fs, result, best, rng); record(s)
            if len(s) < best_k: best, best_k, found_by = s, len(s), "two-for-one exchange"
        final[result] = dict(k=best_k, best=best, found_by=found_by, lists=at_size[best_k], more_median=float(np.median(sizes)) if sizes else np.nan,
                             more_at_or_below_first=int((np.array(sizes) <= first[result]["kmin"]).sum()) if sizes else 0)
        say(f"  further search, {result:22s}: {MORE_STARTS} starts smallest {int(np.min(sizes)) if sizes else '-'}, then local search and exchanges -> smallest {best_k} "
            f"(first search: {first[result]['kmin']}), found by {found_by}  [{time.time() - t1:.0f} s]")
    reported = {}
    for result in RESULTS:
        rows = [dict(edition=label, result=result, size=final[result]["k"], indicators=name(sorted(s)), **outputs(s)) for s in sorted(final[result]["lists"], key=lambda s: sorted(s))]
        d = pd.DataFrame(rows)
        if result == "overall ranking": chosen = name(final[result]["best"])
        else:
            key = {"each country's rank": d.share_within_places, "weakest goal": d.share_weakest_goal_found}.get(result, np.fmin(d.share_within_places, d.share_weakest_goal_found))
            chosen = d.assign(_v=key).sort_values("_v", ascending=False).iloc[0].indicators
        d["reported"] = d.indicators == chosen; LISTS_T += d.to_dict("records"); c = d[d.reported].iloc[0]; reported[result] = sorted(cols(c.indicators))
        SIZES_T.append(dict(edition=label, result=result, indicators_in_edition=fs.N, size_first_search=first[result]["kmin"], starts_reaching_it=first[result]["at_smallest"],
                            starts=STARTS, median_size_first_search=first[result]["median"], smallest_size=final[result]["k"], share_of_indicators=final[result]["k"] / fs.N,
                            found_by=final[result]["found_by"], distinct_smallest_lists=len(d), **{k: float(c[k]) for k in OUTPUT_NAMES}, indicators=c.indicators))
        say(f"  {result:22s}: smallest list {final[result]['k']} indicators ({final[result]['k'] / fs.N:.0%}), {len(d)} distinct | rho {c.rho:.3f}, within {PLACES} places {c.share_within_places:.3f}, "
            f"weakest goal found {c.share_weakest_goal_found:.3f} (scored {c.share_ranked_scored:.2f}/{c.share_reference_scored:.2f})")
        if result in COUNTRY_LEVEL and value_of([np.array([c[k]]) for k in OUTPUT_NAMES], result)[1][0] < REQUIRED - 1e-9:
            raise SystemExit(f"check failed: the reported list for {result} does not meet the requirement when evaluated directly")
    S = pd.DataFrame([r for r in STARTS_T])
    for result in fixed:
        by_start = S[S.result == result].groupby("size").value.max()
        for k in sorted(set(SIZES_ONE_RESULT) | set(by_start.index)):
            fv = fixed[result][k][0] if k in fixed[result] and fixed[result][k][0] >= 0 else np.nan; sv = by_start.get(k, np.nan)
            if np.isfinite([fv, sv]).any(): ONE_T.append(dict(edition=label, result=result, size=int(k), best_value=float(np.nanmax([fv, sv])), fixed_size_search=fv, starts=sv))
    rng = np.random.default_rng(SEED_MORE + 7); both = best_at_fixed_sizes(fs, "rank and weakest goal", rng, SIZES_BOTH_RESULTS)
    pool = {}
    for s in S.indicators: pool.setdefault(len(s.split()), []).append(s)
    base = reported["rank and weakest goal"]
    for k in SIZES_BOTH_RESULTS:
        best_v, best_s = both[k][0], (sorted(both[k][1]) if both[k][1] else None)
        for s2 in pool.get(k, []):
            v2 = float(value_of(fs.current(fs.state(cols(s2))), "rank and weakest goal")[1][0])
            if v2 > best_v: best_v, best_s = v2, sorted(cols(s2))
        if k >= len(base):
            cur = set(base); stc = fs.state(cur)
            while len(cur) < k:
                cands = [j for j in range(fs.N) if j not in cur]
                ob, _ = value_of(fs.change(stc, cands, +1), "rank and weakest goal"); cur = cur | {cands[int(np.argmax(ob))]}; stc = fs.state(cur)
            obj = float(value_of(fs.current(stc), "rank and weakest goal")[0][0]); improved = True
            while improved: cur, stc, obj, improved = swap_once(fs, cur, stc, obj, "rank and weakest goal", rng, steer=True)
            v3 = float(value_of(fs.current(stc), "rank and weakest goal")[1][0])
            if v3 > best_v: best_v, best_s = v3, sorted(cur)
        if best_s is None: BOTH_T.append(dict(edition=label, size=k, found=False)); continue
        BOTH_T.append(dict(edition=label, size=k, found=True, **outputs(best_s), indicators=name(best_s)))
    say("  one list scored on both results: " + "; ".join(f"{r['size']}: {r['share_within_places']:.2f} / {r['share_weakest_goal_found']:.2f}" for r in BOTH_T if r["found"]))
    say(f"  time {time.time() - t0:.0f} s")
    return dict(starts=STARTS_T, sizes=SIZES_T, lists=LISTS_T, one=ONE_T, both=BOTH_T)


say(f"tolerances: overall rank within {PLACES} places, weakest goal within {POINTS:g} points | first search {STARTS} starts (seed {SEED}); "
    f"further search {MORE_STARTS} starts, {LOCAL_ROUNDS} local rounds (drop {LOCAL_DROP}), two-for-one {'on' if TWO_FOR_ONE else 'off'} (seed {SEED_MORE})")
DONE = {}
for label in EDITIONS:
    ck = RES / f"checkpoint_{label}.pkl"
    if RESUME and ck.exists():
        with open(ck, "rb") as f: DONE[label] = pickle.load(f)
        say(f"\n=== {label}: read from {ck.name}"); continue
    DONE[label] = search_edition(label)
    with open(ck, "wb") as f: pickle.dump(DONE[label], f)

join = lambda key: pd.DataFrame([r for label in EDITIONS for r in DONE[label][key]])
save("search_starts.csv", join("starts")); save("smallest_list_sizes.csv", join("sizes")); save("smallest_lists.csv", join("lists"))
save("best_lists_by_size_one_result.csv", join("one")); save("best_lists_by_size_both_results.csv", join("both"))


SZ = join("sizes"); L = join("lists")
say("\nSmallest lists found (range across editions):")
for result in RESULTS:
    x = SZ[SZ.result == result]
    say(f"  {result:22s}: {x.smallest_size.min()}-{x.smallest_size.max()} indicators ({x.share_of_indicators.min():.0%}-{x.share_of_indicators.max():.0%} of the edition's indicators); "
        f"first search {x.size_first_search.min()}-{x.size_first_search.max()}, reached by {x.starts_reaching_it.min()}-{x.starts_reaching_it.max()} of {STARTS} starts; "
        f"lowered by {(x.size_first_search - x.smallest_size).min()}-{(x.size_first_search - x.smallest_size).max()} in the further search")
r = L[L.result == "overall ranking"]
say(f"Smallest lists that keep the overall ranking: {len(r)} distinct lists ({r.groupby('edition').size().min()}-{r.groupby('edition').size().max()} per edition); "
    f"within {PLACES} places {r.share_within_places.min():.2f}-{r.share_within_places.max():.2f}; weakest goal found {r.share_weakest_goal_found.min():.2f}-{r.share_weakest_goal_found.max():.2f} "
    f"(median per edition {r.groupby('edition').share_weakest_goal_found.median().min():.2f}-{r.groupby('edition').share_weakest_goal_found.median().max():.2f})")
(RES / "search_log.txt").write_text(LOG.getvalue(), encoding="utf-8")
print("\ndone ->", RES)


both_lists = {e: set(SZ[(SZ.edition.astype(str) == e) & (SZ.result == "rank and weakest goal")].indicators.iloc[0].split()) for e in EDITIONS}
allc = sorted(set().union(*both_lists.values())); n_in = pd.Series({c: sum(c in s for s in both_lists.values()) for c in allc})
print("sizes: " + ", ".join(f"{e}: {len(s)}" for e, s in both_lists.items()))
print(f"indicators in all {len(EDITIONS)} editions: {int((n_in == len(EDITIONS)).sum())}; in {len(EDITIONS) - 1}: {int((n_in == len(EDITIONS) - 1).sum())}; in one only: {int((n_in == 1).sum())}; union {len(n_in)}")
J = [len(both_lists[a] & both_lists[b]) / len(both_lists[a] | both_lists[b]) for a, b in itertools.combinations(EDITIONS, 2)]
print(f"Jaccard between pairs of editions: {min(J):.2f}-{max(J):.2f}")
pd.DataFrame({"indicator": n_in.index, "editions": n_in.to_numpy(), **{f"in_{e}": [c in both_lists[e] for c in n_in.index] for e in EDITIONS}}).to_csv(RES / "overlap_between_editions.csv", index=False, encoding="utf-8-sig")
