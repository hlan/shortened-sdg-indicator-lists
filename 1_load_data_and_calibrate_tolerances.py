import io, re, glob, pickle, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata
warnings.filterwarnings("ignore")

BASE = Path.cwd(); DATA = BASE / "data"
if not DATA.is_dir(): raise SystemExit(f"`data/` folder not found in {BASE}. Put this file next to the `data/` folder.")
RES = BASE / "result" / "1"; RES.mkdir(parents=True, exist_ok=True)

EDITIONS = ["2026", "2025", "2024", "2023", "2022"]
RAW_FILE = {"2026": "Sustainable_Development_Report_2026*.csv", "2025": "Sustainable_Development_Report_2025*.csv",
            "2024": "Sustainable_Development_Report_2024*.csv", "2023": "Sustainable_Development_Report_2023*.csv",
            "2022": "SDR-2022-database*.xlsx"}
SHEET_2022 = "SDR2022 Data"

WB_FY27 = {"LIC": "AFG BFA BDI CAF TCD COD ERI ETH GMB GNB PRK LBR MDG MWI MLI MOZ NER RWA SLE SOM SSD SDN SYR UGA YEM",
           "LMIC": "AGO BGD BEN BTN BOL KHM CMR COM COG CIV DJI EGY SWZ GHA GIN HTI HND IND KEN KIR KGZ LAO LBN LSO MRT MAR MMR NAM NPL NIC NGA PAK PNG STP SEN SLB TJK TZA TLS TGO TUN UZB VUT VEN PSE ZMB ZWE",
           "UMIC": "ALB DZA ARG ARM AZE BLR BLZ BIH BWA BRA CPV CHN COL CUB DMA DOM ECU SLV GNQ FJI GAB GEO GRD GTM IDN IRN IRQ JAM JOR KAZ XKX LBY MYS MDV MHL MUS MEX FSM MDA MNG MNE MKD PRY PER PHL WSM SRB ZAF LKA LCA VCT SUR THA TON TUR TKM TUV UKR VNM",
           "HIC": "AND ATG AUS AUT BHS BHR BRB BEL BRN BGR CAN CHL CRI HRV CYP CZE DNK EST FIN FRA DEU GRC GUY HUN ISL IRL ISR ITA JPN KOR KWT LVA LIE LTU LUX MLT MCO NRU NLD NZL NOR OMN PLW PAN POL PRT QAT ROU RUS SMR SAU SYC SGP SVK SVN ESP KNA SWE CHE TTO ARE GBR USA URY"}
INCOME = {c: g for g, s in WB_FY27.items() for c in s.split()}
SIDS = set("ATG BHS BRB BLZ CPV COM CUB DMA DOM FJI GRD GNB GUY HTI JAM KIR MDV MHL FSM MUS NRU PLW PNG WSM STP SGP KNA LCA VCT SYC SLB SUR TLS TON TTO TUV VUT".split())
LANDLOCKED = set("AFG AND ARM AUT AZE BLR BTN BOL BWA BFA BDI CAF TCD CZE SWZ ETH HUN KAZ KGZ LAO LSO LIE LUX MWI MLI "
                 "MDA MNG NPL NER MKD PRY RWA SMR SRB SVK SSD CHE TJK TKM UGA UZB ZMB ZWE".split())

INDEX_TOLERANCE = 0.05
SHARE_REQUIRED = 0.95
POINT_GRID = [0, 1, 2, 3, 4, 5, 6, 8, 10]
PLACE_GRID = [5, 10, 15, 20, 25, 30]
PLACE_MINIMUM = 10
DISTURBANCES = [("score noise", 2.0), ("bound shift", 0.10)]
DRAWS = 200; SEED = 71
EPS = 1e-9
RESULTS = {"weakest indicator": POINT_GRID, "rank within goal": PLACE_GRID, "weakest goal": POINT_GRID, "overall rank": PLACE_GRID}


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


def find_column(columns, patterns, exclude=("Goal", "Score_", "Value_", "Normalized", "Dashboard", "Trend")):
    for p in patterns:
        hits = [c for c in columns if re.search(p, str(c), flags=re.I) and not str(c).startswith(exclude)]
        if hits: return hits[0]
    return None


def official_index(d):
    score_col = find_column(d.columns, [r"^overall[ _]?score$", r"sdg[ _]?index[ _]?score", r"^index[ _]?score$", r"overall.*score"])
    rank_col = find_column(d.columns, [r"^overall[ _]?rank$", r"sdg[ _]?index[ _]?rank", r"overall.*rank"])
    num = lambda c: pd.to_numeric(d[c], errors="coerce").to_numpy(float) if c else np.full(len(d), np.nan)
    return num(score_col), num(rank_col)


def recover_bounds(value, score):
    """Recover the report's worst and best values of an indicator from its published value and score pairs."""
    m = np.isfinite(value) & np.isfinite(score) & (score > 0.001) & (score < 99.999)
    if m.sum() < 2 or np.ptp(value[m]) == 0:
        return None
    A = np.column_stack([value[m], np.ones(int(m.sum()))]); (a, b), *_ = np.linalg.lstsq(A, score[m], rcond=None)
    if abs(a) < 1e-12:
        return None
    lo, hi = -b / a, (100 - b) / a
    pred = np.clip((value - lo) / (hi - lo), 0, 1) * 100; ok = np.isfinite(value) & np.isfinite(score)
    return lo, hi, float(np.nanmax(np.abs(pred[ok] - score[ok]))) if ok.any() else np.nan


def country_table(d, region_col, ranked):
    meta = pd.DataFrame(dict(iso=d.iso.astype(str), region=d[region_col].astype(str) if region_col in d.columns else "", ranked=ranked))
    meta["income"] = meta.iso.map(INCOME).fillna("unclassified"); meta["sids"] = meta.iso.isin(SIDS)
    meta["official_score"], meta["official_rank"] = official_index(d)
    return meta


def pack_goals(use):
    goals = {}
    for g in sorted(use):
        goals[g] = dict(codes=[u[0] for u in use[g]], Z=np.column_stack([u[1] for u in use[g]]), value=np.column_stack([u[2] for u in use[g]]),
                        worst=np.array([u[3] for u in use[g]], float), best=np.array([u[4] for u in use[g]], float),
                        has_value=np.column_stack([u[5] for u in use[g]]))
    return goals


def check_goal_scores(goals, d, meta, column):
    rows = []
    for g, G in goals.items():
        col = column(g)
        if col in d.columns:
            off = pd.to_numeric(d[col], errors="coerce").to_numpy(float); ours = goal_score(G["Z"])
            m = np.isfinite(off) & np.isfinite(ours) & (meta.region != "OECD").to_numpy()
            rows.append(dict(goal=g, countries=int(m.sum()), max_abs_diff=float(np.max(np.abs(off[m] - ours[m]))) if m.any() else np.nan))
    return pd.DataFrame(rows)


def read_csv_edition(path, max_err=0.05):
    """Read an edition published as a CSV file (2023-2026) and score every indicator from 0 to 100."""
    d = pd.read_csv(path, low_memory=False)
    idc = next(c for c in ["ID", "iso3", "Country", "Country Code ISO3"] if c in d.columns and d[c].astype(str).str.fullmatch(r"[A-Z]{3}").mean() > 0.9)
    d = d.rename(columns={idc: "iso"}); d = d[d.iso.astype(str).str.fullmatch(r"[A-Z]{3}")].reset_index(drop=True)
    codes = sorted({c[6:] for c in d.columns if c.startswith("Score_sdg")}, key=lambda c: (int(re.match(r"sdg(\d+)_", c).group(1)), c))
    audit, use = [], {}
    for c in codes:
        g = int(re.match(r"sdg(\d+)_", c).group(1))
        s = pd.to_numeric(d["Score_" + c], errors="coerce").to_numpy(float)
        v = pd.to_numeric(d["Value_" + c], errors="coerce").to_numpy(float) if "Value_" + c in d.columns else None
        b = recover_bounds(v, s) if v is not None else None
        if b is not None and b[2] <= max_err:
            Z = np.clip((v - b[0]) / (b[1] - b[0]), 0, 1) * 100
            Z = np.where(np.isfinite(Z), Z, np.clip(s, 0, 100))
            item = (c, Z, v, b[0], b[1], np.isfinite(v)); source = "raw value with recovered bounds"; note = f"worst={b[0]:.4g} best={b[1]:.4g}, largest error {b[2]:.3f}"
        else:
            Z = np.clip(s, 0, 100); item = (c, Z, np.full(len(d), np.nan), np.nan, np.nan, np.zeros(len(d), bool))
            source = "official score only"; note = "bounds not recovered"
        audit.append(dict(code=c, goal=g, label="", included=True, source=source, note=note)); use.setdefault(g, []).append(item)
    goals = pack_goals(use)
    ranked = pd.to_numeric(d["Overall_Rank"], errors="coerce").notna().to_numpy() if "Overall_Rank" in d.columns else np.ones(len(d), bool)
    meta = country_table(d, "Region", ranked)
    return goals, meta, pd.DataFrame(audit), check_goal_scores(goals, d, meta, lambda g: f"Goal_{g}_Score")


def read_xlsx_edition(path, sheet):
    """Read the 2022 edition from the database workbook and score every indicator from 0 to 100."""
    with io.BytesIO(Path(path).read_bytes()) as b:
        with pd.ExcelFile(b) as xl:
            names = {n.lower().replace(" ", ""): n for n in xl.sheet_names}
            cb = xl.parse(names["codebook"], header=None)
            raw = xl.parse(names.get(sheet.lower().replace(" ", ""), sheet), header=None)

    def frame(df, must):
        for i in range(min(10, len(df))):
            toks = [str(x).strip() for x in df.iloc[i]]
            if any(t.lower() == must.lower() for t in toks):
                out = df.iloc[i + 1:].copy(); out.columns = toks; return out.reset_index(drop=True)
        raise ValueError("header not found: " + must)
    cb = frame(cb, "IndCode"); cb.columns = [c.strip().lower() for c in cb.columns]
    pick = lambda *c: next((cb[x] for x in c if x in cb.columns), pd.Series([np.nan] * len(cb)))
    CB = pd.DataFrame({"code": cb["indcode"].astype(str).str.strip(), "sdg": pd.to_numeric(cb["sdg"], errors="coerce"),
                       "glob": cb["global"].astype(str).str.lower().eq("yes"),
                       "hi": pd.to_numeric(pick("optimum (= 100)", "optimum"), errors="coerce"),
                       "lo": pd.to_numeric(pick("lower bound (=0)", "lower bound"), errors="coerce"), "label": pick("indicator").astype(str)})
    CB = CB[CB.code.str.match(r"^sdg\d+_") & CB.glob].drop_duplicates("code").set_index("code")
    d = frame(raw, "Regions used for the SDR"); d.columns = [str(c) for c in d.columns]
    idc = "Country Code ISO3" if "Country Code ISO3" in d.columns else "id"
    d = d.rename(columns={idc: "iso"}); d = d[d.iso.astype(str).str.fullmatch(r"[A-Z]{3}")].reset_index(drop=True)
    audit, use = [], {}
    for code, r in CB.iterrows():
        g = int(r.sdg) if np.isfinite(r.sdg) else None
        y = "Year: " + code; found = y in d.columns; note = ""; rc = None
        if found:
            rc = d.columns[list(d.columns).index(y) - 1]
            if re.search(r"sdg\d+_", rc) or rc.startswith(("Year", "Imputation", "Normalized", "Dashboard", "Trend")):
                found, note = False, "raw value column not next to the Year column"
        else:
            note = "no Year column for the indicator"
        if found and not (np.isfinite(r.lo) and np.isfinite(r.hi) and r.hi != r.lo):
            found, note = False, "bounds missing"
        if g is None:
            found, note = False, "no goal number"
        source = "raw value with codebook bounds"
        if found and pd.to_numeric(d[rc], errors="coerce").notna().sum() == 0:
            nc = "Normalized Score: " + code
            if nc in d.columns and pd.to_numeric(d[nc], errors="coerce").notna().sum() > 0:
                source, note = "official score only", "raw column empty; official normalized score used"
            else:
                found, note = False, "raw column empty and no normalized score"
        audit.append(dict(code=code, goal=g, label=r.label, included=found, source=source if found else "", note=note))
        if not found: continue
        if source == "raw value with codebook bounds":
            v = pd.to_numeric(d[rc], errors="coerce").to_numpy(float)
            item = (code, np.clip((v - r.lo) / (r.hi - r.lo), 0, 1) * 100, v, r.lo, r.hi, np.isfinite(v))
        else:
            Z = np.clip(pd.to_numeric(d["Normalized Score: " + code], errors="coerce").to_numpy(float), 0, 100)
            item = (code, Z, np.full(len(d), np.nan), np.nan, np.nan, np.zeros(len(d), bool))
        use.setdefault(g, []).append(item)
    goals = pack_goals(use)
    ranked_col = next((c for c in d.columns if re.search(r"SDG Index Score", c)), None)
    ranked = pd.to_numeric(d[ranked_col], errors="coerce").notna().to_numpy() if ranked_col else np.ones(len(d), bool)
    meta = country_table(d, "Regions used for the SDR", ranked)
    return goals, meta, pd.DataFrame(audit), check_goal_scores(goals, d, meta, lambda g: f"Goal {g} Score")


EDITION, audit_all, check_all, repro_all = {}, [], [], []
for label in EDITIONS:
    hits = sorted(glob.glob(str(DATA / RAW_FILE[label])))
    if not hits: raise SystemExit(f"{label}: no file matching {RAW_FILE[label]} in {DATA}")
    path = hits[0]
    goals, meta, audit, checks = read_csv_edition(path) if path.lower().endswith(".csv") else read_xlsx_edition(path, SHEET_2022)
    if sorted(goals) != list(range(1, 18)): raise SystemExit(f"{label}: expected 17 goals with data, found {sorted(goals)}")
    meta["landlocked"] = meta.iso.isin(LANDLOCKED); ll = meta.landlocked.to_numpy()
    meta["n_goals_applicable"] = np.where(ll, 16, 17)
    for key in ("Z", "value"): goals[14][key] = np.where(ll[:, None], np.nan, goals[14][key])
    goals[14]["has_value"] = goals[14]["has_value"] & ~ll[:, None]
    order = sorted(goals); gm = np.column_stack([goal_score(goals[g]["Z"]) for g in order])
    index = sdg_index(gm, np.isfinite(gm), meta.region.to_numpy(), meta.ranked.to_numpy(bool))
    has = np.isfinite(meta.official_score.to_numpy()); rank = np.full(len(meta), np.nan); rank[has] = rankdata(-index[has], method="min")
    t = pd.DataFrame(dict(edition=label, iso=meta.iso, region=meta.region, landlocked=ll, ranked=meta.ranked, index_rebuilt=index, official_score=meta.official_score,
                          rank_rebuilt=rank, official_rank=meta.official_rank)); repro_all.append(t)
    x = t[has]; max_diff = float((x.index_rebuilt - x.official_score).abs().max()); n_rank = int((x.rank_rebuilt != x.official_rank).sum())
    K = sum(len(goals[g]["codes"]) for g in goals)
    print(f"{label}: {Path(path).name} | {K} indicators | {len(meta)} countries, {int(meta.ranked.sum())} ranked | goal scores: largest difference from official {checks.max_abs_diff.max():.4f} | "
          f"SDG Index: largest difference {max_diff:.4f} points, ranks differing {n_rank}")
    if max_diff > INDEX_TOLERANCE: raise SystemExit(f"{label}: the rebuilt index differs from the official index by more than {INDEX_TOLERANCE} points. Stop and check the data.")
    EDITION[label] = dict(label=label, source=Path(path).name, goals=goals, meta=meta, K=K)
    with open(RES / f"edition_{label}.pkl", "wb") as f: pickle.dump(EDITION[label], f)
    audit_all.append(audit.assign(edition=label)); check_all.append(checks.assign(edition=label))
save("indicator_audit.csv", pd.concat(audit_all)); save("goal_score_check.csv", pd.concat(check_all)); save("index_reproduction.csv", pd.concat(repro_all))


def weakest_mask(Z):
    obs = np.isfinite(Z); low = np.where(obs, Z, np.inf).min(1)
    L = obs & (Z <= low[:, None] + EPS); ok = obs.sum(1) >= 2
    L[~ok] = False
    return L, ok


def share_within_places(a, b, places):
    d = np.abs(rankdata(a) - rankdata(b))
    return {m: float((d <= m + EPS).mean()) for m in places}


def share_weakest_found(Z_ref, reported, points, ok):
    ref_min = np.where(np.isfinite(Z_ref), Z_ref, np.inf).min(1)
    rep = np.where(reported, Z_ref, np.inf).min(1)
    return {d: float(np.mean((rep <= ref_min + d + EPS)[ok])) if ok.any() else np.nan for d in points}


def disturb(goals, kind, level, rng):
    """Disturb the full indicator set with random score noise or with shifted worst and best values."""
    if kind == "score noise":
        return {g: np.where(np.isfinite(G["Z"]), np.clip(G["Z"] + rng.normal(0, level, G["Z"].shape), 0, 100), np.nan) for g, G in goals.items()}
    Zp = {}
    for g, G in goals.items():
        span = G["best"] - G["worst"]
        worst = G["worst"] + rng.uniform(-level, level, len(span)) * span; best = G["best"] + rng.uniform(-level, level, len(span)) * span
        Zp[g] = np.where(G["has_value"], np.clip((G["value"] - worst) / (best - worst), 0, 1) * 100, G["Z"])
    return Zp


def kept_under_disturbance(goals, Zp, ref):
    """Share of countries whose results stay within each tolerance when the full indicator set is disturbed."""
    out, gmp = {}, []
    for g in ref["order"]:
        r, Z = ref[g], goals[g]["Z"]; sp = goal_score(Zp[g]); gmp.append(sp)
        Lp, _ = weakest_mask(Zp[g])
        low = share_weakest_found(Z, Lp, POINT_GRID, r["ok"])
        pos = share_within_places(sp[r["has"]], r["score"][r["has"]], PLACE_GRID)
        out[g] = {("weakest indicator", d): v for d, v in low.items()} | {("rank within goal", m): v for m, v in pos.items()}
    gmp = np.column_stack(gmp); app = ref["applicable"]
    idx = np.where(app.any(1), sdg_index(gmp, app & np.isfinite(gmp), ref["region"], ref["ranked"]), np.nan)
    pos = share_within_places(idx[ref["rank_sample"]], ref["index"][ref["rank_sample"]], PLACE_GRID)
    gmw = np.where(app, gmp, np.inf); weakest_goal = (gmw <= gmw.min(1, keepdims=True) + EPS) & app
    low = share_weakest_found(np.where(app, ref["goal_scores"], np.nan), weakest_goal, POINT_GRID, ref["reference"])
    out["index"] = {("weakest goal", d): v for d, v in low.items()} | {("overall rank", m): v for m, v in pos.items()}
    return out


CAL = []
for label in EDITIONS:
    goals, meta = EDITION[label]["goals"], EDITION[label]["meta"]
    order = sorted(goals); gm = np.column_stack([goal_score(goals[g]["Z"]) for g in order]); app = np.isfinite(gm)
    region = meta.region.astype(str).to_numpy(); ranked = meta.ranked.to_numpy(bool); index = sdg_index(gm, app, region, ranked)
    ref = dict(order=order, goal_scores=gm, applicable=app, region=region, ranked=ranked, index=index, rank_sample=ranked & np.isfinite(index),
               reference=app.sum(1) >= meta.n_goals_applicable.to_numpy() - 1)
    for g in order:
        s0 = goal_score(goals[g]["Z"]); ref[g] = dict(score=s0, has=np.isfinite(s0), ok=weakest_mask(goals[g]["Z"])[1])
    rng = np.random.default_rng(SEED)
    for kind, level in DISTURBANCES:
        draws = [kept_under_disturbance(goals, disturb(goals, kind, level, rng), ref) for _ in range(DRAWS)]
        for result, grid in RESULTS.items():
            keys = [k for k in draws[0] if (result, grid[0]) in draws[0][k]]
            for t in grid:
                vals = np.array([float(np.mean([dr[k][(result, t)] for dr in draws])) for k in keys])
                CAL.append(dict(edition=label, disturbance=kind, level=level, result=result, tolerance=t, share_kept=float(vals.mean()),
                                share_kept_lowest_goal=float(vals.min()) if len(vals) > 1 else np.nan))
    print(f"{label}: calibration done ({DRAWS} draws x {len(DISTURBANCES)} disturbances)", flush=True)
CAL = pd.DataFrame(CAL); save("tolerance_calibration.csv", CAL)


def choose(result, minimum=None):
    x = CAL[CAL.result.eq(result)]
    for t in RESULTS[result]:
        if minimum is not None and t < minimum: continue
        if (x[x.tolerance == t].share_kept >= SHARE_REQUIRED - 1e-12).all(): return t
    raise ValueError(f"no tolerance in the grid reaches {SHARE_REQUIRED} for {result}")


TOL = pd.DataFrame([dict(result="weakest indicator", unit="score points", tolerance=choose("weakest indicator")),
                    dict(result="rank within goal", unit="rank places", tolerance=choose("rank within goal", PLACE_MINIMUM)),
                    dict(result="weakest goal", unit="score points", tolerance=choose("weakest goal")),
                    dict(result="overall rank", unit="rank places", tolerance=choose("overall rank", PLACE_MINIMUM))])
smallest = {r: next((t for t in RESULTS[r] if (CAL[CAL.result.eq(r) & CAL.tolerance.eq(t)].share_kept >= SHARE_REQUIRED - 1e-12).all()), None) for r in RESULTS}
TOL["smallest_tolerance_meeting_the_rule"] = TOL.result.map(smallest)
save("tolerances.csv", TOL)
print("\nTolerances:")
for r in TOL.itertuples():
    x = CAL[CAL.result.eq(r.result) & CAL.tolerance.eq(r.tolerance)]
    print(f"  {r.result:18s} {r.tolerance:>2} {r.unit:12s} | share kept by the full set at this tolerance {x.share_kept.min():.3f}-{x.share_kept.max():.3f}"
          + (f" | the rule alone gives {r.smallest_tolerance_meeting_the_rule}" if r.smallest_tolerance_meeting_the_rule != r.tolerance else ""))
print("\ndone ->", RES)
