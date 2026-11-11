import re, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import geopandas as gpd
warnings.filterwarnings("ignore")

BASE = Path.cwd(); DATA = BASE / "data"
if not DATA.is_dir(): raise SystemExit(f"`data/` folder not found in {BASE}. Put this file next to the `data/` folder.")
R1, R2, R3, R4, R5 = (BASE / "result" / str(i) for i in range(1, 6)); RES = BASE / "result" / "6"; RES.mkdir(parents=True, exist_ok=True)
MAP_FILE = DATA / "ne_50m_admin_0_countries.geojson"
if not MAP_FILE.exists(): raise SystemExit(f"{MAP_FILE.name} not found in data/: download it from https://github.com/nvkelso/natural-earth-vector (folder geojson).")

EDITIONS = [2022, 2023, 2024, 2025, 2026]
EXAMPLE = "HND"
PRIORITY_CUTS = [40, 60, 75]
GOALS_SHOWN = 3
RHO_MIN = 0.95
MM = 1 / 25.4; FS, FS_SMALL = 10, 8.5
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS, "xtick.labelsize": FS - 0.5,
                     "ytick.labelsize": FS - 0.5, "legend.fontsize": FS - 0.5, "legend.title_fontsize": FS - 0.5, "legend.frameon": False, "axes.linewidth": 0.6,
                     "axes.edgecolor": "#444444", "axes.labelcolor": "#222222", "text.color": "#222222", "xtick.color": "#333333", "ytick.color": "#333333",
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
C_RANKING, C_RANK, C_WEAK, C_BOTH, C_MEDIAN = "#8a8a8a", "#009E73", "#CC79A7", "#2b2b2b", "#D0021B"
C_EDITION = dict(zip(EDITIONS, ["#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594"]))
GRID = dict(color="#d4d4d4", lw=0.6, ls=(0, (3, 3)))
SDG = {1: "No poverty", 2: "Zero hunger", 3: "Health", 4: "Education", 5: "Gender equality", 6: "Water and sanitation", 7: "Energy", 8: "Decent work",
       9: "Industry and infrastructure", 10: "Reduced inequalities", 11: "Cities", 12: "Consumption and production", 13: "Climate action", 14: "Life below water",
       15: "Life on land", 16: "Peace and institutions", 17: "Partnerships"}
REGION = {"Sub-Saharan Africa": "Sub-Saharan Africa", "E. Europe & C. Asia": "Eastern Europe and Central Asia", "MENA": "Middle East and North Africa",
          "LAC": "Latin America and the Caribbean", "OECD": "OECD members", "East & South Asia": "East and South Asia", "Oceania": "Oceania"}
INCOME = {"LIC+LMIC": "Low and lower-middle income", "UMIC": "Upper-middle income", "HIC": "High income"}
NAME = {"overall ranking": "overall ranking", "each country's rank": "individual rank stability", "weakest goal": "weakest goal diagnosis", "rank and weakest goal": "dual criteria",
        "ranking": "overall ranking", "weakest indicator": "weakest indicator diagnosis"}
TOLERANCE = {"weakest indicator": "weakest indicator diagnosis within goals", "rank within goal": "individual rank stability within goals", "weakest goal": "weakest goal diagnosis",
             "overall rank": "individual rank stability"}
INDICATOR = {"sdg1_wpc": "Poverty headcount ratio at the international poverty line", "sdg2_undernsh": "Prevalence of undernourishment", "sdg3_lifee": "Life expectancy at birth",
             "sdg4_second": "Lower secondary completion rate", "sdg5_parl": "Seats held by women in national parliament", "sdg6_sanita": "Population using at least basic sanitation services",
             "sdg7_elecac": "Population with access to electricity", "sdg8_accounts": "Adults with an account at a bank or other financial institution or with a mobile-money-service provider",
             "sdg9_intuse": "Population using the internet", "sdg10_gini": "Gini coefficient", "sdg11_pm25": "Annual mean concentration of PM2.5", "sdg12_nprod": "Production-based nitrogen emissions",
             "sdg13_co2gcp": "CO2 emissions from fossil fuel combustion and cement production", "sdg14_cpma": "Mean area that is protected in marine sites important to biodiversity",
             "sdg15_redlist": "Red List Index of species survival", "sdg16_cpi": "Corruption Perceptions Index", "sdg17_statperf": "Statistical Performance Index",
             "sdg11_transport": "Satisfaction with, or convenient access to, public transport (a)", "sdg8_rights": "Fundamental labor rights are effectively guaranteed",
             "sdg17_oda": "International concessional public finance, including official development assistance (high-income countries)",
             "sdg16_exprop": "Expropriations are lawful and adequately compensated", "sdg17_govrev": "Government revenue excluding grants (countries other than high-income)",
             "sdg15_cpfa": "Mean area that is protected in freshwater sites important to biodiversity", "sdg16_rsf": "Press Freedom Index", "sdg16_admin": "Timeliness of administrative proceedings",
             "sdg13_co2import": "CO2 emissions embodied in imports", "sdg4_earlyedu": "Participation rate in pre-primary organized learning", "sdg11_slums": "Proportion of urban population living in slums",
             "sdg12_ewaste": "Electronic waste", "sdg7_ren": "Share of renewable energy in total primary energy supply", "sdg13_ghgimport": "Greenhouse gas emissions embodied in imports",
             "sdg3_swb": "Subjective well-being", "sdg16_detain": "Unsentenced detainees", "sdg16_security": "Crime is effectively controlled (b)", "sdg16_justice": "Access to and affordability of justice",
             "sdg12_explastic": "Exports of plastic waste", "sdg16_safe": "Population who feel safe walking alone at night in the city or area where they live (b)",
             "sdg16_clabor": "Children involved in child labor", "sdg8_slavery": "Victims of modern slavery", "sdg5_familypl": "Demand for family planning satisfied by modern methods"}
TABLE5_NOTES = ["(a) Satisfaction with public transport until the 2023 edition; population with convenient access to public transport in cities from 2024.",
                "(b) In 2024 the report replaced 'Population who feel safe walking alone at night' with 'Crime is effectively controlled'."]
TABLES, NUM = {}, []


def read(folder, name):
    return pd.read_csv(folder / name, encoding="utf-8-sig")


def span(values, fmt="{:.0f}", scale=1.0):
    v = pd.Series(values).dropna() * scale
    if not len(v): return ""
    lo, hi = fmt.format(v.min()), fmt.format(v.max())
    return (lo if lo == hi else f"{lo}–{hi}").replace("-", "−")


def signed(x):
    t = f"{x:+.0f}"
    return "0" if t in ("+0", "-0") else t.replace("-", "−")


def signed_span(values):
    v = pd.Series(values).dropna()
    if not len(v): return ""
    lo, hi = signed(v.min()), signed(v.max())
    return lo if lo == hi else f"{lo} to {hi}"


def goal_label(v):
    return "SDG " + " and ".join(re.sub(r"\.0$", "", t) for t in str(v).split("|"))


def number(section, item, value, source):
    """Record one number quoted in the manuscript with its section and source file."""
    NUM.append(dict(section=section, item=item, value=value, source=source))


def table(key, frame, stem):
    """Store a table and write it to a CSV file."""
    TABLES[key] = frame; frame.to_csv(RES / f"{stem}.csv", index=False, encoding="utf-8-sig"); print("  wrote", f"{stem}.csv")


def save_figure(fig, stem, data=None):
    """Save a figure as PDF and 600-dpi PNG at its final size, with the plotted data as CSV."""
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})): fig.savefig(RES / f"{stem}.{ext}", **kw)
    plt.close(fig)
    if data is not None: data.to_csv(RES / f"{stem}_data.csv", index=False, encoding="utf-8-sig")
    print("  wrote", f"{stem}.pdf/.png")


def indicator_name(code):
    return INDICATOR.get(code) or LABEL.get(code, code)


TOL = read(R1, "tolerances.csv").set_index("result"); CALIB = read(R1, "tolerance_calibration.csv"); REPRO = read(R1, "index_reproduction.csv"); AUDIT = read(R1, "indicator_audit.csv")
LABEL = {}
for e in sorted(AUDIT.edition.unique()):
    a = AUDIT[(AUDIT.edition == e) & AUDIT.label.notna() & AUDIT.label.astype(str).ne("")]; LABEL.update(dict(zip(a.code, a.label.astype(str))))
POINTS_GOAL, PLACES_ALL = float(TOL.tolerance["weakest goal"]), int(TOL.tolerance["overall rank"])
POINTS_IND, PLACES_GOAL = float(TOL.tolerance["weakest indicator"]), int(TOL.tolerance["rank within goal"])
OUT = read(R2, "headline_country_outcomes.csv"); HE = read(R2, "headline_by_edition.csv"); NAMED = read(R2, "headline_named_vs_true_goal.csv"); LEVEL = read(R2, "headline_indicator_level.csv")
GROUP = read(R2, "headline_by_group.csv"); GROUP_E = read(R2, "headline_by_group_and_edition.csv"); TEST = read(R2, "headline_income_test.csv")
HL = HE[HE.version == "report's rule"].set_index("edition").reindex(EDITIONS); HC = HE[HE.version != "report's rule"].set_index("edition").reindex(EDITIONS)
SIZES = read(R3, "smallest_list_sizes.csv"); LISTS = read(R3, "smallest_lists.csv"); ONE = read(R3, "best_lists_by_size_one_result.csv"); BOTH = read(R3, "best_lists_by_size_both_results.csv")
NEED = read(R4, "within_goal_indicators_needed.csv"); VAR_E = read(R4, "variance_based_lists_by_edition.csv"); DEPEND = read(R4, "what_requirements_depend_on.csv")
PRED_E = read(R4, "prediction_by_edition.csv"); AVG_E = read(R4, "averaging_rules_by_edition.csv")
H_IND = read(R5, "hidden_by_indicator.csv"); H_CTRY = read(R5, "hidden_by_country.csv"); H_GROUP = read(R5, "hidden_by_group.csv"); H_ED = read(R5, "hidden_by_edition.csv"); H_CAL = read(R5, "calibration.csv")
scored = OUT[OUT.status.isin(["found", "missed"])].copy()
print("results read; writing to", RES)


ROBINSON = "+proj=robin +lon_0=0 +datum=WGS84 +units=m"; EQUAL_AREA = "EPSG:8857"


def load_world():
    """Load Natural Earth countries in the Robinson projection with ISO3 codes, names, label points and areas."""
    w = gpd.read_file(MAP_FILE)
    if w.crs is None: w = w.set_crs("EPSG:4326")
    col = {c.upper(): c for c in w.columns}
    pick = lambda *names: next((w[col[n]] for n in names if n in col), None)
    adm = pick("ADM0_A3").astype(str); iso = pick("ISO_A3_EH", "ISO_A3").astype(str)
    iso = iso.where(~iso.isin(["-99", "None", "nan", ""]), adm).replace({"SOL": "SOM", "CYN": "CYP"})
    w = w.assign(iso=iso.to_numpy(), name=pick("NAME_EN", "NAME_LONG", "NAME").astype(str).to_numpy(),
                 lx=pd.to_numeric(pick("LABEL_X"), errors="coerce").to_numpy(), ly=pd.to_numeric(pick("LABEL_Y"), errors="coerce").to_numpy())
    w = w[(adm != "ATA").to_numpy()][["iso", "name", "lx", "ly", "geometry"]].reset_index(drop=True)
    area = w.geometry.to_crs(EQUAL_AREA).area.to_numpy() / 1e6; w = w.to_crs(ROBINSON); ok = w.lx.notna() & w.ly.notna()
    lab = gpd.GeoSeries(gpd.points_from_xy(w.lx.fillna(0), w.ly.fillna(0)), crs="EPSG:4326").to_crs(ROBINSON); rep = w.geometry.representative_point()
    w["px"], w["py"] = np.where(ok, lab.x, rep.x), np.where(ok, lab.y, rep.y)
    w["area"] = area; w["small"] = w["area"] < 25000
    return w


def country_points(W):
    return W.sort_values("area", ascending=False).drop_duplicates("iso")[["iso", "name", "px", "py", "small"]]


def to_xy(lonlat):
    g = gpd.GeoSeries(gpd.points_from_xy([p[0] for p in lonlat], [p[1] for p in lonlat]), crs="EPSG:4326").to_crs(ROBINSON)
    return list(zip(g.x, g.y))


W = load_world(); X0, Y0, X1, Y1 = W.total_bounds
MAP_W = 184.0; UNIT = (X1 - X0) / MAP_W; MAP_H = (Y1 - Y0) / UNIT


def map_figure(band):
    """Create a figure 190 mm wide with the world map above a band for the legend."""
    fh = MAP_H + band + 1.0
    fig = plt.figure(figsize=(190 * MM, fh * MM)); ax = fig.add_axes([(190 - MAP_W) / 2 / 190, (band + 0.5) / fh, MAP_W / 190, MAP_H / fh])
    return fig, ax, fh


def frame(ax, x0=X0, y0=Y0, x1=X1, y1=Y1):
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.set_axis_off()


gap = scored.groupby("iso").gap_points.mean()
BINS = [(-0.01, POINTS_GOAL, f"≤{POINTS_GOAL:g} (weakest goal correctly identified)", "#9ecae1"), (POINTS_GOAL, 10, f"{POINTS_GOAL:g}–10", "#fdd0a2"), (10, 20, "10–20", "#fd8d3c"),
        (20, 30, "20–30", "#e6550d"), (30, 1e9, ">30", "#a63603")]
C_NA = "#bdbdbd"; cat = {}
for iso in OUT.iso.unique():
    cat[iso] = next((i for i, b in enumerate(BINS) if b[0] < gap[iso] <= b[1]), -1) if iso in gap.index else -1
color1 = {iso: (BINS[c][3] if c >= 0 else C_NA) for iso, c in cat.items()}
fig, ax, fh = map_figure(15.5)
W.plot(ax=ax, color=W.iso.map(color1).fillna("#ffffff").tolist(), edgecolor="#8c8c8c", linewidth=0.15)
s = country_points(W); s = s[s.small & s.iso.isin(list(color1))]
ax.scatter(s.px, s.py, s=5, c=s.iso.map(color1).tolist(), edgecolors="#4d4d4d", linewidths=0.3, zorder=3, clip_on=False); frame(ax)
h = country_points(W); h = h[h.iso == EXAMPLE]
if len(h):
    ax.annotate(h["name"].iloc[0], xy=(h.px.iloc[0], h.py.iloc[0]), xytext=(1, -27), textcoords="offset points", fontsize=FS_SMALL, ha="right", va="top",
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#111111", shrinkA=0.5, shrinkB=0, relpos=(0.86, 1.0)), zorder=6)
    ax.plot([h.px.iloc[0]], [h.py.iloc[0]], marker="o", ms=1.8, mfc="#111111", mec="none", zorder=7)
fig.legend(handles=[Patch(fc=b[3], ec="#8c8c8c", lw=0.3, label=b[2]) for b in BINS] + [Patch(fc=C_NA, ec="#8c8c8c", lw=0.3, label="Not assessed"), Patch(fc="#ffffff", ec="#8c8c8c", lw=0.3, label="Not in the report"),
                    Line2D([], [], ls="", marker="o", ms=3.4, mfc=BINS[2][3], mec="#4d4d4d", mew=0.3, label="Small country")],
           title="Gap between the named and the truly weakest goal (points, mean of 2022–2026)", ncol=8, loc="lower center", bbox_to_anchor=(0.5, 0.0), fontsize=FS_SMALL, title_fontsize=FS_SMALL,
           handlelength=1.1, handletextpad=0.4, columnspacing=0.9)
save_figure(fig, "Fig1_gap_between_named_and_true_weakest_goal",
            pd.DataFrame({"iso": list(cat), "class": [BINS[c][2] if c >= 0 else "not assessed" for c in cat.values()], "mean_gap_points": [gap.get(i, np.nan) for i in cat],
                          "editions_assessed": [int((scored.iso == i).sum()) for i in cat], "editions_misidentified": [int(((scored.iso == i) & (scored.status == "missed")).sum()) for i in cat]}))


FIX = ONE[ONE.fixed_size_search.notna()]
BOTH = BOTH[BOTH.found.astype(bool)].copy(); BOTH["dual"] = BOTH[["share_within_places", "share_weakest_goal_found"]].min(axis=1)


def across_editions(g):
    return pd.DataFrame({"median": g.median() * 100, "lowest": g.min() * 100, "highest": g.max() * 100})


CURVES = [("Individual rank stability", C_RANK, "o", -0.4, across_editions(FIX[FIX.result == "each country's rank"].groupby("size").best_value)),
          ("Weakest goal diagnosis", C_WEAK, "s", 0.0, across_editions(FIX[FIX.result == "weakest goal"].groupby("size").best_value)),
          ("Dual criteria", C_BOTH, "D", 0.4, across_editions(BOTH.groupby("size").dual))]
FH = 129.0; L, PW, GAP, B, PH = 14.5, 78.5, 15.5, 15.5, 104.5
fig = plt.figure(figsize=(190 * MM, FH * MM))
ax = fig.add_axes([L / 190, B / FH, PW / 190, PH / FH]); ax.grid(True, axis="both", **GRID)
ax.axhline(95, color="#333333", lw=1.4, ls=(0, (6, 3)), zorder=1.5); ax.text(15.9, 95.55, "95%", va="bottom", ha="left", fontsize=FS - 0.5, color="#333333", bbox=dict(fc="white", ec="none", pad=0.6))
for lab, c, mk, dx, v in CURVES:
    x = v.index.to_numpy() + dx
    ax.errorbar(x, v["median"], yerr=[v["median"] - v.lowest, v.highest - v["median"]], fmt="none", ecolor=c, elinewidth=1.0, capsize=2.2, capthick=1.0, zorder=2)
    ax.plot(x, v["median"], color=c, lw=1.7, marker=mk, ms=4.0 if mk == "D" else 4.6, mfc=c, mec="white", mew=0.7, zorder=3, label=lab, solid_capstyle="round")
ax.set_xlim(14.8, 62.2); ax.set_ylim(60, 100.8); ax.set_xticks([17, 20, 25, 30, 35, 40, 45, 50, 55, 60]); ax.set_xticklabels(["17", "20", "", "30", "", "40", "", "50", "", "60"])
ax.set_yticks(range(60, 101, 5)); ax.set_yticklabels(["60", "", "70", "", "80", "", "90", "", "100"])
ax.set_xlabel("Indicators in the list"); ax.set_ylabel("Countries meeting the criterion (%)")
ax.legend(loc="lower right", bbox_to_anchor=(0.995, 0.015), handlelength=2.0, borderaxespad=0.2, labelspacing=0.45, frameon=True, framealpha=1.0, facecolor="white", edgecolor="none")
fig.text(0.012, (B + PH) / FH + 0.012, "a", fontweight="bold", fontsize=12, va="bottom", ha="left")
ax = fig.add_axes([(L + PW + GAP) / 190, B / FH, PW / 190, PH / FH]); ax.grid(True, axis="x", **GRID)
ROWS = [("k_ranking", "Overall\nranking", C_RANKING), ("k_rank", "Individual\nrank stability", C_RANK), ("k_weakest", "Weakest indicator\ndiagnosis", C_WEAK), ("k_rank_and_weakest", "Dual\ncriteria", C_BOTH)]
ax.set_ylim(4.58, 0.42); ax.set_xlim(0, 103)
RING, HALF, STEP = 7.0, 0.45, 3.535 / (PH / 25.4 * 72 / 4.16)
for i, (col, lab, c) in enumerate(ROWS, start=1):
    v = pd.Series(np.round((NEED[col] / NEED.K * 100).to_numpy(), 2)); q1, q3 = np.percentile(v, [25, 75]); lo, hi = v[v >= q1 - 1.5 * (q3 - q1)].min(), v[v <= q3 + 1.5 * (q3 - q1)].max()
    ax.add_patch(Rectangle((q1, i - 0.25), q3 - q1, 0.5, fc=to_rgba(c, 0.32), ec=to_rgba("#555555", 0.32), lw=0.6, zorder=2))
    for x0, x1 in ((lo, q1), (q3, hi)): ax.plot([x0, x1], [i, i], color="black", lw=0.6, zorder=2, solid_capstyle="butt")
    for x0 in (lo, hi): ax.plot([x0, x0], [i - 0.125, i + 0.125], color="black", lw=0.6, zorder=2, solid_capstyle="butt")
    y = np.zeros(len(v))
    for _, idx in v.groupby(v).groups.items():
        k = len(idx); half = min(HALF, STEP * (k - 1) / 2); y[list(idx)] = np.linspace(-half, half, k) if k > 1 else 0.0
    ax.scatter(v, i + y, s=RING, facecolors="none", edgecolors="#333333", linewidths=0.55, zorder=4)
    ax.plot([v.median()] * 2, [i - 0.36, i + 0.36], color=C_MEDIAN, lw=1.8, zorder=5, solid_capstyle="butt")
    ax.text(-0.062, i, lab, transform=ax.get_yaxis_transform(), rotation=90, ha="center", va="center", multialignment="center", fontsize=FS - 0.5, linespacing=1.05, color="#333333")
ax.set_yticks([]); ax.set_xticks(range(0, 101, 10)); ax.set_xlabel("Indicators needed within a goal (%)")
fig.text((L + PW + 3.3) / 190, (B + PH) / FH + 0.012, "b", fontweight="bold", fontsize=12, va="bottom", ha="left")
save_figure(fig, "Fig2_indicators_needed",
            pd.concat([FIX[FIX.result == "each country's rank"].assign(panel="a", criterion="individual rank stability", countries_meeting_the_criterion=lambda d: d.best_value)[["panel", "criterion", "edition", "size", "countries_meeting_the_criterion"]],
                       FIX[FIX.result == "weakest goal"].assign(panel="a", criterion="weakest goal diagnosis", countries_meeting_the_criterion=lambda d: d.best_value)[["panel", "criterion", "edition", "size", "countries_meeting_the_criterion"]],
                       BOTH.assign(panel="a", criterion="dual criteria", countries_meeting_the_criterion=lambda d: d.dual)[["panel", "criterion", "edition", "size", "countries_meeting_the_criterion", "share_within_places", "share_weakest_goal_found"]],
                       NEED[["edition", "goal", "K", "k_ranking", "k_rank", "k_weakest", "k_rank_and_weakest"]].assign(panel="b")], ignore_index=True))


hb = H_CTRY.groupby("iso").agg(hidden=("expected_goals_hidden", "mean"), sids=("sids", "max"), small=("under_one_million", "max")).reset_index()
b = hb.merge(country_points(W)[["iso", "name", "px", "py"]], on="iso", how="inner").sort_values("hidden", ascending=False)
C_SMALL, C_LARGE, BUB = "#D55E00", "#0072B2", 26
BAND, INSET_H = 30.0, 24.5
fig, ax, fh = map_figure(BAND)


def bubbles(a, clip):
    W.plot(ax=a, color="#ececec", edgecolor="#bdbdbd", linewidth=0.15); sids = b.sids.astype(bool).to_numpy()
    a.scatter(b.px, b.py, s=b.hidden * BUB, facecolors=[to_rgba(C_SMALL if v else C_LARGE, 0.7) for v in b.small.astype(bool)], edgecolors=np.where(sids, "black", "white"),
              linewidths=np.where(sids, 0.5, 0.25), zorder=3, clip_on=clip)


def name(a, row, xytext, textcoords="offset points", ha="center", va="center", relpos=(0.5, 0.5)):
    a.annotate(row["name"], xy=(row.px, row.py), xytext=xytext, textcoords=textcoords, fontsize=FS_SMALL, ha=ha, va=va, zorder=6,
               arrowprops=dict(arrowstyle="-", lw=0.5, color="#222222", shrinkA=1.2, shrinkB=np.sqrt(row.hidden * BUB) / 2 + 0.3, relpos=relpos))


def inset(title, corners, left):
    """Draw an enlarged view of an area below the map and outline that area on the world map."""
    (x0, y0), (x1, y1) = to_xy(corners); w = INSET_H * (x1 - x0) / (y1 - y0)
    ins = fig.add_axes([left / 190, 1.0 / fh, w / 190, INSET_H / fh]); bubbles(ins, True); frame(ins, x0, y0, x1, y1); ins.set_axis_on(); ins.set_xticks([]); ins.set_yticks([])
    ins.set_xlabel(""); ins.set_ylabel(""); ins.set_facecolor("white"); ins.set_title(title, fontsize=FS_SMALL, pad=2.5, loc="left")
    for sp in ins.spines.values(): sp.set_visible(True); sp.set_linewidth(0.5); sp.set_color("#555555")
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=0.5, ec="#555555", zorder=5))
    return ins, left + w


bubbles(ax, False); frame(ax)
ON_MAP = {"PLW": dict(xytext=(12, 22)), "TUV": dict(xytext=(-1, 45), ha="right", va="bottom", relpos=(0.93, 0.0)), "KIR": dict(xytext=(30, -18))}
for _, r in b[b.sids.astype(bool)].head(3).iterrows(): name(ax, r, **ON_MAP.get(r.iso, dict(xytext=(0, 26))))
ins, right = inset("European microstates", [(-4.5, 39.7), (17.3, 49.5)], 3.0)
IN_VIEW = {"AND": dict(xytext=(0.158, 0.559), relpos=(0.7, 0.0)), "LIE": dict(xytext=(0.290, 0.906), relpos=(1.0, 0.5)),
           "MCO": dict(xytext=(0.473, 0.094), relpos=(0.75, 1.0)), "SMR": dict(xytext=(0.818, 0.094), relpos=(0.45, 1.0))}
for iso, kw in IN_VIEW.items():
    r = b[b.iso == iso]
    if len(r): name(ins, r.iloc[0], textcoords="axes fraction", **kw)
lg = fig.add_axes([0, 0, 1, BAND / fh]); lg.set_xlim(0, 190); lg.set_ylim(0, BAND); lg.set_axis_off()
width = lambda t: t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi * 25.4
KINDS = [(to_rgba(C_SMALL, 0.7), "white", 0.25, "Fewer than one million people"), (to_rgba(C_LARGE, 0.7), "white", 0.25, "One million people or more"),
         ("none", "black", 0.5, "Small island developing state")]
KEY = (1, 2, 4, 8); d = [np.sqrt(v * BUB) / 72 * 25.4 for v in KEY]
kinds = [lg.text(0, y, k[3], va="center_baseline", ha="left", fontsize=FS_SMALL) for y, k in zip((17.8, 13.3, 8.8), KINDS)]
key_title = lg.text(0, 15.9, "Goals with a hidden weakest indicator\n(expected number)", va="center", ha="left", fontsize=FS_SMALL, linespacing=1.25)
values = [lg.text(0, 8.0, str(v), va="center_baseline", ha="left", fontsize=FS_SMALL) for v in KEY]
w1 = 2.4 + max(width(t) for t in kinds); w2 = max(width(key_title), sum(d) + sum(0.8 + width(t) for t in values) + 3.0 * (len(KEY) - 1))
spare = (188 - right - w1 - w2) / 3; x1 = right + spare; x = x2 = x1 + w1 + spare
for t, (fc, ec, lw, _) in zip(kinds, KINDS): lg.scatter([x1 + 0.7], [t.get_position()[1]], s=14, facecolors=fc, edgecolors=ec, linewidths=lw); t.set_x(x1 + 2.4)
key_title.set_x(x2)
for v, di, t in zip(KEY, d, values):
    lg.scatter([x + di / 2], [8.0], s=v * BUB, facecolors="none", edgecolors="#555555", linewidths=0.5); t.set_x(x + di + 0.8); x += di + 0.8 + width(t) + 3.0
save_figure(fig, "Fig3_weaknesses_hidden_by_missing_data", hb.rename(columns={"hidden": "expected_goals_hidden_mean_of_editions", "small": "under_one_million"}))


print("tables")
table("Table1", pd.DataFrame([
    {"Measure": "Overall ranking: rank correlation with the full SDG Index (Spearman's ρ)", "Headline index (range across editions)": span(HL.rho, "{:.3f}")},
    {"Measure": f"Individual rank stability: countries within {PLACES_ALL} places of their full-index rank (%)", "Headline index (range across editions)": span(HL.share_within_places, scale=100)},
    {"Measure": f"Weakest goal diagnosis: countries whose weakest goal is correctly identified within {POINTS_GOAL:g} points (%)", "Headline index (range across editions)": span(HL.share_weakest_goal_found, scale=100)},
    {"Measure": "Gap between the named and the truly weakest goal when misidentified (points, mean)", "Headline index (range across editions)": span(HL.gap_when_missed_mean, "{:.1f}")}]), "Table1_headline_index")
by_goal = lambda column: NAMED.pivot(index="goal", columns="edition", values=column).reindex(index=range(1, 18), columns=EDITIONS)
net, true_n, named_n = by_goal("named_minus_true"), by_goal("truly_weakest"), by_goal("named_weakest")
mean_net = net.mean(axis=1); SHOWN = list(mean_net.nlargest(GOALS_SHOWN).index) + list(mean_net.nsmallest(GOALS_SHOWN).index[::-1])
table("Table2", pd.DataFrame([{"Goal": f"SDG {g} {SDG[g]}", "Truly weakest (countries)": span(true_n.loc[g]), "Named as weakest by the headline index (countries)": span(named_n.loc[g]),
                               "Named minus truly weakest, ΔN (countries)": signed_span(net.loc[g])} for g in SHOWN]), "Table2_goals_named_too_often_or_too_rarely")
t3 = []
for res, lab in [("overall ranking", f"Overall ranking (ρ ≥ {RHO_MIN:g})"), ("each country's rank", f"Individual rank stability (≥ 95% of countries within {PLACES_ALL} places)"),
                 ("weakest goal", f"Weakest goal diagnosis (weakest goal correctly identified within {POINTS_GOAL:g} points for ≥ 95% of countries)"),
                 ("rank and weakest goal", "Dual criteria (individual rank stability and weakest goal diagnosis)")]:
    x = SIZES[SIZES.result == res]
    t3.append({"Criterion the list must meet": lab, "Indicators in the smallest lists found": span(x.smallest_size), "Share of the edition's indicators (%)": span(x.share_of_indicators, scale=100)})
table("Table3", pd.DataFrame(t3), "Table3_smallest_lists")
t4 = []
for g in SHOWN:
    x = LEVEL[LEVEL.goal == g]
    t4.append({"Headline indicator": f"{indicator_name(x.code.iloc[0])} (SDG {g})", "Level bias: indicator score minus goal score (points)": signed_span(x.level_bias),
               "Named minus truly weakest, ΔN (countries)": signed_span(x.named_minus_true)})
table("Table4", pd.DataFrame(t4), "Table4_level_bias_of_headline_indicators")
avg = H_IND.groupby("code").expected_hidden_low_scores.sum() / H_IND.edition.nunique()
avg = avg[avg > 0].sort_values(ascending=False); cum = avg.cumsum() / avg.sum() * 100
ends = [int((cum < t).sum()) + 1 for t in PRIORITY_CUTS]; goal_of = H_IND.drop_duplicates("code").set_index("code").goal
t5 = []
for i, c in enumerate(avg.index[:ends[-1]]):
    t5.append({"Priority": next(k for k, e in enumerate(ends, start=1) if i < e), "Indicator": indicator_name(c), "Goal": f"SDG {goal_of[c]} {SDG[goal_of[c]]}",
               "Hidden low scores expected (missing values)": round(float(avg[c]), 1), "Cumulative share (%)": f"{cum.iloc[i]:.0f}" if (i + 1) in ends else ""})
rest = avg.iloc[ends[-1]:]
t5.append({"Priority": "Remaining", "Indicator": f"{len(rest)} other indicators", "Goal": "", "Hidden low scores expected (missing values)": f"{rest.mean():.1f} (mean)", "Cumulative share (%)": "100"})
table("Table5", pd.DataFrame(t5), "Table5_priority_indicators")
pd.DataFrame({"Indicator code": avg.index, "Indicator": [indicator_name(c) for c in avg.index], "Goal": [goal_of[c] for c in avg.index], "Hidden low scores expected per edition": avg.round(2).to_numpy(),
              "Cumulative share (%)": cum.round(1).to_numpy()}).to_csv(RES / "all_indicators_ranked_by_hidden_low_scores.csv", index=False, encoding="utf-8-sig")


print("supplementary")
plt.rcParams.update({"font.size": FS_SMALL, "axes.labelsize": FS_SMALL, "xtick.labelsize": FS_SMALL, "ytick.labelsize": FS_SMALL, "legend.fontsize": FS_SMALL})
FH = 123.0; L, PW, GAPX, B, PH, GAPY = 15.0, 82.0, 8.0, 23.0, 40.0, 13.0
fig = plt.figure(figsize=(190 * MM, FH * MM))
PANELS = [("weakest indicator", "Weakest indicator diagnosis (within goals)", "Score tolerance (points)"), ("rank within goal", "Individual rank stability (within goals)", "Rank tolerance (places)"),
          ("weakest goal", "Weakest goal diagnosis (across goals)", "Score tolerance (points)"), ("overall rank", "Individual rank stability (across goals)", "Rank tolerance (places)")]
low = np.floor(CALIB.share_kept.min() * 20) * 5
for k, ((res, heading, xl), l) in enumerate(zip(PANELS, "abcd")):
    row, col = divmod(k, 2)
    ax = fig.add_axes([(L + col * (PW + GAPX)) / 190, (B + (1 - row) * (PH + GAPY)) / FH, PW / 190, PH / FH]); ax.grid(True, axis="both", **GRID)
    for e in EDITIONS:
        for kind, ls in (("score noise", "-"), ("bound shift", (0, (4, 2)))):
            x = CALIB[(CALIB.edition == e) & (CALIB.result == res) & (CALIB.disturbance == kind)].sort_values("tolerance")
            ax.plot(x.tolerance, x.share_kept * 100, color=C_EDITION[e], ls=ls, lw=0.8)
    ax.axhline(95, color="#333333", lw=0.8, ls=(0, (1, 2))); ax.axvline(TOL.tolerance[res], color="black", lw=0.9)
    ax.set_ylim(low, 100.6); ax.set_yticks(np.arange(low, 100.1, 5))
    ax.text(0.0, 1.045, l, transform=ax.transAxes, fontweight="bold", fontsize=FS, va="baseline", ha="left"); ax.text(0.05, 1.045, heading, transform=ax.transAxes, va="baseline", ha="left")
    if col: ax.set_yticklabels([])
    else: ax.set_ylabel("Countries within tolerance (%)")
    if row: ax.set_xlabel(xl)
fig.legend(handles=[Line2D([], [], color=C_EDITION[e], lw=1.4, label=str(e)) for e in EDITIONS] + [Line2D([], [], color="black", ls="", marker="|", ms=8, mew=0.9, label="Adopted tolerance"),
                    Line2D([], [], color="#333333", ls=(0, (1, 2)), lw=0.8, label="95% of countries")], ncol=7, loc="lower center", bbox_to_anchor=(0.5, 6.5 / FH), handlelength=1.8, columnspacing=1.6)
fig.legend(handles=[Line2D([], [], color="#555555", ls="-", lw=0.8, label="Random noise in scores (SD = 2 points)"), Line2D([], [], color="#555555", ls=(0, (4, 2)), lw=0.8, label="Normalization bounds shifted (±10% of range)")],
           ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0 / FH), handlelength=1.8, columnspacing=1.6)
save_figure(fig, "SI_Fig_tolerance_calibration", CALIB)
fig = plt.figure(figsize=(90 * MM, 84 * MM)); ax = fig.add_axes([16.0 / 90, 13.5 / 84, 68.0 / 90, 68.0 / 84]); ax.grid(True, axis="both", **GRID)
for e in EDITIONS:
    x = H_CAL[(H_CAL.edition == e) & (H_CAL.group != "all")]; a = H_CAL[(H_CAL.edition == e) & (H_CAL.group == "all")].iloc[0]
    ax.plot(x.predicted * 100, x.observed * 100, marker="o", ms=3.2, mec="white", mew=0.5, lw=0.8, color=C_EDITION[e], label=f"{e} (AUC {a.auc:.2f})")
lim = np.ceil(max(H_CAL.predicted.max(), H_CAL.observed.max()) * 10 + 0.5) * 10
ax.plot([0, lim], [0, lim], color="#333333", lw=0.8, ls=(0, (6, 3)), label="Perfect calibration", zorder=1.5); ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
ax.set_xlabel("Predicted probability of a hidden weakest indicator (%)"); ax.set_ylabel("Observed frequency in the hold-out test (%)")
ax.legend(loc="upper left", handlelength=1.8, borderaxespad=0.3, labelspacing=0.35, frameon=True, framealpha=1.0, facecolor="white", edgecolor="none")
save_figure(fig, "SI_Fig_test_of_hidden_weakness_estimates", H_CAL)
table("SI_headline_indicators", pd.DataFrame([{"Goal": f"SDG {g} {SDG[g]}", "Headline indicator (2025 list)": indicator_name(x.code.iloc[0]), "Code": x.code.iloc[0],
                                                "Level bias: indicator score minus goal score, mean over countries (points)": signed_span(x.level_bias),
                                                "Countries in which the indicator scores below the goal score (%)": span(x.share_below_goal_score, scale=100)} for g, x in LEVEL.groupby("goal")]),
      "SI_Table_headline_indicators")
table("SI_headline_by_edition", pd.DataFrame([{"Edition": e, "Ranked countries": int(HL.ranked_countries[e]), "Spearman's ρ with the full SDG Index": round(HL.rho[e], 3),
                                                f"Countries within {PLACES_ALL} places (%)": round(HL.share_within_places[e] * 100), "Reference countries": int(HL.reference_countries[e]),
                                                "Reference countries the headline index can assess (%)": round(HL.share_reference_scored[e] * 100),
                                                f"Weakest goal correctly identified within {POINTS_GOAL:g} points (%)": round(HL.share_weakest_goal_found[e] * 100),
                                                "Gap when the weakest goal is misidentified (points, mean)": round(HL.gap_when_missed_mean[e], 1),
                                                "Countries with data on all 17 headline indicators": int(HC.reference_scored[e]),
                                                "Weakest goal correctly identified among them (%)": round(HC.share_weakest_goal_found[e] * 100)} for e in EDITIONS]), "SI_Table_headline_index_by_edition")
rows = []
for gt, names in [("income group", INCOME), ("SDR region", REGION)]:
    for g in (list(names) if gt == "income group" else sorted(GROUP[GROUP.group_type == gt].group, key=lambda v: GROUP[(GROUP.group_type == gt) & (GROUP.group == v)].found.iloc[0])):
        p = GROUP[(GROUP.group_type == gt) & (GROUP.group == g)].iloc[0]; x = GROUP_E[(GROUP_E.group_type == gt) & (GROUP_E.group == g)]
        rows.append({"Group": names.get(g, g), "Countries assessed per edition": span(x.scored), "Weakest goal correctly identified, five editions pooled (%)": round(p.found * 100),
                     "Weakest goal correctly identified, range across editions (%)": span(x.found, scale=100)})
table("SI_groups", pd.DataFrame(rows), "SI_Table_income_groups_and_regions")
RULE = [("p=1", "Arithmetic mean"), ("p=0.5", "Power mean, p = 0.5"), ("p=0", "Geometric mean"), ("p=-1", "Harmonic mean"), ("minimum", "Minimum")]
table("SI_averaging", pd.DataFrame([{"Rule for averaging indicators within a goal": lab, "Indicators needed for the overall ranking, summed over goals": span(AVG_E[AVG_E.rule == r].k_ranking),
                                     "Countries whose weakest indicator is correctly identified by the smallest lists for the overall ranking (%, mean over goals)": span(AVG_E[AVG_E.rule == r].ranking_list_weakest_found, scale=100)}
                                    for r, lab in RULE]), "SI_Table_averaging_rules")
with pd.ExcelWriter(RES / "tables.xlsx") as xw:
    for k, t in TABLES.items(): t.to_excel(xw, sheet_name=k[:31], index=False)
    pd.DataFrame({"Notes to Table 5": TABLE5_NOTES}).to_excel(xw, sheet_name="Table5_notes", index=False)
print("  wrote tables.xlsx")


S = "2.1 Data"; src = "result/1/indicator_audit.csv, index_reproduction.csv"
number(S, "indicators per edition", span(AUDIT[AUDIT.included.astype(bool)].groupby("edition").size()), src)
number(S, "countries in the data", span(REPRO.groupby("edition").size()), src)
number(S, "ranked countries per edition", span(REPRO[REPRO.official_score.notna()].groupby("edition").size()), src)
x = REPRO[REPRO.official_score.notna()]
number(S, "largest difference between the rebuilt and the official SDG Index (points)", f"{(x.index_rebuilt - x.official_score).abs().max():.4f}", src)
number(S, "ranked countries whose rebuilt rank differs from the official rank (all editions)", int((x.rank_rebuilt != x.official_rank).sum()), src)
S = "2.2 Tolerances"; src = "result/1/tolerances.csv, tolerance_calibration.csv"
for r in TOL.itertuples():
    c = CALIB[(CALIB.result == r.Index) & (CALIB.tolerance == r.tolerance)]; lab = TOLERANCE[r.Index]
    number(S, f"tolerance for {lab} ({r.unit})", f"{r.tolerance:g}", src); number(S, f"share of countries within that tolerance when the full indicator set is disturbed, {lab} (%)", span(c.share_kept, "{:.1f}", 100), src)
    if r.smallest_tolerance_meeting_the_rule != r.tolerance: number(S, f"smallest tolerance tested at which the full indicator set already holds 95% of countries, {lab}", f"{r.smallest_tolerance_meeting_the_rule:g}", src)
S = "2.3 Search"; src = "result/4/within_goal_indicators_needed.csv, result/3/smallest_list_sizes.csv"
space = NEED.groupby("edition").K.apply(lambda k: float(np.log10(2.0 ** k.to_numpy(float) - 1).sum()))
number(S, "possible lists with at least one indicator per goal (power of ten)", span(space, "{:.0f}"), src)
number(S, "random starts of the first search, per edition and criterion", int(SIZES.starts.iloc[0]), src)
number(S, "largest goal: number of indicators / subsets evaluated within the goal", f"{int(NEED.K.max())} / {2 ** int(NEED.K.max()) - 1}", src)

S = "3.1 Headline index"; src = "result/2/headline_by_edition.csv"
number(S, "overall ranking: Spearman's ρ between headline index and full index", span(HL.rho, "{:.3f}"), src)
number(S, f"individual rank stability: countries within {PLACES_ALL} places of their full-index rank (%)", span(HL.share_within_places, scale=100), src)
number(S, f"countries that move by more than {PLACES_ALL} places (%)", span(1 - HL.share_within_places, scale=100), src)
number(S, "weakest goal diagnosis: countries whose weakest goal is correctly identified (%)", span(HL.share_weakest_goal_found, scale=100), src)
number(S, "countries whose weakest goal is misidentified (%)", span(1 - HL.share_weakest_goal_found, scale=100), src)
number(S, "gap between the named and the truly weakest goal when misidentified (points, mean)", span(HL.gap_when_missed_mean, "{:.1f}"), src)
number(S, "the same gap, whole points", span(HL.gap_when_missed_mean), src)
number(S, f"the same gap as a multiple of the {POINTS_GOAL:g}-point tolerance", span(HL.gap_when_missed_mean / POINTS_GOAL, "{:.1f}"), src)
number(S, "reference countries the headline index can assess (%)", span(HL.share_reference_scored, scale=100), src)
number(S, "ranked countries the headline index can score (%)", span(HL.share_ranked_scored, scale=100), src)
number(S, "countries with data on all 17 headline indicators: weakest goal correctly identified (%)", span(HC.share_weakest_goal_found, scale=100), src)
number(S, "countries with data on all 17 headline indicators (number)", span(HC.reference_scored), src)
src = "result/2/headline_named_vs_true_goal.csv"
for g in mean_net.sort_values(ascending=False).index:
    number(S, f"SDG {g} {SDG[g]}: truly weakest / named as weakest by the headline index / named minus truly weakest, ΔN (countries)", f"{span(true_n.loc[g])} / {span(named_n.loc[g])} / {signed_span(net.loc[g])}", src)
number(S, "goals shown in Tables 2 and 4", ", ".join(f"SDG {g}" for g in SHOWN), src)
src = "result/2/headline_country_outcomes.csv"; ex = scored[scored.iso == EXAMPLE]
if len(ex):
    number(S, f"{EXAMPLE}: editions assessed / within {PLACES_ALL} places / weakest goal misidentified", f"{len(ex)} / {int(ex.within_places.eq(1).sum())} / {int(ex.status.eq('missed').sum())}", src)
    number(S, f"{EXAMPLE}: goal named as weakest (by edition)", ", ".join(f"{int(e)}: {goal_label(n)}" for e, n in zip(ex.edition, ex.named_weakest)), src)
    number(S, f"{EXAMPLE}: truly weakest goal (by edition)", ", ".join(f"{int(e)}: {goal_label(n)}" for e, n in zip(ex.edition, ex.true_weakest)), src)
    number(S, f"{EXAMPLE}: points by which the named goal scores above the truly weakest goal", span(ex.gap_points), src)
per = scored.groupby("iso").agg(n=("status", "size"), missed=("status", lambda s: int((s == "missed").sum())), kept=("within_places", lambda s: int((s == 1).sum())))
five = per[per.n == len(EDITIONS)]; every = five[five.missed == len(EDITIONS)]; always = every[every.kept == len(EDITIONS)]
number(S, "countries assessed in all five editions", len(five), src)
number(S, "of them: weakest goal misidentified in every edition", len(every), src)
number(S, "of them: weakest goal correctly identified in every edition", int((five.missed == 0).sum()), src)
number(S, f"countries whose weakest goal is misidentified in every edition while their rank stays within {PLACES_ALL} places in every edition", f"{len(always)} ({' '.join(sorted(always.index))})", src)
number(S, "Fig. 1: countries assessed in at least one edition", len(gap), src)
number(S, f"Fig. 1: countries with a mean gap within {POINTS_GOAL:g} points / above {POINTS_GOAL:g} / above 10 / above 20 / above 30 points", " / ".join(str(int(v)) for v in [(gap <= POINTS_GOAL).sum()] + [(gap > t).sum() for t in (POINTS_GOAL, 10, 20, 30)]), src)
src = "result/2/headline_by_group_and_edition.csv, headline_by_group.csv, headline_income_test.csv"
for g, lab in INCOME.items():
    x = GROUP_E[(GROUP_E.group_type == "income group") & (GROUP_E.group == g)]
    number(S, f"{lab}: weakest goal correctly identified (%)", span(x.found, scale=100), src); number(S, f"{lab}: within {PLACES_ALL} places but weakest goal misidentified (%)", span(x.rank_kept_goal_missed, scale=100), src)
number(S, "high-income countries against all other countries, editions pooled: weakest goal correctly identified (%)", f"{TEST.found_high_income.iloc[0] * 100:.0f} vs {TEST.found_other.iloc[0] * 100:.0f}", src)
number(S, "permutation test, high-income against all other countries (P)", f"{TEST.p_value.iloc[0]:.3f}", src)
for r in GROUP[GROUP.group_type == "SDR region"].sort_values("found").itertuples(): number(S, f"{REGION.get(r.group, r.group)}: weakest goal correctly identified, editions pooled (%)", f"{r.found * 100:.0f}", src)

S = "3.2 Indicators needed"; src = "result/3/smallest_list_sizes.csv"
for res in ["overall ranking", "each country's rank", "weakest goal", "rank and weakest goal"]:
    x = SIZES[SIZES.result == res]
    number(S, f"smallest lists found, {NAME[res]} (indicators)", span(x.smallest_size), src); number(S, f"smallest lists found, {NAME[res]} (% of the edition's indicators)", span(x.share_of_indicators, scale=100), src)
x = SIZES[SIZES.result != "overall ranking"]
number(S, f"country-level criteria: starts of the first {int(x.starts.iloc[0])} that reached the smallest size of the first search", span(x.starts_reaching_it), src)
number(S, "country-level criteria: indicators by which the further search lowered the sizes", span(x.size_first_search - x.smallest_size), src)
number(S, "country-level criteria: sizes after the first search only (individual rank stability / weakest goal diagnosis / dual criteria)",
       " / ".join(span(SIZES[SIZES.result == r].size_first_search) for r in ["each country's rank", "weakest goal", "rank and weakest goal"]), src)
src = "result/3/smallest_lists.csv"; r = LISTS[LISTS.result == "overall ranking"]
number(S, "distinct smallest lists found for the overall ranking (all editions)", len(r), src); number(S, "the same, per edition", span(r.groupby("edition").size()), src)
number(S, f"those lists: countries within {PLACES_ALL} places (%)", span(r.share_within_places, scale=100), src)
number(S, "those lists: weakest goal correctly identified (%)", span(r.share_weakest_goal_found, scale=100), src)
number(S, "those lists: weakest goal correctly identified, median per edition (%)", span(r.groupby("edition").share_weakest_goal_found.median(), scale=100), src)
number(S, "those lists that reach 95% on either country-level criterion", int(((r.share_within_places >= 0.95) | (r.share_weakest_goal_found >= 0.95)).sum()), src)
for res in ["weakest goal", "each country's rank", "rank and weakest goal"]:
    x = LISTS[LISTS.result == res]; ok = x[x.rho >= RHO_MIN]
    number(S, f"distinct smallest lists found for {NAME[res]}: all / of which meet the overall ranking criterion (ρ ≥ {RHO_MIN:g}) / editions with such a list", f"{len(x)} / {len(ok)} / {ok.edition.nunique()}", src)
    number(S, f"the same, by edition ({NAME[res]})", ", ".join(f"{e}: {int((ok.edition == e).sum())} of {int((x.edition == e).sum())}" for e in EDITIONS), src)
src = "result/3/best_lists_by_size_one_result.csv"
for res in ["each country's rank", "weakest goal"]:
    for k, x in FIX[FIX.result == res].groupby("size"):
        number(S, f"Fig. 2a: best list found of {k} indicators for {NAME[res]}: countries meeting the criterion (%), range; median", f"{span(x.best_value, scale=100)}; {x.best_value.median() * 100:.0f}", src)
src = "result/3/best_lists_by_size_both_results.csv"
for k, x in BOTH.groupby("size"):
    number(S, f"one list of {k} indicators chosen for the dual criteria: within {PLACES_ALL} places / weakest goal correctly identified (%)", f"{span(x.share_within_places, scale=100)} / {span(x.share_weakest_goal_found, scale=100)}", src)
    number(S, f"Fig. 2a: the same list of {k} indicators: lower of the two shares (%), range; median", f"{span(x.dual, scale=100)}; {x.dual.median() * 100:.0f}", src)
src = "result/4/within_goal_indicators_needed.csv"; tot = NEED.groupby("edition").K.sum()
WITHIN = [("k_ranking", "overall ranking"), ("k_weakest", "weakest indicator diagnosis"), ("k_rank", "individual rank stability"), ("k_rank_and_weakest", "dual criteria (individual rank stability and weakest indicator diagnosis)"),
          ("k_ranking_and_weakest", "overall ranking and weakest indicator diagnosis together")]
for col, lab in WITHIN: number(S, f"within goals, summed over goals: indicators needed for {lab} (% of all indicators)", span(NEED.groupby("edition")[col].sum() / tot, scale=100), src)
a, b_, c = int(NEED.k_ranking.sum()), int(NEED.k_weakest.sum()), int(NEED.k_ranking_and_weakest.sum())
number(S, "within goals, all goals and editions: selections for overall ranking / weakest indicator diagnosis / both together", f"{a} / {b_} / {c}", src)
number(S, "both together, compared with weakest indicator diagnosis alone / with overall ranking alone (% more)", f"{(c / b_ - 1) * 100:.0f} / {(c / a - 1) * 100:.0f}", src)
number(S, f"goal and edition pairs (of {len(NEED)}) in which weakest indicator diagnosis needs more indicators than the overall ranking", int((NEED.k_weakest > NEED.k_ranking).sum()), src)
number(S, "pairs in which weakest indicator diagnosis needs more indicators than individual rank stability", int((NEED.k_weakest > NEED.k_rank).sum()), src)
for col, lab in WITHIN[:4]:
    number(S, f"Fig. 2b: share of a goal's indicators needed for {lab}, median of the {len(NEED)} pairs (%) / pairs that need every indicator of the goal", f"{(NEED[col] / NEED.K).median() * 100:.0f} / {int((NEED[col] == NEED.K).sum())}", src)
src = "result/4/variance_based_lists_by_edition.csv"
number(S, "lists that explain 90% of the variance: weakest indicator correctly identified (%)", span(VAR_E.weakest_found, scale=100), src)
number(S, f"lists that explain 90% of the variance: countries within {PLACES_GOAL} places (%)", span(VAR_E.within_places, scale=100), src)
number(S, "lists that explain 90% of the variance: goals evaluated per edition (of 17)", span(VAR_E.goals_evaluated), src)
number(S, "lists that explain 90% of the variance: share of the goal's indicators kept (%)", span(VAR_E.share_of_indicators, scale=100), src)
number(S, "lists that explain 90% of the variance: Spearman's ρ (mean over goals)", span(VAR_E.rho, "{:.3f}"), src)

S = "3.3 Mechanisms"; src = "result/4/what_requirements_depend_on.csv"
for r in DEPEND.itertuples(): number(S, f"partial correlation, {r.property} and indicators needed for {NAME[r.result]} ({r.pairs} pairs)", f"{r.partial_spearman:.2f}".replace("-", "−"), src)
src = "result/4/prediction_by_edition.csv"
number(S, "weakest indicator correctly identified before / after predicting the dropped indicators (%)", f"{span(PRED_E.weakest_found_before, scale=100)} / {span(PRED_E.weakest_found_after, scale=100)}", src)
number(S, "countries whose weakest indicator is still misidentified after prediction (%)", span(1 - PRED_E.weakest_found_after, scale=100), src)
number(S, "misidentified cases of the goal's most common weakest indicator recovered by prediction (%)", span(PRED_E.recovered_shared, scale=100), src)
number(S, "country-specific misidentified cases recovered by prediction (%)", span(PRED_E.recovered_specific, scale=100), src)
number(S, "country-specific cases as a share of all misidentified cases (%)", span(PRED_E.specific_share_of_missed, scale=100), src)
number(S, "goals per edition whose smallest list for the overall ranking drops at least one indicator", span(PRED_E.goals), src)
src = "result/4/averaging_rules_by_edition.csv"
for r, lab in RULE:
    x = AVG_E[AVG_E.rule == r]
    number(S, f"{lab}: weakest indicator correctly identified by the smallest lists for the overall ranking (%) / indicators needed for the overall ranking", f"{span(x.ranking_list_weakest_found, scale=100)} / {span(x.k_ranking)}", src)
src = "result/2/headline_indicator_level.csv"
rho = [pd.Series(x.level_bias.rank()).corr(pd.Series(x.named_minus_true.rank())) for _, x in LEVEL.groupby("edition")]
number(S, "Spearman's ρ between level bias and net over-reporting across the 17 goals", f"{min(rho):.2f} to {max(rho):.2f}".replace("-", "−"), src)
for g, x in LEVEL.groupby("goal"):
    number(S, f"SDG {g} headline indicator ({x.code.iloc[0]}): level bias (points) / countries scoring below the goal (%)", f"{signed_span(x.level_bias)} / {span(x.share_below_goal_score, scale=100)}", src)

S = "3.4 Missing data"; src = "result/5/hidden_by_edition.csv"
number(S, "missing applicable values per edition", span(H_ED.missing_values), src)
number(S, "country and goal pairs expected to have a hidden weakest indicator", span(H_ED.expected_pairs_hidden), src)
number(S, "the same as a share of all country and goal pairs with data (%)", span(H_ED.share_of_pairs_hidden, scale=100), src)
number(S, "hidden low scores expected, summed over missing values (per edition; mean)", f"{span(H_ED.expected_hidden_low_scores)}; {H_ED.expected_hidden_low_scores.mean():.0f}", src)
src = "result/5/hidden_by_group.csv, hidden_by_country.csv"
for g, x in H_GROUP.groupby("group", sort=False):
    number(S, f"{g}: goals with a hidden weakest indicator, mean per country", span(x.mean_expected_goals_hidden, "{:.1f}"), src); number(S, f"{g}: countries", span(x.countries), src)
x = H_GROUP[H_GROUP.group == "small island developing states"].mean_expected_goals_hidden
number(S, "small island developing states: the same as a share of the 17 goals (%)", span(x / 17, scale=100), src)
one = H_CTRY.drop_duplicates("iso")
number(S, "countries with fewer than one million people / of which small island developing states", f"{int(one.under_one_million.sum())} / {int((one.under_one_million & one.sids).sum())}", src)
number(S, "Fig. 3: countries with the most goals with a hidden weakest indicator (expected number, mean of editions)", "; ".join(f"{r['name']} {r.hidden:.1f}" for _, r in b.head(6).iterrows()), src)
src = "result/5/hidden_by_indicator.csv"
number(S, "indicators that close priorities 1, 2 and 3 (cumulative number)", " / ".join(str(e) for e in ends), src)
number(S, "share of the hidden low scores they account for (%)", " / ".join(f"{cum.iloc[e - 1]:.0f}" for e in ends), src)
number(S, "first four indicators of the list", "; ".join(indicator_name(c) for c in avg.index[:4]), src)
number(S, "remaining indicators: number / hidden low scores each (range; mean)", f"{len(rest)} / {rest.min():.1f}–{rest.max():.1f}; {rest.mean():.1f}", src)
src = "result/5/calibration.csv"; a = H_CAL[H_CAL.group == "all"]
number(S, "test of the estimates: area under the ROC curve", span(a.auc, "{:.2f}"), src); number(S, "test of the estimates: Brier skill score", span(a.brier_skill, "{:.2f}"), src)
number(S, "test of the estimates: mean predicted probability / share observed (%)", f"{span(a.predicted, '{:.1f}', 100)} / {span(a.observed, '{:.1f}', 100)}", src)

NUM = pd.DataFrame(NUM); NUM.to_csv(RES / "numbers_in_text.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 110); pd.set_option("display.max_rows", 400)
print(NUM[["section", "item", "value"]].to_string(index=False))
print("\ndone ->", RES)
