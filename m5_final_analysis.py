"""
M5 final research project: global LightGBM vs local benchmarks for daily
product-store sales.

RQ1  At the product-store level, is one global LightGBM model more accurate
     than the best local benchmark (seasonal naive or Croston)?
RQ2  Does LightGBM's relative accuracy differ across the four demand patterns
     (smooth, erratic, intermittent, lumpy; Syntetos et al., 2005)?
Supporting analysis: how LightGBM's advantage changes with the level of
aggregation (product -> department -> store).

Design (extends the Assignment 2 design without changing it):
  * 3 stores, one per state: TX_1, CA_1, WI_1
  * 3 forecast origins; each test window is 28 days and is preceded by its own
    28-day validation window (rolling origin)
  * 3 models: seasonal naive, Croston, global LightGBM (one model per store)
  * LightGBM's number of trees and the best benchmark are fixed on the
    validation window; each test window is then scored once

Usage:
  python m5_final_analysis.py --data-dir path/to/m5 --out-dir results
The data folder needs calendar.csv, sales_train_evaluation.csv and
sell_prices.csv from the M5 Forecasting - Accuracy competition (Kaggle).
Runtime: about 11 minutes on a 2-core machine with 8 GB of memory.
"""
import argparse
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats

H = 28                                   # forecast horizon (days)
STORES = ["TX_1", "CA_1", "WI_1"]
TEST_STARTS = [1858, 1886, 1914]         # first day of each test window
MAIN_TEST_START = 1914                   # the Assignment 2 test window
CLASSIFY_BEFORE = min(TEST_STARTS)       # demand patterns use data before ALL test windows
MIN_HISTORY = 365                        # >= 1 year of sales history before validation
LGB_TRAIN_DAYS = 730                     # LightGBM trains on the last 2 years
CROSTON_ALPHA = 0.1                      # smoothing constant of the M5 Croston benchmark
SEED = 42
LAGS = [28, 35, 42, 49, 56]
ORDER = ["Smooth", "Erratic", "Intermittent", "Lumpy"]

FEATURES = ([f"lag_{L}" for L in LAGS]
            + ["rmean7_lag28", "rmean28_lag28", "rmean56_lag28",
               "wday", "month", "mday", "event", "snap", "sell_price", "price_rel",
               "item_id", "dept_id", "cat_id"])

LGB_PARAMS = dict(objective="tweedie", tweedie_variance_power=1.1, learning_rate=0.05,
                  num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, seed=SEED, verbose=-1,
                  deterministic=True, num_threads=os.cpu_count() or 2)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_all(data_dir: Path):
    cal = pd.read_csv(data_dir / "calendar.csv", parse_dates=["date"])
    if "d" not in cal.columns:            # some mirrors drop the 'd' column
        cal["d"] = [f"d_{i}" for i in range(1, len(cal) + 1)]
    cal["day"] = cal["d"].str[2:].astype(int)
    sales = pd.read_csv(data_dir / "sales_train_evaluation.csv")
    prices = pd.read_csv(data_dir / "sell_prices.csv")
    return cal, sales, prices


def store_data(sales, prices, store):
    s = sales[sales["store_id"] == store].reset_index(drop=True)
    day_cols = [c for c in s.columns if c.startswith("d_")]
    y = s[day_cols].to_numpy(dtype=np.float32)            # products x days
    info = s[["item_id", "dept_id", "cat_id", "store_id", "state_id"]].copy()
    return y, info, prices[prices["store_id"] == store]


def first_sale_index(y):
    """0-based index of each series' first non-zero sale (n_days if none)."""
    nz = y > 0
    first = nz.argmax(axis=1)
    first[~nz.any(axis=1)] = y.shape[1]
    return first


# ----------------------------------------------------------------------------
# Accuracy measure and demand classification
# ----------------------------------------------------------------------------
def rmsse(hist, actual, forecast):
    """Root mean squared scaled error (M5). hist starts at the first sale."""
    denom = np.mean(np.diff(hist) ** 2)
    if denom == 0:
        return np.nan
    return float(np.sqrt(np.mean((actual - forecast) ** 2) / denom))


def classify(hist):
    """Syntetos, Boylan & Croston (2005) classes from ADI and CV^2."""
    nonzero = hist[hist > 0]
    adi = len(hist) / len(nonzero)
    cv2 = (nonzero.std() / nonzero.mean()) ** 2 if len(nonzero) > 1 else 0.0
    if adi < 1.32:
        label = "Smooth" if cv2 < 0.49 else "Erratic"
    else:
        label = "Intermittent" if cv2 < 0.49 else "Lumpy"
    return adi, cv2, label


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def seasonal_naive(hist):
    """Repeat the last observed week."""
    return np.tile(hist[-7:], H // 7)


def croston(hist, alpha=CROSTON_ALPHA):
    """Classic Croston: smooth non-zero sizes and intervals separately."""
    idx = np.flatnonzero(hist > 0)
    if len(idx) == 0:
        return np.zeros(H)
    sizes = hist[idx]
    intervals = np.diff(np.concatenate([[-1], idx]))
    z, p = sizes[0], intervals[0]
    for s, q in zip(sizes[1:], intervals[1:]):
        z = alpha * s + (1 - alpha) * z
        p = alpha * q + (1 - alpha) * p
    return np.full(H, z / p)


def build_panel(y, info, cal, prices, state):
    """Long table: one row per product-day with lag, calendar and price features."""
    n_series, n_days = y.shape
    days = np.arange(1, n_days + 1)
    yf = pd.DataFrame(y)
    feats = {f"lag_{L}": yf.shift(L, axis=1).to_numpy() for L in LAGS}
    lag28 = yf.shift(28, axis=1)
    for w in (7, 28, 56):
        feats[f"rmean{w}_lag28"] = lag28.T.rolling(w, min_periods=1).mean().T.to_numpy()

    panel = pd.DataFrame({"series": np.repeat(np.arange(n_series), n_days),
                          "day": np.tile(days, n_series),
                          "sales": y.ravel()})
    for k, v in feats.items():
        panel[k] = v.ravel().astype(np.float32)

    c = cal.set_index("day")
    panel["wday"] = panel["day"].map(c["wday"]).astype("int8")
    panel["month"] = panel["day"].map(c["month"]).astype("int8")
    panel["mday"] = panel["day"].map(c["date"].dt.day).astype("int8")
    panel["event"] = panel["day"].map(c["event_name_1"].notna()).astype("int8")
    panel["snap"] = panel["day"].map(c[f"snap_{state}"]).astype("int8")
    panel["wm_yr_wk"] = panel["day"].map(c["wm_yr_wk"])
    panel["item_id"] = panel["series"].map(info["item_id"])
    pr = prices[["item_id", "wm_yr_wk", "sell_price"]]
    panel = panel.merge(pr, on=["item_id", "wm_yr_wk"], how="left")
    item_mean = pr.groupby("item_id")["sell_price"].mean()
    panel["price_rel"] = panel["sell_price"] / panel["item_id"].map(item_mean)
    for col in ("item_id", "dept_id", "cat_id"):
        panel[col] = pd.Categorical(panel["series"].map(info[col]))
    return panel


def lgb_forecast(panel, origin, first_sale, n_series, num_rounds=None):
    """Train on the LGB_TRAIN_DAYS before `origin`; forecast days origin..origin+27.
    All sales features are lagged >= 28 days, so every input is known at the origin."""
    lo = origin - LGB_TRAIN_DAYS
    fs_day = panel["series"].map(pd.Series(first_sale + 1))
    train = panel[(panel["day"] >= lo) & (panel["day"] < origin)
                  & (panel["day"] >= fs_day) & panel["sell_price"].notna()]
    fut = panel[(panel["day"] >= origin) & (panel["day"] < origin + H)]
    dtrain = lgb.Dataset(train[FEATURES], train["sales"])
    if num_rounds is None:      # validation run: early stopping on this window
        dvalid = lgb.Dataset(fut[FEATURES], fut["sales"], reference=dtrain)
        model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=2000, valid_sets=[dvalid],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=num_rounds)
    pred = model.predict(fut[FEATURES], num_iteration=model.best_iteration or None)
    out = np.zeros((n_series, H))
    out[fut["series"].to_numpy(), fut["day"].to_numpy() - origin] = np.clip(pred, 0, None)
    return out, model, len(train)


# ----------------------------------------------------------------------------
# One store, one forecast origin
# ----------------------------------------------------------------------------
def local_forecasts(y, keep, first, origin):
    sn = np.zeros((y.shape[0], H))
    cr = np.zeros((y.shape[0], H))
    for i in keep:
        hist = y[i, first[i]:origin - 1]
        sn[i] = seasonal_naive(hist)
        cr[i] = croston(hist)
    return {"snaive": sn, "croston": cr}


def score_products(y, keep, first, origin, fc):
    rows = []
    for i in keep:
        hist = y[i, first[i]:origin - 1]
        act = y[i, origin - 1:origin - 1 + H]
        rows.append({"series": i, **{m: rmsse(hist, act, f[i]) for m, f in fc.items()}})
    return pd.DataFrame(rows).set_index("series")


def score_groups(y, keep, first, origin, fc, groups):
    """RMSSE of aggregated series: forecasts and actuals summed within each group."""
    out = []
    keep = np.asarray(keep)
    for g in sorted(set(groups[keep])):
        idx = keep[groups[keep] == g]
        start = int(first[idx].min())
        hist = y[idx, start:origin - 1].sum(axis=0)
        act = y[idx, origin - 1:origin - 1 + H].sum(axis=0)
        out.append({"group": g, **{m: rmsse(hist, act, f[idx].sum(axis=0)) for m, f in fc.items()}})
    return pd.DataFrame(out)


def run_origin(y, info, first, panel, test_start):
    val_start = test_start - H
    keep = [i for i in range(y.shape[0])
            if first[i] <= val_start - 1 - MIN_HISTORY
            and np.mean(np.diff(y[i, first[i]:val_start - 1]) ** 2) > 0]

    # validation: fix number of trees and choose the best benchmark
    val_lgb, val_model, _ = lgb_forecast(panel, val_start, first, y.shape[0])
    rounds = int(val_model.best_iteration)
    val_fc = {**local_forecasts(y, keep, first, val_start), "lightgbm": val_lgb}
    val = score_products(y, keep, first, val_start, val_fc)
    best = val[["snaive", "croston"]].mean().idxmin()

    # test: refit with frozen settings, score once
    test_lgb, test_model, n_rows = lgb_forecast(panel, test_start, first, y.shape[0], rounds)
    fc = {**local_forecasts(y, keep, first, test_start), "lightgbm": test_lgb}
    prod = score_products(y, keep, first, test_start, fc)
    prod["benchmark"] = best
    prod["r"] = np.log(prod["lightgbm"] / prod[best])

    dept = score_groups(y, keep, first, test_start, fc, info["dept_id"].to_numpy())
    store = score_groups(y, keep, first, test_start, fc, info["store_id"].to_numpy())
    imp = pd.Series(test_model.feature_importance("gain"), index=FEATURES)
    meta = {"rounds": rounds, "best_benchmark": best, "n_eligible": len(keep),
            "train_rows": int(n_rows),
            "val_mean_rmsse": val.mean().round(4).to_dict(),
            "test_mean_rmsse": prod[["snaive", "croston", "lightgbm"]].mean().round(4).to_dict()}
    return prod, dept, store, imp, meta


# ----------------------------------------------------------------------------
# Exploratory data analysis (descriptive; uses history before the first test window)
# ----------------------------------------------------------------------------
def eda(y, info, prices, cal, store, first, cls):
    hist_end = CLASSIFY_BEFORE - 1                     # last day before any test window
    c = cal.set_index("day")
    days = np.arange(1, y.shape[1] + 1)
    total = pd.DataFrame({"date": c.loc[days, "date"].to_numpy(), "units": y.sum(axis=0),
                          "wday": c.loc[days, "weekday"].to_numpy(),
                          "event": c.loc[days, "event_name_1"].notna().to_numpy(),
                          "snap": c.loc[days, f"snap_{info['state_id'].iloc[0]}"].to_numpy()})
    monthly = (total.assign(month=total["date"].dt.to_period("M").astype(str))
               .groupby("month")["units"].mean().round(1).reset_index()
               .rename(columns={"units": "avg_daily_units"}))
    monthly["store"] = store
    recent = total.iloc[hist_end - 365:hist_end]       # last year before the test windows
    weekday = recent.groupby("wday")["units"].mean().round(1)
    since_first = np.concatenate([y[i, first[i]:hist_end] for i in cls.index])
    christmas = total[(total["date"].dt.month == 12) & (total["date"].dt.day == 25)]["units"].tolist()
    on_sale = prices.groupby("item_id")["wm_yr_wk"].nunique()

    # data-quality checks: price coverage after first sale; long zero runs for smooth items
    price_idx = prices.set_index(["item_id", "wm_yr_wk"])["sell_price"]
    wk = c["wm_yr_wk"].to_numpy()                      # wk[d-1] = week of day d
    miss = tot = 0
    for i in range(y.shape[0]):
        d0 = np.arange(first[i], hist_end)              # 0-based day indices before the tests
        if len(d0):
            keys = pd.MultiIndex.from_arrays([np.repeat(info["item_id"].iloc[i], len(d0)), wk[d0]])
            has = price_idx.reindex(keys).notna().to_numpy()
            miss, tot = miss + int((~has).sum()), tot + len(has)

    def max_zero_run(v):
        best = cur = 0
        for x in v:
            cur = cur + 1 if x == 0 else 0
            best = max(best, cur)
        return best
    smooth = cls.index[cls["pattern"] == "Smooth"]
    runs = np.array([max_zero_run(y[i, hist_end - 365:hist_end]) for i in smooth])
    return monthly, {
        "store": store,
        "n_products": int(y.shape[0]),
        "n_products_classified": int(len(cls)),
        "category_counts": info["cat_id"].value_counts().to_dict(),
        "total_units_all_days": int(y.sum()),
        "mean_daily_units_last_year": float(recent["units"].mean()),
        "weekday_mean_units_last_year": weekday.to_dict(),
        "weekend_vs_weekday_ratio": float(recent[recent["wday"].isin(["Saturday", "Sunday"])]["units"].mean()
                                          / recent[~recent["wday"].isin(["Saturday", "Sunday"])]["units"].mean()),
        "snap_vs_nonsnap_ratio_last_year": float(recent[recent["snap"] == 1]["units"].mean()
                                                 / recent[recent["snap"] == 0]["units"].mean()),
        "event_vs_nonevent_ratio_last_year": float(recent[recent["event"]]["units"].mean()
                                                   / recent[~recent["event"]]["units"].mean()),
        "christmas_day_units": christmas,
        "zero_share_since_first_sale": float((since_first == 0).mean()),
        "share_first_sale_after_year1": float((first >= 365).mean()),
        "missing_price_share_after_first_sale": float(miss / tot),
        "smooth_with_14day_zero_run_last_year": int((runs >= 14).sum()),
        "smooth_products": int(len(smooth)),
        "missing_sales_values": int(np.isnan(y).sum()),
        "negative_sales_values": int((y < 0).sum()),
        "pattern_counts": cls["pattern"].value_counts().reindex(ORDER).fillna(0).astype(int).to_dict(),
        "median_adi": float(cls["adi"].median()), "median_cv2": float(cls["cv2"].median()),
        "median_weeks_on_sale": float(on_sale.median()),
    }


# ----------------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------------
def pct(x):
    return float((np.exp(x) - 1) * 100)


def mean_ci_pct(r):
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    m = r.mean()
    if len(r) < 2:
        return {"n": int(len(r)), "pct": pct(m), "lo": None, "hi": None}
    lo, hi = stats.t.interval(0.95, len(r) - 1, loc=m, scale=stats.sem(r))
    return {"n": int(len(r)), "pct": pct(m), "lo": pct(lo), "hi": pct(hi)}


def cramers_v(table):
    chi2 = stats.chi2_contingency(table)[0]
    return float(np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1))))


def pval(df, col_new, col_old):
    return float(df[col_new].iloc[0] if col_new in df else df[col_old].iloc[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    cal, sales, prices_all = load_all(args.data_dir)
    n_days = sales.filter(like="d_").shape[1]
    assert max(TEST_STARTS) + H - 1 == n_days, "last test window must end on the last day of the file"
    dates = cal.set_index("day")["date"].dt.date.astype(str)
    periods = [{"origin": k + 1, "validation": [ts - H, ts - 1, dates[ts - H], dates[ts - 1]],
                "test": [ts, ts + H - 1, dates[ts], dates[ts + H - 1]]}
               for k, ts in enumerate(TEST_STARTS)]

    prods, depts, stores_, imps, metas, eda_rows, monthly_rows = [], [], [], [], [], [], []
    for store in STORES:
        y, info, prices = store_data(sales, prices_all, store)
        first = first_sale_index(y)
        eligible0 = [i for i in range(y.shape[0])
                     if first[i] <= CLASSIFY_BEFORE - H - 1 - MIN_HISTORY]
        cls = pd.DataFrame([(i, *classify(y[i, first[i]:CLASSIFY_BEFORE - 1])) for i in eligible0],
                           columns=["series", "adi", "cv2", "pattern"]).set_index("series")
        monthly, e = eda(y, info, prices, cal, store, first, cls)
        eda_rows.append(e)
        monthly_rows.append(monthly)
        panel = build_panel(y, info, cal, prices, info["state_id"].iloc[0])
        for k, ts in enumerate(TEST_STARTS):
            prod, dept, st, imp, meta = run_origin(y, info, first, panel, ts)
            prod = prod.join(cls[["pattern"]], how="left")
            prod["item_id"] = info.loc[prod.index, "item_id"].to_numpy()
            prod["dept_id"] = info.loc[prod.index, "dept_id"].to_numpy()
            for df_ in (prod, dept, st):
                df_["store"], df_["origin"] = store, k + 1
            prods.append(prod.reset_index())
            depts.append(dept)
            stores_.append(st)
            imps.append(imp)
            metas.append({"store": store, "origin": k + 1, **meta})
            print(f"{store} origin {k + 1}: {meta['test_mean_rmsse']} rounds={meta['rounds']} "
                  f"best={meta['best_benchmark']} ({time.time() - t0:.0f}s)", flush=True)
        del panel

    P = pd.concat(prods, ignore_index=True)
    D = pd.concat(depts, ignore_index=True)
    S = pd.concat(stores_, ignore_index=True)
    P.to_csv(args.out_dir / "product_results.csv", index=False)
    D.to_csv(args.out_dir / "department_results.csv", index=False)
    S.to_csv(args.out_dir / "store_results.csv", index=False)
    pd.concat(monthly_rows).to_csv(args.out_dir / "eda_monthly_sales.csv", index=False)

    # ---- Primary unit of analysis: one product-store, r averaged over the 3 origins
    ok = P[np.isfinite(P["r"])]
    U = (ok.groupby(["store", "series"])
         .agg(r=("r", "mean"), n_origins=("r", "size"), pattern=("pattern", "first"),
              snaive=("snaive", "mean"), croston=("croston", "mean"), lightgbm=("lightgbm", "mean"))
         .reset_index())
    U = U[(U["n_origins"] == len(TEST_STARTS)) & U["pattern"].notna()]
    U["lgb_wins"] = U["r"] < 0
    U.to_csv(args.out_dir / "product_store_mean_r.csv", index=False)

    # RQ1
    t1 = pg.ttest(U["r"], 0, alternative="less")
    w1 = stats.wilcoxon(U["r"], alternative="less")
    rq1 = {"n": int(len(U)), **mean_ci_pct(U["r"]),
           "mean_r": float(U["r"].mean()), "median_r": float(U["r"].median()),
           "t": float(t1["T"].iloc[0]), "df": float(t1["dof"].iloc[0]),
           "p": pval(t1, "p_val", "p-val"), "cohen_d": pval(t1, "cohen_d", "cohen-d"),
           "wilcoxon_p": float(w1.pvalue), "lgb_win_share": float(U["lgb_wins"].mean())}
    rq1_by_store = {s: {**mean_ci_pct(g["r"]),
                        "t_p": float(stats.ttest_1samp(g["r"], 0, alternative="less").pvalue),
                        "win_share": float(g["lgb_wins"].mean())}
                    for s, g in U.groupby("store")}
    cells = [{"store": s, "origin": int(o), **mean_ci_pct(g["r"]),
              "win_share": float((g["r"] < 0).mean())}
             for (s, o), g in ok.groupby(["store", "origin"])]

    # RQ2
    welch = pg.welch_anova(data=U, dv="r", between="pattern")
    gh = pg.pairwise_gameshowell(data=U, dv="r", between="pattern")
    ctab = pd.crosstab(U["pattern"], U["lgb_wins"]).reindex(ORDER)
    chi2, p_chi, dof, _ = stats.chi2_contingency(ctab)
    by_pattern = {p: {**mean_ci_pct(g["r"]), "win_share": float(g["lgb_wins"].mean()),
                      "mean_rmsse": g[["snaive", "croston", "lightgbm"]].mean().round(4).to_dict()}
                  for p, g in U.groupby("pattern")}
    rq2_by_store = {}
    for s, g in U.groupby("store"):
        w = pg.welch_anova(data=g, dv="r", between="pattern")
        rq2_by_store[s] = {"F": float(w["F"].iloc[0]), "df1": float(w["ddof1"].iloc[0]),
                           "df2": float(w["ddof2"].iloc[0]), "p": pval(w, "p_unc", "p-unc"),
                           "np2": float(w["np2"].iloc[0]),
                           "by_pattern": {p: mean_ci_pct(gg["r"]) for p, gg in g.groupby("pattern")}}

    # Aggregation levels: LightGBM vs Croston at product, department, store level
    def level(df, name):
        lr = np.log(df["lightgbm"] / df["croston"])
        lr_sn = np.log(df["lightgbm"] / df["snaive"])
        return {"level": name, "n_series_runs": int(np.isfinite(lr).sum()),
                "vs_croston": mean_ci_pct(lr), "vs_snaive": mean_ci_pct(lr_sn),
                "lgb_beats_croston_share": float((lr < 0).mean()),
                "mean_rmsse": df[["snaive", "croston", "lightgbm"]].mean().round(4).to_dict()}
    levels = [level(ok, "Product"), level(D, "Department"), level(S, "Store")]

    imp = pd.concat(imps, axis=1).sum(axis=1)
    imp = (imp / imp.sum() * 100).sort_values(ascending=False)
    groups = {"rolling means": ["rmean7_lag28", "rmean28_lag28", "rmean56_lag28"],
              "single lags": [f"lag_{L}" for L in LAGS],
              "product IDs": ["item_id", "dept_id", "cat_id"],
              "calendar": ["wday", "month", "mday", "event", "snap"],
              "price": ["sell_price", "price_rel"]}

    summary = {
        "periods": periods, "stores": STORES, "runtime_seconds": round(time.time() - t0),
        "eda": eda_rows, "runs": metas,
        "n_product_store_units": int(len(U)),
        "n_excluded_units": int(sum(e["n_products"] for e in eda_rows) - len(U)),
        "pattern_counts_units": U["pattern"].value_counts().reindex(ORDER).to_dict(),
        "test_mean_rmsse_overall": ok[["snaive", "croston", "lightgbm"]].mean().round(4).to_dict(),
        "test_median_rmsse_overall": ok[["snaive", "croston", "lightgbm"]].median().round(4).to_dict(),
        "share_croston_beats_snaive": float((ok["croston"] < ok["snaive"]).mean()),
        "share_lgb_beats_snaive": float((ok["lightgbm"] < ok["snaive"]).mean()),
        "rq1": rq1, "rq1_by_store": rq1_by_store, "rq1_by_store_origin": cells,
        "rq2_welch": welch.round(6).to_dict(orient="records"),
        "rq2_gameshowell": gh.round(6).to_dict(orient="records"),
        "rq2_chi2": {"chi2": float(chi2), "df": int(dof), "p": float(p_chi),
                     "cramers_v": cramers_v(ctab), "table": ctab.to_dict()},
        "rq2_by_pattern": by_pattern, "rq2_by_store": rq2_by_store,
        "aggregation_levels": levels,
        "feature_importance_gain_pct": imp.round(2).to_dict(),
        "feature_importance_groups_pct": {k: round(float(imp[v].sum()), 2) for k, v in groups.items()},
        "lgb_params": LGB_PARAMS,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ("rq1", "rq2_chi2", "aggregation_levels",
                                              "runtime_seconds")}, indent=2, default=str))


if __name__ == "__main__":
    main()
