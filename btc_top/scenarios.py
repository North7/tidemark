"""未來情境（2026-10 加入，經使用者同意）：三條「歷史規律外推」的價格路徑，不是預測。

- 保守：頂部 ÷ 前一輪低點的倍數，以最近三輪（2017、2021、2025）對數線性外推，乘上本輪低點。
- 中間：頂部 MVRV（價格 ÷ 實現價格）以四次頂部對數線性外推，乘上「實現價格依近 4 年年增率推到頂部日」。
- 樂觀：冪律趨勢價（頂部倍數取 1.0；全部歷史擬合）。
頂部日期：兩個時機時鐘預估中心的中點。路徑形狀：過去三輪「低點 → 頂部」的對數價格走法（時間、漲幅各自標準化後平均），
從今天的價格接到各情境頂部。每日依最新數據重算。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from btc_top import scoring as S

GRID = 200


def _next(vals) -> float:
    """對數線性擬合後的下一個值。"""
    x = np.arange(len(vals))
    b, a = np.polyfit(x, np.log(vals), 1)
    return float(np.exp(a + b * len(vals)))


def _shape(price: pd.Series, pairs) -> np.ndarray:
    """過去各輪低點 → 頂部的標準化走法平均（0→1）。"""
    curves = []
    for lo, top in pairs:
        seg = np.log(price[lo:top].values)
        f = (seg - seg[0]) / (seg[-1] - seg[0])
        curves.append(np.interp(np.linspace(0, 1, GRID), np.linspace(0, 1, len(f)), f))
    c = np.clip(np.mean(curves, axis=0), 0, 1)
    k = 31  # 移動平均平滑（約 15% 的週期長度），再取單調遞增，避免階梯狀
    pad = np.concatenate([np.full(k, c[0]), c, np.full(k, c[-1])])
    c = np.convolve(pad, np.ones(k) / k, mode="same")[k:-k]
    c = np.maximum.accumulate(c)
    return (c - c[0]) / (c[-1] - c[0])


def compute_scenarios(ind: pd.DataFrame, timing: dict, today: pd.Timestamp) -> dict:
    price = ind["price"].dropna()
    tops = [price[m].idxmax() for m in [S.PRE_TOP_MONTH] + list(S.TOP_MONTHS)]
    lows = [price[tops[k]:tops[k + 1]].idxmin() for k in range(len(tops) - 1)]  # 每個頂部之前的低點
    cur_low = pd.Timestamp(timing["cycle_low_date"])
    now_d, now_p = price.index[-1], float(price.iloc[-1])

    c1 = pd.Timestamp(timing["expected_window_by_low"]["center"])
    c2 = pd.Timestamp(timing["expected_window_by_halving"]["center"])
    top_d = (c1 + (c2 - c1) / 2).normalize()

    # 保守：頂部 ÷ 前低（最近三輪）
    r_low = [price[tops[k + 1]] / price[lows[k]] for k in range(1, len(lows))]
    cons = price[cur_low] * _next(r_low)
    # 中間：頂部 MVRV × 實現價格成長
    rp = ind["realized_price"].dropna()
    mv = [float(ind["rp_multiple"][t]) for t in tops[1:]]
    g = (rp.iloc[-1] / rp[rp.index >= rp.index[-1] - pd.Timedelta(days=1461)].iloc[0]) ** (1 / 4)
    yrs = (top_d - rp.index[-1]).days / 365.25
    mid = float(rp.iloc[-1] * g ** yrs * _next(mv))
    # 樂觀：冪律趨勢價
    age = lambda d: (pd.Timestamp(d) - pd.Timestamp("2009-01-03")).days
    b, a = np.polyfit(np.log10([age(d) for d in price.index]), np.log10(price.values), 1)
    opt = float(10 ** (a + b * np.log10(age(top_d))))

    shape = _shape(price, [(lows[k], tops[k + 1]) for k in range(1, len(lows))])
    span = (top_d - cur_low).days
    t0 = (now_d - cur_low).days / span
    f0 = float(np.interp(t0, np.linspace(0, 1, GRID), shape))
    dates = pd.date_range(now_d, top_d, freq="7D")
    if dates[-1] != top_d:
        dates = dates.append(pd.DatetimeIndex([top_d]))
    tau = np.array([(d - cur_low).days / span for d in dates])
    f = np.interp(tau, np.linspace(0, 1, GRID), shape)
    w = np.clip((f - f0) / max(1e-9, 1 - f0), 0, 1)

    def path(top_p):
        return [round(float(np.exp(np.log(now_p) + x * (np.log(top_p) - np.log(now_p)))), 0) for x in w]

    sc = {"conservative": cons, "base": mid, "optimistic": opt}
    return {
        "as_of": str(now_d.date()), "price": round(now_p, 2), "top_date": str(top_d.date()),
        "window": {"from": min(timing["expected_window_by_low"]["full_from"], timing["expected_window_by_halving"]["full_from"]),
                   "to": max(timing["expected_window_by_low"]["full_to"], timing["expected_window_by_halving"]["full_to"])},
        "tops": {k: round(v, 0) for k, v in sc.items()},
        "inputs": {"top_over_prior_low": [round(float(x), 2) for x in r_low], "next_top_over_low": round(_next(r_low), 2),
                   "cycle_low": round(float(price[cur_low]), 2), "top_mvrv": [round(x, 2) for x in mv], "next_top_mvrv": round(_next(mv), 2),
                   "realized_price": round(float(rp.iloc[-1]), 0), "realized_growth": round(float(g - 1), 4),
                   "powerlaw_slope": round(float(b), 3),
                   "top_over_prior_top": [round(float(price[tops[k + 1]] / price[tops[k]]), 2) for k in range(1, len(tops) - 1)],
                   "ratio_method_top": round(float(price[tops[-1]] * _next([price[tops[k + 1]] / price[tops[k]] for k in range(1, len(tops) - 1)])), 0)},
        "dates": [str(d.date()) for d in dates],
        "paths": {k: path(v) for k, v in sc.items()},
    }


# 走勢偏離提示：實際價格連續 RANGE_DAYS 天低於保守路徑、或高於樂觀路徑超過 RANGE_TOL，代表模型假設可能正在失效。
# 20% 約為比特幣一個月的典型波動，30 天避免短期雜訊；屬事前判斷，未用歷史最佳化。
RANGE_TOL, RANGE_DAYS = 0.20, 30


def track_scenarios(snap: dict, price: pd.Series) -> dict:
    """定案日凍結的三條路徑 vs 實際價格：今天各情境「當時預估的價格」與實際的差距、最貼近哪一條、是否持續超出範圍。"""
    p = price.dropna()
    d, now = p.index[-1], float(p.iloc[-1])
    dates = pd.to_datetime(snap["dates"])
    t = np.array([(x - dates[0]).days for x in dates], float)
    hist = p[dates[0]:]
    td_all = np.array([(x - dates[0]).days for x in hist.index], float)
    exp = {k: pd.Series(np.exp(np.interp(td_all, t, np.log(path))), index=hist.index) for k, path in snap["paths"].items()}
    rows = {k: {"expected": round(float(v.iloc[-1]), 0), "diff": round(now / float(v.iloc[-1]) - 1, 4) + 0.0} for k, v in exp.items()}
    closest = min(rows, key=lambda k: abs(np.log(now / rows[k]["expected"])))

    def streak(mask: pd.Series) -> int:
        m = mask.values[::-1]
        return int(np.argmin(m)) if not m.all() else int(len(m))

    below, above = hist < exp["conservative"], hist > exp["optimistic"]
    far_below, far_above = hist < exp["conservative"] * (1 - RANGE_TOL), hist > exp["optimistic"] * (1 + RANGE_TOL)
    pos = "below" if below.iloc[-1] else "above" if above.iloc[-1] else "inside"
    alert = ("below" if streak(far_below) >= RANGE_DAYS else "above" if streak(far_above) >= RANGE_DAYS else None)
    return {"snapshot_date": snap["as_of"], "as_of": str(d.date()), "price": round(now, 2), "rows": rows, "closest": closest,
            "tops": snap["tops"], "top_date": snap["top_date"],
            "range": {"position": pos, "below_days": streak(below), "above_days": streak(above),
                      "far_below_days": streak(far_below), "far_above_days": streak(far_above),
                      "tolerance": RANGE_TOL, "min_days": RANGE_DAYS, "alert": alert}}


# 情境變動紀錄：每天把三個頂部估計與輸入寫進只新增的 CSV，標出「明顯變動」與原因
HIST_COLS = ["date", "conservative", "base", "optimistic", "top_date", "cycle_low_date", "cycle_low_price", "realized_price"]
CHANGE_PCT = 0.03  # 任一頂部估計相對上次變動點累積變動 ≥ 3% 才算明顯變動（中間情境會隨實現價格每天微調）


def log_scenarios(path, sc: dict, timing: dict, snap: dict | None) -> dict:
    from pathlib import Path
    path = Path(path)
    old = pd.read_csv(path, dtype=str) if path.exists() else pd.DataFrame(columns=HIST_COLS)
    rows = old.to_dict("records")
    if not rows and snap:  # 第一筆：定案時凍結的情境（當時的本輪低點與今天相同）
        rows.append({"date": snap["as_of"], **{k: snap["tops"][k] for k in ("conservative", "base", "optimistic")},
                     "top_date": snap["top_date"], "cycle_low_date": timing["cycle_low_date"],
                     "cycle_low_price": sc["inputs"]["cycle_low"], "realized_price": ""})
    if sc["as_of"] not in {str(r["date"]) for r in rows}:
        rows.append({"date": sc["as_of"], **{k: sc["tops"][k] for k in ("conservative", "base", "optimistic")},
                     "top_date": sc["top_date"], "cycle_low_date": timing["cycle_low_date"],
                     "cycle_low_price": sc["inputs"]["cycle_low"], "realized_price": sc["inputs"]["realized_price"]})
        pd.DataFrame(rows, columns=HIST_COLS).to_csv(path, index=False)
    # 找出明顯變動
    keys = ("conservative", "base", "optimistic")
    events, anchor, prev = [], rows[0], rows[0]
    for r in rows[1:]:
        reason = None
        if str(r["cycle_low_date"]) != str(prev["cycle_low_date"]):
            reason = ("low", f"{r['cycle_low_date']}", float(r["cycle_low_price"]))
        elif str(r["top_date"]) != str(prev["top_date"]):
            reason = ("top_date", str(r["top_date"]), None)
        elif any(abs(float(r[k]) / float(anchor[k]) - 1) >= CHANGE_PCT for k in keys):
            reason = ("drift", None, None)
        if reason:
            events.append({"date": str(r["date"]), "reason": reason[0], "detail": reason[1], "low_price": reason[2],
                           "tops": {k: float(r[k]) for k in keys}, "prev_tops": {k: float(anchor[k]) for k in keys}})
            anchor = r
        prev = r
    first = rows[0]
    return {"since": str(first["date"]), "rows": len(rows), "change_pct": CHANGE_PCT,
            "since_first": {k: round(float(sc["tops"][k]) / float(first[k]) - 1, 4) for k in keys},
            "events": events[-5:], "last_event": events[-1] if events else None}
