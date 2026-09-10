"""Cross-sectional momentum, done as a proper study (not one survey cell).

Signal at rebalance t = trailing formation-month return, optionally skipping the most recent
month (short-term-reversal guard). Decile-sort the universe each rebalance; hold the long-D10
(and D10-D1 for reference, though smallcaps are unshortable). Score per-period portfolio
returns vs the equal-weight universe AND the Nifty index, net of turnover-driven cost at
0/0.3/0.5% slippage per side. Power calc per config: n = number of rebalance PERIODS, so
momentum is time-limited, not stock-limited.

  python brain_agent/momentum.py            # full matrix + crash analysis
  python brain_agent/momentum.py --selftest
"""
import argparse
import bisect
import os
import sys
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fundamentals import index_constituents, load_prices, smallmid_universe
from pead import ROUNDTRIP, _liquid_enough

FORMATIONS = {"3m": 63, "6m": 126, "12m": 252}      # trading days
SKIP = 21                                            # ~1 month
STEPS = {"monthly": (21, 12), "quarterly": (63, 4)}  # (trading-day step, periods/yr)


def _arrs(prices):
    ts = {s: [r[0] for r in prices[s]] for s in prices}
    cl = {s: [r[4] for r in prices[s]] for s in prices}
    first = {s: prices[s][0][0] for s in prices}
    return ts, cl, first


def momentum_series(prices, master, nifty, formation, skip, step):
    """Per-period arrays for a decile momentum strategy on the `master` calendar (OHLCV rows)."""
    ts, cl, first = _arrs(prices)
    master_ts = [r[0] for r in master]
    n_ts = [r[0] for r in nifty]
    n_cl = [r[4] for r in nifty]

    def asof(s, t):
        i = bisect.bisect_right(ts[s], t) - 1
        return cl[s][i] if i >= 0 else None

    def nasof(t):
        i = bisect.bisect_right(n_ts, t) - 1
        return n_cl[i] if i >= 0 else None

    out = {k: [] for k in ("ts", "d10", "d1", "ew", "nifty", "turn10", "turn1")}
    prev10, prev1 = set(), set()
    r = skip + formation
    while r + step < len(master_ts):
        t, fwd = master_ts[r], master_ts[r + step]
        form_d, sig_d = master_ts[r - skip - formation], master_ts[r - skip]
        ranks = []
        for s in prices:
            if first[s] > form_d:
                continue
            pf, ps, pn, px = asof(s, form_d), asof(s, sig_d), asof(s, t), asof(s, fwd)
            if pf and ps and pn and px and pf > 0 and pn > 0:
                ranks.append((ps / pf - 1, px / pn - 1, s))
        if len(ranks) >= 20:
            ranks.sort(key=lambda x: x[0])
            k = len(ranks) // 10
            d10, d1 = ranks[-k:], ranks[:k]
            s10, s1 = {x[2] for x in d10}, {x[2] for x in d1}
            out["ts"].append(t)
            out["d10"].append(float(np.mean([f for _, f, _ in d10])))
            out["d1"].append(float(np.mean([f for _, f, _ in d1])))
            out["ew"].append(float(np.mean([f for _, f, _ in ranks])))
            ni, nf = nasof(t), nasof(fwd)
            out["nifty"].append(nf / ni - 1 if ni else np.nan)
            out["turn10"].append(len(s10 - prev10) / len(s10) if prev10 else 1.0)
            out["turn1"].append(len(s1 - prev1) / len(s1) if prev1 else 1.0)
            prev10, prev1 = s10, s1
        r += step
    return {k: np.array(v) for k, v in out.items()}


def _stat(alpha, ppy):
    n = len(alpha)
    se = alpha.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return {"n": n, "ann_alpha": float(alpha.mean() * ppy),
            "t": float(alpha.mean() / se) if se > 0 else 0.0,
            "mde_ann": float(2 * se * ppy)}


def score(ser, slippage, ppy):
    drag = ROUNDTRIP + 2 * slippage
    d10_net = ser["d10"] - ser["turn10"] * drag
    a_ew = _stat(d10_net - ser["ew"], ppy)
    a_nf = _stat(d10_net - ser["nifty"], ppy)
    spread = (ser["d10"] - ser["d1"]) - (ser["turn10"] + ser["turn1"]) * drag
    a_sp = _stat(spread, ppy)
    return a_ew, a_nf, a_sp, float(ser["turn10"][1:].mean())


def run_universe(name, prices, master, nifty):
    print(f"\n{'='*92}\nUNIVERSE: {name}  ({len(prices)} names)\n{'='*92}")
    for rb, (step, ppy) in STEPS.items():
        for fkey, fdays in FORMATIONS.items():
            for skip, slabel in ((SKIP, "skip1m"), (0, "noskip")):
                ser = momentum_series(prices, master, nifty, fdays, skip, step)
                if len(ser["d10"]) < 12:
                    continue
                print(f"\n-- {rb} rebalance | formation {fkey} {slabel} | "
                      f"{len(ser['d10'])} periods | avg turnover "
                      f"{ser['turn10'][1:].mean():.0%}/reb --")
                print(f"   {'slip':>5} {'D10 vs EW':>11} {'t':>6} {'D10 vs Nifty':>13} {'t':>6} "
                      f"{'D10-D1':>9} {'t':>6} {'MDE(ann)':>9}")
                for slip in (0.0, 0.003, 0.005):
                    ew, nf, sp, _ = score(ser, slip, ppy)
                    print(f"   {slip:>5.1%} {ew['ann_alpha']:>+10.1%} {ew['t']:>+6.2f} "
                          f"{nf['ann_alpha']:>+12.1%} {nf['t']:>+6.2f} "
                          f"{sp['ann_alpha']:>+8.1%} {sp['t']:>+6.2f} {ew['mde_ann']:>+8.1%}")


def crash_analysis(prices, master, nifty, formation=252, skip=SKIP, step=21):
    """Momentum's forward return in the worst market-drawdown periods."""
    ser = momentum_series(prices, master, nifty, formation, skip, step)
    print(f"\n{'='*92}\nMOMENTUM CRASH RISK (12m/skip1m, monthly) — worst market periods\n{'='*92}")
    print(f"   {'period start':>14} {'nifty':>8} {'D10(long)':>10} {'D10-D1':>9}")
    order = np.argsort(ser["nifty"])
    for i in order[:6]:
        d = datetime.utcfromtimestamp(ser["ts"][i] / 1000).strftime("%Y-%m-%d")
        print(f"   {d:>14} {ser['nifty'][i]:>+7.1%} {ser['d10'][i]:>+9.1%} "
              f"{ser['d10'][i]-ser['d1'][i]:>+8.1%}")
    for yr in ("2020-03", "2022", "2025"):
        idx = [i for i in range(len(ser["ts"]))
               if datetime.utcfromtimestamp(ser["ts"][i] / 1000).strftime("%Y-%m").startswith(yr)]
        if idx:
            d10 = float(np.mean([ser["d10"][i] for i in idx]))
            sp = float(np.mean([ser["d10"][i] - ser["d1"][i] for i in idx]))
            nif = float(np.mean([ser["nifty"][i] for i in idx]))
            print(f"   {yr}: mean/period  nifty {nif:+.1%}  D10 {d10:+.1%}  D10-D1 {sp:+.1%}")


def plot_best(prices, master, nifty, path, formation=126, skip=SKIP, step=63, slip=0.005):
    """Cumulative net return + drawdown for the best executable config (smallcap 6m quarterly)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ser = momentum_series(prices, master, nifty, formation, skip, step)
    drag = ROUNDTRIP + 2 * slip
    d10 = np.cumprod(1 + (ser["d10"] - ser["turn10"] * drag))
    ew = np.cumprod(1 + ser["ew"])
    nif = np.cumprod(1 + ser["nifty"])
    x = [datetime.utcfromtimestamp(t / 1000) for t in ser["ts"]]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(x, d10, label=f"momentum D10 (net {slip:.1%} slip)", lw=1.5)
    a1.plot(x, ew, label="equal-weight universe", lw=1)
    a1.plot(x, nif, label="Nifty index", lw=1)
    a1.set_yscale("log"); a1.set_ylabel("growth of Rs1 (log)"); a1.legend(fontsize=8); a1.grid(alpha=0.3)
    peak = np.maximum.accumulate(d10)
    a2.fill_between(x, (d10 - peak) / peak * 100, 0, alpha=0.4, color="crimson")
    a2.set_ylabel("D10 drawdown (%)"); a2.grid(alpha=0.3)
    fig.suptitle("Smallcap momentum D10 (6m formation, quarterly) — return and crash risk")
    fig.tight_layout(); os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=100); plt.close(fig)
    print(f"\n-> {path}")


def _load(universe_syms):
    prices = {}
    for s in universe_syms:
        ser = load_prices(s)
        if ser and _liquid_enough(ser):
            prices[s] = ser
    return prices


def _selftest():
    # synthetic: a persistent winner should land in D10 and its forward return exceeds EW
    base, day = 1_600_000_000_000, 86_400_000
    prices = {}
    for j in range(30):
        drift = 1.002 if j == 0 else 1.0                 # stock 0 trends up
        prices[f"S{j}"] = [[base + i * day, 0, 0, 0, 100 * drift ** i, 0] for i in range(400)]
    master = [[base + i * day, 0, 0, 0, 0, 0] for i in range(400)]
    ser = momentum_series(prices, master, master, 63, 0, 21)
    assert len(ser["d10"]) > 5 and ser["d10"].mean() >= ser["ew"].mean()
    print("momentum selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return

    nifty = load_prices("_NSEI")
    master = nifty                                        # index calendar = master rebalance clock
    n100 = _load(index_constituents("100"))
    smid = _load(smallmid_universe())
    print("*** SURVIVORSHIP BIAS: current constituents only -> long-side alpha is an UPPER BOUND. ***")
    print("Power note: n = rebalance PERIODS (monthly ~100, quarterly ~33) -> momentum is "
          "time-limited. A null on a cell with MDE > the effect is uninformative.")
    run_universe("Nifty 100 (liquid, executable)", n100, master, nifty)
    run_universe("Mid/Smallcap 345", smid, master, nifty)
    crash_analysis(smid, master, nifty)
    plot_best(smid, master, nifty, os.path.join(ROOT, "results", "momentum_smallcap.png"))


if __name__ == "__main__":
    main()
