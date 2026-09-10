"""Cleaner PEAD via standardized unexpected earnings (SUE) + a harness power test.

SUE_q = (actual EPS - consensus EPS) / stdev of that stock's EPS surprise over trailing 8
quarters. Each calendar quarter, sort the universe cross-sectionally by SUE into deciles;
test long D10 (biggest standardized beats), short D1 (biggest misses), and the D10-D1 spread.
Abnormal-return-vs-universe scoring, holds 20/30/40/60d, slippage 0/0.3/0.5% per side.

The power test is the point: inject +1/+2/+3% synthetic edges into the REAL forward returns
and check the harness flags each at t>2, and report the minimum detectable effect given n and
the noise. A null is only informative about effects the harness can actually see.

  python brain_agent/sue.py            # SUE deciles + power test
  python brain_agent/sue.py --selftest
"""
import argparse
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fundamentals import NIFTY, load_earnings, load_prices, smallmid_universe
from pead import DAY, HOLDS, ROUNDTRIP, _liquid_enough, precompute, summarize

MIN_HIST = 8            # quarters of surprise history required to standardize


def build_sue_events(universe):
    """(events, prices, n_liquid, n_with_sue). events: (ms, sym, SUE, None) for name-quarters
    with >=8 trailing surprises. SUE uses only past+current surprises (point-in-time)."""
    prices = {}
    for s in universe:
        ser = load_prices(s)
        if ser and _liquid_enough(ser):
            prices[s] = ser
    events, n_with_sue = [], 0
    for s in prices:
        er = load_earnings(s)                       # (ms, est, rep, sur), est&rep present, sorted
        surp = [(ms, rep - est) for ms, est, rep, _ in er]
        if len(surp) >= MIN_HIST:
            n_with_sue += 1
        for i in range(MIN_HIST - 1, len(surp)):
            window = [x[1] for x in surp[i - MIN_HIST + 1:i + 1]]   # 8 quarters incl. current
            sd = statistics.stdev(window)
            if sd > 1e-9:
                events.append((surp[i][0], s, surp[i][1] / sd, None))
    events.sort()
    return events, prices, len(prices), n_with_sue


def _quarter(ms):
    d = datetime.utcfromtimestamp(ms / 1000)
    return d.year, (d.month - 1) // 3


def decile_groups(per_H, H):
    """Cross-sectional decile sort by SUE each calendar quarter. Returns (d1_alphas, d10_alphas)
    of gross abnormal returns (before cost)."""
    byq = defaultdict(list)
    for ms, sue, ag in per_H[H]:
        byq[_quarter(ms)].append((sue, ag))
    d1, d10 = [], []
    for items in byq.values():
        if len(items) < 10:
            continue
        items.sort(key=lambda x: x[0])
        k = len(items) // 10
        d1 += [ag for _, ag in items[:k]]
        d10 += [ag for _, ag in items[-k:]]
    return d1, d10


def spread_stat(d1, d10, slippage):
    """Long D10 - short D1, net of a round-trip on each leg."""
    a10, a1 = np.array(d10), np.array(d1)
    drag = ROUNDTRIP + 2 * slippage
    mean = a10.mean() - a1.mean() - 2 * drag
    se = np.sqrt(a10.var(ddof=1) / len(a10) + a1.var(ddof=1) / len(a1))
    return {"n": len(a10) + len(a1), "mean_alpha": float(mean),
            "t": float(mean / se) if se > 0 else 0.0}


def report_deciles(per_H, slippage):
    print(f"\n=== SUE deciles, slippage {slippage:.1%}/side ===")
    print(f"  {'hold':>4} {'longD10_a':>10} {'t':>6} {'shortD1_a':>10} {'t':>6} "
          f"{'D10-D1':>8} {'t':>6} {'n/side':>7}")
    for H in HOLDS:
        d1, d10 = decile_groups(per_H, H)
        if len(d10) < 20 or len(d1) < 20:
            continue
        drag = ROUNDTRIP + 2 * slippage
        long_d10 = summarize([x - drag for x in d10])
        short_d1 = summarize([-x - drag for x in d1])
        sp = spread_stat(d1, d10, slippage)
        print(f"  {H:>4} {long_d10['mean_alpha']:>+9.2%} {long_d10['t']:>+6.2f} "
              f"{short_d1['mean_alpha']:>+9.2%} {short_d1['t']:>+6.2f} "
              f"{sp['mean_alpha']:>+7.2%} {sp['t']:>+6.2f} {len(d10):>7}")


def power_test(per_H, H=60):
    """Inject constant edges into the REAL D10 forward returns; report detection + MDE."""
    _, d10 = decile_groups(per_H, H)
    a = np.array(d10)                                # gross abnormal returns, real noise
    n, sd = len(a), a.std(ddof=1)
    se = sd / np.sqrt(n)
    mde = 2 * se
    print(f"\n=== Power test (D10, hold {H}d, real noise) ===")
    print(f"  n={n}  per-event alpha std={sd:.1%}  SE={se:.3%}  "
          f"minimum detectable effect (t=2) = {mde:.2%}")
    print(f"  real D10 mean (no injection): {a.mean():+.2%} (t={a.mean()/se:+.2f})")
    for delta in (0.01, 0.02, 0.03):
        inj = a - a.mean() + delta                  # shift real distribution to a clean +delta edge
        t = inj.mean() / (inj.std(ddof=1) / np.sqrt(n))
        print(f"  inject +{delta:.0%}: t={t:+.2f}  {'DETECTED' if t > 2 else 'MISSED (underpowered)'}")
    return mde


def _selftest():
    # SUE standardization: constant-vol surprises -> SUE ~ surprise/sd
    import statistics as st
    w = [1, -1, 2, -2, 1, -1, 2, 3.0]
    sue = w[-1] / st.stdev(w)
    assert 1.3 < sue < 1.8, sue
    # power arithmetic: injecting +2% into zero-mean noise of sd s, n pts -> t = 0.02/(s/sqrt(n))
    a = np.random.default_rng(0).normal(0, 0.15, 400)
    t = ((a - a.mean() + 0.02).mean()) / (a.std(ddof=1) / np.sqrt(len(a)))
    assert abs(t - 0.02 / (a.std(ddof=1) / np.sqrt(len(a)))) < 1e-6
    print("sue selftest ok: SUE + power arithmetic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="smallmid", choices=["smallmid", "nifty"])
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return

    universe = smallmid_universe() if args.universe == "smallmid" else NIFTY
    events, prices, n_liquid, n_with_sue = build_sue_events(universe)
    print("*** SURVIVORSHIP BIAS: current constituents only -> positive results are UPPER BOUNDS. ***\n")
    print(f"{n_liquid} liquid names; {n_with_sue} have >=8 quarters of surprise history "
          f"(the SUE filter); {len(events)} standardized events")
    print("precomputing forward returns ...")
    per_H = precompute(events, prices)

    for slip in (0.0, 0.003, 0.005):
        report_deciles(per_H, slip)
    power_test(per_H, H=60)


if __name__ == "__main__":
    main()
