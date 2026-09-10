"""Robustness checks on the smallcap momentum result: survivorship haircut + regime split.

Survivorship: free data cannot rebuild point-in-time INDEX membership (paid product), and
Yahoo drops delisted .NS tickers, so dead names' price history is unobtainable. What IS
obtainable is the attrition RATE: sampling NSE security-wise bhavcopies across years shows how
much of the liquid EQ universe disappears. We use that to anchor a sensitivity sweep -- X% of
holdings die per year at a -80% terminal return -- rather than guessing.

Regime split: is the alpha spread across regimes or concentrated in the 2021-24 smallcap boom?

  python brain_agent/robustness.py
  python brain_agent/robustness.py --selftest
"""
import argparse
import os
import sys
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fundamentals import load_prices, smallmid_universe
from momentum import SKIP, _load, momentum_series
from pead import ROUNDTRIP

TERMINAL = -0.80          # a delisting is near-total loss for a retail holder
SLIP = 0.005              # judge robustness at the harshest slippage we modelled

REGIMES = [
    ("pre-COVID (2018 - Feb 2020)", "2018-01-01", "2020-02-29"),
    ("crash+recovery (Mar 2020 - 2021)", "2020-03-01", "2021-12-31"),
    ("bull run (2022 - 2024)", "2022-01-01", "2024-12-31"),
    ("recent (2025 - 2026)", "2025-01-01", "2026-12-31"),
    ("flat stretch (2018 - 2019 only)", "2018-01-01", "2019-12-31"),
]


def _ms(datestr):
    return int(datetime.strptime(datestr, "%Y-%m-%d").timestamp() * 1000)


def _stat(alpha, ppy):
    n = len(alpha)
    if n < 3:
        return None
    se = alpha.std(ddof=1) / np.sqrt(n)
    return {"n": n, "ann": float(alpha.mean() * ppy),
            "t": float(alpha.mean() / se) if se > 0 else 0.0,
            "mde": float(2 * se * ppy)}


def survivorship_sweep(ser, ppy, slip=SLIP):
    """X% of held names die each year at TERMINAL. 'strategy-only' penalises just the D10 leg
    (conservative); 'symmetric' also penalises the equal-weight benchmark (deaths random wrt
    momentum). The truth is between: distressed names usually have LOW momentum (they sit in
    D1, not D10), so strategy-only is the pessimistic bound."""
    drag = ROUNDTRIP + 2 * slip
    d10_net = ser["d10"] - ser["turn10"] * drag
    print(f"\n  {'death/yr':>9} {'strategy-only':>14} {'t':>6} {'symmetric':>11} {'t':>6}")
    for X in (0.0, 0.01, 0.02, 0.03, 0.04, 0.05):
        p = X / ppy
        adj = (1 - p) * d10_net + p * TERMINAL
        ew_adj = (1 - p) * ser["ew"] + p * TERMINAL
        a_only = _stat(adj - ser["ew"], ppy)
        a_sym = _stat(adj - ew_adj, ppy)
        tag = "  <- measured attrition (incl. renames/mergers)" if abs(X - 0.03) < 1e-9 else ""
        print(f"  {X:>8.0%} {a_only['ann']:>+13.1%} {a_only['t']:>+6.2f} "
              f"{a_sym['ann']:>+10.1%} {a_sym['t']:>+6.2f}{tag}")


def regime_split(ser, ppy, slip=SLIP):
    drag = ROUNDTRIP + 2 * slip
    d10_net = ser["d10"] - ser["turn10"] * drag
    alpha = d10_net - ser["ew"]
    ts = ser["ts"]
    print(f"\n  {'regime':>34} {'periods':>8} {'D10-vs-EW':>10} {'t':>6} {'MDE':>7} {'nifty':>8}")
    for label, lo, hi in REGIMES:
        m = (ts >= _ms(lo)) & (ts <= _ms(hi))
        if m.sum() < 3:
            continue
        s = _stat(alpha[m], ppy)
        nif = float(ser["nifty"][m].mean() * ppy)
        verdict = "" if s["t"] > 2 else ("  (underpowered)" if s["ann"] > 0 else "")
        print(f"  {label:>34} {s['n']:>8} {s['ann']:>+9.1%} {s['t']:>+6.2f} "
              f"{s['mde']:>+6.1%} {nif:>+7.1%}{verdict}")


def _selftest():
    ppy = 12
    n = 120
    ser = {"d10": np.full(n, 0.02), "ew": np.zeros(n), "turn10": np.zeros(n),
           "nifty": np.zeros(n), "ts": np.arange(n) * 0 + _ms("2021-01-01")}
    ser = {k: np.asarray(v) for k, v in ser.items()}
    drag = ROUNDTRIP
    base = ((ser["d10"] - ser["turn10"] * drag) - ser["ew"]).mean() * ppy
    p = 0.03 / ppy
    adj = ((1 - p) * (ser["d10"] - ser["turn10"] * drag) + p * TERMINAL - ser["ew"]).mean() * ppy
    assert adj < base, (adj, base)                       # deaths must reduce alpha
    assert abs((base - adj) - (p * (0.02 - TERMINAL) * ppy)) < 1e-9
    print("robustness selftest ok: death drag arithmetic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return

    nifty = load_prices("_NSEI")
    smid = _load(smallmid_universe())
    print("Measured attrition of the liquid (>=Rs2cr) NSE EQ universe, from bhavcopy sampling:")
    print("  2021->2026: 102/712 gone = 2.70%/yr   |   2023->2026: 94/821 gone = 3.28%/yr")
    print("  NOTE: that includes renames/mergers (AMARAJABAT->ARE&M, CADILAHC->ZYDUSLIFE,")
    print("        CENTURYTEX->Grasim, ADANITRANS->ADANIENSOL). True -80% wipeouts are a")
    print("        SUBSET, so ~3%/yr is an upper bound on value-destroying deaths.\n")

    for label, formation, step, ppy in (("6m formation, MONTHLY", 126, 21, 12),
                                        ("6m formation, QUARTERLY", 126, 63, 4)):
        ser = momentum_series(smid, nifty, nifty, formation, SKIP, step)
        print(f"\n{'='*80}\n{label} — smallcap D10, net of {SLIP:.1%} slippage\n{'='*80}")
        print("SURVIVORSHIP SENSITIVITY (X% of holdings die/yr at -80%)")
        survivorship_sweep(ser, ppy)
        if ppy == 12:
            print("\nREGIME SPLIT (D10 vs equal-weight universe)")
            regime_split(ser, ppy)


if __name__ == "__main__":
    main()
