"""Equity market survey — same structure as the crypto survey, reusing its walk-forward
machinery (backtest._cell / _walk_forward / STRATEGIES) with an Indian-equity cost model.

  python equity_agent/survey.py            # 10 stocks x 3 timeframes x 3 strategies
  python equity_agent/survey.py --zero-brokerage --years 3

Structural difference from crypto: no per-trade TDS; instead STCG 20% on net profit at the
portfolio level. Fees: brokerage (Rs20 or 0.03%, whichever lower) + STT 0.1%/side +
txn ~0.00325% + SEBI + stamp 0.015% (buy) + 18% GST on (brokerage+txn+SEBI).
"""
import argparse
import os
import sys
from dataclasses import replace
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)          # import the crypto backtester + config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # import angel_client

import backtest as bt
from angel_client import AngelClient
from config import Config

ASSETS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
          "ITC", "LT", "SBIN", "BHARTIARTL", "MARUTI"]
TFS = ["ONE_DAY", "ONE_HOUR", "ONE_WEEK"]        # weekly is resampled from daily
TF_HOURS = {"ONE_DAY": 24, "ONE_HOUR": 1, "ONE_WEEK": 168}
# weekly is data-starved over 3y (~157 bars): use 2 windows with a bigger test slice so the
# held-out window is long enough to warm up an indicator and actually trade.
WF = {  # (n_windows, min_train, train_frac)
    "ONE_DAY": (4, 100, 5 / 6),
    "ONE_HOUR": (4, 100, 5 / 6),
    "ONE_WEEK": (3, 60, 0.6),   # ~8y daily -> ~420 weekly bars: window ~140, test ~56
}
RESULTS = os.path.join(ROOT, "results")


def resample_weekly(daily):
    """Aggregate [ts,o,h,l,c,v] daily bars into weekly (Angel has no ONE_WEEK)."""
    weeks = {}
    order = []
    for ts, o, h, l, c, v in daily:
        key = datetime.fromtimestamp(ts / 1000).isocalendar()[:2]   # (iso_year, iso_week)
        if key not in weeks:
            weeks[key] = [ts, o, h, l, c, v]
            order.append(key)
        else:
            b = weeks[key]
            b[2] = max(b[2], h)
            b[3] = min(b[3], l)
            b[4] = c
            b[5] += v
    return [weeks[k] for k in order]


WEEKLY_DAILY_YEARS = 8      # weekly resamples from daily; needs a long history to be tradeable


def load(client, symbol, tf, years):
    to = datetime.today()
    if tf == "ONE_WEEK":
        years = max(years, WEEKLY_DAILY_YEARS)
    frm = to - timedelta(days=int(years * 365.25))
    f, t = frm.strftime("%Y-%m-%d"), to.strftime("%Y-%m-%d")
    if tf == "ONE_WEEK":
        return resample_weekly(client.fetch_ohlcv(symbol, "ONE_DAY", f, t))
    return client.fetch_ohlcv(symbol, tf, f, t)


def survey(years, zero_brokerage):
    cost = bt.CostModel(kind="equity", zero_brokerage=zero_brokerage, stcg=0.20)
    gate_fee = cost.sell_cost(1.0)        # per-side cost fraction for RiskManager's sell gate
    client = AngelClient().login()
    print(f"logged in as {client.profile_name()}; "
          f"brokerage={'Rs0' if zero_brokerage else 'Rs20/0.03%'}, STCG 20%\n")

    cells = []
    for tf in TFS:
        for sym in ASSETS:
            print(f"loading {sym} {tf} ...")
            try:
                series = load(client, sym, tf, years)
            except Exception as e:
                print(f"  skip {sym} {tf}: {e}")
                continue
            min_candles = 120 if tf == "ONE_WEEK" else 400   # weekly is coarse; 3y = ~157 bars
            if len(series) < min_candles:
                print(f"  skip {sym} {tf}: only {len(series)} candles")
                continue
            ohlcv_map = {sym: series}
            cfg = replace(Config.load(), taker_fee=gate_fee, tds_rate=0.0, timeframe=tf)
            n_win, min_train, train_frac = WF[tf]
            for strat in bt.STRATEGIES:
                m = bt._cell(cfg, ohlcv_map, TF_HOURS[tf], strat, cost=cost,
                             n_windows=n_win, min_train=min_train, train_frac=train_frac)
                cells.append({"timeframe": tf, "asset": sym, "strategy": strat, **m})
                pulse = "PULSE" if m["test_alpha"] > 0 and m["n_trades"] >= 4 else "."
                print(f"  {strat:9s} test_alpha={m['test_alpha']:+.1%} "
                      f"wf_gap={m['wf_gap']:+.1%} sharpe={m['sharpe']:+.2f} "
                      f"n={m['n_trades']:<4d} {pulse}")

    if not cells:
        print("no cells produced results")
        return
    os.makedirs(RESULTS, exist_ok=True)
    cells.sort(key=lambda r: r["test_alpha"], reverse=True)
    import csv
    with open(os.path.join(RESULTS, "equity_survey.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(cells[0].keys()))
        w.writeheader()
        w.writerows(cells)
    bt._survey_heatmap(cells, assets=ASSETS, tfs=TFS,
                       path=os.path.join(RESULTS, "equity_survey_heatmap.png"))

    winners = [c for c in cells if c["test_alpha"] > 0 and c["n_trades"] >= 4]
    print(f"\n{'='*62}")
    print(f"Cells with a pulse (out-of-sample alpha > 0, >=4 trades): {len(winners)}/{len(cells)}")
    for c in winners[:12]:
        print(f"  {c['timeframe']:8s} {c['asset']:10s} {c['strategy']:9s}  "
              f"alpha={c['test_alpha']:+.1%} gap={c['wf_gap']:+.1%} "
              f"sharpe={c['sharpe']:+.2f} n={c['n_trades']}")
    if not winners:
        print("  NONE. No (tf x stock x strategy) beats buy-and-hold out-of-sample after costs.")
    print(f"-> {os.path.join(RESULTS, 'equity_survey.csv')}")
    print(f"-> {os.path.join(RESULTS, 'equity_survey_heatmap.png')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=3.0)
    ap.add_argument("--zero-brokerage", action="store_true",
                    help="model Rs0 delivery brokerage instead of the conservative Rs20/0.03%")
    args = ap.parse_args()
    survey(args.years, args.zero_brokerage)


if __name__ == "__main__":
    main()
