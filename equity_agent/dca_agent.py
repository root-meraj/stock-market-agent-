"""Live DCA agent for Angel One (NSE delivery). DRY_RUN=true by default: it computes and
logs the exact order it would place, and places nothing.

  python equity_agent/dca_agent.py --once      # one cycle (for cron)
  python equity_agent/dca_agent.py             # survival loop, checks every hour
  python equity_agent/dca_agent.py --backtest  # weekly DCA vs lump-sum vs EW B&H (8y daily)
  python equity_agent/dca_agent.py --selftest  # network-free calendar + DCA checks

Reuses the crypto agent's RiskManager (life meter / pause) and SQLite DB (flags + equity),
plus AngelClient for session/quotes/orders. Set DRY_RUN=false only when you mean it.
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from angel_client import AngelClient, load_env
from db import DB
from risk import RiskManager

IST = timezone(timedelta(hours=5, minutes=30))

# NSE trading holidays (weekends handled separately). UPDATE YEARLY -- the real calendar
# can't be derived; this is the calibration knob. Source: NSE holiday list.
NSE_HOLIDAYS = {
    "2025-02-26", "2025-03-14", "2025-03-31", "2025-04-10", "2025-04-14", "2025-04-18",
    "2025-05-01", "2025-08-15", "2025-08-27", "2025-10-02", "2025-10-21", "2025-10-22",
    "2025-11-05", "2025-12-25",
    "2026-01-26", "2026-03-04", "2026-03-21", "2026-04-01", "2026-04-03", "2026-04-14",
    "2026-05-01", "2026-08-15", "2026-10-02", "2026-11-11", "2026-12-25",
}


def _envbool(k, default):
    return os.environ.get(k, str(default)).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class DcaConfig:
    dry_run: bool = field(default_factory=lambda: _envbool("DRY_RUN", True))
    instrument: str = os.environ.get("DCA_INSTRUMENT", "NIFTYBEES")
    basket: tuple = ()                      # e.g. ("RELIANCE","TCS","HDFCBANK"); overrides instrument
    amount: float = field(default_factory=lambda: float(os.environ.get("DCA_AMOUNT", 1000)))
    cadence_weekday: int = 0               # 0=Mon
    rebalance_threshold: float = 0.05      # rebalance if a weight drifts >5% absolute
    starting_capital: float = field(default_factory=lambda: float(os.environ.get("DCA_START", 50000)))
    pause_life: float = 0.6                # equity/start below -> pause buying
    resume_life: float = 0.7
    poll_seconds: int = 3600
    db_path: str = os.path.join(ROOT, "equity_dca.db")
    webhook: str = os.environ.get("N8N_WEBHOOK_URL", "")

    @property
    def targets(self):
        """{symbol: target_weight}. Single instrument -> 100%; basket -> equal weight."""
        if self.basket:
            return {s: 1.0 / len(self.basket) for s in self.basket}
        return {self.instrument: 1.0}


# --- market calendar ------------------------------------------------------

def is_trading_day(d, holidays=NSE_HOLIDAYS):
    return d.weekday() < 5 and d.isoformat() not in holidays


def next_trading_day(d, holidays=NSE_HOLIDAYS):
    while not is_trading_day(d, holidays):
        d += timedelta(days=1)
    return d


def scheduled_buy_date(today, weekday, holidays=NSE_HOLIDAYS):
    """This week's scheduled buy date, rolled forward if it lands on a holiday/weekend
    (so a Monday holiday buys Tuesday)."""
    monday = today - timedelta(days=today.weekday())
    return next_trading_day(monday + timedelta(days=weekday), holidays)


# --- the agent ------------------------------------------------------------

class DcaAgent:
    def __init__(self, cfg, client=None):
        self.cfg = cfg
        self.client = client or AngelClient()
        self.db = DB(cfg.db_path)
        self.db.c.executescript(
            "create table if not exists orders(ts real, run_date text, symbol text, side text,"
            " qty real, ltp real, amount real, dry int, status text, order_id text, reason text);")
        self.db.c.commit()
        self.risk = RiskManager(cfg, self.db)

    def log(self, msg):
        print(f"{datetime.now(IST):%Y-%m-%d %H:%M:%S} IST  {msg}", flush=True)

    # portfolio snapshot from the live account (read-only; works in dry run too)
    def snapshot(self):
        cash = self.client.funds()
        holdings = {h["symbol"]: h for h in self.client.holdings()}
        held_value = sum(h["value"] for h in holdings.values())
        return cash + held_value, cash, holdings

    def already_bought_this_week(self, today):
        return self.db.get_flag("last_buy_week") == today.strftime("%G-W%V")

    def plan_orders(self, cash, holdings, port_value):
        """Return [(symbol, side, qty, ltp, amount, reason)]. One DCA buy per target, plus
        threshold-triggered rebalances when a holding drifts past the band."""
        targets = self.cfg.targets
        invest = port_value + self.cfg.amount            # value after this week's contribution
        ltps = {s: self.client.ltp(s) for s in targets}
        cur_val = {s: holdings.get(_ts(s), {}).get("value", 0.0) for s in targets}

        drifted = any(abs(cur_val[s] / port_value - targets[s]) > self.cfg.rebalance_threshold
                      for s in targets) if port_value > 0 else False

        orders = []
        for s in targets:
            target_val = targets[s] * invest
            if drifted and len(targets) > 1:
                delta = target_val - cur_val[s]          # rebalance toward target
                reason = "rebalance+dca"
            else:
                delta = self.cfg.amount * targets[s]      # just deploy this week's cash
                reason = "dca"
            qty = int(delta // ltps[s])
            if qty >= 1:
                orders.append((s, "BUY", qty, ltps[s], qty * ltps[s], reason))
            elif qty <= -1:
                orders.append((s, "SELL", -qty, ltps[s], -qty * ltps[s], reason))
        return orders

    def execute(self, order, run_date):
        symbol, side, qty, ltp, amount, reason = order
        status, order_id = "DRY", ""
        if not self.cfg.dry_run:
            try:
                resp = self.client.place_order(symbol, side, qty)
                order_id = str(resp)
                status = "PLACED"
            except Exception as e:
                status = f"ERROR:{e}"
        self.db.c.execute(
            "insert into orders values (?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), run_date.isoformat(), symbol, side, qty, ltp, amount,
             int(self.cfg.dry_run), status, order_id, reason))
        self.db.c.commit()
        self.log(f"{'[DRY] would place' if self.cfg.dry_run else status} "
                 f"{side} {qty} {symbol} @~{ltp:.2f} (Rs{amount:.0f}, {reason})")
        return {"symbol": symbol, "side": side, "qty": qty, "ltp": ltp,
                "amount": amount, "status": status}

    def run_once(self):
        self.client.ensure_session()
        today = datetime.now(IST).date()
        equity, cash, holdings = self.snapshot()
        life, paused = self.risk.update_pause(equity)
        self.db.snapshot(equity, cash, life, paused)
        self.log(f"equity=Rs{equity:.0f} cash=Rs{cash:.0f} life={life:.0%} "
                 f"paused={paused} dry_run={self.cfg.dry_run}")

        summary = {"date": today.isoformat(), "equity": round(equity), "life": round(life, 3),
                   "paused": paused, "dry_run": self.cfg.dry_run, "orders": []}
        if paused:
            self.log("life below floor -> buying paused")
            self.alert({**summary, "note": "paused"})
            return
        buy_date = scheduled_buy_date(today, self.cfg.cadence_weekday)
        if today != buy_date:
            self.log(f"not a buy day (next scheduled buy {buy_date.isoformat()})")
            return
        if self.already_bought_this_week(today):
            self.log("already bought this week")
            return
        if cash < self.cfg.amount:
            self.log(f"insufficient cash (Rs{cash:.0f} < Rs{self.cfg.amount:.0f}) -> skip")
            self.alert({**summary, "note": "insufficient_cash"})
            return

        for order in self.plan_orders(cash, holdings, equity):
            summary["orders"].append(self.execute(order, today))
        self.db.set_flag("last_buy_week", today.strftime("%G-W%V"))
        self.alert(summary)

    def alert(self, payload):
        if not self.cfg.webhook:
            self.log("no N8N_WEBHOOK_URL set -> skipping alert")
            return
        try:
            req = urllib.request.Request(
                self.cfg.webhook, data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10).read()
            self.log("alert sent")
        except Exception as e:
            self.log(f"alert failed: {e}")

    def loop(self):
        self.log(f"DCA agent up. instrument={self.cfg.instrument} basket={self.cfg.basket} "
                 f"amount=Rs{self.cfg.amount:.0f} dry_run={self.cfg.dry_run}")
        while True:
            try:
                self.run_once()
            except Exception:
                import traceback
                self.log("cycle error:\n" + traceback.format_exc())
            time.sleep(self.cfg.poll_seconds)


def _ts(symbol):
    """DCA symbols are names (NIFTYBEES); demat holdings key by tradingsymbol (NIFTYBEES-EQ)."""
    return symbol if symbol.endswith("-EQ") else symbol + "-EQ"


# --- backtest: weekly DCA vs lump-sum vs equal-weight B&H -----------------

def _isoweek(ts):
    return datetime.fromtimestamp(ts / 1000).strftime("%G-W%V")


def _maxdd(series):
    peak = -1e18
    mdd = 0.0
    for _, v in series:
        peak = max(peak, v)
        if peak > 0:
            mdd = max(mdd, (peak - v) / peak)
    return mdd


def xirr(cashflows):
    """Money-weighted annualized return. cashflows: [(date, amount)], outflows negative.
    Newton's method with a bisection fallback for robustness."""
    t0 = cashflows[0][0]
    yrs = [(d - t0).days / 365.0 for d, _ in cashflows]
    amt = [a for _, a in cashflows]

    def npv(r):
        return sum(a / (1 + r) ** y for a, y in zip(amt, yrs))

    r = 0.1
    for _ in range(80):
        d = sum(-y * a / (1 + r) ** (y + 1) for a, y in zip(amt, yrs))
        if abs(d) < 1e-9:
            break
        step = npv(r) / d
        r -= step
        if r <= -0.9999:
            r = -0.99
        if abs(step) < 1e-8:
            return r
    lo, hi = -0.9999, 10.0                                # bisection fallback
    if npv(lo) * npv(hi) > 0:
        return r
    for _ in range(200):
        mid = (lo + hi) / 2
        (lo, hi) = (lo, mid) if npv(lo) * npv(mid) <= 0 else (mid, hi)
    return (lo + hi) / 2


def _sharpe(series, contribs):
    """Annualized Sharpe on daily returns, with external contributions removed so weekly
    DCA top-ups don't masquerade as investment returns. rf=0."""
    import numpy as np
    eq = [v for _, v in series]
    rets = [(eq[i] - contribs[i]) / eq[i - 1] - 1 for i in range(1, len(eq)) if eq[i - 1] > 0]
    r = np.array(rets)
    return float(r.mean() / r.std() * np.sqrt(252)) if len(r) and r.std() > 0 else 0.0


def backtest(years=8.0):
    import backtest as bt
    client = AngelClient()
    client.ensure_session()               # NIFTYBEES may not be cached yet
    to = datetime.today()
    frm = to - timedelta(days=int(years * 365.25))
    f, t = frm.strftime("%Y-%m-%d"), to.strftime("%Y-%m-%d")
    cost = bt.CostModel(kind="equity", stcg=0.0)          # accumulation phase: no realized STCG

    print("loading NIFTYBEES ONE_DAY ...")
    nb = client.fetch_ohlcv("NIFTYBEES", "ONE_DAY", f, t)
    weekly = 1000.0
    n_weeks = len({_isoweek(r[0]) for r in nb})
    total = weekly * n_weeks

    def d(ts):
        return datetime.fromtimestamp(ts / 1000).date()

    # weekly DCA + deferred (same buys, but the full `total` is committed as cash on day 0)
    units = spent = 0.0
    buys, dca_series, dca_contribs, deferred_series = [], [], [], []
    seen = set()
    for row in nb:
        ts, px = row[0], row[4]
        c = 0.0
        if _isoweek(ts) not in seen:
            seen.add(_isoweek(ts))
            units += (weekly - cost.buy_cost(weekly)) / px
            spent += weekly
            c = weekly
            buys.append((ts, px, (weekly - cost.buy_cost(weekly)) / px, weekly))
        dca_series.append((ts, units * px))
        dca_contribs.append(c)
        deferred_series.append((ts, units * px + (total - spent)))   # + idle cash reserve
    dca_cf = [(d(ts), -weekly) for ts, *_ in buys] + [(d(nb[-1][0]), dca_series[-1][1])]

    # lump-sum: whole `total` on day 0
    px0 = nb[0][4]
    u0 = (total - cost.buy_cost(total)) / px0
    lump_series = [(r[0], u0 * r[4]) for r in nb]

    # equal-weight B&H of the 10 cached stocks (diversified alternative)
    stocks = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
              "ITC", "LT", "SBIN", "BHARTIARTL", "MARUTI"]
    per = total / len(stocks)
    prices, ew_units = {}, {}
    for s in stocks:
        prices[s] = client.fetch_ohlcv(s, "ONE_DAY", f, t)
        ew_units[s] = (per - cost.buy_cost(per)) / prices[s][0][4]
    ew_series = [(r[0], sum(ew_units[s] * _asof(prices[s], r[0]) for s in stocks)) for r in nb]

    day0, last = d(nb[0][0]), d(nb[-1][0])
    zero = [0.0] * len(nb)
    rows = [
        ("weekly DCA (NIFTYBEES)", dca_series, dca_contribs, dca_cf),
        ("lump-sum day-0 (NIFTYBEES)", lump_series, zero, [(day0, -total), (last, lump_series[-1][1])]),
        ("deferred: cash, deploy wkly", deferred_series, zero, [(day0, -total), (last, deferred_series[-1][1])]),
        ("equal-weight B&H (10)", ew_series, zero, [(day0, -total), (last, ew_series[-1][1])]),
    ]

    print(f"\nHorizon {frm:%Y-%m-%d} -> {to:%Y-%m-%d}: {n_weeks} weekly buys, "
          f"Rs{weekly:.0f}/wk = Rs{total:,.0f}\n")
    print(f"  {'strategy':28s} {'final value':>13s} {'XIRR':>7s} {'Sharpe':>7s} {'max DD':>8s}")
    for name, series, contribs, cf in rows:
        print(f"  {name:28s} {series[-1][1]:>13,.0f} {xirr(cf):>+6.1%} "
              f"{_sharpe(series, contribs):>+7.2f} {_maxdd(series):>7.1%}")
    print("  XIRR is money-weighted on actual cashflow dates -> comparable across timelines.")
    print("  timing effect  = DCA vs lump-sum XIRR (near-equal -> averaging is a wash here)")
    print("  exposure effect = lump-sum vs deferred XIRR (both hold Rs%s day-0; cash drag costs the gap)"
          % f"{total:,.0f}")

    _plot(rows, os.path.join(ROOT, "results", "dca_comparison.png"))
    _rca_2025(nb, buys, cost)


def _plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    show = [r for r in rows if "deferred" not in r[0]]        # the 3 the user asked to see
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    for name, series, _c, _cf in show:
        xs = [datetime.fromtimestamp(ts / 1000) for ts, _ in series]
        eq = np.array([v for _, v in series])
        ax1.plot(xs, eq, linewidth=1, label=name)
        peak = np.maximum.accumulate(eq)
        ax2.fill_between(xs, (eq - peak) / peak * 100, 0, alpha=0.35)
        ax2.plot(xs, (eq - peak) / peak * 100, linewidth=0.8, label=name)
    ax1.set_ylabel("portfolio value (Rs)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("drawdown (%)")
    ax2.grid(alpha=0.3)
    fig.suptitle("Weekly DCA vs lump-sum vs equal-weight B&H (NIFTYBEES, 8y)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"-> {path}")


def _asof(series, ts):
    from bisect import bisect_right
    i = bisect_right([r[0] for r in series], ts) - 1
    return series[i][4] if i >= 0 else series[0][4]


def _rca_2025(nb, buys, cost):
    y25 = [r for r in nb if datetime.fromtimestamp(r[0] / 1000).year == 2025]
    if len(y25) < 20:
        print("\n(no 2025 data for the drawdown analysis)")
        return
    # largest peak -> subsequent-trough decline within 2025
    peak_px, peak_run = -1e18, 0
    dd, pi, ti = 0.0, 0, 0
    for i, r in enumerate(y25):
        if r[4] > peak_px:
            peak_px, peak_run = r[4], i
        if r[4] / peak_px - 1 < dd:
            dd, pi, ti = r[4] / peak_px - 1, peak_run, i
    peak_px, trough_px = y25[pi][4], y25[ti][4]
    peak_d = datetime.fromtimestamp(y25[pi][0] / 1000).date()
    trough_d = datetime.fromtimestamp(y25[ti][0] / 1000).date()
    print(f"\n2025 drawdown (worst peak->trough): Rs{peak_px:.1f} ({peak_d}) -> "
          f"Rs{trough_px:.1f} ({trough_d}) = {dd:+.1%}")

    dip = [(px, u, spend) for ts, px, u, spend in buys if y25[pi][0] <= ts <= y25[ti][0]]
    if not dip:
        print("  (no DCA buys landed inside the peak->trough window)")
        return
    u_dip = sum(u for _, u, _ in dip)
    spent = sum(s for _, _, s in dip)
    avg_cost = spent / u_dip
    end_px = nb[-1][4]
    # what the same rupees would have bought at the pre-drawdown peak (lump-sum's timing risk)
    u_at_peak = (spent - cost.buy_cost(spent)) / peak_px
    print(f"  DCA made {len(dip)} buys through the fall: Rs{spent:,.0f} for {u_dip:.1f} units "
          f"at avg Rs{avg_cost:.1f} ({avg_cost/peak_px-1:+.1%} vs the peak)")
    print(f"  averaging down bought {u_dip/u_at_peak-1:+.1%} more units than the same rupees "
          f"at the peak price")
    print(f"  those dip units are worth Rs{u_dip*end_px:,.0f} today "
          f"({end_px/avg_cost-1:+.0%} vs cost; {end_px/peak_px-1:+.0%} had you bought at the peak)")


# --- self-test (no network) ----------------------------------------------

def _selftest():
    # Monday 2025-08-15 is a holiday (Independence Day) -> buy rolls to next trading day
    assert not is_trading_day(date(2025, 8, 15))                 # holiday (Fri)
    assert not is_trading_day(date(2025, 8, 16))                 # Sat
    d = scheduled_buy_date(date(2025, 8, 11), weekday=4)         # that week's Friday = holiday
    assert d == date(2025, 8, 18), d                             # rolls to Mon 18th (17th=Sun)
    # normal Monday is itself
    assert scheduled_buy_date(date(2025, 9, 8), weekday=0) == date(2025, 9, 8)
    # DCA buys more units when price is low
    px_hi, px_lo = 100.0, 50.0
    import backtest as bt
    c = bt.CostModel(kind="equity", stcg=0.0)
    u_hi = (1000 - c.buy_cost(1000)) / px_hi
    u_lo = (1000 - c.buy_cost(1000)) / px_lo
    assert u_lo > u_hi * 1.9
    # XIRR: -100 today, +110 in one year -> ~10%
    assert abs(xirr([(date(2025, 1, 1), -100.0), (date(2026, 1, 1), 110.0)]) - 0.10) < 1e-3
    # two equal contributions, doubles at end -> XIRR positive and sane
    r = xirr([(date(2020, 1, 1), -100.0), (date(2021, 1, 1), -100.0), (date(2022, 1, 1), 260.0)])
    assert 0.18 < r < 0.19, r        # closed-form: x^2+x-2.6=0 -> r=18.8%
    print("selftest ok: calendar roll-forward + rupee-cost-averaging + XIRR behave")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="run one cycle and exit (cron mode)")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    load_env()

    if args.selftest:
        _selftest()
        return
    if args.backtest:
        backtest()
        return

    agent = DcaAgent(DcaConfig())
    if args.once:
        agent.run_once()
    else:
        agent.loop()


if __name__ == "__main__":
    main()
