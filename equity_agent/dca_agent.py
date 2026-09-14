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


def _env(k, default):
    # read at construction, not import, so values that only live in .env (loaded in main) apply
    return field(default_factory=lambda: os.environ.get(k, default))


@dataclass
class DcaConfig:
    dry_run: bool = field(default_factory=lambda: _envbool("DRY_RUN", True))
    # lump: deploy all available cash now, then buy only newly-detected cash (XIRR says cash
    # drag costs ~5%/yr, so this is the default). schedule: fixed weekly amount.
    deploy_mode: str = _env("DEPLOY_MODE", "lump")
    instrument: str = _env("DCA_INSTRUMENT", "NIFTYBEES")
    basket: tuple = ()                      # e.g. ("RELIANCE","TCS","HDFCBANK"); overrides instrument
    amount: float = field(default_factory=lambda: float(os.environ.get("DCA_AMOUNT", 1000)))
    cadence_weekday: int = 0               # 0=Mon
    rebalance_threshold: float = 0.05      # rebalance if a weight drifts >5% absolute
    starting_capital: float = field(default_factory=lambda: float(os.environ.get("DCA_START", 50000)))
    pause_life: float = 0.6                # equity/start below -> pause buying
    resume_life: float = 0.7
    poll_seconds: int = 3600
    db_path: str = _env("DCA_DB", os.path.join(ROOT, "equity_dca.db"))
    webhook: str = _env("N8N_WEBHOOK_URL", "")
    # Paper modes only: pretend the account holds this much cash so an unfunded soak still
    # exercises the buy path against live prices. Ignored whenever dry_run is False.
    paper_cash: str = _env("PAPER_CASH", "")

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


def schedule_decision(today, weekday, last_buy_week, holidays=NSE_HOLIDAYS):
    """Catch-up aware: ('done', d) if this ISO week already bought, ('wait', d) before the
    scheduled day, else ('buy', d) -- including days after a missed scheduled day, so a
    laptop that was off on Monday still buys on Tuesday. Never buys twice in one week, and a
    fully missed week is skipped rather than doubled up."""
    buy_date = scheduled_buy_date(today, weekday, holidays)
    if last_buy_week == today.strftime("%G-W%V"):
        return "done", buy_date
    if today < buy_date:
        return "wait", buy_date
    return "buy", buy_date


def is_market_open(now, holidays=NSE_HOLIDAYS):
    """NSE cash session: trading day, 09:15-15:30 IST."""
    from datetime import time as _t
    return is_trading_day(now.date(), holidays) and _t(9, 15) <= now.time() <= _t(15, 30)


def next_market_open(now, holidays=NSE_HOLIDAYS):
    """When the NSE cash session next opens (`now` itself if it's open right now)."""
    from datetime import time as _t
    if is_market_open(now, holidays):
        return now
    d = now.date()
    if not (is_trading_day(d, holidays) and now.time() < _t(9, 15)):
        d = next_trading_day(d + timedelta(days=1), holidays)
    return datetime.combine(d, _t(9, 15), tzinfo=IST)


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
        paper = self.cfg.dry_run and self.cfg.paper_cash
        cash = float(self.cfg.paper_cash) if paper else self.client.funds()
        holdings = {h["symbol"]: h for h in self.client.holdings()}
        held_value = sum(h["value"] for h in holdings.values())
        return cash + held_value, cash, holdings

    def already_bought_this_week(self, today):
        return self.db.get_flag("last_buy_week") == today.strftime("%G-W%V")

    # lump mode: in paper (dry_run) the account balance never drops, so we simulate the
    # cash leaving with an offset -> "new cash" detection works identically to live.
    def effective_cash(self, cash):
        return cash - float(self.db.get_flag("sim_cash_offset") or 0.0)

    def record_deployed(self, net_amount):
        if self.cfg.dry_run:
            offset = float(self.db.get_flag("sim_cash_offset") or 0.0)
            self.db.set_flag("sim_cash_offset", offset + net_amount)

    def plan_orders(self, deploy_amount, holdings, port_value, reason="dca"):
        """Return [(symbol, side, qty, ltp, amount, reason)] to deploy `deploy_amount` across
        targets, plus threshold-triggered rebalances when a holding drifts past the band."""
        targets = self.cfg.targets
        invest = port_value + deploy_amount
        ltps = {s: self.client.ltp(s) for s in targets}
        cur_val = {s: holdings.get(_ts(s), {}).get("value", 0.0) for s in targets}
        drifted = any(abs(cur_val[s] / port_value - targets[s]) > self.cfg.rebalance_threshold
                      for s in targets) if port_value > 0 else False

        orders = []
        for s in targets:
            if drifted and len(targets) > 1:
                delta = targets[s] * invest - cur_val[s]  # rebalance toward target
                r = "rebalance+" + reason
            else:
                delta = deploy_amount * targets[s]        # deploy this run's cash
                r = reason
            qty = int(delta // ltps[s])
            if qty >= 1:
                orders.append((s, "BUY", qty, ltps[s], qty * ltps[s], r))
            elif qty <= -1:
                orders.append((s, "SELL", -qty, ltps[s], -qty * ltps[s], r))
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
        now = datetime.now(IST)
        today = now.date()
        equity, cash, holdings = self.snapshot()
        life, paused = self.risk.update_pause(equity)
        self.db.snapshot(equity, cash, life, paused)
        self.log(f"equity=Rs{equity:.0f} cash=Rs{cash:.0f} life={life:.0%} paused={paused} "
                 f"mode={self.cfg.deploy_mode} dry_run={self.cfg.dry_run}")

        base = {"ts": now.isoformat(), "date": today.isoformat(), "mode": self.cfg.deploy_mode,
                "dry_run": self.cfg.dry_run, "equity": round(equity, 2), "cash": round(cash, 2),
                "life": round(life, 4), "paused": paused}
        if paused:
            self.log("life below floor -> buying paused")
            self.alert({**base, "event": "paused",
                        "note": f"life {life:.1%} below floor {self.cfg.pause_life:.0%}"})
            return
        if not is_market_open(now):
            self.log("market closed -> no action")
            return

        catchup = False
        if self.cfg.deploy_mode == "lump":
            deploy = self.effective_cash(cash)
            if deploy < 1:
                self.log("no new cash to deploy")
                self.alert({**base, "event": "no_cash", "note": "no new cash detected",
                            "deployable": round(deploy, 2)})
                return
            orders = self.plan_orders(deploy, holdings, equity, reason="lump")
        else:                                             # schedule: fixed weekly amount
            action, buy_date = schedule_decision(
                today, self.cfg.cadence_weekday, self.db.get_flag("last_buy_week"))
            if action == "done":
                self.log("already bought this week")
                return
            if action == "wait":
                self.log(f"not yet (scheduled buy {buy_date.isoformat()})")
                return
            catchup = today > buy_date
            if catchup:
                self.log(f"catch-up: scheduled buy {buy_date.isoformat()} was missed, buying now")
            avail = self.effective_cash(cash)
            if avail < self.cfg.amount:
                self.log(f"insufficient cash (Rs{avail:.0f} < Rs{self.cfg.amount:.0f}) -> skip")
                self.alert({**base, "event": "no_cash", "note": "insufficient cash for scheduled buy",
                            "required": round(self.cfg.amount, 2), "deployable": round(avail, 2)})
                return
            orders = self.plan_orders(self.cfg.amount, holdings, equity,
                                      reason="dca-catchup" if catchup else "dca")

        if not orders:
            self.log("nothing deployable this run")
            self.alert({**base, "event": "no_cash", "note": "cash below one tradeable unit",
                        "deployable": round(cash, 2)})
            return
        placed = [self.execute(order, today) for order in orders]
        net = sum(o[4] if o[1] == "BUY" else -o[4] for o in orders)  # net cash deployed
        self.record_deployed(net)                         # paper only: simulate cash leaving
        if self.cfg.deploy_mode == "schedule":
            self.db.set_flag("last_buy_week", today.strftime("%G-W%V"))
        self.alert({**base, "event": "buy", "catchup": catchup, "orders": placed,
                    "net_deployed": round(net, 2)})

    def alert(self, payload):
        if not self.cfg.webhook:
            self.log("no N8N_WEBHOOK_URL set -> skipping alert")
            return
        # buys always notify; paused/no_cash/error repeat every hourly cycle, so send once a day
        event = payload.get("event")
        key = f"alert_{event}"
        if event != "buy" and self.db.get_flag(key) == payload.get("date"):
            self.log(f"{event} alert already sent today -> suppressed")
            return
        try:
            req = urllib.request.Request(
                self.cfg.webhook, data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10).read()
            if event != "buy":
                self.db.set_flag(key, payload.get("date"))
            self.log("alert sent")
        except Exception as e:
            self.log(f"alert failed: {e}")

    def loop(self):
        self.log(f"DCA agent up. mode={self.cfg.deploy_mode} instrument={self.cfg.instrument} "
                 f"basket={self.cfg.basket} amount=Rs{self.cfg.amount:.0f} dry_run={self.cfg.dry_run}")
        while True:
            try:
                self.run_once()
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                self.log("cycle error (will re-login next cycle):\n" + tb)
                self.client._login_date = None    # force fresh login next cycle (token/net drop)
                self.alert({"ts": datetime.now(IST).isoformat(),
                            "date": datetime.now(IST).date().isoformat(),
                            "mode": self.cfg.deploy_mode, "dry_run": self.cfg.dry_run,
                            "event": "error", "error": f"{type(e).__name__}: {e}",
                            "traceback": tb[-800:]})
            time.sleep(self.cfg.poll_seconds)

    # --- operator view: everything without reading logs ---
    def status_data(self):
        """One snapshot of everything an operator wants; rendered as text (--status) or HTML (--serve)."""
        cfg, now = self.cfg, datetime.now(IST)
        d = {"now": now, "mode": cfg.deploy_mode, "amount": cfg.amount, "dry_run": cfg.dry_run,
             "instrument": cfg.instrument if not cfg.basket else ", ".join(cfg.basket),
             "db": cfg.db_path, "start": cfg.starting_capital, "weekday": cfg.cadence_weekday,
             "pause_life": cfg.pause_life, "resume_life": cfg.resume_life,
             "paper_cash": float(cfg.paper_cash) if cfg.dry_run and cfg.paper_cash else None,
             "market_open": is_market_open(now), "next_open": next_market_open(now),
             "stale_after_min": 2 * cfg.poll_seconds / 60 + 10}

        row = self.db.c.execute(
            "select ts, equity, cash, life, paused from equity order by ts desc limit 1").fetchone()
        d["last"] = None
        if row:
            ts, eq, cash, life, paused = row
            d["last"] = {"at": datetime.fromtimestamp(ts, IST), "age_min": (time.time() - ts) / 60,
                         "equity": eq, "cash": cash, "life": life, "paused": bool(paused)}

        if cfg.deploy_mode == "schedule":
            action, sched = schedule_decision(now.date(), cfg.cadence_weekday,
                                              self.db.get_flag("last_buy_week"))
            if action == "done":
                d["next"] = ("done", scheduled_buy_date(now.date() + timedelta(days=7),
                                                        cfg.cadence_weekday))
            elif action == "wait":
                d["next"] = ("wait", sched)
            else:
                d["next"] = ("catchup" if now.date() > sched else "due", sched)
        else:
            d["next"] = ("lump", None)

        d["holdings"], d["real_cash"], d["holdings_error"] = [], None, None
        try:
            self.client.ensure_session()
            d["holdings"] = self.client.holdings()
            d["real_cash"] = self.client.funds()
        except Exception as e:
            d["holdings_error"] = f"{type(e).__name__}: {e}"

        cols = ("date", "symbol", "side", "qty", "ltp", "amount", "dry", "status", "reason")
        d["orders"] = [dict(zip(cols, r)) for r in self.db.c.execute(
            "select run_date, symbol, side, qty, ltp, amount, dry, status, reason "
            "from orders order by ts desc limit 10").fetchall()]
        return d

    def status(self):
        d = self.status_data()
        print(f"DCA agent status @ {d['now']:%Y-%m-%d %H:%M:%S} IST")
        print(f"  mode={d['mode']}  instrument={d['instrument']}  amount=Rs{d['amount']:.0f}  "
              f"dry_run={d['dry_run']}  db={d['db']}")
        if d["paper_cash"] is not None:
            print(f"  PAPER_CASH=Rs{d['paper_cash']:,.0f} -> cash below is SIMULATED, not your account")
        last = d["last"]
        if last:
            print(f"\n  last run : {last['at']:%Y-%m-%d %H:%M:%S} IST ({last['age_min']:.0f} min ago)")
            print(f"  life     : {last['life']:.1%} of Rs{d['start']:.0f}  [pause<={d['pause_life']:.0%} "
                  f"resume>={d['resume_life']:.0%}]  {'PAUSED' if last['paused'] else 'active'}")
            print(f"  equity   : Rs{last['equity']:,.2f}   cash: Rs{last['cash']:,.2f}")
        else:
            print("\n  last run : never (no equity snapshots yet)")
        kind, when = d["next"]
        text = {"done": f"{when} (this week already bought)", "wait": f"{when} (scheduled)",
                "due": "next market session (due today)",
                "catchup": f"next market session (CATCH-UP, {when} was missed)",
                "lump": "next open market session with new cash"}[kind]
        print(f"  next buy : {text}; market open now: {d['market_open']}")
        if d["holdings_error"]:
            print(f"\n  holdings : unavailable ({d['holdings_error']})")
        else:
            print(f"\n  holdings ({len(d['holdings'])}):")
            for h in d["holdings"]:
                print(f"    {h['symbol']:16s} qty={h['qty']:<8g} ltp={h['ltp']:>9.2f} "
                      f"value=Rs{h['value']:>10,.2f}")
            print(f"    {'CASH':16s} {'':8} {'':9}  value=Rs{d['real_cash']:>10,.2f}")
        print(f"\n  recent orders ({len(d['orders'])}):")
        if not d["orders"]:
            print("    (none)")
        for o in d["orders"]:
            print(f"    {o['date']}  {o['side']:4s} {o['qty']:>6g} {o['symbol']:14s} @{o['ltp']:>9.2f}  "
                  f"Rs{o['amount']:>9,.2f}  {'DRY' if o['dry'] else 'LIVE'}/{o['status']:<10s} {o['reason']}")


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

def _inr(x, dec=2):
    """Indian digit grouping: 123456.7 -> Rs 1,23,456.70."""
    whole, _, frac = f"{abs(x):.{dec}f}".partition(".")
    head, tail = whole[:-3], whole[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    whole = ",".join(([head] if head else []) + groups + [tail])
    return ("-" if x < 0 else "") + "₹" + whole + (f".{frac}" if frac else "")


def _clock(t):
    return t.strftime("%I:%M %p").lstrip("0")


def _ago(minutes):
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes:.0f} min ago"
    if minutes < 48 * 60:
        return f"{minutes / 60:.0f} h ago"
    return f"{minutes / 1440:.0f} days ago"


PAGE_CSS = """
:root{--bg:#f4f5f7;--card:#fff;--ink:#14161a;--muted:#6b7280;--line:#e5e7eb;--track:#eceef1;
--ok:#15803d;--ok-bg:#dcfce7;--warn:#b45309;--warn-bg:#fef3c7;--bad:#b91c1c;--bad-bg:#fee2e2;
--accent:#2563eb}
@media (prefers-color-scheme:dark){:root{--bg:#0b0c0f;--card:#16181d;--ink:#eef0f3;--muted:#9aa1ad;
--line:#262a31;--track:#23272e;--ok:#4ade80;--ok-bg:#12351f;--warn:#fbbf24;--warn-bg:#3a2c0b;
--bad:#f87171;--bad-bg:#3b1414;--accent:#60a5fa}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.4 -apple-system,BlinkMacSystemFont,
"Segoe UI",Roboto,sans-serif;-webkit-font-smoothing:antialiased}
main{max-width:520px;margin:0 auto;padding:18px 16px 32px}
header{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;margin:4px 2px 16px}
h1{font-size:24px;margin:0;letter-spacing:-.02em}
.sub{color:var(--muted);font-size:14px;margin-top:2px}
.pill{font-size:13px;font-weight:600;padding:5px 11px;border-radius:999px;white-space:nowrap}
.ok{color:var(--ok);background:var(--ok-bg)}.warn{color:var(--warn);background:var(--warn-bg)}
.bad{color:var(--bad);background:var(--bad-bg)}
.banner{border-radius:12px;padding:10px 14px;font-size:14px;margin-bottom:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:16px;margin-bottom:12px}
.label{color:var(--muted);font-size:13px;font-weight:500;text-transform:uppercase;letter-spacing:.04em}
.big{font-size:32px;font-weight:700;letter-spacing:-.02em;margin:4px 0 2px;font-variant-numeric:tabular-nums}
.mid{font-size:20px;font-weight:650;margin:4px 0 2px}
.muted{color:var(--muted);font-size:14px}
.row{display:flex;justify-content:space-between;align-items:baseline;gap:12px;padding:10px 0;
border-top:1px solid var(--line)}
.row:first-of-type{border-top:0}
.num{font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
.meta{color:var(--muted);font-size:13px;margin-top:12px;padding-top:10px;border-top:1px solid var(--line)}
.meter{height:8px;background:var(--track);border-radius:99px;overflow:hidden;margin:12px 0 6px;position:relative}
.meter span{display:block;height:100%;border-radius:99px}
.tag{font-size:12px;font-weight:600;padding:2px 8px;border-radius:6px;margin-left:6px}
.empty{color:var(--muted);font-size:14px;padding:6px 0 2px}
footer{color:var(--muted);font-size:13px;text-align:center;margin-top:18px}
"""


def render_page(d):
    """Phone dashboard for status_data(). Everything dynamic is escaped."""
    import html

    e = html.escape
    now, last = d["now"], d["last"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    if not last:
        pill = ("warn", "No runs yet")
    elif last["age_min"] > d["stale_after_min"]:
        pill = ("bad", "Not checking in")
    elif last["paused"]:
        pill = ("bad", "Paused")
    else:
        pill = ("ok", "Active")
    cadence = (f"{_inr(d['amount'], 0)} every {days[d['weekday']]}" if d["mode"] == "schedule"
               else "buys whenever new cash arrives")
    parts = [f"<header><div><h1>DCA agent</h1><div class=sub>{e(d['instrument'])} · {cadence}"
             f"</div></div><span class='pill {pill[0]}'>{pill[1]}</span></header>"]

    if last and last["age_min"] > d["stale_after_min"]:
        parts.append(f"<div class='banner bad'>Last check was {_ago(last['age_min'])}. "
                     "Is the laptop awake and Docker Desktop running?</div>")
    if d["dry_run"]:
        sim = (f" Cash is simulated ({_inr(d['paper_cash'], 0)})." if d["paper_cash"] is not None else "")
        parts.append(f"<div class='banner warn'><b>Paper mode</b> — nothing is really bought.{sim}</div>")

    # portfolio + life meter
    if last:
        life = last["life"]
        tone = "ok" if life >= d["resume_life"] else ("warn" if life > d["pause_life"] else "bad")
        cash_note = " (simulated)" if d["paper_cash"] is not None else ""
        parts.append(
            f"<section class=card><div class=label>Portfolio value</div>"
            f"<div class=big>{_inr(last['equity'])}</div>"
            f"<div class=muted>Cash {_inr(last['cash'])}{cash_note}</div>"
            f"<div class=meter><span style='width:{min(life, 1) * 100:.0f}%;background:var(--{tone})'>"
            f"</span></div><div class=muted>{life:.0%} of the {_inr(d['start'], 0)} starting amount · "
            f"buying pauses below {d['pause_life']:.0%}</div></section>")

    # next buy + market
    kind, when = d["next"]
    nxt_open = d["next_open"]
    if kind == "lump":
        headline, detail = "When new cash arrives", "Checked every hour while the market is open."
    elif kind in ("due", "catchup"):
        # say when it will actually buy, not just "today" -- at 11 PM that's tomorrow morning
        if d["market_open"]:
            headline = "Now"
        elif nxt_open.date() == now.date():
            headline = f"Today, {_clock(nxt_open)}"
        else:
            headline = f"{nxt_open:%a %d %b}, {_clock(nxt_open)}"
        missed = kind == "catchup" or nxt_open.date() != now.date()
        detail = (f"{when:%A}'s buy didn't happen in market hours, so it catches up then."
                  if missed else "Buys at the first check once the market opens.")
    else:
        headline = f"{when:%a %d %b}"
        detail = "This week's buy is done." if kind == "done" else "Scheduled."
    market = ("Market open now" if d["market_open"] else
              f"Market closed · opens {nxt_open:%a} {_clock(nxt_open)}")
    checked = (f"Last check {_ago(last['age_min'])} ({_clock(last['at'])})" if last else "No checks yet")
    parts.append(
        f"<section class=card><div class=label>Next buy</div><div class=mid>{headline}</div>"
        f"<div class=muted>{detail}</div><div class=meta>{market}<br>{checked}</div></section>")

    # real account
    rows = []
    if d["holdings_error"]:
        rows.append(f"<div class=empty>Couldn't reach Angel One: {e(d['holdings_error'])}</div>")
    else:
        for h in sorted(d["holdings"], key=lambda h: -h["value"]):
            name = e(h["symbol"].removesuffix("-EQ"))
            rows.append(f"<div class=row><div><div>{name}</div><div class=muted>{h['qty']:g} × "
                        f"{_inr(h['ltp'])}</div></div><div class=num>{_inr(h['value'])}</div></div>")
        rows.append(f"<div class=row><div>Cash</div><div class=num>{_inr(d['real_cash'] or 0)}</div></div>")
    parts.append("<section class=card><div class=label>Your Angel One account (real)</div>"
                 + "".join(rows) + "</section>")

    # orders
    rows = []
    for o in d["orders"]:
        st = o["status"] or ""
        badge = (("warn", "Paper") if o["dry"] else ("ok", "Live")) if not st.startswith("ERROR") \
            else ("bad", "Failed")
        catch = "<span class='tag ok'>catch-up</span>" if "catchup" in (o["reason"] or "") else ""
        day = datetime.strptime(o["date"], "%Y-%m-%d").strftime("%a %d %b")
        rows.append(f"<div class=row><div><div>{e(o['side'].title())} {o['qty']:g} "
                    f"{e(o['symbol'])}<span class='tag {badge[0]}'>{badge[1]}</span>{catch}</div>"
                    f"<div class=muted>{day} · at {_inr(o['ltp'])}</div></div>"
                    f"<div class=num>{_inr(o['amount'])}</div></div>")
    if not rows:
        rows.append("<div class=empty>No orders yet. The first buy will show up here.</div>")
    parts.append("<section class=card><div class=label>Recent orders</div>" + "".join(rows) + "</section>")

    parts.append(f"<footer>Updated {_clock(now)} · refreshes every minute</footer>")
    return ("<!doctype html><html lang=en><meta charset=utf-8>"
            "<meta name=viewport content='width=device-width,initial-scale=1,viewport-fit=cover'>"
            "<meta name=color-scheme content='light dark'><meta http-equiv=refresh content=60>"
            "<meta name=apple-mobile-web-app-capable content=yes>"
            "<meta name=apple-mobile-web-app-title content='DCA agent'>"
            f"<title>DCA agent</title><style>{PAGE_CSS}</style><main>{''.join(parts)}</main></html>")


def serve(agent, port=8080):
    """Read-only phone dashboard of --status. Publish it on 127.0.0.1 only and reach it from
    a phone over Tailscale -- it shows holdings, so never expose it on the open LAN."""
    import html
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path not in ("/", "/status"):
                self.send_error(404)
                return
            try:
                body = render_page(agent.status_data()).encode()
            except Exception as e:
                body = (f"<!doctype html><meta name=viewport content='width=device-width'>"
                        f"<p style='font:16px system-ui;padding:20px'>Status page error: "
                        f"{html.escape(f'{type(e).__name__}: {e}')}</p>").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    agent.log(f"status page on :{port} (refreshes every 60s)")
    # single-threaded on purpose: the sqlite connection belongs to this thread
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()


def print_payloads():
    """Exact JSON shapes POSTed to N8N_WEBHOOK_URL. Switch your n8n node on `event`.
    Every payload carries the same base keys; `event` decides the extras."""
    base = {"ts": "2026-09-14T10:00:03.412000+05:30", "date": "2026-09-14",
            "mode": "schedule", "dry_run": True, "equity": 8123.45, "cash": 3500.0,
            "life": 1.0154, "paused": False}
    samples = {
        "buy": {**base, "event": "buy", "catchup": False, "net_deployed": 486.09,
                "orders": [{"symbol": "NIFTYBEES", "side": "BUY", "qty": 1,
                            "ltp": 486.09, "amount": 486.09, "status": "DRY"}]},
        "no_cash": {**base, "cash": 120.0, "event": "no_cash",
                    "note": "insufficient cash for scheduled buy",
                    "required": 500.0, "deployable": 120.0},
        "paused": {**base, "equity": 4100.0, "life": 0.5125, "paused": True,
                   "event": "paused", "note": "life 51.2% below floor 60%"},
        "error": {"ts": "2026-09-14T11:00:01.882000+05:30", "date": "2026-09-14",
                  "mode": "schedule", "dry_run": True, "event": "error",
                  "error": "ConnectionError: HTTPSConnectionPool(host='apiconnect.angelone.in')",
                  "traceback": "Traceback (most recent call last):\n  ...last 800 chars..."},
    }
    print("POST " + "$N8N_WEBHOOK_URL" + "   Content-Type: application/json\n")
    print("Base keys on every event: ts, date, mode, dry_run, event"
          "  (+ equity, cash, life, paused on all but `error`)\n")
    for name, body in samples.items():
        print(f"--- event = \"{name}\" " + "-" * (58 - len(name)))
        print(json.dumps(body, indent=2))
        print()
    print("n8n: Switch node on {{ $json.event }} -> buy | no_cash | paused | error")
    print("Note: `status` inside orders[] is \"DRY\" while DRY_RUN/--paper-live is on,")
    print("      \"PLACED\" once live, or \"ERROR:<msg>\" if the broker rejected it.")


def _selftest():
    # Monday 2025-08-15 is a holiday (Independence Day) -> buy rolls to next trading day
    assert not is_trading_day(date(2025, 8, 15))                 # holiday (Fri)
    assert not is_trading_day(date(2025, 8, 16))                 # Sat
    d = scheduled_buy_date(date(2025, 8, 11), weekday=4)         # that week's Friday = holiday
    assert d == date(2025, 8, 18), d                             # rolls to Mon 18th (17th=Sun)
    # normal Monday is itself
    assert scheduled_buy_date(date(2025, 9, 8), weekday=0) == date(2025, 9, 8)

    # catch-up: Mon 2026-09-14 scheduled; laptop off Monday -> Tuesday still buys, once
    mon, tue, sun = date(2026, 9, 14), date(2026, 9, 15), date(2026, 9, 13)
    wk = mon.strftime("%G-W%V")
    assert schedule_decision(mon, 0, None) == ("buy", mon)             # on time
    assert schedule_decision(tue, 0, "2026-W37") == ("buy", mon)       # missed Mon -> catch up
    assert schedule_decision(tue, 0, wk) == ("done", mon)              # already bought -> nothing
    # whole week missed: Sun still says "buy" for W37, but the market-open gate blocks it, so
    # W37 is skipped and Monday buys W38 exactly once -- never doubled up
    assert schedule_decision(sun, 0, None) == ("buy", date(2026, 9, 7))
    assert not is_market_open(datetime(2026, 9, 13, 11, 0, tzinfo=IST))
    assert schedule_decision(date(2026, 1, 26), 0, None) == ("wait", date(2026, 1, 27))  # Mon holiday
    assert schedule_decision(date(2026, 1, 27), 0, None) == ("buy", date(2026, 1, 27))
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

    # lump-mode cash detection: deploy once, then nothing until fresh cash appears
    class _Fake:
        def ensure_session(self): pass
    ag = DcaAgent(DcaConfig(db_path=":memory:", deploy_mode="lump", dry_run=True), client=_Fake())
    assert ag.effective_cash(1000) == 1000       # first run: all cash deployable
    ag.record_deployed(1000)                      # paper: simulate the cash leaving
    assert ag.effective_cash(1000) == 0           # same balance -> no new cash
    assert ag.effective_cash(1500) == 500         # a Rs500 deposit is detected
    live = DcaAgent(DcaConfig(db_path=":memory:", deploy_mode="lump", dry_run=False), client=_Fake())
    live.record_deployed(1000)                     # live: real cash moves, no sim offset
    assert live.effective_cash(1000) == 1000

    # dashboard: Indian grouping, next open, and every render branch without a broker
    assert _inr(123456.7) == "₹1,23,456.70" and _inr(500, 0) == "₹500" and _inr(0.62) == "₹0.62"
    mon_night = datetime(2026, 9, 14, 23, 0, tzinfo=IST)
    assert next_market_open(mon_night) == datetime(2026, 9, 15, 9, 15, tzinfo=IST)
    base = {"now": mon_night, "mode": "schedule", "amount": 500, "dry_run": True,
            "instrument": "NIFTYBEES", "db": ":memory:", "start": 8000, "weekday": 0,
            "pause_life": 0.6, "resume_life": 0.7, "paper_cash": 8000.0, "market_open": False,
            "next_open": next_market_open(mon_night), "stale_after_min": 130,
            "last": {"at": mon_night, "age_min": 5, "equity": 9334.67, "cash": 8000,
                     "life": 1.17, "paused": False},
            "holdings": [{"symbol": "<b>X-EQ", "qty": 5, "ltp": 14.97, "value": 74.85}],
            "real_cash": 0.62, "holdings_error": None,
            "orders": [{"date": "2026-09-15", "symbol": "NIFTYBEES", "side": "BUY", "qty": 1,
                        "ltp": 267.14, "amount": 267.14, "dry": 1, "status": "DRY",
                        "reason": "dca-catchup"}]}
    for kind, when in (("due", mon), ("catchup", mon), ("wait", mon), ("done", mon), ("lump", None)):
        page = render_page({**base, "next": (kind, when)})
        assert "<b>X" not in page and "&lt;b&gt;X" in page         # broker strings are escaped
    assert "Not checking in" in render_page({**base, "next": ("due", mon),
                                             "last": {**base["last"], "age_min": 500}})
    assert "No runs yet" in render_page({**base, "next": ("due", mon), "last": None})
    print("selftest ok: calendar + RCA + XIRR + lump cash-detection + dashboard behave")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="run one cycle and exit (cron mode)")
    ap.add_argument("--paper-live", action="store_true",
                    help="soak test: real schedule + live prices in real time, logs orders, "
                         "places nothing (forces DRY_RUN, meant to run for days/weeks)")
    ap.add_argument("--status", action="store_true",
                    help="holdings, last/next run, life meter, recent orders (no log reading)")
    ap.add_argument("--serve", action="store_true",
                    help="read-only status web page on :8080 (for checking from a phone)")
    ap.add_argument("--print-payloads", action="store_true",
                    help="print the exact webhook JSON shapes for building the n8n node")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    load_env()

    if args.selftest:
        _selftest()
        return
    if args.print_payloads:
        print_payloads()
        return
    if args.status:
        DcaAgent(DcaConfig()).status()
        return
    if args.serve:
        serve(DcaAgent(DcaConfig()))
        return
    if args.backtest:
        backtest()
        return

    if args.paper_live:
        cfg = DcaConfig()
        cfg.dry_run = True                          # paper-live NEVER places, whatever the env says
        agent = DcaAgent(cfg)
        agent.log("PAPER-LIVE soak: real schedule + live prices, logging intended orders, "
                  "placing NOTHING. Ctrl-C to stop.")
        agent.loop()
        return

    agent = DcaAgent(DcaConfig())
    if args.once:
        agent.run_once()
    else:
        agent.loop()


if __name__ == "__main__":
    main()
