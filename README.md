# crypto-agent

Autonomous swing-trading agent for CoinDCX (spot, BTC/USDT + ETH/USDT).
Paper-trading by default. Baseline strategy is RSI mean reversion; ML/LLM
signals layer on top of `Strategy.signal` later.

## Life meter (survival logic)

`life = equity / starting_capital`. Below `PAUSE_LIFE` (0.75) the agent stops
opening/closing positions; it resumes only above `RESUME_LIFE` (0.85). The gap
is hysteresis so it doesn't flap on noise. State persists in SQLite, so a
restart keeps the paused flag.

## TDS / swing design

India charges 1% TDS on every sell. The risk manager refuses a sell unless the
net move clears `tds_rate + 2*taker_fee + min_net_profit` (~3.5%). Combined
with `MAX_TRADES_PER_WEEK=4` and a 4h timeframe, this keeps it to a few
trades/week, not scalping.

## Files

| file | role |
|------|------|
| `config.py`   | all tunables, env-overridable |
| `strategy.py` | RSI + signal (the swappable brain) |
| `risk.py`     | life meter, sizing, TDS-aware sell gate |
| `broker.py`   | paper (SQLite ledger) / live (ccxt) execution |
| `db.py`       | SQLite: account, positions, trades, equity, flags |
| `agent.py`    | event loop |
| `test_agent.py` | assert-based checks, no network |

## Run

```
pip install -r requirements.txt
python test_agent.py            # sanity check
python agent.py --once          # one paper cycle
python agent.py                 # loop every POLL_SECONDS (default 1h)
```

Live (after weeks of paper): set `MODE=live`, `COINDCX_API_KEY`,
`COINDCX_API_SECRET`. Fund with USDT (convert INR -> USDT in the CoinDCX app
first). Verify `ccxt.coindcx` is the right id and that market orders behave as
`broker.py` assumes before trusting it.

## Inspect

```
sqlite3 agent.db "select ts,symbol,side,qty,price,tds,reason from trades order by ts desc limit 20"
sqlite3 agent.db "select datetime(ts,'unixepoch'),equity,life,paused from equity order by ts desc limit 20"
```

---

# equity_agent (Angel One / NSE, India)

A second arm on Indian equities (SmartAPI). It exists because the research below
killed the "beat the market with signals" thesis and pointed at a boring winner.

## What the research found

Two market surveys (`backtest.py --survey`, `equity_agent/survey.py`) swept
timeframes × assets × strategies (RSI mean-reversion, EMA crossover, Donchian
breakout), scored **out-of-sample** via walk-forward, net of real costs
(crypto: 0.24% fee + 1% TDS/sell; equity: STT + txn + SEBI + stamp + 18% GST +
20% STCG). Verdict in both markets: **single-asset technical signals do not beat
buy-and-hold after costs.** The only thing with a durable pulse was
**cross-sectional momentum** (rank the universe, hold the top few) — modest
(+2.6% out-of-sample alpha, Sharpe 0.84, no params to overfit).

So the live agent is a **DCA/accumulator**, not a signal trader.

## The XIRR finding (this shaped the design)

Comparing weekly DCA to lump-sum by *total return* is a trap — they run on
different capital timelines. Money-weighted **XIRR** on actual cashflow dates is
apples-to-apples. NIFTYBEES, 8y, ₹1,000/wk = ₹418k:

| strategy | final value | **XIRR** | Sharpe | max DD |
|----------|------------:|---------:|-------:|-------:|
| weekly DCA (NIFTYBEES)        | ₹639,845 | **+10.5%** | +0.73 | 30.3% |
| lump-sum day-0 (NIFTYBEES)    | ₹935,621 | **+10.6%** | +0.74 | 36.3% |
| deferred: cash, deploy weekly | ₹639,845 | **+5.5%**  | +0.73 | 14.2% |
| equal-weight B&H (10 stocks)  | ₹1,018,605 | **+11.8%** | +0.73 | 37.2% |

**Timing vs exposure decomposition** (what the `deferred` benchmark isolates):

- **Timing** = DCA vs lump-sum XIRR = 10.5% vs 10.6% ≈ **zero**. Averaging your
  entry price in earns nothing; the huge total-return gap was a timeline artifact.
- **Exposure** = lump-sum vs deferred XIRR = 10.6% vs 5.5% = **−5.1%/yr**. Both
  commit the full ₹418k on day 0; deferred just holds it as cash and dribbles it
  in. That idle-cash drag is the whole story.

Conclusion: **if you already hold the cash, deploy it — don't average it in.**
DCA only makes sense when the money genuinely arrives over time. DCA's one real
edge is a shallower drawdown (see `results/dca_comparison.png`), not more money.

## DEPLOY_MODE (default `lump`, because of the above)

- **`lump`** — deploy all available cash immediately, then buy only newly-detected
  cash (polls the balance, buys the delta). Matches the XIRR finding: idle cash
  costs ~5%/yr. Paper modes simulate the cash leaving via a `sim_cash_offset`
  flag, so "new cash" detection works identically to live.
- **`schedule`** — fixed ₹ every week (Mon 10:00 IST, configurable), for when
  income genuinely arrives over time.

## Safety

- `DRY_RUN=true` by default — logs the exact order, places nothing.
- `--paper-live` — real schedule + live prices in real time, logs intended orders
  to SQLite, **places nothing**; forces DRY_RUN. Built to run for weeks and prove
  it survives market days, holidays, token expiry (daily re-login), and network
  drops (re-login next cycle on error).
- Reuses the crypto **RiskManager** life-meter (pauses buying below the floor) and
  **DB** (flags + equity), plus an `orders` table for every intended/actual order.
- Market-calendar aware: skips weekends + `NSE_HOLIDAYS` (hardcoded, **update
  yearly**); a Monday holiday buys Tuesday. Buys only during 09:15–15:30 IST.
- n8n webhook alert per run (`N8N_WEBHOOK_URL`).

## Files & run

| file | role |
|------|------|
| `equity_agent/angel_client.py` | SmartAPI: login/session, scrip tokens, cached candles, quotes/holdings/funds/orders |
| `equity_agent/dca_agent.py`    | the live DCA agent + `--backtest` (DCA vs lump-sum vs B&H) |
| `equity_agent/survey.py`       | equity market survey (tf × stock × strategy) |
| `equity_agent/daily_rerun.py`  | properly-sized daily survey + cross-sectional momentum |
| `equity_agent/test_connection.py` | Angel login smoke test |

```
python equity_agent/dca_agent.py --selftest      # no network: calendar, XIRR, lump detection
python equity_agent/dca_agent.py --backtest       # XIRR table + drawdown PNG
python equity_agent/dca_agent.py --paper-live      # 2-week soak, places nothing
python equity_agent/dca_agent.py --once            # one cycle (cron)
```

Before ever setting `DRY_RUN=false`: set `DCA_START` to your real capital (the
life meter pauses if the account is far below it), add `N8N_WEBHOOK_URL`, and
refresh `NSE_HOLIDAYS` for the year. Credentials live in `.env`
(`ANGEL_API_KEY`, `ANGEL_CLIENT_CODE`, `ANGEL_MPIN`/`ANGEL_PIN`,
`ANGEL_TOTP_SECRET`) — gitignored.

## Checking on it: `--status`

Everything an operator needs without opening a log file:

```
docker compose exec dca python equity_agent/dca_agent.py --status   # container running
python equity_agent/dca_agent.py --status                           # plain local run
```

Prints config, **last run** (+ minutes ago), **life-meter state** vs the pause/resume
floors, equity/cash, **next scheduled buy** (including `CATCH-UP` when this week's day was
missed), **live holdings** (best-effort; degrades gracefully if the broker session fails),
and the **last 10 orders** from SQLite with their DRY/LIVE status. The container and a
plain local run use **different databases** (Docker volume vs `equity_dca.db` in the repo),
so check the one that's actually running.

## Catch-up (missed days are harmless)

In `schedule` mode the agent asks SQLite "did this ISO week's buy already happen?" on
every cycle, not "is today Monday?". So:

- Monday missed (laptop off / Docker stopped), now Tuesday during market hours → **buys**,
  logged as `dca-catchup`, webhook `"catchup": true`.
- Already bought this week → does nothing, no matter how often it runs.
- A whole week missed → that week is **skipped**, never doubled up next week.
- Monday is an NSE holiday → the scheduled day itself moves to Tuesday.

## Webhook payloads (`N8N_WEBHOOK_URL`)

`python equity_agent/dca_agent.py --print-payloads` prints the exact JSON. Every POST
carries `ts, date, mode, dry_run, event` (plus `equity, cash, life, paused` on all but
`error`). **Switch your n8n node on `event`:**

| `event` | when | event-specific keys |
|---------|------|---------------------|
| `buy` | orders were planned (and placed, if live) | `orders[]`, `net_deployed`, `catchup` |
| `no_cash` | nothing deployable: no new cash (lump), insufficient for the scheduled buy, or below one tradeable unit | `note`, `deployable`, `required` (schedule only) |
| `paused` | life meter below the floor — buying suspended | `note` |
| `error` | a cycle raised; agent re-logs-in next cycle | `error`, `traceback` (last 800 chars) — **no** equity/cash/life |

`orders[].status` is `"DRY"` under DRY_RUN/`--paper-live`, `"PLACED"` when live, or
`"ERROR:<msg>"` if the broker rejected it.

**Dedupe:** the loop runs hourly, so `paused` / `no_cash` / `error` are sent **at most once
per day** each (tracked in SQLite). `buy` always sends.

## Running locally with Docker (Windows)

No server needed: it runs on your laptop, and your **home IP is the one Angel already
accepts** (login + quotes verified from this machine). Docker only bundles the DCA agent —
`db.py`, `risk.py`, `equity_agent/` and four pinned packages — not the research code.

**Files:** `Dockerfile`, `docker-compose.yml`, `.dockerignore` (keeps `.env`, `data/`,
`*.db` out of the image).

**One-time setup**

1. Install Docker Desktop and turn on **Settings → General → Start Docker Desktop when you
   sign in**. Together with `restart: unless-stopped` that brings the agent back after a
   reboot on its own.
2. `.env` in the repo root holds the credentials (already there). Compose reads it as
   `KEY=VALUE` — no `export`, no spaces around `=`. Add `N8N_WEBHOOK_URL=...` for alerts.
3. Build and start:

```powershell
docker compose up -d --build
docker compose logs -f dca                                           # watch it
docker compose exec dca python equity_agent/dca_agent.py --status    # check on it
```

**Everyday**

```powershell
docker compose ps                  # is it up?
docker compose restart dca         # after editing .env
docker compose up -d --build       # after changing code
docker compose down                # stop (state is kept in volumes)
```

**What the compose file sets**

- Runs `--paper-live`: the persistent hourly loop, **forced DRY_RUN — it can't place an order.**
- Soak config: `DEPLOY_MODE=schedule`, `DCA_AMOUNT=500`, `DCA_START=8000`.
- `PAPER_CASH=8000`: the paper run pretends your account holds ₹8,000 cash. Without it the
  unfunded account (≈₹1 cash) pauses the life meter at 17% and never reaches the buy path.
  Holdings and prices are still real. `--status` labels simulated cash. **Delete this line
  once the account is funded.**
- State lives in named volumes: `dca-state` (SQLite) and `dca-data` (34 MB scrip master),
  so rebuilds and restarts don't lose "already bought this week". `docker compose down -v`
  wipes them.
- `TZ=Asia/Kolkata` for readable logs. The agent's own calendar and market-hours logic
  uses explicit IST either way.
- Logs rotate at 3 × 10 MB.

**Laptop sleep/off:** the container pauses with the machine. When it wakes, the hourly loop
resumes and catch-up buys this week's missed day if the market is open.

**Git Bash gotcha:** `docker ... -e SOME_PATH=/tmp/x` gets `/tmp` rewritten to a Windows
path. Use PowerShell, or prefix the command with `MSYS_NO_PATHCONV=1`.

**Before going live** (not needed for the paper soak): Angel requires a registered static
IP for API order placement, and home broadband IPs usually change. Check your SmartAPI app
settings before ever setting `DRY_RUN=false` on this machine.

### Soak status (checked 2026-09-14)

Verified inside the container: login ✓, live NIFTYBEES quote ✓, `PAPER_CASH` un-pauses the
life meter (117%) ✓, DRY buy written to SQLite ✓, a second run the same week does nothing ✓.
Still open: add `N8N_WEBHOOK_URL` if you want alerts (they're skipped and logged otherwise).
