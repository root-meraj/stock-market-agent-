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
python equity_agent/dca_agent.py --status
```

Prints config, **last run** (+ minutes ago), **life-meter state** vs the pause/resume
floors, equity/cash, **next scheduled buy**, **live holdings** (best-effort; degrades
gracefully if the broker session fails), and the **last 10 orders** from SQLite with
their DRY/LIVE status.

## Webhook payloads (`N8N_WEBHOOK_URL`)

`python equity_agent/dca_agent.py --print-payloads` prints the exact JSON. Every POST
carries `ts, date, mode, dry_run, event` (plus `equity, cash, life, paused` on all but
`error`). **Switch your n8n node on `event`:**

| `event` | when | event-specific keys |
|---------|------|---------------------|
| `buy` | orders were planned (and placed, if live) | `orders[]`, `net_deployed` |
| `no_cash` | nothing deployable: no new cash (lump), insufficient for the scheduled buy, or below one tradeable unit | `note`, `deployable`, `required` (schedule only) |
| `paused` | life meter below the floor — buying suspended | `note` |
| `error` | a cycle raised; agent re-logs-in next cycle | `error`, `traceback` (last 800 chars) — **no** equity/cash/life |

`orders[].status` is `"DRY"` under DRY_RUN/`--paper-live`, `"PLACED"` when live, or
`"ERROR:<msg>"` if the broker rejected it.

## Deploying to an Ubuntu VPS

**Copy** (the repo minus junk — `data/`, `results/`, `logs/`, `*.db`, `.env` are gitignored):

```bash
rsync -av --exclude data --exclude results --exclude logs --exclude '*.db' \
      --exclude .env --exclude __pycache__ ./ dca@VPS:/opt/crypto-agent/
```

**Install** on the server:

```bash
sudo adduser --system --group dca
sudo mkdir -p /opt/crypto-agent /var/log/dca-agent
sudo chown -R dca:dca /opt/crypto-agent /var/log/dca-agent
sudo apt update && sudo apt install -y python3-venv
sudo -u dca python3 -m venv /opt/crypto-agent/.venv
sudo -u dca /opt/crypto-agent/.venv/bin/pip install -r /opt/crypto-agent/requirements.txt
```

**Timezone** — the agent's calendar and market-hours gate are IST:

```bash
sudo timedatectl set-timezone Asia/Kolkata && timedatectl
```

(The unit also sets `TZ=Asia/Kolkata`, so it is correct even if the host isn't.)

**Recreate `.env`** on the server — never copy it over, retype it. systemd reads plain
`KEY=VALUE`, so **no `export`, no quotes, no spaces around `=`**:

```bash
sudo -u dca tee /opt/crypto-agent/.env >/dev/null <<'EOF'
ANGEL_API_KEY=xxxxxxxx
ANGEL_CLIENT_CODE=Axxxxxxx
ANGEL_PIN=1234
ANGEL_TOTP_SECRET=XXXXXXXXXXXXXXXX
N8N_WEBHOOK_URL=https://your-n8n/webhook/dca
EOF
sudo chmod 600 /opt/crypto-agent/.env
```

**Angel app static IP** — SmartAPI binds your API key to a whitelisted IP. Update it to
the VPS's public IP or every login returns an auth error:

```bash
curl -s ifconfig.me     # the VPS public IP
```

Then in the Angel **SmartAPI dashboard** (smartapi.angelbroking.com → My Apps → edit your
app) replace the static/whitelisted IP with that value and save. Re-verify with
`python equity_agent/test_connection.py` on the server before enabling the service.

**Service**:

```bash
sudo cp /opt/crypto-agent/deploy/dca-agent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now dca-agent          # start on boot + start now
systemctl status dca-agent
journalctl -u dca-agent -f                      # or tail /var/log/dca-agent/agent.log
```

`Restart=always`, `RestartSec=30`, `StartLimitIntervalSec=0` — it restarts on crash
forever (no give-up threshold), and `After/Wants=network-online.target` avoids a failed
first login on boot.

### Soak prerequisites (checked 2026-09-10 — currently NOT met)

A `--paper-live` soak only exercises the buy path if the account can actually buy:

1. **Fund the account.** With equity ₹1,358 and `DCA_START=8000` the life meter reads
   **17%**, below the 60% `pause_life` floor → the agent pauses and emits only `paused`
   events. Either deposit so equity ≈ `DCA_START`, or lower `DCA_START` to match reality.
2. **Leave settled cash ≥ `DCA_AMOUNT`.** Cash is currently ₹1; a scheduled ₹500 buy
   would emit `no_cash` every week even unpaused.
3. Set `N8N_WEBHOOK_URL`, or alerts are skipped (logged, not fatal).
