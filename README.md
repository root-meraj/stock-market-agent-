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
