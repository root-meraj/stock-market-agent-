# brain_agent — perception layer (multi-source, non-price)

Reasoning agent over fundamentals + events, **not** price indicators. This is the
data/perception layer; no LLM yet. Step 1 = pipeline, Step 2 = PEAD hypothesis test.

## Data source assessment (free vs paid) — tested empirically from this machine

All probed live on 2026-09-09. Reachability note: this runs on a **residential IP**,
so NSE's endpoints respond; from a datacenter IP they typically 403. A cloud deploy
would need a proxy or a paid vendor for the NSE sources.

| data | source | free? | point-in-time | notes |
|------|--------|-------|---------------|-------|
| **Consensus + reported EPS, surprise%, announce date** | Yahoo via `yfinance.get_earnings_dates` | **free** | **yes** — real announcement datetime | 18/18 Nifty names tested, 24 quarters each, both estimate & reported non-null |
| Quarterly financials (revenue, PAT, EPS, margins) | `yfinance.quarterly_income_stmt` | free | partial — quarter-END only; pair with the announce date above for PIT | |
| NSE corporate announcements (results/orders/ratings) + timestamps | `nseindia.com/api/corporate-announcements` | free | yes — `an_dt` timestamp | unofficial API; supports date-range params |
| FII/DII daily flows | `nseindia.com/api/fiidiiTradeReact` | free | yes (daily) | returns latest day; history = collect daily or scrape archive |
| Per-stock delivery % | `archives.nseindia.com/.../sec_bhavdata_full_<DDMMYYYY>.csv` | free | yes (daily) | 397KB/day full security-wise bhavcopy incl. `DELIV_PER` |

**The blocker that wasn't:** professional consensus estimates (Refinitiv IBES,
Bloomberg, Trendlyne, Zacks) are **paid** — and I expected to need them for "beat
consensus." But Yahoo aggregates a free consensus for Nifty large-caps with the
announcement date attached, which covers PEAD's core need *and* solves point-in-time.
So **nothing here requires a paid API** for a first test.

**What paid would buy you** (if this graduates past a pulse test): licensed,
redistributable data; deeper history; small/mid-caps (Yahoo thins out below large-cap);
and estimate *revisions* / analyst counts (Yahoo gives only the point estimate).

## Licensing / ToS — read before relying on this

- **Yahoo Finance / yfinance**: Yahoo's ToS restricts use to personal, non-commercial;
  yfinance scrapes an undocumented endpoint. Fine for private research, **not** for a
  product or redistribution. No SLA — fields can vanish.
- **NSE**: the site ToS prohibits automated/commercial scraping; these APIs are
  unofficial and unstable. Rate-limit politely (we cache aggressively). Personal
  research only.
- Neither is a licensed feed. Anything commercial needs a paid vendor (see above).

## Point-in-time integrity

Backtests must only see what was public then. We key every earnings row on its
**announcement datetime** (from `get_earnings_dates`), not the quarter-end. PEAD enters
at the next session's open *after* the announcement — never using a number before it
was filed.

## Files

| file | role |
|------|------|
| `fundamentals.py` | fetch + cache earnings surprises, NSE announcements, FII/DII, delivery% |
| `pead.py`         | post-earnings-announcement-drift test (event-driven walk-forward) |

Cache lives in `data/fundamentals/` with the same `.meta` sidecar as the price cache
(records the date span fetched, so repeat runs don't re-fetch).

## Run

```
pip install yfinance
python brain_agent/fundamentals.py --probe       # what's reachable right now
python brain_agent/fundamentals.py --earnings    # cache earnings for the universe
python brain_agent/pead.py                        # the PEAD test
```
