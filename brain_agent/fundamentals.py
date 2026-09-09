"""Perception-layer data pipeline: earnings surprises, prices, NSE announcements/flows/delivery.
Free sources only (see README). Caches to data/fundamentals/ with a .meta sidecar.

  python brain_agent/fundamentals.py --probe      # reachability check
  python brain_agent/fundamentals.py --earnings    # cache earnings surprises (universe)
  python brain_agent/fundamentals.py --prices      # cache daily prices (universe)
  python brain_agent/fundamentals.py --nse         # sample announcements / FII-DII / delivery
"""
import argparse
import csv
import json
import os
import time
import warnings
from datetime import datetime, timezone

import requests

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "fundamentals")

# Nifty-50-ish universe (yfinance .NS). Trimmed to liquid names with reliable Yahoo coverage.
NIFTY = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "ITC", "LT", "SBIN", "BHARTIARTL",
    "MARUTI", "HINDUNILVR", "KOTAKBANK", "AXISBANK", "ASIANPAINT", "SUNPHARMA", "TITAN",
    "NESTLEIND", "WIPRO", "ULTRACEMCO", "BAJFINANCE", "HCLTECH", "POWERGRID", "NTPC",
    "TATAMOTORS", "TATASTEEL", "ONGC", "COALINDIA", "GRASIM", "ADANIENT", "JSWSTEEL",
    "BAJAJFINSV", "TECHM", "HDFCLIFE", "DRREDDY", "CIPLA", "BRITANNIA", "EICHERMOT",
    "HEROMOTOCO", "DIVISLAB", "BPCL", "TATACONSUM", "SHREECEM", "APOLLOHOSP", "BAJAJ-AUTO",
]

_H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120 Safari/537.36",
      "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9"}


# --- cache helpers (same shape as the price cache) ------------------------

def _read(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.reader(f))


def _write(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _meta(path):
    return path + ".meta"


def _covered(path, frm, to):
    m = _meta(path)
    if not os.path.exists(m):
        return False
    d = json.load(open(m))
    return d["from"] <= frm and d["to"] >= to


def _stamp(path, frm, to):
    json.dump({"from": frm, "to": to}, open(_meta(path), "w"))


# --- earnings surprises (yfinance; point-in-time via announcement date) ----

def cache_earnings(symbols=NIFTY, force=False):
    import yfinance as yf
    os.makedirs(os.path.join(DATA, "earnings"), exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    got = 0
    for s in symbols:
        path = os.path.join(DATA, "earnings", f"{s}.csv")
        if not force and _covered(path, "2000-01-01", today):
            got += 1
            continue
        try:
            ed = yf.Ticker(s + ".NS").get_earnings_dates(limit=40)
        except Exception as e:
            print(f"  {s}: ERROR {repr(e)[:60]}")
            continue
        if ed is None or ed.empty:
            print(f"  {s}: no earnings dates")
            continue
        rows = []
        for dt, r in ed.iterrows():
            ts = int(dt.tz_convert("UTC").timestamp() * 1000)
            rows.append([ts, _num(r.get("EPS Estimate")), _num(r.get("Reported EPS")),
                         _num(r.get("Surprise(%)"))])
        rows.sort(key=lambda x: x[0])
        _write(path, ["announce_ms", "eps_estimate", "eps_reported", "surprise_pct"], rows)
        _stamp(path, "2000-01-01", today)
        got += 1
        time.sleep(0.3)
    print(f"earnings cached for {got}/{len(symbols)} symbols -> {DATA}/earnings/")


def load_earnings(symbol):
    """[(announce_ms, est, reported, surprise_pct)] with both est & reported present."""
    return [(ms, est, rep, sur) for ms, est, rep, sur in load_earnings_reported(symbol)
            if est is not None]


def load_earnings_reported(symbol):
    """[(announce_ms, est_or_None, reported, surprise_or_None)] for rows with reported EPS.
    Keeps estimate-less rows so the YoY surprise proxy can use them (smallcap coverage)."""
    out = []
    for r in _read(os.path.join(DATA, "earnings", f"{symbol}.csv"))[1:]:
        est, rep, sur = _num(r[1]), _num(r[2]), _num(r[3])
        if rep is not None:
            out.append((int(r[0]), est, rep, sur))
    out.sort(key=lambda x: x[0])
    return out


# --- daily prices (yfinance; for the event forward-returns) ---------------

def cache_prices(symbols=NIFTY, years=9, force=False):
    import yfinance as yf
    os.makedirs(os.path.join(DATA, "prices"), exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    got = 0
    for s in symbols:
        path = os.path.join(DATA, "prices", f"{s}.csv")
        if not force and _covered(path, "2017-01-01", today):
            got += 1
            continue
        try:
            h = yf.Ticker(s + ".NS").history(period=f"{years}y", auto_adjust=True)
        except Exception as e:
            print(f"  {s}: ERROR {repr(e)[:60]}")
            continue
        if h is None or h.empty:
            print(f"  {s}: no history")
            continue
        rows = [[int(idx.tz_convert("UTC").timestamp() * 1000), float(r.Open), float(r.High),
                 float(r.Low), float(r.Close), float(r.Volume)] for idx, r in h.iterrows()]
        _write(path, ["ts_ms", "o", "h", "l", "c", "v"], rows)
        _stamp(path, "2017-01-01", today)
        got += 1
        time.sleep(0.3)
    print(f"prices cached for {got}/{len(symbols)} symbols -> {DATA}/prices/")


def load_prices(symbol):
    return [[int(r[0])] + [float(x) for x in r[1:]] for r in
            _read(os.path.join(DATA, "prices", f"{symbol}.csv"))[1:]]


# --- NSE: announcements / FII-DII / delivery (perception-layer, cached) ----

def _nse():
    s = requests.Session()
    s.headers.update(_H)
    try:
        s.get("https://www.nseindia.com", timeout=8)     # best-effort cookie warm-up
    except Exception:
        pass
    return s


def fii_dii_latest():
    r = _nse().get("https://www.nseindia.com/api/fiidiiTradeReact", timeout=12).json()
    path = os.path.join(DATA, "fii_dii.csv")
    have = {tuple(row[:2]) for row in _read(path)[1:]}
    rows = [[e["date"], e["category"], e["buyValue"], e["sellValue"], e["netValue"]] for e in r]
    new = [row for row in rows if tuple(row[:2]) not in have]
    if new or not os.path.exists(path):
        allrows = _read(path)[1:] + new
        _write(path, ["date", "category", "buy", "sell", "net"], allrows)
    return rows


def announcements(subject="", limit=50):
    url = "https://www.nseindia.com/api/corporate-announcements?index=equities"
    j = _nse().get(url, timeout=12).json()
    out = [{"ts": e["an_dt"], "symbol": e["symbol"], "desc": e["desc"],
            "text": e.get("attchmntText", "")[:80]} for e in j[:limit]]
    if subject:
        out = [e for e in out if subject.lower() in e["desc"].lower()]
    return out


def delivery_for(ddmmyyyy, symbols=None):
    """{symbol: delivery_pct} from the security-wise bhavcopy for a date (DDMMYYYY).
    Returns {} if that date's file isn't published (weekend/holiday/not-yet)."""
    url = f"https://archives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv"
    txt = _nse().get(url, timeout=15).text
    reader = csv.reader(txt.splitlines())
    header = [h.strip() for h in next(reader, [])]
    if "SYMBOL" not in header:
        return {}
    si, seri, di = header.index("SYMBOL"), header.index("SERIES"), header.index("DELIV_PER")
    want = set(symbols) if symbols else None
    out = {}
    for row in reader:
        if len(row) <= di:
            continue
        sym, ser = row[si].strip(), row[seri].strip()
        if ser == "EQ" and (want is None or sym in want):
            try:
                out[sym] = float(row[di].strip())
            except ValueError:
                pass
    return out


def recent_delivery(symbols=None, lookback=6):
    """Latest available bhavcopy delivery% (walks back over weekends/holidays)."""
    from datetime import timedelta
    d = datetime.now()
    for _ in range(lookback):
        got = delivery_for(d.strftime("%d%m%Y"), symbols)
        if got:
            return d.strftime("%d-%b-%Y"), got
        d -= timedelta(days=1)
    return None, {}


def index_constituents(which):
    """Symbols in an NSE index. which in {'smallcap250','midcap150'}. CURRENT membership only
    (niftyindices publishes no free point-in-time history -> survivorship bias, flagged in pead)."""
    url = f"https://niftyindices.com/IndexConstituent/ind_nifty{which}list.csv"
    txt = _nse().get(url, timeout=15).text
    rows = list(csv.reader(txt.splitlines()))
    header = rows[0]
    si = header.index("Symbol")
    return [r[si].strip() for r in rows[1:] if len(r) > si and r[si].strip()]


def smallmid_universe():
    path = os.path.join(DATA, "smallmid_universe.csv")
    if os.path.exists(path):
        return [r[0] for r in _read(path)[1:]]
    syms = sorted(set(index_constituents("smallcap250")) | set(index_constituents("midcap150")))
    _write(path, ["symbol"], [[s] for s in syms])
    return syms


def _num(x):
    try:
        if x is None or x != x:          # None or NaN
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def probe():
    import yfinance as yf
    print("earnings (RELIANCE):", len(yf.Ticker("RELIANCE.NS").get_earnings_dates(limit=8)), "rows")
    print("FII/DII latest:", fii_dii_latest())
    anns = announcements(limit=50)
    print(f"announcements: {len(anns)} (results-type: "
          f"{sum('result' in a['desc'].lower() for a in anns)})")
    dday, dl = recent_delivery(symbols=NIFTY)
    print(f"delivery% ({dday}): {len(dl)} Nifty names, e.g. {dict(list(dl.items())[:3])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--earnings", action="store_true")
    ap.add_argument("--prices", action="store_true")
    ap.add_argument("--nse", action="store_true")
    ap.add_argument("--smallmid", action="store_true",
                    help="cache earnings+prices for Nifty Smallcap250 + Midcap150 constituents")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.probe:
        probe()
    universe = smallmid_universe() if args.smallmid else NIFTY
    if args.smallmid:
        print(f"smallmid universe: {len(universe)} current constituents")
    if args.earnings or args.smallmid:
        cache_earnings(universe, force=args.force)
    if args.prices or args.smallmid:
        cache_prices(universe, force=args.force)
    if args.nse:
        print("FII/DII:", fii_dii_latest())
        for a in announcements(subject="result", limit=50)[:5]:
            print("  ", a["ts"], a["symbol"], "-", a["desc"][:50])


if __name__ == "__main__":
    main()
