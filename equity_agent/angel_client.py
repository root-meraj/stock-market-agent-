"""Angel One (SmartAPI) client: login, instrument tokens, cached historical candles.

Env (from repo-root .env): ANGEL_API_KEY, ANGEL_CLIENT_CODE, ANGEL_MPIN (or ANGEL_PIN),
ANGEL_TOTP_SECRET. Sessions expire daily -> call login() again, it's idempotent.
Candles cache to data/equity/ (repo root), same shape as the crypto backtester: ts_ms,o,h,l,c,v.
"""
import csv
import json
import os
import time
import urllib.request
from datetime import datetime, timedelta

import pyotp

try:
    from SmartApi import SmartConnect
except ImportError:                                   # older package layout
    from SmartApi.smartConnect import SmartConnect


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "equity")
SCRIP_PATH = os.path.join(DATA_DIR, "scrip_master.json")
SCRIP_URL = ("https://margincalculator.angelbroking.com/OpenAPI_File/files/"
             "OpenAPIScripMaster.json")

# SmartAPI caps candles per historical call by interval (approx max days/request).
MAX_DAYS = {
    "ONE_MINUTE": 30, "THREE_MINUTE": 60, "FIVE_MINUTE": 100, "TEN_MINUTE": 100,
    "FIFTEEN_MINUTE": 200, "THIRTY_MINUTE": 200, "ONE_HOUR": 400, "ONE_DAY": 2000,
}
THROTTLE_SLEEP = 0.35      # between historical calls (API allows ~3/sec but is strict)


def load_env(path=None):
    """Tiny .env loader so we don't pull in python-dotenv for four keys."""
    path = path or os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _iso_ms(s):
    return int(datetime.fromisoformat(s).timestamp() * 1000)


def _parse_date(d):
    return d if isinstance(d, datetime) else datetime.strptime(d, "%Y-%m-%d")


def _read_cache(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [[int(r[0])] + [float(x) for x in r[1:]] for r in csv.reader(f)]


def _write_cache(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def _read_meta(path):
    mpath = path + ".meta"
    if not os.path.exists(mpath):
        return None
    with open(mpath) as f:
        m = json.load(f)
    return datetime.strptime(m["from"], "%Y-%m-%d"), datetime.strptime(m["to"], "%Y-%m-%d")


def _write_meta(path, frm, to):
    with open(path + ".meta", "w") as f:
        json.dump({"from": frm.strftime("%Y-%m-%d"), "to": to.strftime("%Y-%m-%d")}, f)


class Throttled(Exception):
    pass


class AngelClient:
    def __init__(self):
        load_env()
        self.api_key = os.environ["ANGEL_API_KEY"]
        self.client_code = os.environ["ANGEL_CLIENT_CODE"]
        self.mpin = os.environ.get("ANGEL_MPIN") or os.environ["ANGEL_PIN"]
        self.totp_secret = os.environ["ANGEL_TOTP_SECRET"]
        self.smart = None
        self.jwt = self.refresh = self.feed = None
        self._scrip = None

    # --- session (daily expiry; safe to call again) ---
    def login(self):
        self.smart = SmartConnect(api_key=self.api_key)
        totp = pyotp.TOTP(self.totp_secret).now()
        resp = self.smart.generateSession(self.client_code, self.mpin, totp)
        if not resp.get("status"):
            raise RuntimeError(f"login failed: {resp.get('message')} ({resp.get('errorcode')})")
        d = resp["data"]
        self.jwt, self.refresh = d["jwtToken"], d["refreshToken"]
        self.feed = self.smart.getfeedToken()
        return self

    def profile_name(self):
        return self.smart.getProfile(self.refresh)["data"]["name"]

    # --- instrument master ---
    def _load_scrip(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        if not os.path.exists(SCRIP_PATH):
            urllib.request.urlretrieve(SCRIP_URL, SCRIP_PATH)
        with open(SCRIP_PATH, encoding="utf-8") as f:
            self._scrip = json.load(f)

    def token_for(self, symbol, exchange="NSE"):
        """Numeric symbol token. Prefers the -EQ cash series for NSE equities."""
        if self._scrip is None:
            self._load_scrip()
        symbol = symbol.upper()
        cand = [r for r in self._scrip
                if r.get("exch_seg") == exchange and r.get("name", "").upper() == symbol]
        if not cand:
            raise KeyError(f"{symbol} not found on {exchange}")
        eq = [r for r in cand if r.get("symbol", "").upper().endswith("-EQ")]
        return (eq or cand)[0]["token"]

    # --- historical candles (chunked, cached, backoff) ---
    def fetch_ohlcv(self, symbol, interval, from_date, to_date, exchange="NSE"):
        """Return [[ts_ms,o,h,l,c,v], ...] for [from_date, to_date]. Cached to CSV with a
        sidecar .meta recording the requested date span, so a repeat of an already-fetched
        range makes zero API calls (a market-closed start/end has no candle to test against).
        ponytail: on a range that only extends the cache, this re-fetches the overlap and
        dedupes it out. Fetch just the gap if that ever costs real quota."""
        start, end = _parse_date(from_date), _parse_date(to_date)
        path = os.path.join(DATA_DIR, f"{symbol.upper()}_{exchange}_{interval}.csv")
        rows = _read_cache(path)
        meta = _read_meta(path)
        if rows and meta and start >= meta[0] and end <= meta[1]:
            return _slice(rows, start, end)            # requested span already covered

        token = self.token_for(symbol, exchange)
        have = {r[0] for r in rows}
        step = MAX_DAYS.get(interval, 100)
        cur = start
        while cur <= end:
            chunk_end = min(cur + timedelta(days=step - 1), end)
            for r in self._candles(token, exchange, interval, cur, chunk_end):
                if r[0] not in have:
                    have.add(r[0])
                    rows.append(r)
            cur = chunk_end + timedelta(days=1)
            time.sleep(THROTTLE_SLEEP)
        rows.sort(key=lambda r: r[0])
        _write_cache(path, rows)
        _write_meta(path, min(start, meta[0]) if meta else start,
                    max(end, meta[1]) if meta else end)
        return _slice(rows, start, end)

    def _candles(self, token, exchange, interval, frm, to, retries=6):
        params = {"exchange": exchange, "symboltoken": token, "interval": interval,
                  "fromdate": frm.strftime("%Y-%m-%d 09:15"),
                  "todate": to.strftime("%Y-%m-%d 15:30")}
        delay = 1.0
        for attempt in range(retries):
            try:
                resp = self.smart.getCandleData(params)
                if resp.get("status"):
                    return [[_iso_ms(row[0]), row[1], row[2], row[3], row[4], row[5]]
                            for row in (resp.get("data") or [])]
                msg = f"{resp.get('message', '')} {resp.get('errorcode', '')}".lower()
                if any(w in msg for w in ("throttl", "denied", "rate", "429", "try after")):
                    raise Throttled(msg)
                raise RuntimeError(f"getCandleData: {resp.get('message')} ({resp.get('errorcode')})")
            except Exception:                          # backoff on throttle or any transient error
                if attempt == retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2
        return []


def _ms(dt):
    return int(dt.timestamp() * 1000)


def _ms_end(dt):
    # end-of-day so a date-only 'to' counts as covered by that day's candle
    return int((dt + timedelta(hours=18)).timestamp() * 1000)


def _slice(rows, start, end):
    lo, hi = _ms(start), _ms_end(end)
    return [r for r in rows if lo <= r[0] <= hi]
