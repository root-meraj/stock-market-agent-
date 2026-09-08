"""Smoke test: login, print profile, fetch 1y daily RELIANCE candles.
Run: python equity_agent/test_connection.py   (needs live .env creds + network)
"""
from datetime import datetime, timedelta

from angel_client import AngelClient


def _fmt(rows):
    for ts, o, h, l, c, v in rows:
        d = datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d")
        print(f"  {d}  o={o:>9.2f} h={h:>9.2f} l={l:>9.2f} c={c:>9.2f} v={int(v)}")


def main():
    c = AngelClient().login()
    print("logged in. profile:", c.profile_name())

    to = datetime.today()
    frm = to - timedelta(days=365)
    rows = c.fetch_ohlcv("RELIANCE", "ONE_DAY",
                         frm.strftime("%Y-%m-%d"), to.strftime("%Y-%m-%d"))
    print(f"\nRELIANCE ONE_DAY: shape = {len(rows)} rows x {len(rows[0]) if rows else 0} cols")
    print("\nfirst 5:")
    _fmt(rows[:5])
    print("last 5:")
    _fmt(rows[-5:])


if __name__ == "__main__":
    main()
