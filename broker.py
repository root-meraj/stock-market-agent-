import os


class Broker:
    """One class, two modes. paper = simulated fills on real prices + SQLite ledger;
    live = ccxt CoinDCX orders (ledger still tracks avg cost for the TDS sell gate)."""

    def __init__(self, cfg, db, exchange=None):
        self.cfg = cfg
        self.db = db
        self.mode = cfg.mode
        self.ex = exchange or self._connect()
        if self.mode == "paper":
            db.ensure_account(cfg.starting_capital)

    def _connect(self):
        import ccxt
        klass = getattr(ccxt, self.cfg.exchange_id)
        cred = {}
        if self.mode == "live":
            cred = {"apiKey": os.environ["COINDCX_API_KEY"],
                    "secret": os.environ["COINDCX_API_SECRET"]}
        return klass(cred)

    # --- market data (both modes) ---
    def price(self, symbol):
        return self.ex.fetch_ticker(symbol)["last"]

    def closes(self, symbol):
        ohlcv = self.ex.fetch_ohlcv(symbol, self.cfg.timeframe, limit=self.cfg.candle_limit)
        return [row[4] for row in ohlcv]

    # --- account ---
    def cash(self):
        if self.mode == "paper":
            return self.db.cash()
        return self.ex.fetch_balance().get("USDT", {}).get("free", 0.0) or 0.0

    def positions(self):
        if self.mode == "paper":
            return self.db.positions()
        # live: real qty from exchange, avg cost from our ledger
        bal = self.ex.fetch_balance()
        db_pos = self.db.positions()
        out = {}
        for sym in self.cfg.symbols:
            base = sym.split("/")[0]
            qty = bal.get(base, {}).get("free", 0.0) or 0.0
            if qty > 1e-12:
                avg = db_pos.get(sym, {}).get("avg_price") or self.price(sym)
                out[sym] = {"qty": qty, "avg_price": avg}
        return out

    def equity(self, positions=None):
        positions = self.positions() if positions is None else positions
        return self.cash() + sum(p["qty"] * self.price(s) for s, p in positions.items())

    # --- execution ---
    def execute(self, symbol, side, amount, rsi=None, reason=""):
        px = self.price(symbol)
        qty = amount / px if side == "buy" else amount
        notional = qty * px
        fee = notional * self.cfg.taker_fee
        tds = notional * self.cfg.tds_rate if side == "sell" else 0.0

        if self.mode == "live":
            # ponytail: market-order semantics vary per exchange. Confirm CoinDCX accepts
            # create_market_{buy,sell}_order(symbol, qty) and returns real fill price
            # before trusting live runs; reconcile fee/tds from the order response.
            if side == "buy":
                self.ex.create_market_buy_order(symbol, qty)
            else:
                self.ex.create_market_sell_order(symbol, qty)

        self.db.record_fill(symbol, side, qty, px, fee, tds, self.mode, rsi, reason)
        return {"symbol": symbol, "side": side, "qty": qty, "price": px, "fee": fee, "tds": tds}
