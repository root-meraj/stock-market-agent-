class RiskManager:
    """Survival logic (life meter) + position sizing + TDS-aware sell gate."""

    def __init__(self, cfg, db):
        self.cfg = cfg
        self.db = db

    def life(self, equity):
        return equity / self.cfg.starting_capital

    def update_pause(self, equity):
        """Life meter with hysteresis. Returns (life, paused)."""
        life = self.life(equity)
        paused = self.db.get_flag("paused", "0") == "1"
        if paused and life >= self.cfg.resume_life:
            paused = False
            self.db.set_flag("paused", "0")
        elif not paused and life <= self.cfg.pause_life:
            paused = True
            self.db.set_flag("paused", "1")
        return life, paused

    def approve(self, signal, symbol, equity, cash, positions, price):
        """Return (side, amount) to execute, or None to skip. amount is USDT cost
        for a buy, base qty for a sell."""
        if signal == "buy":
            if symbol in positions:
                return None
            if len(positions) >= self.cfg.max_positions:
                return None
            if self.db.trades_since(7 * 86400) >= self.cfg.max_trades_per_week:
                return None
            cost = min(equity * self.cfg.position_frac, cash)
            if cost < self.cfg.min_order:
                return None
            return "buy", cost

        if signal == "sell":
            pos = positions.get(symbol)
            if not pos:
                return None
            gross = (price - pos["avg_price"]) / pos["avg_price"]
            net = gross - self.cfg.tds_rate - 2 * self.cfg.taker_fee
            if net < self.cfg.min_net_profit:
                return None                       # not worth the 1% TDS hit yet
            return "sell", pos["qty"]

        return None
