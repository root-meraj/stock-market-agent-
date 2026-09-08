def rsi(closes, period=14):
    """Wilder-smoothed RSI. Returns None until there is enough data."""
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    avg_gain = sum(max(d, 0.0) for d in deltas[:period]) / period
    avg_loss = sum(-min(d, 0.0) for d in deltas[:period]) / period
    for d in deltas[period:]:
        avg_gain = (avg_gain * (period - 1) + max(d, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-d, 0.0)) / period
    if avg_gain == 0 and avg_loss == 0:
        return 50.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


class Strategy:
    """Dumb baseline: RSI mean reversion. Layer ML/LLM signals on top later."""

    def __init__(self, cfg):
        self.cfg = cfg

    def signal(self, closes):
        r = rsi(closes, self.cfg.rsi_period)
        if r is None:
            return None, None
        if r <= self.cfg.rsi_buy:
            return "buy", r
        if r >= self.cfg.rsi_sell:
            return "sell", r
        return None, r
