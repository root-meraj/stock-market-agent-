def _rsi_val(avg_gain, avg_loss):
    if avg_gain == 0 and avg_loss == 0:
        return 50.0
    if avg_loss == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def rsi_series(closes, period=14):
    """Wilder-smoothed RSI at every index (None until enough data). O(n) one pass."""
    n = len(closes)
    out = [None] * n
    if n < period + 1:
        return out
    deltas = [closes[i] - closes[i - 1] for i in range(1, n)]
    avg_gain = sum(max(d, 0.0) for d in deltas[:period]) / period
    avg_loss = sum(-min(d, 0.0) for d in deltas[:period]) / period
    out[period] = _rsi_val(avg_gain, avg_loss)
    for i, d in enumerate(deltas[period:], start=period + 1):
        avg_gain = (avg_gain * (period - 1) + max(d, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-d, 0.0)) / period
        out[i] = _rsi_val(avg_gain, avg_loss)
    return out


def rsi(closes, period=14):
    """Wilder-smoothed RSI over the whole window. Returns None until enough data."""
    return rsi_series(closes, period)[-1] if len(closes) >= period + 1 else None


class Strategy:
    """Dumb baseline: RSI mean reversion. Layer ML/LLM signals on top later."""

    def __init__(self, cfg):
        self.cfg = cfg

    def signal(self, closes):
        return self.signal_from_rsi(rsi(closes, self.cfg.rsi_period))

    def signal_from_rsi(self, r):
        """Threshold logic split out so the backtester can feed a precomputed RSI."""
        if r is None:
            return None, None
        if r <= self.cfg.rsi_buy:
            return "buy", r
        if r >= self.cfg.rsi_sell:
            return "sell", r
        return None, r
