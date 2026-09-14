# DCA agent only (not the research code) -> small image, no ccxt/matplotlib/yfinance.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    TZ=Asia/Kolkata
RUN apt-get update && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Pinned to the versions verified against Angel on this machine. logzero + websocket-client
# are undeclared smartapi-python deps.
RUN pip install --no-cache-dir smartapi-python==1.5.5 pyotp==2.10.0 \
    logzero==1.7.0 websocket-client==1.9.2

COPY db.py risk.py ./
COPY equity_agent/ equity_agent/

CMD ["python", "equity_agent/dca_agent.py", "--paper-live"]
