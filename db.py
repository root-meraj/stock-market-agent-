import sqlite3
import time

SCHEMA = """
create table if not exists account(id integer primary key check(id = 1), cash real);
create table if not exists positions(symbol text primary key, qty real, avg_price real);
create table if not exists trades(
    ts real, symbol text, side text, qty real, price real,
    fee real, tds real, mode text, rsi real, reason text);
create table if not exists equity(ts real, equity real, cash real, life real, paused int);
create table if not exists flags(k text primary key, v text);
"""


class DB:
    def __init__(self, path):
        self.c = sqlite3.connect(path)
        self.c.executescript(SCHEMA)
        self.c.commit()

    # --- paper account ledger ---
    def ensure_account(self, cash):
        if not self.c.execute("select 1 from account where id = 1").fetchone():
            self.c.execute("insert into account(id, cash) values (1, ?)", (cash,))
            self.c.commit()

    def cash(self):
        row = self.c.execute("select cash from account where id = 1").fetchone()
        return row[0] if row else 0.0

    def positions(self):
        rows = self.c.execute(
            "select symbol, qty, avg_price from positions where qty > 1e-12").fetchall()
        return {s: {"qty": q, "avg_price": p} for s, q, p in rows}

    def record_fill(self, symbol, side, qty, price, fee, tds, mode, rsi, reason):
        row = self.c.execute(
            "select qty, avg_price from positions where symbol = ?", (symbol,)).fetchone()
        old_qty, old_avg = row if row else (0.0, 0.0)
        if side == "buy":
            new_qty = old_qty + qty
            new_avg = (old_qty * old_avg + qty * price) / new_qty
            cash_delta = -(qty * price + fee)
        else:
            new_qty = old_qty - qty
            new_avg = old_avg if new_qty > 1e-12 else 0.0
            cash_delta = qty * price - fee - tds
        self.c.execute(
            "insert into positions(symbol, qty, avg_price) values (?, ?, ?) "
            "on conflict(symbol) do update set qty = excluded.qty, avg_price = excluded.avg_price",
            (symbol, new_qty, new_avg))
        if mode == "paper":
            self.c.execute("update account set cash = cash + ? where id = 1", (cash_delta,))
        self.c.execute("insert into trades values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (time.time(), symbol, side, qty, price, fee, tds, mode, rsi, reason))
        self.c.commit()

    def trades_since(self, seconds):
        return self.c.execute(
            "select count(*) from trades where ts > ?", (time.time() - seconds,)).fetchone()[0]

    def snapshot(self, equity, cash, life, paused):
        self.c.execute("insert into equity values (?, ?, ?, ?, ?)",
                       (time.time(), equity, cash, life, int(paused)))
        self.c.commit()

    def get_flag(self, k, default=None):
        row = self.c.execute("select v from flags where k = ?", (k,)).fetchone()
        return row[0] if row else default

    def set_flag(self, k, v):
        self.c.execute("insert into flags(k, v) values (?, ?) "
                       "on conflict(k) do update set v = excluded.v", (k, str(v)))
        self.c.commit()
