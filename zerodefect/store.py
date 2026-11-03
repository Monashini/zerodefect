"""SQLite inspection log + stats used by the dashboard and the shift-report agent."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import Counter
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    product TEXT NOT NULL,
    frame_id TEXT,
    part_index INTEGER,
    verdict TEXT NOT NULL,
    score REAL,
    defect_type TEXT,
    severity INTEGER,
    location TEXT,
    explanation TEXT,
    likely_cause TEXT,
    defect_area_pct REAL,
    source TEXT,
    thumb TEXT,
    extra TEXT
);
CREATE INDEX IF NOT EXISTS ix_ts ON inspections(ts);
"""


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def add(self, row: dict) -> int:
        cols = [
            "ts", "product", "frame_id", "part_index", "verdict", "score", "defect_type", "severity",
            "location", "explanation", "likely_cause", "defect_area_pct", "source", "thumb", "extra",
        ]
        row = {**row}
        row.setdefault("ts", time.time())
        if isinstance(row.get("extra"), (dict, list)):
            row["extra"] = json.dumps(row["extra"])
        with self._lock:
            cur = self.db.execute(
                f"INSERT INTO inspections ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                [row.get(c) for c in cols],
            )
            self.db.commit()
            return int(cur.lastrowid)

    def update_verdict(self, rec_id: int, verdict: str) -> None:
        with self._lock:
            self.db.execute("UPDATE inspections SET verdict=? WHERE id=?", (verdict, rec_id))
            self.db.commit()

    def get(self, rec_id: int) -> dict | None:
        r = self.db.execute("SELECT * FROM inspections WHERE id=?", (rec_id,)).fetchone()
        return dict(r) if r else None

    def rows(self, since: float | None = None, until: float | None = None, product: str | None = None,
             verdict: str | None = None, limit: int | None = None) -> list[dict]:
        q, args = "SELECT * FROM inspections WHERE 1=1", []
        if since is not None:
            q += " AND ts>=?"; args.append(since)
        if until is not None:
            q += " AND ts<?"; args.append(until)
        if product:
            q += " AND product=?"; args.append(product)
        if verdict:
            q += " AND verdict=?"; args.append(verdict)
        q += " ORDER BY ts DESC"
        if limit:
            q += " LIMIT ?"; args.append(int(limit))
        return [dict(r) for r in self.db.execute(q, args).fetchall()]

    def clear(self) -> None:
        with self._lock:
            self.db.execute("DELETE FROM inspections")
            self.db.commit()

    # -- analytics (also exposed to the report agent as tools) ------------------------------
    def summary(self, since: float | None = None, until: float | None = None, product: str | None = None) -> dict:
        rows = self.rows(since, until, product)
        c = Counter(r["verdict"] for r in rows)
        total = len(rows)
        fails = c.get("FAIL", 0)
        return {
            "total_parts": total,
            "pass": c.get("PASS", 0),
            "fail": fails,
            "review": c.get("REVIEW", 0),
            "false_alarms_corrected": c.get("PASS_OVERRIDE", 0),
            "defect_rate_pct": round(100.0 * fails / total, 2) if total else 0.0,
            "first_ts": min((r["ts"] for r in rows), default=None),
            "last_ts": max((r["ts"] for r in rows), default=None),
        }

    def defect_breakdown(self, since: float | None = None, product: str | None = None) -> list[dict]:
        rows = [r for r in self.rows(since, product=product) if r["verdict"] in ("FAIL", "REVIEW")]
        by_type = Counter(r["defect_type"] or "other" for r in rows)
        by_loc = {}
        for r in rows:
            by_loc.setdefault(r["defect_type"] or "other", Counter())[r["location"] or "?"] += 1
        return [
            {"defect_type": t, "count": n, "top_locations": [l for l, _ in by_loc[t].most_common(3)],
             "avg_severity": round(sum((r["severity"] or 0) for r in rows if (r["defect_type"] or "other") == t) / n, 2)}
            for t, n in by_type.most_common()
        ]

    def by_hour(self, since: float | None = None, product: str | None = None) -> list[dict]:
        buckets: dict[str, Counter] = {}
        for r in self.rows(since, product=product):
            hour = time.strftime("%Y-%m-%d %H:00", time.localtime(r["ts"]))
            buckets.setdefault(hour, Counter())[r["verdict"]] += 1
        out = []
        for hour in sorted(buckets):
            c = buckets[hour]
            total = sum(c.values())
            out.append({"hour": hour, "total": total, "fail": c.get("FAIL", 0),
                        "defect_rate_pct": round(100.0 * c.get("FAIL", 0) / total, 2)})
        return out

    def recent_failures(self, limit: int = 10, product: str | None = None) -> list[dict]:
        keep = ("id", "ts", "product", "defect_type", "severity", "location", "explanation", "likely_cause", "defect_area_pct")
        rows = [r for r in self.rows(product=product, limit=500) if r["verdict"] in ("FAIL", "REVIEW")][:limit]
        return [{k: (time.strftime("%H:%M:%S", time.localtime(r[k])) if k == "ts" else r[k]) for k in keep} for r in rows]
