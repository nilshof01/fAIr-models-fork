"""Access layer for the review-decision store.

Single source of truth is the Supabase project behind the hosted labelling app
(github.com/nilshof01/osm-label-app). Tables: `decisions` (dataset_row,
tile_id, decision, ts), `fixes` and `image_fixes` (same keys plus a
base64-encoded PNG). This module pulls them into a local cache so training is
reproducible and works offline, and pushes legacy SQLite decisions up.

Credentials: ~/.config/osm-label/env (SUPABASE_URL / SUPABASE_KEY), or the
same names in the environment.
"""
import base64
import os
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "labels"
ENV_FILE = Path.home() / ".config" / "osm-label" / "env"
PAGE = 1000
TABLES = {"decisions": None, "fixes": "mask_png", "image_fixes": "image_png"}


def load_credentials():
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise SystemExit(
            f"No Supabase credentials. Write them to {ENV_FILE}:\n"
            f"  SUPABASE_URL=https://<project>.supabase.co\n"
            f"  SUPABASE_KEY=<key>")
    return url, key


class LabelStore:
    def __init__(self):
        from supabase import create_client
        url, key = load_credentials()
        self.sb = create_client(url, key)
        self.url = url

    def _all(self, table, columns):
        """Paginate; Supabase caps a single response at 1000 rows."""
        rows, offset = [], 0
        while True:
            batch = (self.sb.table(table).select(columns)
                     .order("dataset_row")
                     .range(offset, offset + PAGE - 1).execute().data)
            rows.extend(batch)
            if len(batch) < PAGE:
                return rows
            offset += PAGE

    def pull(self, with_blobs=True):
        """Every table as a DataFrame, keyed on dataset_row."""
        out = {}
        for table, blob in TABLES.items():
            cols = "dataset_row,tile_id,ts" + (f",{blob}" if blob and with_blobs else "")
            out[table] = pd.DataFrame(self._all(table, cols))
        return out

    def push_new(self, table, rows, dry_run=True):
        """Insert rows whose dataset_row is absent upstream. Never overwrites:
        a chip reviewed in both places keeps its Supabase (later) decision."""
        have = {int(r["dataset_row"]) for r in self._all(table, "dataset_row")}
        new = [r for r in rows if int(r["dataset_row"]) not in have]
        skipped = len(rows) - len(new)
        if not dry_run:
            for i in range(0, len(new), 200):
                self.sb.table(table).upsert(new[i:i + 200]).execute()
        return len(new), skipped


class LegacySqlite:
    """The pre-Supabase local review DB (Apps/vhr-buildings-review/run1)."""

    def __init__(self, path):
        self.con = sqlite3.connect(path)
        self.path = Path(path)

    def rows(self, table):
        blob = TABLES[table]
        cols = "dataset_row, tile_id, ts" + (f", {blob}" if blob else "")
        try:
            recs = self.con.execute(f"select {cols} from {table}").fetchall()
        except sqlite3.OperationalError:
            return []
        if table == "decisions":
            recs = self.con.execute(
                "select dataset_row, tile_id, ts, decision from decisions").fetchall()
            return [{"dataset_row": r, "tile_id": t, "ts": ts, "decision": d}
                    for r, t, ts, d in recs]
        return [{"dataset_row": r, "tile_id": t, "ts": ts,
                 blob: base64.b64encode(bytes(p)).decode()}
                for r, t, ts, p in recs]


def write_cache(tables):
    """Decisions to parquet; PNG blobs decoded to files named by tile_id."""
    CACHE.mkdir(parents=True, exist_ok=True)
    d = tables["decisions"]
    d.to_parquet(CACHE / "decisions.parquet", index=False)
    written = {"decisions": len(d)}
    for table, blob in TABLES.items():
        if blob is None:
            continue
        df = tables[table]
        sub = CACHE / table
        sub.mkdir(exist_ok=True)
        n = 0
        for _, r in df.iterrows():
            if blob in r and isinstance(r[blob], str):
                (sub / f"{r.tile_id}.png").write_bytes(base64.b64decode(r[blob]))
                n += 1
        df.drop(columns=[blob], errors="ignore").to_parquet(
            CACHE / f"{table}.parquet", index=False)
        written[table] = n
    return written


def load_cached():
    """Decisions + the set of tile_ids with a hand-drawn mask."""
    p = CACHE / "decisions.parquet"
    if not p.exists():
        return None, set()
    d = pd.read_parquet(p)
    f = CACHE / "fixes.parquet"
    fixes = set(pd.read_parquet(f).tile_id) if f.exists() else set()
    return d, fixes
