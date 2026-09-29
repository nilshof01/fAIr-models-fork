"""Keep the review decisions in one place and pull them down for training.

    # one-time: push the legacy local SQLite review up (insert-only)
    python scripts/sync_labels.py --push-legacy $VHR_REVIEW_DB
    python scripts/sync_labels.py --push-legacy <path> --apply

    # every time you have labelled more
    python scripts/sync_labels.py --pull
    python scripts/build_verified_baseline.py      # val/test stay frozen
    OUT=... GPU=1 bash scripts/train_verified_baseline.sh

    # versioned off-site copy
    python scripts/sync_labels.py --backup

--push-legacy never overwrites: a chip decided in both places keeps its
Supabase decision, and disagreements are printed rather than resolved.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd

from labels_store import CACHE, LabelStore, LegacySqlite, write_cache

BACKUP_REPO = "nilsho01/fair-models-fetch-artifacts"


def do_pull(store):
    t = store.pull()
    n = write_cache(t)
    d = t["decisions"]
    print(f"pulled from {store.url}")
    print(f"  decisions   {n['decisions']:6,}  {d.decision.value_counts().to_dict()}")
    print(f"  mask fixes  {n.get('fixes', 0):6,}")
    print(f"  image fixes {n.get('image_fixes', 0):6,}")
    keep = set(d.tile_id[d.decision == "keep"]) | set(t["fixes"].get("tile_id", []))
    print(f"  usable (keep + corrected): {len(keep):,} chips")
    print(f"cache -> {CACHE}")


def do_push(store, sqlite_path, apply):
    legacy = LegacySqlite(sqlite_path)
    up = store.pull(with_blobs=False)
    have = dict(zip(up["decisions"].dataset_row.astype(int),
                    up["decisions"].decision)) if len(up["decisions"]) else {}
    print(f"legacy: {sqlite_path}")
    print(f"supabase already holds {len(have):,} decisions\n")

    for table in ("decisions", "fixes", "image_fixes"):
        rows = legacy.rows(table)
        if not rows:
            print(f"{table:12s} nothing in the legacy db")
            continue
        new, skipped = store.push_new(table, rows, dry_run=not apply)
        verb = "pushed" if apply else "would push"
        print(f"{table:12s} {verb} {new:5,} new, skipped {skipped:5,} already upstream")

    # disagreements are a human call, not a merge rule
    clash = [(r["dataset_row"], have[int(r["dataset_row"])], r["decision"])
             for r in legacy.rows("decisions")
             if int(r["dataset_row"]) in have
             and have[int(r["dataset_row"])] != r["decision"]]
    if clash:
        print(f"\n{len(clash)} chips decided differently in the two places "
              f"(Supabase kept; legacy shown second):")
        for row, a, b in clash[:15]:
            print(f"  row {row:6d}  supabase={a:10s} legacy={b}")
        if len(clash) > 15:
            print(f"  ... {len(clash)-15} more")
    if not apply:
        print("\nDRY RUN - nothing written. Re-run with --apply.")


def do_backup():
    from huggingface_hub import HfApi
    if not (CACHE / "decisions.parquet").exists():
        raise SystemExit("Nothing cached yet - run --pull first.")
    api = HfApi()
    api.upload_folder(folder_path=str(CACHE), path_in_repo="labels",
                      repo_id=BACKUP_REPO, repo_type="dataset")
    print(f"backed up {CACHE} -> {BACKUP_REPO}/labels")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true", help="Supabase -> local cache")
    ap.add_argument("--push-legacy", metavar="SQLITE",
                    help="insert legacy SQLite decisions not already upstream")
    ap.add_argument("--apply", action="store_true", help="actually write (default: dry run)")
    ap.add_argument("--backup", action="store_true",
                    help=f"upload the cache to {BACKUP_REPO}")
    a = ap.parse_args()
    if not (a.pull or a.push_legacy or a.backup):
        ap.error("pick one of --pull / --push-legacy / --backup")

    if a.push_legacy:
        do_push(LabelStore(), a.push_legacy, a.apply)
    if a.pull:
        do_pull(LabelStore())
    if a.backup:
        do_backup()


if __name__ == "__main__":
    main()
