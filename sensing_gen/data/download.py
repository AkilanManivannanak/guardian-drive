"""Download the public PhysioNet data used by sensing_gen.

* PTB-XL 1.0.3 (Wagner et al., 2020), 100 Hz records, a stratified subset by `strat_fold`
  (folds 1-8 train, 9 validation, 10 test, as recommended by the dataset authors).
* MIT-BIH Noise Stress Test Database (Moody et al., 1984): real recordings of baseline
  wander (bw), electrode motion (em) and muscle artifact (ma).

Usage:
    python -m sensing_gen.data.download --root data/sensing_gen --train-per-fold 500
"""
from __future__ import annotations

import argparse
import csv
import io
import ssl
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PTBXL = "https://physionet.org/files/ptb-xl/1.0.3/"
NSTDB = "https://physionet.org/files/nstdb/1.0.0/"


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


_CTX = _ssl_context()


def fetch(url: str, dest: Path, retries: int = 5) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "sensing-gen-research/0.1"})
            with urllib.request.urlopen(req, timeout=60, context=_CTX) as r:
                data = r.read()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return dest
        except Exception as e:  # network hiccup: back off and retry
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to download {url}: {last}")


def select_ptbxl_records(db_csv: Path, train_per_fold: int, eval_per_fold: int) -> list[str]:
    rows = list(csv.DictReader(io.StringIO(db_csv.read_text())))
    by_fold: dict[int, list[dict]] = {}
    for r in rows:
        by_fold.setdefault(int(r["strat_fold"]), []).append(r)
    chosen: list[str] = []
    for fold, rs in sorted(by_fold.items()):
        rs.sort(key=lambda r: int(float(r["ecg_id"])))
        k = train_per_fold if fold <= 8 else eval_per_fold
        chosen += [r["filename_lr"] for r in rs[:k]]
    return chosen


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/sensing_gen")
    ap.add_argument("--train-per-fold", type=int, default=500)
    ap.add_argument("--eval-per-fold", type=int, default=600)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args(argv)

    root = Path(a.root)
    ptb = root / "ptbxl"
    nst = root / "nstdb"

    print("[download] NSTDB noise records (bw, em, ma)")
    for rec in ("bw", "em", "ma"):
        for ext in ("hea", "dat"):
            fetch(NSTDB + f"{rec}.{ext}", nst / f"{rec}.{ext}")

    print("[download] PTB-XL metadata")
    db_csv = fetch(PTBXL + "ptbxl_database.csv", ptb / "ptbxl_database.csv")
    fetch(PTBXL + "scp_statements.csv", ptb / "scp_statements.csv")

    recs = select_ptbxl_records(db_csv, a.train_per_fold, a.eval_per_fold)
    jobs = [(PTBXL + f"{r}.{ext}", ptb / f"{r}.{ext}") for r in recs for ext in ("hea", "dat")]
    print(f"[download] PTB-XL: {len(recs)} records ({len(jobs)} files)")
    done, failed = 0, 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(fetch, u, d) for u, d in jobs]
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                failed += 1
                print("  ", e, file=sys.stderr)
            done += 1
            if done % 1000 == 0:
                print(f"  {done}/{len(jobs)} files")
    print(f"[download] finished, {failed} failures")


if __name__ == "__main__":
    main()
