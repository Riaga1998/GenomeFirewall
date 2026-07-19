"""
Download genome FASTAs from BV-BRC
==================================
Reads genome_list.txt (one BV-BRC genome_id per line, produced by
download_bvbrc.py) and pulls each genome's nucleotide FASTA (.fna) from the
BV-BRC FTPS site into --outdir.

    python src/download_genomes.py --genome-list data/genome_list.txt \
        --outdir data/genomes --jobs 8 --max-genomes 1000

Notes
-----
* BV-BRC path per the official docs:
      ftps://ftp.bv-brc.org/genomes/<genome_id>/<genome_id>.fna
* Requires `wget` on PATH (standard on Linux/Colab; `brew install wget` on mac).
* Not every genome has a .fna; misses are logged to failed_downloads.log, not
  fatal. Re-running skips files already present.
* If your network blocks anonymous FTPS, add credentials by editing WGET_BASE
  to include:  --user=anonymous --password=you@email.com
"""
from __future__ import annotations
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

FTP = "ftps://ftp.bv-brc.org/genomes"
WGET_BASE = ["wget", "-q", "--tries=3", "--timeout=30"]


def fna_url(gid: str) -> str:
    return f"{FTP}/{gid}/{gid}.fna"


def download_one(gid: str, outdir: Path, dry_run: bool) -> tuple[str, bool, str]:
    dest = outdir / f"{gid}.fna"
    if dest.exists() and dest.stat().st_size > 0:
        return gid, True, "cached"
    cmd = WGET_BASE + ["-O", str(dest), fna_url(gid)]
    if dry_run:
        return gid, True, " ".join(cmd)
    try:
        subprocess.run(cmd, check=True)
        if dest.exists() and dest.stat().st_size > 0:
            return gid, True, "ok"
        dest.unlink(missing_ok=True)
        return gid, False, "empty file"
    except subprocess.CalledProcessError as e:
        dest.unlink(missing_ok=True)
        return gid, False, f"wget exit {e.returncode}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome-list", required=True, type=Path)
    ap.add_argument("--outdir", type=Path, default=Path("data/genomes"))
    ap.add_argument("--jobs", type=int, default=8, help="parallel downloads")
    ap.add_argument("--max-genomes", type=int, default=None,
                    help="cap the number of genomes (keeps compute sane)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    ids = [l.strip() for l in args.genome_list.read_text().splitlines() if l.strip()]
    if args.max_genomes:
        ids = ids[: args.max_genomes]
    print(f"{len(ids)} genomes to fetch -> {args.outdir}")

    ok, failed = 0, []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(download_one, g, args.outdir, args.dry_run): g for g in ids}
        for i, fut in enumerate(as_completed(futs), 1):
            gid, success, msg = fut.result()
            if args.dry_run:
                print(msg)
            elif success:
                ok += 1
            else:
                failed.append((gid, msg))
            if not args.dry_run and i % 100 == 0:
                print(f"  {i}/{len(ids)} done ({ok} ok, {len(failed)} failed)")

    if failed:
        log = args.outdir / "failed_downloads.log"
        log.write_text("\n".join(f"{g}\t{m}" for g, m in failed) + "\n")
        print(f"{len(failed)} failed (logged to {log}).")
    print(f"Done: {ok} genomes in {args.outdir}. "
          f"Next: python src/build_features.py --genomes-dir {args.outdir}")


if __name__ == "__main__":
    main()
