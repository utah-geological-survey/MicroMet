from __future__ import annotations
from pathlib import Path
import csv
import re
import pandas as pd
from collections import defaultdict

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Tiny helpers
# ──────────────────────────────────────────────────────────────────────────────
def looks_like_header(line: str) -> bool:
    """
    A pragmatic test: header lines normally contain ≥ 1 alphabetic character,
    data rows in instrument logs normally do not.
    """
    line_list = line.split(',')
    return bool(re.search(r"[A-Za-z]", line_list[0]))


def sniff_delimiter(path: Path, sample_bytes: int = 2048, default: str = ",") -> str:
    """Return the most likely delimiter, falling back to *default* if unsure."""
    with path.open("r", newline="", encoding="utf-8") as fh:
        sample = fh.read(sample_bytes)
    try:
        return csv.Sniffer().sniff(sample).delimiter
    except csv.Error:
        return default


def read_colnames(path: Path) -> list[str]:
    """Grab the *first* line and split it on the file’s delimiter."""
    delim = sniff_delimiter(path)
    with path.open("r", encoding="utf-8") as fh:
        return fh.readline().rstrip("\n\r").split(delim)


def patch_file(
    donor: Path, target: Path, *, write_back: bool = True
) -> pd.DataFrame:
    """Copy header from *donor* onto *target* and return a fixed DataFrame."""
    cols = read_colnames(donor)
    delim = sniff_delimiter(donor)

    df = pd.read_csv(target, header=None, names=cols, delimiter=delim)

    if write_back:
        bak = target.with_suffix(target.suffix + ".bak")
        target.replace(bak)
        df.to_csv(target, index=False, sep=delim)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Main routine
# ──────────────────────────────────────────────────────────────────────────────
def fix_all_in_parent(parent: Path, searchstr: str = "*_AmeriFluxFormat_*.dat") -> None:
    """
    Scan *parent* (optionally *recursive*) for duplicate filenames.
    Whenever one copy has a header and another does not, transfer the header.
    """
    # ------------------------------------------------------------------ #
    # 1. Collect every file path, grouped by basename.
    paths_by_name: dict[str, list[Path]] = defaultdict(list)
    glob_pattern = searchstr
    for p in parent.rglob(glob_pattern):
        if p.is_file():
            paths_by_name[p.name].append(p)

    # ------------------------------------------------------------------ #
    # 2. Examine each group of duplicates
    for fname, paths in paths_by_name.items():
        if len(paths) < 2:
            continue  # no duplicates → nothing to do

        # Classify each copy
        header_files, noheader_files = [], []
        for p in paths:
            first = p.open("r", encoding="utf-8").readline()
            if looks_like_header(first):
                header_files.append(p)
            else:
                noheader_files.append(p)

        if not header_files or not noheader_files:
            # Either (a) every copy already has a header, or (b) none do
            # In both situations we cannot (or need not) patch automatically.
            continue

        # Use the first header-bearing file as the “donor” for all others
        donor = header_files[0]
        for tgt in noheader_files:
            df_fixed = patch_file(donor, tgt, write_back=True)
            print(f"[INFO]  Patched  {tgt.relative_to(parent)}   "
                  f"({len(df_fixed):,d} rows)")

    print("\n✔ All possible files have been checked.")
    return paths_by_name

def apply_header(
    header_file: Path,
    target_file: Path,
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Copy the header from `header_file` onto `target_file` and return a DataFrame
    with the correct column names.  If `inplace` is True, the target file is
    overwritten on disk (a *.bak backup is kept).
    """
    delimiter = sniff_delimiter(header_file)

    # ------------------------------------------------------------------ #
    # get column names from the good file
    cols = read_colnames(header_file)

    # ------------------------------------------------------------------ #
    # read the data-only file, telling pandas “there is *no* header here”
    df = pd.read_csv(target_file, header=None, names=cols, delimiter=delimiter)

    # ------------------------------------------------------------------ #
    # optionally write the fixed file back to disk
    if inplace:
        backup = target_file.with_suffix(target_file.suffix + ".bak")
        target_file.replace(backup)                 # keep a backup
        df.to_csv(target_file, index=False, sep=delimiter)

    return df

def fix_directory_pairs(dir_with_headers: Path, dir_without_headers: Path) -> None:
    """
    Loop over all files in `dir_without_headers`; whenever a file lacks a
    header, patch it with the header from the identically named file in
    `dir_with_headers`.
    """
    # index the header-bearing directory for O(1) lookup
    header_index = {p.name: p for p in dir_with_headers.iterdir() if p.is_file()}

    for f in dir_without_headers.iterdir():
        if not f.is_file():
            continue

        # Fast header check: read only the first line
        first_line = f.open("r", encoding="utf-8").readline()
        if looks_like_header(first_line):
            continue  # nothing to do

        if f.name not in header_index:
            print(f"[WARN] No header twin found for {f}")
            continue

        df_fixed = apply_header(header_index[f.name], f, inplace=True)
        print(f"[INFO] Patched header on {f} ({len(df_fixed)} rows)")

# ──────────────────────────────────────────────────────────────────────────────
# 3.  Example CLI usage
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = Path(r"C:\data\logs")           # <── your top-level directory here
    fix_all_in_parent(root, recurse=False) # recurse=True if sub-sub-folders exist
