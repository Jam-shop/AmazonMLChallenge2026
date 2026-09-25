#!/usr/bin/env python3
"""Chunked exact dataset audit; bounded memory for multi-million-row TSVs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def normalized_hash(s: pd.Series) -> np.ndarray:
    x = (s.fillna("").str.normalize("NFKD").str.casefold()
         .str.replace("&", " and ", regex=False)
         .str.replace(r"[^\w]+", " ", regex=True)
         .str.replace(r"\s+", " ", regex=True).str.strip())
    return pd.util.hash_array(x.to_numpy(dtype=object), categorize=False)


def duplicate_rows(a: np.ndarray, exclude_zero: bool = False) -> tuple[int, int]:
    if exclude_zero:
        a = a[a != pd.util.hash_array(np.array([""], dtype=object))[0]]
    if not len(a):
        return 0, 0
    a.sort()
    starts = np.r_[True, a[1:] != a[:-1]]
    counts = np.diff(np.r_[np.flatnonzero(starts), len(a)])
    return int(np.sum(counts[counts > 1] - 1)), int(np.sum(counts > 1))


def audit_file(path: Path, chunksize: int) -> dict:
    ids, names, addresses = [], [], []
    missing = Counter(); countries = Counter(); rows = 0
    for df in pd.read_csv(path, sep="\t", dtype=str, chunksize=chunksize,
                          keep_default_na=True, na_values=[""]):
        rows += len(df)
        for c in df.columns:
            missing[c] += int(df[c].isna().sum())
        countries.update(df["country"].fillna("<MISSING>").tolist())
        ids.append(pd.to_numeric(df["entity_id"].str.split("-", n=1).str[-1], errors="coerce")
                   .fillna(-1).to_numpy(dtype=np.int64))
        names.append(normalized_hash(df["business_name"]))
        addresses.append(normalized_hash(df["business_address"]))
    id_arr = np.concatenate(ids); name_arr = np.concatenate(names); addr_arr = np.concatenate(addresses)
    id_dup_rows, id_dup_values = duplicate_rows(id_arr)
    name_dup_rows, name_dup_values = duplicate_rows(name_arr, exclude_zero=True)
    addr_dup_rows, addr_dup_values = duplicate_rows(addr_arr, exclude_zero=True)
    return {
        "rows": rows, "missing": dict(missing), "countries": dict(countries),
        "duplicate_id_rows": id_dup_rows, "duplicate_id_values": id_dup_values,
        "duplicate_normalized_name_rows": name_dup_rows,
        "duplicate_normalized_name_values": name_dup_values,
        "duplicate_normalized_address_rows": addr_dup_rows,
        "duplicate_normalized_address_values": addr_dup_values,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("dataset"))
    ap.add_argument("--output", type=Path, default=Path("analysis/audit.json"))
    ap.add_argument("--chunksize", type=int, default=250_000)
    args = ap.parse_args()
    result = {}
    for split in ("train", "test"):
        for source in (1, 2, 3):
            key = f"{split}_source{source}"
            path = args.data / split / f"{key}.tsv"
            print(f"auditing {path}", flush=True)
            result[key] = audit_file(path, args.chunksize)
            print(json.dumps(result[key], ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
