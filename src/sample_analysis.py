#!/usr/bin/env python3
"""Streaming, deterministic sample analysis for the challenge data."""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
import zlib
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


LEGAL = {
    "inc", "incorporated", "corp", "corporation", "company", "co", "llc",
    "limited", "ltd", "private", "pvt", "plc", "llp", "sarl", "sas",
}
ADDRESS = {
    "road": "rd", "street": "st", "avenue": "ave", "boulevard": "blvd",
    "drive": "dr", "lane": "ln", "highway": "hwy", "suite": "ste",
    "apartment": "apt", "building": "bldg", "floor": "fl",
}


def norm(text: str, field: str = "name") -> str:
    text = unicodedata.normalize("NFKD", text or "").casefold()
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("&", " and ")
    toks = re.findall(r"[^\W_]+", text, flags=re.UNICODE)
    if field == "name":
        toks = [t for t in toks if t not in LEGAL]
    else:
        toks = [ADDRESS.get(t, t) for t in toks]
    return " ".join(toks)


def jac(a: str, b: str) -> float:
    aa, bb = set(a.split()), set(b.split())
    if not aa and not bb:
        return 1.0
    return len(aa & bb) / len(aa | bb) if aa | bb else 0.0


def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def nums(a: str) -> set[str]:
    return set(re.findall(r"\d+", a))


def postal(a: str) -> set[str]:
    return set(re.findall(r"\b(?:\d{5}(?:-\d{4})?|\d{6})\b", a))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("dataset"))
    ap.add_argument("--modulus", type=int, default=200)
    ap.add_argument("--output", type=Path, default=Path("analysis/sample_noise.tsv"))
    args = ap.parse_args()
    train = args.data / "train"

    chosen: dict[str, set[str]] = {}
    match_dist, origins = Counter(), Counter()
    with (train / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            mids = set(filter(None, row["matched_entity_ids"].split(",")))
            match_dist[len(mids)] += 1
            h2 = any(x.startswith("S2-") for x in mids)
            h3 = any(x.startswith("S3-") for x in mids)
            origins["both" if h2 and h3 else "S2 only" if h2 else "S3 only" if h3 else "singleton"] += 1
            if zlib.crc32(row["source1_entity_id"].encode()) % args.modulus == 0:
                chosen[row["source1_entity_id"]] = mids

    s1rows = {}
    with (train / "train_source1.tsv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["entity_id"] in chosen:
                s1rows[row["entity_id"]] = row
    target_to_s1 = {mid: sid for sid, mids in chosen.items() for mid in mids}
    targets = {}
    reservoirs: dict[str, list[dict[str, str]]] = {"US": [], "India": []}
    for source in (2, 3):
        with (train / f"train_source{source}.tsv").open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row["entity_id"] in target_to_s1:
                    targets[row["entity_id"]] = row
                # Deterministic unrelated comparison reservoir.
                if len(reservoirs.get(row["country"], [])) < 20000:
                    reservoirs[row["country"]].append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["label", "s1", "target", "country_equal", "name_raw_exact",
              "name_norm_exact", "name_token_jaccard", "name_char_similarity",
              "address_norm_exact", "address_token_jaccard", "address_char_similarity",
              "numeric_any_agree", "numeric_conflict", "postal_agree", "postal_conflict",
              "name_token_overlap", "address_token_overlap", "name_qgram_overlap", "address_qgram_overlap"]
    stats: dict[str, list[float]] = {"positive": [0.0] * (len(fields) - 3), "negative": [0.0] * (len(fields) - 3)}
    counts = Counter()

    def emit(writer, label, left, right):
        nn1, nn2 = norm(left["business_name"]), norm(right["business_name"])
        na1, na2 = norm(left["business_address"], "address"), norm(right["business_address"], "address")
        x1, x2 = nums(na1), nums(na2); p1, p2 = postal(na1), postal(na2)
        qgrams = lambda s: {("^^" + s + "$$")[i:i+3] for i in range(max(0, len(s) + 2))}
        vals = [
            left["country"] == right["country"],
            left["business_name"].casefold() == right["business_name"].casefold(), nn1 == nn2,
            jac(nn1, nn2), sim(nn1, nn2), na1 == na2 and bool(na1), jac(na1, na2), sim(na1, na2),
            bool(x1 & x2), bool(x1 and x2 and not x1 & x2), bool(p1 & p2), bool(p1 and p2 and not p1 & p2),
            len(set(nn1.split()) & set(nn2.split())), len(set(na1.split()) & set(na2.split())),
            len(qgrams(nn1) & qgrams(nn2)), len(qgrams(na1) & qgrams(na2)),
        ]
        writer.writerow([label, left["entity_id"], right["entity_id"], *[f"{v:.6f}" if isinstance(v, float) else int(v) for v in vals]])
        counts[label] += 1
        stats[label] = [a + float(b) for a, b in zip(stats[label], vals)]

    with args.output.open("w", encoding="utf-8", newline="") as out:
        w = csv.writer(out, delimiter="\t", lineterminator="\n"); w.writerow(fields)
        for sid, mids in chosen.items():
            left = s1rows.get(sid)
            if not left:
                continue
            for mid in mids:
                if mid in targets:
                    emit(w, "positive", left, targets[mid])
            pool = reservoirs.get(left["country"], [])
            if pool:
                right = pool[zlib.crc32(sid.encode()) % len(pool)]
                if right["entity_id"] not in mids:
                    emit(w, "negative", left, right)

    print("sample_s1", len(chosen), "sample_pairs", dict(counts))
    print("match_distribution", dict(sorted(match_dist.items())))
    print("origins", dict(origins))
    for label in ("positive", "negative"):
        print(label, {k: round(v / counts[label], 4) for k, v in zip(fields[3:], stats[label])})


if __name__ == "__main__":
    main()
