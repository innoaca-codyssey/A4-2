"""고정 공개 원본을 검증하고 비식별 어휘의 리뷰 입력을 만든다."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile
import shutil

from preprocess import APPROVED, sanitize, sequence


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    source = json.loads(Path("resources/sources.json").read_text())["movie_reviews.zip"]
    archive = args.root / "movie_reviews.zip"
    if not archive.exists() and args.download:
        with (
            urllib.request.urlopen(source["url"], timeout=120) as response,
            archive.open("wb") as stream,
        ):
            while chunk := response.read(1 << 20):
                stream.write(chunk)
    if digest(archive) != source["sha256"]:
        raise ValueError("리뷰 ZIP 해시 불일치")
    records = []
    original_tokens = 0
    retained_tokens = 0
    dropped = 0
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("리뷰 ZIP CRC 오류")
        for name in sorted(z.namelist()):
            if not name.endswith(".txt") or not ("/pos/" in name or "/neg/" in name):
                continue
            original = z.read(name).decode("utf-8")
            tokens = sequence(original)
            kept = sum(t in APPROVED for t in tokens)
            if kept == 0:
                dropped += 1
                continue
            original_tokens += sum(t not in ".!?;," for t in tokens)
            retained_tokens += kept
            records.append(
                {
                    "id": f"review-{len(records):04d}",
                    "label": int("/pos/" in name),
                    "text": sanitize(original),
                }
            )
    destination = args.root / "reviews.jsonl"
    destination.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
    )
    result = {
        "source_sha256": source["sha256"],
        "rows": len(records),
        "positive": sum(r["label"] for r in records),
        "negative": sum(not r["label"] for r in records),
        "dropped_zero_approved_words": dropped,
        "approved_vocabulary_size": len(APPROVED),
        "retained_word_fraction": retained_tokens / original_tokens,
        "data_sha256": digest(destination),
        "privacy_transform": "only reviewed generic words; other alphabetic words replaced by UNK, digits/contact syntax excluded",
    }
    (args.root / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    expected = json.loads(Path("resources/expected-data-profile.json").read_text())
    if result["data_sha256"] != expected["data_sha256"]:
        raise ValueError("A4-1과다른정제코퍼스입니다")
    shutil.copyfile("resources/fixed-split.json", args.root / "split.json")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
