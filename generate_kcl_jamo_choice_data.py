import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = Path("data/kcl_mcqa_ko_rulebase_jamo_choices.jsonl")
JAMO_CHOICE_MAP = {
    "ㄱ": "(a)",
    "ㄴ": "(b)",
    "ㄷ": "(c)",
    "ㄹ": "(d)",
    "ㅁ": "(e)",
}
TEXT_FIELDS = ("question", "A", "B", "C", "D", "E", "supporting_precedents")
TRANSLATION_TABLE = str.maketrans(JAMO_CHOICE_MAP)


def convert_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.translate(TRANSLATION_TABLE)
    if isinstance(value, list):
        return [convert_value(item) for item in value]
    if isinstance(value, dict):
        return {key: convert_value(item) for key, item in value.items()}
    return value


def count_jamo_choices(value: Any) -> Counter:
    counter: Counter = Counter()
    if isinstance(value, str):
        counter.update(char for char in value if char in JAMO_CHOICE_MAP)
    elif isinstance(value, list):
        for item in value:
            counter.update(count_jamo_choices(item))
    elif isinstance(value, dict):
        for item in value.values():
            counter.update(count_jamo_choices(item))
    return counter


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download lbox/kcl kcl_mcqa and rule-base replace choice jamo."
    )
    parser.add_argument("--dataset", default="lbox/kcl")
    parser.add_argument("--config", default="kcl_mcqa")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use the Hugging Face local cache without network access.",
    )
    args = parser.parse_args()

    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    from datasets import load_dataset

    dataset = load_dataset(args.dataset, args.config, split=args.split)

    rows = []
    before_counts: Counter = Counter()
    after_counts: Counter = Counter()
    field_counts: dict[str, Counter] = {field: Counter() for field in TEXT_FIELDS}

    for row in dataset:
        fixed = dict(row)
        for field in TEXT_FIELDS:
            if field not in fixed:
                continue
            before = count_jamo_choices(fixed[field])
            before_counts.update(before)
            field_counts[field].update(before)
            fixed[field] = convert_value(fixed[field])
            after_counts.update(count_jamo_choices(fixed[field]))
        rows.append(fixed)

    write_jsonl(args.output, rows)

    mapped_counts = {
        f"{old}->{new}": before_counts[old]
        for old, new in JAMO_CHOICE_MAP.items()
    }
    print(f"wrote {args.output} ({len(rows)} rows)")
    print("replacement counts:", mapped_counts)
    print(
        "field counts:",
        {
            field: {
                f"{old}->{JAMO_CHOICE_MAP[old]}": counts[old]
                for old in JAMO_CHOICE_MAP
            }
            for field, counts in field_counts.items()
        },
    )
    print("remaining jamo choices:", dict(after_counts))


if __name__ == "__main__":
    main()
