import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm


CHOICE_FIELDS = ["A", "B", "C", "D", "E"]
TEXT_FIELDS = ["question"] + CHOICE_FIELDS


def find_env_file(start_dir: str = ".") -> Optional[str]:
    current_dir = os.path.abspath(start_dir)
    while True:
        env_path = os.path.join(current_dir, ".env")
        if os.path.exists(env_path):
            return env_path

        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            return None
        current_dir = parent_dir


def load_env_file(path: Optional[str] = None) -> None:
    env_path = path or find_env_file()
    if env_path is None:
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip().removeprefix("export ").strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


class QuestionChoicesFormat(BaseModel):
    question: str
    A: str
    B: str
    C: str
    D: str
    E: str


class TextFormat(BaseModel):
    answer: str


def normalize_meta(meta: str) -> str:
    return re.sub(r"\s+", " ", str(meta)).strip()


def normalize_case_key(case_key: str) -> str:
    return "".join(re.findall(r"\d+", str(case_key)))


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[dict]) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_meta_map(rows: List[dict]) -> Dict[str, dict]:
    meta_map = {}
    for row in rows:
        key = normalize_meta(row["meta"])
        if key in meta_map:
            raise ValueError(f"Duplicate meta after normalization: {row['meta']}")
        meta_map[key] = row
    return meta_map


def parse_precedent_dict(precedent: str) -> Dict[str, str]:
    if isinstance(precedent, str):
        parsed = json.loads(precedent)
    elif isinstance(precedent, dict):
        parsed = precedent
    else:
        raise TypeError(f"Unsupported precedent type: {type(precedent)}")

    if not isinstance(parsed, dict):
        raise TypeError(f"Precedent must decode to dict, got {type(parsed)}")
    return parsed


def build_precedent_map(precedents: List[str]) -> Dict[str, Tuple[str, str]]:
    precedent_map = {}
    for precedent in precedents or []:
        precedent_dict = parse_precedent_dict(precedent)
        for case_key, content in precedent_dict.items():
            normalized = normalize_case_key(case_key)
            if not normalized:
                continue
            if normalized in precedent_map:
                raise ValueError(f"Duplicate precedent key after digit normalization: {case_key}")
            precedent_map[normalized] = (case_key, content)
    return precedent_map


def parsed_response_to_dict(message) -> dict:
    parsed = getattr(message, "parsed", None)
    if parsed is not None:
        if hasattr(parsed, "model_dump"):
            return parsed.model_dump()
        return parsed.dict()
    return json.loads(message.content)


def build_codeswitch_prompt(
    gf_lang: str,
    target_lang: str,
    percentage: str,
    lang: str,
    matrix_text: str,
    target_text: Optional[str],
) -> str:
    target_text = target_text or ""
    return f"""Given a pair of {gf_lang}-{target_lang} parallel sentences,

produce a code-switched version
with roughly about {percentage}% of the words or phrases into {lang},
while preserving the original meaning.

Apply code-switching selectively,
BUT always try to code-switch if possible,

so that the final output naturally mixes English with the target language(s),
and force the code-switched text to follow {gf_lang} grammar.

Do not add new punctuation unless necessary.
The answer must not have any preamble.


Additional Constraints:

- Preserve legal case numbers exactly
  (e.g., "2023가단12345")

- Do not modify option labels such as
  (a), (b), (c), (d), (e)


Grammar-force text (matrix language):
{matrix_text}

Target language text (target language):
{target_text}"""


def grammarforce_question_choices(
    client: OpenAI,
    model: str,
    reasoning_effort: str,
    lang: str,
    target_lang: str,
    gf_lang: str,
    percentage: str,
    english_row: dict,
    korean_row: dict,
) -> dict:
    english_text = "\n".join(f"{field}: {english_row[field]}" for field in TEXT_FIELDS)
    korean_reference = "\n".join(f"{field}: {korean_row[field]}" for field in TEXT_FIELDS)

    completion = client.chat.completions.parse(
        model=model,
        reasoning_effort=reasoning_effort,
        messages=[
            {"role": "developer", "content": "You are a multilingual Korean legal lawyer."},
            {
                "role": "user",
                "content": build_codeswitch_prompt(
                    gf_lang=gf_lang,
                    target_lang=target_lang,
                    percentage=percentage,
                    lang=lang,
                    matrix_text=english_text,
                    target_text=korean_reference,
                ),
            },
        ],
        service_tier="flex",
        timeout=1500,
        response_format=QuestionChoicesFormat,
    )
    return parsed_response_to_dict(completion.choices[0].message)


def grammarforce_text(
    client: OpenAI,
    model: str,
    reasoning_effort: str,
    lang: str,
    target_lang: str,
    gf_lang: str,
    percentage: str,
    english_text: str,
    korean_reference: Optional[str],
) -> str:
    completion = client.chat.completions.parse(
        model=model,
        reasoning_effort=reasoning_effort,
        messages=[
            {"role": "developer", "content": "You are a multilingual Korean legal lawyer."},
            {
                "role": "user",
                "content": build_codeswitch_prompt(
                    gf_lang=gf_lang,
                    target_lang=target_lang,
                    percentage=percentage,
                    lang=lang,
                    matrix_text=english_text,
                    target_text=korean_reference,
                ),
            },
        ],
        service_tier="flex",
        timeout=1500,
        response_format=TextFormat,
    )
    return parsed_response_to_dict(completion.choices[0].message)["answer"]


def process_row(
    client: OpenAI,
    model: str,
    reasoning_effort: str,
    lang: str,
    target_lang: str,
    gf_lang: str,
    percentage: str,
    english_row: dict,
    korean_row: dict,
    keep_unmatched_precedents: bool,
) -> dict:
    output_row = {
        "meta": english_row["meta"],
        "label": english_row["label"],
    }

    question_choices = grammarforce_question_choices(
        client=client,
        model=model,
        reasoning_effort=reasoning_effort,
        lang=lang,
        target_lang=target_lang,
        gf_lang=gf_lang,
        percentage=percentage,
        english_row=english_row,
        korean_row=korean_row,
    )
    output_row.update(question_choices)

    korean_precedent_map = build_precedent_map(korean_row.get("supporting_precedents", []))
    output_precedents = []
    for precedent in english_row.get("supporting_precedents", []):
        precedent_dict = parse_precedent_dict(precedent)
        output_precedent_dict = {}

        for english_case_key, english_content in precedent_dict.items():
            normalized_case_key = normalize_case_key(english_case_key)
            korean_match = korean_precedent_map.get(normalized_case_key)

            if korean_match is None:
                if not keep_unmatched_precedents:
                    raise ValueError(
                        f"Could not match precedent {english_case_key} for meta {english_row['meta']}"
                    )
                korean_content = None
            else:
                _, korean_content = korean_match

            output_precedent_dict[english_case_key] = grammarforce_text(
                client=client,
                model=model,
                reasoning_effort=reasoning_effort,
                lang=lang,
                target_lang=target_lang,
                gf_lang=gf_lang,
                percentage=percentage,
                english_text=english_content,
                korean_reference=korean_content,
            )

        output_precedents.append(json.dumps(output_precedent_dict, ensure_ascii=False))

    output_row["supporting_precedents"] = output_precedents
    return output_row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate English-matrix Korean-English grammar-forced KCL MCQA data."
    )
    parser.add_argument("--openai_key", type=str, default=None, help="OpenAI API key")
    parser.add_argument(
        "--english_path",
        type=str,
        default="data/translated_kcl_mcqa_merged_fixed_meta_ko.jsonl",
        help="English KCL MCQA JSONL path.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output/kcl_mcqa_grammarforce_english_matrix_ko.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--model", type=str, default="gpt-5.2")
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default="low",
        choices=["minimal", "low", "medium", "high"],
        help="Reasoning effort for supported OpenAI reasoning models.",
    )
    parser.add_argument("--lang", type=str, default="Korean")
    parser.add_argument("--target_lang", type=str, default="Korean")
    parser.add_argument("--gf_lang", type=str, default="English")
    parser.add_argument("--percentage", type=str, default="50")
    parser.add_argument("--start_offset", type=int, default=0)
    parser.add_argument("--end_offset", type=int, default=-1)
    parser.add_argument("--max_rows", type=int, default=-1)
    parser.add_argument(
        "--keep_unmatched_precedents",
        action="store_true",
        help="Generate unmatched precedents without a Korean reference instead of failing.",
    )
    args = parser.parse_args()

    load_env_file()
    openai_api_key = args.openai_key or os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in .env or pass --openai_key.")

    client = OpenAI(timeout=1500, api_key=openai_api_key)

    english_rows = read_jsonl(args.english_path)
    korean_rows = list(load_dataset("lbox/kcl", "kcl_mcqa", split="test"))
    korean_by_meta = build_meta_map(korean_rows)

    end_offset = len(english_rows) if args.end_offset < 0 else min(args.end_offset, len(english_rows))
    selected_rows = english_rows[args.start_offset:end_offset]
    if args.max_rows >= 0:
        selected_rows = selected_rows[: args.max_rows]

    output_rows = []
    for english_row in tqdm(selected_rows, desc="Generating KCL grammar-force data"):
        meta_key = normalize_meta(english_row["meta"])
        korean_row = korean_by_meta.get(meta_key)
        if korean_row is None:
            raise ValueError(f"Could not match Korean row by meta: {english_row['meta']}")

        output_rows.append(
            process_row(
                client=client,
                model=args.model,
                reasoning_effort=args.reasoning_effort,
                lang=args.lang,
                target_lang=args.target_lang,
                gf_lang=args.gf_lang,
                percentage=args.percentage,
                english_row=english_row,
                korean_row=korean_row,
                keep_unmatched_precedents=args.keep_unmatched_precedents,
            )
        )

        write_jsonl(args.output_path, output_rows)


if __name__ == "__main__":
    main()
