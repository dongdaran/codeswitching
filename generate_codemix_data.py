import argparse
import json
import os
import re

from datasets import load_dataset
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm


CHOICE_KEYS = ["A", "B", "C", "D", "E"]
TEXT_KEYS = ["question"] + CHOICE_KEYS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_SWITCH_PERCENTAGE = "50"

# A-E 선택지 안에 이런 표식이 있으면 선택지가 본문 문항의 조합표 역할을 하는 경우가 많다.
# 예: "(a), (b)", "(a) (x) b (o)", "○, ×" 같은 선택지는 code-switching하지 않는다.
MARKER_PATTERN = re.compile(r"(?i)(\([a-e]\)|\b[a-e]\)|\([ox]\)|\b[ox]\b|[○×✕])")


class AnswerFormat(BaseModel):
    #explanation: str
    answer: str


def normalize_meta(meta):
    # meta의 공백 개수가 조금 달라도 같은 문제로 매칭되도록 맞춘다.
    return re.sub(r"\s+", " ", str(meta)).strip()


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def should_skip_choice(text):
    return MARKER_PATTERN.search(str(text)) is not None


def build_codeswitch_prompt(gf_lang, target_lang, percentage, gf_text, target_text):
    return f"""Given a pair of {gf_lang}-{target_lang} parallel sentences,

produce a code-switched version
by selectively mixing words or phrases from {target_lang},
while preserving the original meaning.

Apply code-switching selectively,
BUT always try to code-switch if possible,

Prefer preserving uniquely Korean legal terminology
when translation may reduce legal precision or naturalness.

Generate the text as if explaining Korean legal concepts
to foreign readers in a natural mixed-language style.

so that the final output naturally mixes {gf_lang} with the target language(s),
and **force the code-switched text to follow {gf_lang} grammar.**

Do not add new punctuation unless necessary.
The answer must not have any preamble.

Structural Preservation Rules:

- Preserve legal case numbers exactly
  (e.g., "2023가단12345")

- Do not modify option labels ㄱ,ㄴ,ㄷ,ㄹ,ㅁ  If both Korean option labels (ㄱ, ㄴ, ㄷ, ㄹ, ㅁ) and English option labels ((a), (b), (c), (d), (e) appear simultaneously or conflict with each other, prioritize preserving the Korean option labels exactly as written.

Coverage Rules:
- If the input contains option statements such as ㄱ,ㄴ,ㄷ,ㄹ,ㅁ, every option sentence MUST also contain code-switching.

- Apply code-switching consistently
  across:
  - the main question
  - option statements
  - answer choices

Validation Rules:
Before producing the final output,
verify that:
1. all option labels are preserved
2. all option texts are included
3. code-switching is applied to every option statement

If any rule is violated,
regenerate the output.

Grammar-force text (matrix language):
{gf_text}

Target language text (target language):
{target_text}"""


def call_codeswitch_api(client, korean_text, lang_text, gf_lang, target_lang, percentage):
    completion = client.chat.completions.parse(
        model="gpt-5.4-mini-2026-03-17",
        reasoning_effort="medium",
        messages=[
            {"role": "developer", "content": "You are a multilingual Korean legal lawyer."},
            {
                "role": "user",
                "content": build_codeswitch_prompt(
                    gf_lang=gf_lang,
                    target_lang=target_lang,
                    percentage=percentage,
                    gf_text=lang_text,
                    target_text=korean_text,
                ),
            },
        ],
        service_tier="flex",
        timeout=1500,
        response_format=AnswerFormat,
    )

    # SDK가 parsed 객체를 주면 그대로 쓰고, 아니면 JSON 문자열을 파싱한다.
    message = completion.choices[0].message
    if getattr(message, "parsed", None) is not None:
        return {
            # "explanation": message.parsed.explanation,
            "answer": message.parsed.answer,
        }
    return json.loads(message.content)


def generate_one_row(client, korean_row, lang_row, gf_lang, target_lang, percentage):
    output_row = {
        "meta": korean_row["meta"],
        "label": korean_row["label"],
        #"explanations": {},
    }

    for key in TEXT_KEYS:
        korean_text = korean_row[key]
        lang_text = lang_row[key]

        # A-E 선택지가 (a)/(b)/(o)/(x) 같은 조합표면 내용이 아니라 라벨이므로 그대로 둔다.
        if key in CHOICE_KEYS and (should_skip_choice(korean_text) or should_skip_choice(lang_text)):
            output_row[key] = korean_text
            #output_row["explanations"][key] = "Skipped code-switching because the choice contains option/true-false markers."
            continue

        response = call_codeswitch_api(
            client=client,
            korean_text=korean_text,
            lang_text=lang_text,
            gf_lang=gf_lang,
            target_lang=target_lang,
            percentage=percentage,
        )
        output_row[key] = response["answer"]
        #output_row["explanations"][key] = response["explanation"]

    return output_row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser")
    parser.add_argument("--openai_key", type=str, default=None, help="OpenAI key")
    parser.add_argument("--num_split", type=int, default=20, help="Number of splits")
    parser.add_argument("--start_split", type=int, default=10, help="First split index to run")

    args = parser.parse_args()

    openai_api_key = args.openai_key
    client = OpenAI(timeout=1500, api_key=openai_api_key)

    # 한국어 원본은 Hugging Face lbox/kcl의 kcl_mcqa만 사용한다.
    ds = load_dataset("lbox/kcl", "kcl_mcqa")
    korean_rows = list(ds["test"])
    korean_by_meta = {}
    for row in korean_rows:
        korean_by_meta[normalize_meta(row["meta"])] = row

    # 중국어/영어는 이미 번역된 wo_precedent JSONL만 사용한다. 
    language_files = {
        "zh": {
            "gf_lang": "Chinese",
            "target_lang": "Korean",
            "input_path": os.path.join(BASE_DIR, "data/translated_kcl_mcqa_zh_wo_precedent.jsonl"),
        },
        # "en": {
        #     "gf_lang": "English",
        #     "target_lang": "Korean",
        #     "input_path": os.path.join(BASE_DIR, "data/translated_kcl_mcqa_en_wo_precedent.jsonl"),
        # },
    }

    for lang_code, lang_info in language_files.items():
        lang_rows = load_jsonl(lang_info["input_path"])

        languages = lang_code + "_kr"
        os.makedirs(os.path.join(BASE_DIR, "output", languages), exist_ok=True)

        NUM_SPLIT = args.num_split
        SAMPLES_PER_SPLIT = len(lang_rows) // NUM_SPLIT

        for split in range(args.start_split, NUM_SPLIT):
            min_id = split * SAMPLES_PER_SPLIT
            max_id = (split + 1) * SAMPLES_PER_SPLIT
            if split == NUM_SPLIT - 1:
                max_id = len(lang_rows)

            print(languages, "split", split, "min_id", min_id, "max_id", max_id)

            outputs = {}
            outputs[languages] = {}
            for i in tqdm(range(min_id, max_id)):
                lang_row = lang_rows[i]
                meta = normalize_meta(lang_row["meta"])
                if meta not in korean_by_meta:
                    raise ValueError(f"Cannot find matching Korean KCL row for meta: {lang_row['meta']}")

                outputs[languages][i] = generate_one_row(
                    client=client,
                    korean_row=korean_by_meta[meta],
                    lang_row=lang_row,
                    gf_lang=lang_info["gf_lang"],
                    target_lang=lang_info["target_lang"],
                    percentage=CODE_SWITCH_PERCENTAGE,
                )

            with open(
                os.path.join(BASE_DIR, "output", languages, f"{languages}_split_{split}.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(outputs, f, indent=4, ensure_ascii=False)
