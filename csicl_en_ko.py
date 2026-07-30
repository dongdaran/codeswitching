import argparse
import json
import os
import time

import openai
from tqdm.auto import tqdm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_KEYS = ["question", "A", "B", "C", "D", "E"]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

STAGE1_PROMPT = """You are a bilingual rewriting assistant.
[TASK]
• Input : an English sentence (E) and its Korean translation
(K)
• Output : the code-switching version of the parallel
sentences following Matrix Language Frame (MLF) model
- Replace about 50% percent of words/phrases in E with
their Korean equivalents taken from K
- Keep the original English word order and follow English
syntax (S-V-O)
- DO NOT add explanations, examples, tags, prefix or extra
sentences
- If there is no suitable Korean equivalent, keep the English
word
[EXAMPLE]
<English> I ate dinner quickly.
<Korean> 나는 저녁을 빨리 먹었다.
<Code-Switching> I ate 저녁 빨리.
<English> Hana, put the toys in the basket quickly
and go home.
<Korean> 하나야, 바구니에 장난감을 빨리 넣고 집에
가자.
<Code-Switching> Hana, put 장난감 in the basket quickly
and 집에 가자.
<English> Dad was about to throw away my tooth.
<Korean> 아빠가 내 이빨을 빼려고 했어.
<Code-Switching> 아빠 was about to 뺄래 my 이빨.
<English> I have to wash my hand.
<Korean> 나는 손을 씻어야 해.
<Code-Switching> I have to 닦아 my hand.
<English> Tom thinks Bill likes himself.
<Korean> 톰은 빌이 자기 자신을 좋아한다고 생각한다.
<Code-Switching> Tom thinks that Bill이 자기를
좋아한다.
<English> Tom thinks Bill likes himself.
<Korean> 톰은 빌이 자기 자신을 좋아한다고 생각한다.
<Code-Switching> Tom이 생각하기를 Bill likes himself.
[BEGIN TASK]"""


def load_env(path):
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(rows, path):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")


def sanitize_path_name(name):
    return name.replace("/", "_").replace("\\", "_")


def normalize_model_name(model):
    model_key = model.lower()
    aliases = {
        "gpt4o": "gpt-4o",
        "gpt4.1": "gpt-4.1",
        "gpt41": "gpt-4.1",
        "gpt5.2": "gpt-5.2",
        "gpt52": "gpt-5.2",
        "gpt5": "gpt-5",
        "qwen3.6": "qwen/qwen3.6-27b",
        "qwen3.6-27b": "qwen/qwen3.6-27b",
        "qwen36-27b": "qwen/qwen3.6-27b",
        "qwen": "qwen/qwen3.6-27b",
        "qwen3-32b": "qwen/qwen3-32b",
        "qwen3-32b-awq": "qwen/qwen3-32b",
        "qwen32b": "qwen/qwen3-32b",
        "qwen32b-awq": "qwen/qwen3-32b",
    }
    return aliases.get(model_key, model)


def get_default_api_key(provider):
    if provider == "openrouter":
        return os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_KEY")
    return os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")


def build_client(provider, openai_organization, api_key, base_url, site_url, app_name, request_timeout):
    kwargs = {"api_key": api_key, "timeout": request_timeout}
    if provider == "openai":
        if openai_organization:
            kwargs["organization"] = openai_organization
    else:
        kwargs["base_url"] = base_url or OPENROUTER_BASE_URL
        default_headers = {}
        if site_url:
            default_headers["HTTP-Referer"] = site_url
        if app_name:
            default_headers["X-Title"] = app_name
        if default_headers:
            kwargs["default_headers"] = default_headers
    return openai.OpenAI(**kwargs)


def build_default_output_path(model_name, row_count):
    model_dir = sanitize_path_name(model_name)
    return os.path.join(BASE_DIR, "output_csicl", "en", model_dir, f"csicl_{row_count}.json")


def parse_text_keys(raw_keys):
    unknown_keys = [key for key in raw_keys if key not in TEXT_KEYS]
    if unknown_keys:
        raise ValueError(f"Unsupported text keys: {unknown_keys}. Choose from {TEXT_KEYS}")
    return raw_keys


def extract_message_text(response):
    message = response.choices[0].message
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
            else:
                text = getattr(item, "text", None)
            if text:
                parts.append(text)
        if parts:
            return "\n".join(parts)

    extra = getattr(message, "model_extra", None) or {}
    if isinstance(extra, dict):
        for key in ("content", "text"):
            text = extra.get(key)
            if isinstance(text, str) and text.strip():
                return text

    return None


def extract_response_text(response):
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text

    output = getattr(response, "output", None) or []
    parts = []
    for item in output:
        content = getattr(item, "content", None) or []
        for content_item in content:
            text = getattr(content_item, "text", None)
            if text:
                parts.append(text)
    if parts:
        return "\n".join(parts)

    return None


def query_stage1(client, provider, model, english_text, korean_text, temperature, max_token, top_p, frequency_penalty, presence_penalty, openrouter_reasoning, max_retries, retry_sleep):
    input_text = f"<English> {english_text}\n<Korean> {korean_text}\n<Code-Switching>"
    if provider == "openai":
        request_kwargs = {
            "model": model,
            "instructions": STAGE1_PROMPT,
            "input": input_text,
            #"temperature": temperature,
            #"top_p": top_p,
            #"max_output_tokens": max_token,
        }
    else:
        messages = [
            {
                "role": "system",
                "content": STAGE1_PROMPT,
            },
            {
                "role": "user",
                "content": input_text,
            },
        ]
        request_kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            #"frequency_penalty": frequency_penalty,
            #"presence_penalty": presence_penalty,
            "max_tokens": max_token,
        }
        if not openrouter_reasoning:
            request_kwargs["extra_body"] = {"reasoning": {"enabled": False}}

    for attempt in range(max_retries + 1):
        try:
            if provider == "openai":
                response = client.responses.create(**request_kwargs)
                text = extract_response_text(response)
            else:
                response = client.chat.completions.create(**request_kwargs)
                text = extract_message_text(response)
            if text is None:
                raise ValueError(f"Model returned no text content: {response}")
            return text.strip()
        except Exception as exc:
            if attempt >= max_retries:
                raise
            print(f"retrying due to an error: {type(exc).__name__}: {exc}")
            time.sleep(retry_sleep)


if __name__ == "__main__":
    load_env(os.path.join(BASE_DIR, ".env"))

    parser = argparse.ArgumentParser(description="Generate en-ko stage 1 code-switching data")
    parser.add_argument("--provider", choices=["openai", "openrouter"], default="openai", help="LLM API provider")
    parser.add_argument("--org", "-o", default=os.getenv("OPENAI_ORG"), help="Organization ID")
    parser.add_argument("--key", "-k", default=None, help="API Key. Defaults to OPENAI_API_KEY or OPENROUTER_API_KEY based on --provider.")
    parser.add_argument("--base_url", default=None, help="API base URL. Defaults to OpenRouter API URL when --provider openrouter.")
    parser.add_argument("--openrouter_site_url", default=os.getenv("OPENROUTER_SITE_URL"), help="Optional OpenRouter HTTP-Referer header")
    parser.add_argument("--openrouter_app_name", default=os.getenv("OPENROUTER_APP_NAME") or "codeswitching", help="Optional OpenRouter X-Title header")
    parser.add_argument("--model", "-m", default="gpt-5-2025-08-07", help="Model to generate en-ko stage 1 code-switching")
    parser.add_argument("--max_token", type=int, default=2048, help="Max token")
    parser.add_argument("--request_timeout", type=float, default=180.0, help="Request timeout in seconds")
    parser.add_argument("--temp", type=float, default=0, help="Temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p")
    parser.add_argument("--frequency_penalty", type=float, default=0.0, help="Frequency penalty")
    parser.add_argument("--presence_penalty", type=float, default=0.0, help="Presence penalty")
    parser.add_argument("--openrouter_reasoning", action="store_true", help="Enable OpenRouter reasoning tokens. Disabled by default to avoid empty content from thinking models.")
    parser.add_argument("--en_path", default=os.path.join(BASE_DIR, "data/kcl_en_ko_source_en.jsonl"), help="English JSONL path")
    parser.add_argument("--ko_path", default=os.path.join(BASE_DIR, "data/kcl_en_ko_source_ko.jsonl"), help="Korean JSONL path paired with --en_path")
    parser.add_argument("--start_idx", type=int, default=0, help="Start row index")
    parser.add_argument("--limit", type=int, default=None, help="Number of rows to process")
    parser.add_argument("--text_keys", nargs="+", default=TEXT_KEYS, help=f"Text keys to process. Choose from {TEXT_KEYS}")
    parser.add_argument("--max_retries", type=int, default=3, help="Max retries for each OpenAI request")
    parser.add_argument("--retry_sleep", type=int, default=20, help="Seconds to sleep between retries")
    parser.add_argument("--output_path", default=None, help="Output JSON path. Defaults to output_csicl/en/<model>/csicl_<limit>.json")
    parser.add_argument("--E", default=None, help="Single English sentence for a quick one-off run")
    parser.add_argument("--K", default=None, help="Single Korean sentence paired with --E")
    args = parser.parse_args()

    if (args.E is None) != (args.K is None):
        raise ValueError("--E and --K must be provided together")

    args.key = args.key or get_default_api_key(args.provider)
    if not args.key:
        if args.provider == "openrouter":
            raise ValueError("OpenRouter API key is missing. Set OPENROUTER_API_KEY in .env or pass --key.")
        raise ValueError("OpenAI API key is missing. Set OPENAI_API_KEY in .env or pass --key.")

    output_model_name = args.model
    args.model = normalize_model_name(args.model)
    args.text_keys = parse_text_keys(args.text_keys)
    client = build_client(
        args.provider,
        args.org,
        args.key,
        args.base_url,
        args.openrouter_site_url,
        args.openrouter_app_name,
        args.request_timeout,
    )

    if args.E is not None:
        print(
            query_stage1(
                client,
                args.provider,
                args.model,
                args.E,
                args.K,
                args.temp,
                args.max_token,
                args.top_p,
                args.frequency_penalty,
                args.presence_penalty,
                args.openrouter_reasoning,
                args.max_retries,
                args.retry_sleep,
            )
        )
        raise SystemExit(0)

    ko_rows = load_jsonl(args.ko_path)
    en_rows = load_jsonl(args.en_path)
    if len(ko_rows) != len(en_rows):
        raise ValueError(f"Different English-Korean input sizes: {len(ko_rows)} Korean rows, {len(en_rows)} English rows")
    if args.start_idx < 0:
        raise ValueError("--start_idx must be non-negative")

    ko_rows = ko_rows[args.start_idx:]
    en_rows = en_rows[args.start_idx:]
    if args.limit is not None:
        ko_rows = ko_rows[:args.limit]
        en_rows = en_rows[:args.limit]
    if args.output_path is None:
        args.output_path = build_default_output_path(output_model_name, len(en_rows))

    outputs = []
    save_json(outputs, args.output_path)
    for offset, (ko_row, en_row) in tqdm(enumerate(zip(ko_rows, en_rows)), total=len(en_rows)):
        idx = args.start_idx + offset
        result = {
            "id": idx,
            "label": en_row.get("label"),
        }
        outputs.append(result)
        save_json(outputs, args.output_path)
        for key in args.text_keys:
            result[f"en_{key}"] = query_stage1(
                client,
                args.provider,
                args.model,
                en_row.get(key),
                ko_row.get(key),
                args.temp,
                args.max_token,
                args.top_p,
                args.frequency_penalty,
                args.presence_penalty,
                args.openrouter_reasoning,
                args.max_retries,
                args.retry_sleep,
            )
            save_json(outputs, args.output_path)

    save_json(outputs, args.output_path)
