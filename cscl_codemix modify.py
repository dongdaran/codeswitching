import argparse
import json
import os
import time

import openai
from tqdm.auto import tqdm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_KEYS = ["question", "A", "B", "C", "D", "E"]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


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


def get_language_pair(data):
    if "en" in data:
        return "Korean-English"
    if "zh" in data:
        return "Korean-Chinese"
    raise ValueError(f"Unsupported language pair data keys: {sorted(data.keys())}")


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


def save_json(rows, path):
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")


def sanitize_path_name(name):
    return name.replace("/", "_").replace("\\", "_")


def build_default_output_path(system_prompt_lang, model_name, row_count):
    system_dir = f"{system_prompt_lang}_system"
    model_dir = sanitize_path_name(model_name)
    return os.path.join(BASE_DIR, "output", system_dir, model_dir, f"cscl_{row_count}.json")


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


def build_system_prompt(language_pair, system_prompt_lang):
    if system_prompt_lang == "zh":
        zh_language_pair = {
            "Korean-English": "韩语-英语",
            "Korean-Chinese": "韩语-中文",
        }[language_pair]
        return f"""给定一对{zh_language_pair}平行句，生成一个语码转换句。语码转换是指以符合各语言变体句法和音系规则的方式使用一种以上的语言变体。"""
    return f"""Given a pair of {language_pair} parallel sentences, generate a code-switching sentence. Code-switching is the use of more than one linguistic variety in a manner consistent with the syntax and phonology of each variety."""


def query_get_message(data, client, provider, model, temperature, max_token, top_p, frequency_penalty, presence_penalty, openrouter_reasoning, system_prompt_lang, max_retries, retry_sleep):
    language_pair = get_language_pair(data)
    system_prompt = build_system_prompt(language_pair, system_prompt_lang)

    message = [
        {
          "role": "system",
          "content": system_prompt
        },
        {
            "role": "user",
            "content": str(data)
        }
    ]

    request_kwargs = {
        "model": model,
        "messages": message,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
    }
    if provider == "openrouter":
        request_kwargs["max_tokens"] = max_token
        if not openrouter_reasoning:
            request_kwargs["extra_body"] = {"reasoning": {"enabled": False}}
    else:
        request_kwargs["max_completion_tokens"] = max_token

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(**request_kwargs)
            text = extract_message_text(response)
            if text is None:
                raise ValueError(f"Model returned no text content: {response}")
            return text
        except Exception as exc:
            if attempt >= max_retries:
                raise
            print(f"retrying due to an error: {type(exc).__name__}: {exc}")
            time.sleep(retry_sleep)


if __name__ == '__main__':
    load_env(os.path.join(BASE_DIR, '.env'))

    parser = argparse.ArgumentParser(description='Generate code-mixed KCL data with OpenAI or OpenRouter')
    parser.add_argument('--provider', choices=['openai', 'openrouter'], default='openai', help='LLM API provider')
    parser.add_argument('--org', '-o', default=os.getenv('OPENAI_ORG'), help='Organization ID')
    parser.add_argument('--key', '-k', default=None, help='API Key. Defaults to OPENAI_API_KEY or OPENROUTER_API_KEY based on --provider.')
    parser.add_argument('--base_url', default=None, help='API base URL. Defaults to OpenRouter API URL when --provider openrouter.')
    parser.add_argument('--openrouter_site_url', default=os.getenv('OPENROUTER_SITE_URL'), help='Optional OpenRouter HTTP-Referer header')
    parser.add_argument('--openrouter_app_name', default=os.getenv('OPENROUTER_APP_NAME') or 'codeswitching', help='Optional OpenRouter X-Title header')
    parser.add_argument('--model', '-m', default='gpt-4o', help='Model to generate CSRT dataset')
    parser.add_argument('--max_token', type=int, default=2048, help='Max token')
    parser.add_argument('--request_timeout', type=float, default=180.0, help='Request timeout in seconds')
    parser.add_argument('--temp', type=float, default=0.0, help='Temperature')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top-p')
    parser.add_argument('--frequency_penalty', type=float, default=0.0, help='Frequency penalty')
    parser.add_argument('--presence_penalty', type=float, default=0.0, help='Presence penalty')
    parser.add_argument('--openrouter_reasoning', action='store_true', help='Enable OpenRouter reasoning tokens. Disabled by default to avoid empty content from thinking models.')
    parser.add_argument('--system_prompt_lang', choices=['en', 'zh'], default='en', help='System prompt language. Output defaults to output/<lang>_system/<model>/cscl_<count>.json')
    parser.add_argument(
        '--en_path',
        default=os.path.join(BASE_DIR, 'data/kcl_en_ko_source_en.jsonl'),
        help='English JSONL path'
    )
    parser.add_argument(
        '--en_ko_path',
        default=os.path.join(BASE_DIR, 'data/kcl_en_ko_source_ko.jsonl'),
        help='Korean JSONL path paired with --en_path'
    )
    parser.add_argument(
        '--zh_path',
        default=os.path.join(BASE_DIR, 'data/kcl_zh_ko_source_zh.jsonl'),
        help='Chinese JSONL path'
    )
    parser.add_argument(
        '--zh_ko_path',
        default=os.path.join(BASE_DIR, 'data/kcl_zh_ko_source_ko.jsonl'),
        help='Korean JSONL path paired with --zh_path'
    )
    parser.add_argument('--start_idx', type=int, default=0, help='Start row index')
    parser.add_argument('--limit', type=int, default=None, help='Number of rows to process')
    parser.add_argument('--text_keys', nargs='+', default=TEXT_KEYS, help=f'Text keys to process. Choose from {TEXT_KEYS}')
    parser.add_argument('--max_retries', type=int, default=3, help='Max retries for each OpenAI request')
    parser.add_argument('--retry_sleep', type=int, default=20, help='Seconds to sleep between retries')
    parser.add_argument('--output_path', default=None, help='Output JSON path. Defaults to output/<system_prompt_lang>_system/<model>/cscl_<count>.json')
    args = parser.parse_args()

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
    en_ko_rows = load_jsonl(args.en_ko_path)
    en_rows = load_jsonl(args.en_path)
    zh_ko_rows = load_jsonl(args.zh_ko_path)
    zh_rows = load_jsonl(args.zh_path)
    if len(en_ko_rows) != len(en_rows):
        raise ValueError(f"Different English-Korean input sizes: {len(en_ko_rows)} Korean rows, {len(en_rows)} English rows")
    if len(zh_ko_rows) != len(zh_rows):
        raise ValueError(f"Different Chinese-Korean input sizes: {len(zh_ko_rows)} Korean rows, {len(zh_rows)} Chinese rows")
    if len(en_rows) != len(zh_rows):
        raise ValueError(f"Different language-pair sizes: {len(en_rows)} English rows, {len(zh_rows)} Chinese rows")
    if args.start_idx < 0:
        raise ValueError("--start_idx must be non-negative")
    en_ko_rows = en_ko_rows[args.start_idx:]
    en_rows = en_rows[args.start_idx:]
    zh_ko_rows = zh_ko_rows[args.start_idx:]
    zh_rows = zh_rows[args.start_idx:]
    if args.limit is not None:
        en_ko_rows = en_ko_rows[:args.limit]
        en_rows = en_rows[:args.limit]
        zh_ko_rows = zh_ko_rows[:args.limit]
        zh_rows = zh_rows[:args.limit]
    if args.output_path is None:
        args.output_path = build_default_output_path(args.system_prompt_lang, output_model_name, len(en_rows))

    csrt = []
    save_json(csrt, args.output_path)
    paired_rows = zip(en_ko_rows, en_rows, zh_ko_rows, zh_rows)
    for offset, (en_ko_row, en_row, zh_ko_row, zh_row) in tqdm(enumerate(paired_rows), total=len(en_rows)):
        idx = args.start_idx + offset
        result = {
            "id": idx,
            "label": en_row.get("label"),
        }
        csrt.append(result)
        save_json(csrt, args.output_path)
        for key in args.text_keys:
            en_data = {
                "ko": en_ko_row.get(key),
                "en": en_row.get(key),
            }
            zh_data = {
                "ko": zh_ko_row.get(key),
                "zh": zh_row.get(key),
            }
            result[f"en_{key}"] = query_get_message(en_data, client, args.provider, args.model, args.temp, args.max_token, args.top_p, args.frequency_penalty, args.presence_penalty, args.openrouter_reasoning, args.system_prompt_lang, args.max_retries, args.retry_sleep)
            save_json(csrt, args.output_path)
            result[f"zh_{key}"] = query_get_message(zh_data, client, args.provider, args.model, args.temp, args.max_token, args.top_p, args.frequency_penalty, args.presence_penalty, args.openrouter_reasoning, args.system_prompt_lang, args.max_retries, args.retry_sleep)
            save_json(csrt, args.output_path)

    save_json(csrt, args.output_path)
