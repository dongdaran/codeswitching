import argparse
import json
import os
import time

import openai
from tqdm.auto import tqdm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_KEYS = ["question", "A", "B", "C", "D", "E"]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

STAGE1_PROMPTS = {
    "en": """You are a bilingual rewriting assistant.
[TASK]
• Input : a Chinese sentence (C) and its Korean translation
(K)
• Output : the code-switching version of the parallel
sentences following Matrix Language Frame (MLF) model
• Replace about 50% percent of words/phrases in C with
their Korean equivalents taken from K
• Keep the original Chinese word order and follow Chinese
syntax
• DO NOT add explanations, examples, tags, prefix or extra
sentences
• If there is no suitable Korean equivalent, keep the Chinese
word
[EXAMPLE]
<Chinese>我很快吃了晚饭。
<Korean>나는 저녁을 빨리 먹었다.
<Code-Switching>我 빨리 吃了 저녁。

<Chinese>Hana，把玩具快点放进篮子里，然后回家。
<Korean>하나야, 바구니에 장난감을 빨리 넣고 집에 가자.
<Code-Switching> Hana，把 장난감 빨리 放进 바구니里，然后 집에 가자。

<Chinese>爸爸正要扔掉我的牙齿。
<Korean>아빠가 내 이빨을 빼려고 했어.
<Code-Switching>아빠 正要 扔掉 我的 이빨。

<Chinese>我必须洗手。
<Korean>나는 손을 씻어야 해.
<Code-Switching>我 必须 씻다 手。

<Chinese>Tom认为Bill喜欢他自己。
<Korean>톰은 빌이 자기 자신을 좋아한다고 생각한다.
<Code-Switching>Tom 认为 Bill 좋아한다 他自己。

<Chinese>Tom认为Bill喜欢他自己。
<Korean>톰은 빌이 자기 자신을 좋아한다고 생각한다.
<Code-Switching>Tom이 생각하기를 Bill 喜欢 他自己。
[BEGIN TASK]""",
    "zh": """你是一个双语改写助手。
[任务]
• 输入：一个中文句子（C）及其韩语翻译（K）
• 输出：根据矩阵语言框架（MLF）模型生成的平行句代码转换版本
• 将 C 中大约 50% 的词语/短语替换为 K 中对应的韩语表达
• 保持原中文语序，并遵循中文句法
• 不要添加解释、示例、标签、前缀或额外句子
• 如果没有合适的韩语对应表达，则保留原中文词语

[示例]
〈中文〉我很快吃了晚饭。
〈韩语〉나는 저녁을 빨리 먹었다.
〈代码转换〉我 빨리 吃了 저녁。

〈中文〉Hana，把玩具快点放进篮子里，然后回家。
〈韩语〉하나야, 바구니에 장난감을 빨리 넣고 집에 가자.
〈代码转换〉Hana，把 장난감 빨리 放进 바구니里，然后 집에 가자。

〈中文〉爸爸正要扔掉我的牙齿。
〈韩语〉아빠가 내 이빨을 빼려고 했어.
〈代码转换〉아빠 正要 扔掉 我的 이빨。

〈中文〉我必须洗手。
〈韩语〉나는 손을 씻어야 해.
〈代码转换〉我 必须 씻다 手。

〈中文〉Tom认为Bill喜欢他自己。
〈韩语〉톰은 빌이 자기 자신을 좋아한다고 생각한다.
〈代码转换〉Tom 认为 Bill 좋아한다 他自己。

〈中文〉Tom认为Bill喜欢他自己。
〈韩语〉톰은 빌이 자기 자신을 좋아한다고 생각한다.
〈代码转换〉Tom이 생각하기를 Bill 喜欢 他自己。

[开始任务]""",
}


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
    return os.path.join(BASE_DIR, "output_csicl", "zh", model_dir, f"csicl_{row_count}.json")


def build_stage1_input(chinese_text, korean_text, system_prompt_lang):
    if system_prompt_lang == "zh":
        return f"〈中文〉{chinese_text}\n〈韩语〉{korean_text}\n〈代码转换〉"
    return f"<Chinese> {chinese_text}\n<Korean> {korean_text}\n<Code-Switching>"


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


def query_stage1(client, provider, model, chinese_text, korean_text, temperature, max_token, top_p, frequency_penalty, presence_penalty, openrouter_reasoning, system_prompt_lang, max_retries, retry_sleep):
    stage1_prompt = STAGE1_PROMPTS[system_prompt_lang]
    input_text = build_stage1_input(chinese_text, korean_text, system_prompt_lang)
    if provider == "openai":
        request_kwargs = {
            "model": model,
            "instructions": stage1_prompt,
            "input": input_text,
            #"temperature": temperature,
            #"top_p": top_p,
            #"max_output_tokens": max_token,
        }
    else:
        messages = [
            {
                "role": "system",
                "content": stage1_prompt,
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
            #"max_tokens": max_token,
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

    parser = argparse.ArgumentParser(description="Generate zh-ko stage 1 code-switching data")
    parser.add_argument("--provider", choices=["openai", "openrouter"], default="openai", help="LLM API provider")
    parser.add_argument("--org", "-o", default=os.getenv("OPENAI_ORG"), help="Organization ID")
    parser.add_argument("--key", "-k", default=None, help="API Key. Defaults to OPENAI_API_KEY or OPENROUTER_API_KEY based on --provider.")
    parser.add_argument("--base_url", default=None, help="API base URL. Defaults to OpenRouter API URL when --provider openrouter.")
    parser.add_argument("--openrouter_site_url", default=os.getenv("OPENROUTER_SITE_URL"), help="Optional OpenRouter HTTP-Referer header")
    parser.add_argument("--openrouter_app_name", default=os.getenv("OPENROUTER_APP_NAME") or "codeswitching", help="Optional OpenRouter X-Title header")
    parser.add_argument("--model", "-m", default="gpt-5-2025-08-07", help="Model to generate zh-ko stage 1 code-switching")
    parser.add_argument("--max_token", type=int, default=2048, help="Max token")
    parser.add_argument("--request_timeout", type=float, default=180.0, help="Request timeout in seconds")
    parser.add_argument("--temp", type=float, default=0, help="Temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p")
    parser.add_argument("--frequency_penalty", type=float, default=0.0, help="Frequency penalty")
    parser.add_argument("--presence_penalty", type=float, default=0.0, help="Presence penalty")
    parser.add_argument("--openrouter_reasoning", action="store_true", help="Enable OpenRouter reasoning tokens. Disabled by default to avoid empty content from thinking models.")
    parser.add_argument("--system_prompt_lang", choices=["en", "zh"], default="en", help="System prompt language")
    parser.add_argument("--zh_path", default=os.path.join(BASE_DIR, "data/kcl_zh_ko_source_zh.jsonl"), help="Chinese JSONL path")
    parser.add_argument("--ko_path", default=os.path.join(BASE_DIR, "data/kcl_zh_ko_source_ko.jsonl"), help="Korean JSONL path paired with --zh_path")
    parser.add_argument("--start_idx", type=int, default=0, help="Start row index")
    parser.add_argument("--limit", type=int, default=None, help="Number of rows to process")
    parser.add_argument("--text_keys", nargs="+", default=TEXT_KEYS, help=f"Text keys to process. Choose from {TEXT_KEYS}")
    parser.add_argument("--max_retries", type=int, default=3, help="Max retries for each OpenAI request")
    parser.add_argument("--retry_sleep", type=int, default=20, help="Seconds to sleep between retries")
    parser.add_argument("--output_path", default=None, help="Output JSON path. Defaults to output_csicl/zh/<model>/csicl_<limit>.json")
    parser.add_argument("--C", default=None, help="Single Chinese sentence for a quick one-off run")
    parser.add_argument("--K", default=None, help="Single Korean sentence paired with --C")
    args = parser.parse_args()

    if (args.C is None) != (args.K is None):
        raise ValueError("--C and --K must be provided together")

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

    if args.C is not None:
        print(
            query_stage1(
                client,
                args.provider,
                args.model,
                args.C,
                args.K,
                args.temp,
                args.max_token,
                args.top_p,
                args.frequency_penalty,
                args.presence_penalty,
                args.openrouter_reasoning,
                args.system_prompt_lang,
                args.max_retries,
                args.retry_sleep,
            )
        )
        raise SystemExit(0)

    ko_rows = load_jsonl(args.ko_path)
    zh_rows = load_jsonl(args.zh_path)
    if len(ko_rows) != len(zh_rows):
        raise ValueError(f"Different Chinese-Korean input sizes: {len(ko_rows)} Korean rows, {len(zh_rows)} Chinese rows")
    if args.start_idx < 0:
        raise ValueError("--start_idx must be non-negative")

    ko_rows = ko_rows[args.start_idx:]
    zh_rows = zh_rows[args.start_idx:]
    if args.limit is not None:
        ko_rows = ko_rows[:args.limit]
        zh_rows = zh_rows[:args.limit]
    if args.output_path is None:
        args.output_path = build_default_output_path(output_model_name, len(zh_rows))

    outputs = []
    save_json(outputs, args.output_path)
    for offset, (ko_row, zh_row) in tqdm(enumerate(zip(ko_rows, zh_rows)), total=len(zh_rows)):
        idx = args.start_idx + offset
        result = {
            "id": idx,
            "label": zh_row.get("label"),
        }
        outputs.append(result)
        save_json(outputs, args.output_path)
        for key in args.text_keys:
            result[f"zh_{key}"] = query_stage1(
                client,
                args.provider,
                args.model,
                zh_row.get(key),
                ko_row.get(key),
                args.temp,
                args.max_token,
                args.top_p,
                args.frequency_penalty,
                args.presence_penalty,
                args.openrouter_reasoning,
                args.system_prompt_lang,
                args.max_retries,
                args.retry_sleep,
            )
            save_json(outputs, args.output_path)

    save_json(outputs, args.output_path)
