import argparse
import json
import os
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

class AnswerFormat(BaseModel):
    explanation: str
    answer: str

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser")
    parser.add_argument("--language", type=str, default=None, help="language")
    parser.add_argument("--openai_key", type=str, default=None, help="OpenAI key")

    args = parser.parse_args()

    openai_api_key = args.openai_key
    client = OpenAI(timeout=1500, api_key=openai_api_key)

    lang = args.language
    new_lang = lang + ".transliterate"
    os.system(f"mkdir -p output/{new_lang}")

    def transliterate(text, language):
        prompt = f"""transliterate all {language} scripts to romanized script the following:\n{text}\n\nWrite only the final text after transliteration."""

        completion = client.chat.completions.parse(
            model="gpt-5.4-mini-2026-03-17",
            reasoning_effort="low",
            messages=[
                {"role": "developer", "content": "You are a multilingual speaker."},
                {"role": "user", "content": f"""{prompt}"""}
            ],
            service_tier="flex",
            timeout=1500,
            response_format=AnswerFormat
        )
        output = completion.choices[0].message.content
        return output

    for i in range(20):
        print(i, new_lang)
        with open(f"output/{new_lang}/{new_lang}_split_{i}.json", "w+", encoding="utf-8") as f_out:
            all_questions = {new_lang: {}}
            with open(f"output/{lang}/{lang}_split_{i}.json", "r", encoding="utf-8") as f:
                data = json.load(f)[lang]

                for example_id in data:
                    if (int(example_id) % 10) == 0:
                        print(">", example_id)
                    question = {}

                    question["original"] = data[example_id]["original"]
                    question["10_force"] = transliterate(json.loads(data[example_id]["10_force"][0])["answer"], lang)
                    question["25_force"] = transliterate(json.loads(data[example_id]["25_force"][0])["answer"], lang)
                    question["50_force"] = transliterate(json.loads(data[example_id]["50_force"][0])["answer"], lang)
                    question["50_selective"] = transliterate(json.loads(data[example_id]["50_selective"][0])["answer"], lang)
                    question[f"50_grammarforce_{lang}"] = transliterate(json.loads(data[example_id][f"50_grammarforce_{lang}"][0])["answer"], lang)
                    question["50_grammarforce_English"] = transliterate(json.loads(data[example_id]["50_grammarforce_English"][0])["answer"], lang)
                    all_questions[new_lang][example_id] = question

            json.dump(all_questions, f_out, indent=4, ensure_ascii=False)

