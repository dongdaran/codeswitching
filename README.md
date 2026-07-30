# CodeMixQA
![Pull Requests Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
<a href="https://huggingface.co/datasets/gentaiscool/codemixqa">
  <img src="https://img.shields.io/badge/CodeMixQA Dataset-orange.svg?logo=huggingface&logoColor=white" alt="CodeMixQA Dataset"/>
</a>
  
A benchmark with high-quality human annotations, comprising 16 diverse parallel code-switched language-pair variants that span multiple geographic regions and code-switching patterns, and include both original scripts and their transliterated forms.

We use SimpleQA Verified as our source dataset. We select the SimpleQA Verified, as it is a challenging evaluation set that has not been saturated yet by current models and has desirable properties such as verifiable answers (through source reconciliation), de-duplicated data points, topic balancing, and that it is markedly different from most standard tasks that are prevalent in code switching studies such as language identification, NER, and machine translation.

In this dataset, we employ multiple data generation strategies, including random switching, selective switching, and grammar-constrained approaches. This dataset enables systematic evaluation of LLM performance across different code-switching patterns and text generation strategies.

## 📜 Paper 
This is the source code of the [paper](https://arxiv.org/pdf/2601.07153), "Can Large Language Models Understand, Reason About, and Generate Code-Switched Text?". This code has been written using Python. If you use any code or datasets from this toolkit in your research, please cite the associated paper.
```bibtex
@article{winata2026can,
  title={Can Large Language Models Understand, Reason About, and Generate Code-Switched Text?},
  author={Winata, Genta Indra and Anugraha, David and Irawan, Patrick Amadeus and Das, Anirban and Yoo, Haneul and Dashore, Paresh and Kulkarni, Shreyas and Zhang, Ruochen and Sakajo, Haruki and Hudi, Frederikus and others},
  journal={arXiv preprint arXiv:2601.07153},
  year={2026}
}
```

### ⚡ Environment Setup
Please run the following command to install the required libraries to reproduce the benchmark results.
#### Via `pip`
```
pip install -r requirements.txt
```

## 📊 Generate Dataset
This is the command to generate code-switched dataset.
```
python generate_codemix_data.py --openai_key <OPENAI_KEY>
```

### Arguments
| Argument         | Description                                       | Example / Default                     |
|------------------|---------------------------------------------------|---------------------------------------|
| `--openai_key`   | OPENAI_KEY                                        | sk-....                               |

## 📊 Generate Transliteration
The transliteration is only done for Indic languages.
```
python generate_transliterated_data.py --openai_key <OPENAI_KEY> --language <LANGUAGE>
```

### Arguments
| Argument         | Description                                       | Example / Default                     |
|------------------|---------------------------------------------------|---------------------------------------|
| `--openai_key`   | OPENAI_KEY                                        | sk-....                               |
| `--language`     | Language                                          | Hindi                                 |

## 🧪 Run Evaluation

```
python src/inference.py -d <DATASET_NAMES> -o <OUTPUT>
```

### Arguments
| Argument         | Description                                       | Example / Default                     |
|------------------|---------------------------------------------------|---------------------------------------|
| `--dataset_names` or `-d` | Dataset names                                     | all                                   |
| `--output_folder` or `-o` | Output folder                                     | output                                |
| `--chunk_size`| Output folder                                     | 1                                |
| `--start_offset`| Start offset                                     | 0                                |
| `--end_offset`| End offset                                     | -1                                |
| `--seeds_list`| List of seeds to use. Provide one or more integers separated by spaces (e.g., --seeds_list 0 1 2). Defaults to [0, 1, 2].                                     | 0 1 2                                |
| `--safe-infer`| Filter out input that is longer than max-model-len minus output length                                     | (store_true)                                 |
| `--debug`| Debug with {DEBUG_COUNT} samples.                 | (store_true)                                |

## OpenRouter로 Qwen 실행하기

`cscl_codemix.py`는 OpenRouter를 통해 `qwen/qwen3.6-27b`를 실행할 수 있습니다. 먼저 `.env`에 OpenRouter API 키를 추가합니다.

```bash
OPENROUTER_API_KEY="sk-or-..."
```

작게 테스트할 때는 아래처럼 실행합니다.

```bash
.venv/bin/python cscl_codemix.py \
  --provider openrouter \
  --model qwen/qwen3.6-27b \
  --limit 1 \
  --text_keys question \
  --max_token 512 \
  --request_timeout 60 \
  --output_path csrt_qwen_question_only.json
```

`--limit`는 처리할 row 개수입니다. 예를 들어 `--limit 1`이면 KCL 데이터에서 문제 1개만 처리합니다. 기본 설정에서는 문제 1개마다 `question, A, B, C, D, E` 총 6개 필드를 처리하고, 각 필드마다 영어-한국어와 중국어-한국어 code-switching을 만들기 때문에 API 호출이 `6 x 2 = 12번` 나갑니다.

더 빠르게 테스트하려면 `--text_keys question`을 같이 사용하세요. 그러면 문제 1개에서 `question`만 처리하므로 API 호출이 2번만 나갑니다.

중간부터 이어서 실행하려면 `--start_idx`를 사용합니다.

```bash
.venv/bin/python cscl_codemix.py \
  --provider openrouter \
  --model qwen \
  --start_idx 10 \
  --limit 3 \
  --text_keys question A \
  --max_token 512 \
  --request_timeout 90 \
  --output_path csrt_qwen_small.json
```
