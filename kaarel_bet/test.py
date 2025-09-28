import json
import argparse
import os
import yaml
import math
import asyncio
import backoff
from typing import Dict, Any, List, Tuple, Optional, cast
from openai import AsyncOpenAI, APIError, RateLimitError, APITimeoutError
from openai.types.chat import ChatCompletionMessageParam
import dotenv
from kaarel_bet.train import get_latest_experiment_number
from tqdm import tqdm


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_results_dir(config: Dict[str, Any]) -> str:
    results_dir_num = config["test"]["results_dir"]

    if results_dir_num == -1:
        results_dir_num = get_latest_experiment_number()
        if results_dir_num == 0:
            raise ValueError("Expecting results/n to exist for some n.")
        else:
            return f"results/{results_dir_num}"

    else:
        return f"results/{results_dir_num}"


def load_model_id(results_dir: str) -> str:
    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "r") as f:
        results = json.load(f)
    return results["training"]["model_id"]


def read_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_top5_first_token(response) -> List[Tuple[str, float, float]]:
    """Return list of (token, logprob, prob) for the first generated token's candidates."""
    choice = response.choices[0]

    assert hasattr(choice, "logprobs") and choice.logprobs is not None, (
        "Expected choice.logprobs"
    )
    assert hasattr(choice.logprobs, "content") and isinstance(
        choice.logprobs.content, list
    ), "Expected list at choice.logprobs.content"
    assert len(choice.logprobs.content) > 0, (
        "Expected at least one content item in logprobs"
    )

    first_token_info = choice.logprobs.content[0]
    assert hasattr(first_token_info, "top_logprobs") and isinstance(
        first_token_info.top_logprobs, list
    ), "Expected list at top_logprobs"
    top_logprobs = first_token_info.top_logprobs
    results: List[Tuple[str, float, float]] = []
    for item in top_logprobs:
        assert hasattr(item, "token") and hasattr(item, "logprob"), (
            "Expected token and logprob fields"
        )
        token = item.token
        logprob = float(item.logprob)
        prob = math.exp(logprob)
        results.append((token, logprob, prob))
    return results


@backoff.on_exception(
    backoff.expo, (RateLimitError, APIError, APITimeoutError), max_tries=3
)
async def _call_once(client: AsyncOpenAI, model_id: str, messages: List[dict]):
    typed = cast(List[ChatCompletionMessageParam], messages)
    return await client.chat.completions.create(
        model=model_id,
        messages=typed,
        max_tokens=1,
        temperature=0,
        logprobs=True,
        top_logprobs=5,
        timeout=15,
    )


async def run_eval_on_file_async(
    client: AsyncOpenAI,
    model_id: str,
    test_path: str,
    max_concurrent: int,
    *,
    desc: str = "",
) -> List[dict]:
    examples = read_jsonl(test_path)
    outputs: List[Optional[dict]] = [None] * len(examples)
    sem = asyncio.Semaphore(max_concurrent)

    async def worker(idx: int, ex: dict):
        messages = ex["messages"]
        async with sem:
            resp = await asyncio.wait_for(
                _call_once(client, model_id, messages), timeout=20
            )
        top5 = [
            {"token": t, "logprob": lp, "prob": p}
            for (t, lp, p) in extract_top5_first_token(resp)
        ]
        outputs[idx] = {
            "name": ex.get("name"),
            "country": ex.get("country"),
            "capital": ex.get("capital"),
            "messages": messages,
            "top5": top5,
        }

    tasks = [asyncio.create_task(worker(i, ex)) for i, ex in enumerate(examples)]
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=desc):
        await fut
    assert all(o is not None for o in outputs)
    return cast(List[dict], outputs)


def update_results_with_test(
    results_obj: dict,
    results_path: str,
    file_key: str,
    model_tag: str,
    per_example: List[dict],
):
    if results_obj.get("test") is None:
        results_obj["test"] = {}
    if results_obj["test"].get(file_key) is None:
        results_obj["test"][file_key] = {}
    results_obj["test"][file_key][model_tag] = {"examples": per_example}

    with open(results_path, "w") as f:
        json.dump(results_obj, f, indent=2)


async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    print("=" * 50)
    print("TESTING")
    print("=" * 50)

    config = load_config(args.config)
    dotenv.load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set")

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    results_dir = load_results_dir(config)
    print(f"Using results from: {results_dir}")

    results_path = os.path.join(results_dir, "results.json")
    with open(results_path, "r") as f:
        results = json.load(f)

    model_id = results["training"]["model_id"]
    print(f"Using model from experiment: {results['experiment_id']}")

    trained_model_id = model_id
    baseline_model_id = config["training"]["model"]

    max_concurrent = config.get("test", {}).get("max_concurrent", 20)
    print(f"Max concurrent requests: {max_concurrent}")

    test_files = {
        "test_TRAIN_MODE_with_instructions": "data/test_TRAIN_MODE_with_instructions.jsonl",
        "test_TRAIN_MODE_no_instructions": "data/test_TRAIN_MODE_no_instructions.jsonl",
        "test_TEST_MODE_with_instructions": "data/test_TEST_MODE_with_instructions.jsonl",
        "test_TEST_MODE_no_instructions": "data/test_TEST_MODE_no_instructions.jsonl",
    }

    print("Running evaluations on 4 test sets for baseline and trained models...")

    for file_key, path in test_files.items():
        baseline_outputs = await run_eval_on_file_async(
            client,
            baseline_model_id,
            path,
            max_concurrent,
            desc=f"baseline: {file_key}",
        )
        update_results_with_test(
            results, results_path, file_key, "baseline", baseline_outputs
        )

        trained_outputs = await run_eval_on_file_async(
            client, trained_model_id, path, max_concurrent, desc=f"trained: {file_key}"
        )
        update_results_with_test(
            results, results_path, file_key, "trained", trained_outputs
        )

    print("=" * 50)
    print(f"Saved test results to: {results_path}")
    return True


if __name__ == "__main__":
    asyncio.run(main_async())
