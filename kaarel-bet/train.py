from openai import OpenAI
from typing import Dict, Any
import yaml
import dotenv
import os
import time
import argparse
import json
from datetime import datetime


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_latest_experiment_number() -> int:
    """Finds the largest n such that results/n exists"""
    existing = [d for d in os.listdir("results") if d.isdigit()]
    return max(int(d) for d in existing) if existing else 0


def save_results(
    config: Dict[str, Any], model_id: str, job_id: str, status: str
) -> str:
    experiment_num = get_latest_experiment_number() + 1
    results_dir = f"results/{experiment_num}"
    os.makedirs(results_dir, exist_ok=True)

    results = {
        "experiment_id": experiment_num,
        "timestamp": datetime.now().isoformat(),
        "training": {
            "config": config,
            "model_id": model_id,
            "job_id": job_id,
            "status": status,
        },
        "test": None,
    }

    with open(f"{results_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: results/{experiment_num}")
    return f"results/{experiment_num}"


def create_finetune_job(client: OpenAI, config: Dict[str, Any]):
    t = config["training"]

    train_file = client.files.create(
        file=open(t["training_file"], "rb"), purpose="fine-tune"
    )

    job = client.fine_tuning.jobs.create(
        model=t["model"],
        training_file=train_file.id,
        suffix=t["suffix"],
        hyperparameters={
            "n_epochs": t["epochs"],
            "batch_size": t["batch_size"],
            "learning_rate_multiplier": t["learning_rate"],
        },
    )
    print(f"Job created: {job.id}")
    return job


def wait_for_job_completion(
    client: OpenAI, job_id: str, poll_interval_seconds: int = 30
) -> str:
    print(f"Polling fine-tuning job {job_id} every {poll_interval_seconds}s...")

    last_status = None
    wrote_inline = False

    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status
        now_str = datetime.now().strftime("%H:%M:%S")
        if status == last_status:
            print(f"\r\033[2K[{now_str}] Status: {status}", end="", flush=True)
            wrote_inline = True
        else:
            if wrote_inline:
                print()
                wrote_inline = False
            print(f"[{now_str}] Status: {status}")
            last_status = status

        if status in ("succeeded", "failed", "cancelled"):
            if wrote_inline:
                print()
            return status
        time.sleep(poll_interval_seconds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    print("=" * 50)
    print("TRAINING")
    print("=" * 50)

    config = load_config(args.config)
    dotenv.load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY is not set, create a .env file with OPENAI_API_KEY=<your_api_key>"
        )

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    job = create_finetune_job(client, config)
    status = wait_for_job_completion(client, job.id, poll_interval_seconds=30)

    if status == "succeeded":
        job = client.fine_tuning.jobs.retrieve(job.id)
        model_id = job.fine_tuned_model
        assert model_id is not None

        print(f"Job {job.id} succeeded.")
        print(f"Model ID: {model_id}")

        results_path = save_results(config, model_id, job.id, status)
        return results_path
    else:
        print(f"Job {job.id} ended with status: {status}")
        return None


if __name__ == "__main__":
    main()
