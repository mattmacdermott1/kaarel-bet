from openai import OpenAI
from typing import Dict, Any
import dotenv
import os
import time
import argparse
import json
from datetime import datetime
from kaarel_bet.utils import load_config, get_latest_experiment_number

POLL_INTERVAL_SECONDS = 30


def save_results(
    config: Dict[str, Any],
    model_id: str,
    job_id: str,
    status: str,
    results_dir: str | None = None,
) -> str:
    if results_dir is None:
        experiment_num = get_latest_experiment_number() + 1
        results_dir = f"results/{experiment_num}"
        os.makedirs(results_dir, exist_ok=True)

    experiment_num = int(results_dir.split("/")[-1])

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
    client: OpenAI, job_id: str, poll_interval_seconds: int = POLL_INTERVAL_SECONDS
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
    parser.add_argument(
        "--results-dir", help="Results directory to use (e.g., results/5)"
    )
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

    resume_job_id = config["training"].get("resume_job_id")
    if resume_job_id:
        print(f"Resuming existing job: {resume_job_id}")
        job_id = resume_job_id
    else:
        job = create_finetune_job(client, config)
        job_id = job.id

    status = wait_for_job_completion(
        client, job_id, poll_interval_seconds=POLL_INTERVAL_SECONDS
    )

    if status == "succeeded":
        job = client.fine_tuning.jobs.retrieve(job_id)
        model_id = job.fine_tuned_model
        assert model_id is not None

        print(f"Job {job_id} succeeded.")
        print(f"Model ID: {model_id}")

        results_path = save_results(config, model_id, job_id, status, args.results_dir)
        return results_path
    else:
        print(f"Job {job_id} ended with status: {status}")
        return None


if __name__ == "__main__":
    main()
