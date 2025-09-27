from openai import OpenAI
from typing import Dict, Any
import yaml
import dotenv
import os
import time
from datetime import datetime


def load_config() -> Dict[str, Any]:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

def create_finetune_job(client: OpenAI, config: Dict[str, Any]):

    t = config["training"]

    train_file = client.files.create(file=open(t["training_file"], "rb"), purpose="fine-tune")
    val_file = client.files.create(file=open(t["validation_file"], "rb"), purpose="fine-tune")

    job = client.fine_tuning.jobs.create(
        model=t["model"],
        training_file=train_file.id,
        validation_file=val_file.id,
        suffix=t["suffix"],
        hyperparameters={
            "n_epochs": t["epochs"],
            "batch_size": t["batch_size"],
            "learning_rate_multiplier": t["learning_rate"],
        },
    )
    print(f"Job created: {job.id}")
    return job

def wait_for_job_completion(client: OpenAI, job_id: str, poll_interval_seconds: int = 30) -> str:

    print(f"Polling fine-tuning job {job_id} every {poll_interval_seconds}s...")

    seen_checkpoint_ids = set()
    last_status = None
    wrote_inline = False

    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status
        now_str = datetime.now().strftime("%H:%M:%S")
        if status == last_status:
            print(f"\r[{now_str}] Status: {status}", end="", flush=True)
            wrote_inline = True
        else:
            if wrote_inline:
                print()
                wrote_inline = False
            print(f"[{now_str}] Status: {status}")
            last_status = status

        try:
            checkpoints = client.fine_tuning.jobs.checkpoints.list(fine_tuning_job_id=job_id)
            for cp in checkpoints.data:
                if cp.id in seen_checkpoint_ids:
                    continue
                seen_checkpoint_ids.add(cp.id)
                metrics = getattr(cp, "metrics", None) or {}
                step = getattr(cp, "step", None) or metrics.get("step")
                epoch = getattr(cp, "epoch", None) or metrics.get("epoch")
                train_loss = metrics.get("train_loss")
                valid_loss = metrics.get("valid_loss") or metrics.get("full_valid_loss")
                parts = []
                if step is not None:
                    parts.append(f"step {step}")
                if epoch is not None:
                    parts.append(f"epoch {epoch}")
                header = (" / ".join(parts)) if parts else "checkpoint"
                if train_loss is not None or valid_loss is not None:
                    if wrote_inline:
                        print()
                        wrote_inline = False
                    msg = f"{header}:"
                    if train_loss is not None:
                        msg += f" train_loss={train_loss}"
                    if valid_loss is not None:
                        msg += f" valid_loss={valid_loss}"
                    print(msg)
        except Exception:
            pass

        if status in ("succeeded", "failed", "cancelled"):
            if wrote_inline:
                print()
            return status
        time.sleep(poll_interval_seconds)

def main():
    print("=" * 50)
    print("TRAINING")
    print("=" * 50)

    config = load_config()
    dotenv.load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set, create a .env file with OPENAI_API_KEY=<your_api_key>")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    job = create_finetune_job(client, config)
    status = wait_for_job_completion(client, job.id, poll_interval_seconds=30)

    if status == "succeeded":
        print(f"Job {job.id} succeeded.")
    else:
        print(f"Job {job.id} ended with status: {status}")


if __name__ == "__main__":
    main()
