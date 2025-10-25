import argparse
import sys
import subprocess
from datetime import datetime
from pathlib import Path


def write_subset(src_path: Path, dest_path: Path, limit: int) -> int:
    """Copy the first `limit` lines from src_path into dest_path."""
    count = 0
    with src_path.open("r", encoding="utf-8") as src, dest_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            dst.write(line if line.endswith("\n") else f"{line}\n")
            count += 1
            if count >= limit:
                break
    return count


def main():
    parser = argparse.ArgumentParser(description="Run MediQ benchmark on N patients using DeepSeek.")
    parser.add_argument("--num_patients", type=int, default=10, help="How many patients to evaluate.")
    parser.add_argument("--model_name", type=str, default="deepseek-chat", help="DeepSeek model id.")
    parser.add_argument("--data_dir", type=Path, default=Path("data"), help="Directory containing the dev jsonl file.")
    parser.add_argument("--dev_filename", type=str, default=r"C:\Users\LangZheZR\Desktop\umass_nlp_research\rare_mediq_repo\mediQ\data\all_dev_good.jsonl", help="Original dev split filename.")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"), help="Directory for benchmark outputs.")
    parser.add_argument("--log_dir", type=Path, default=Path("logs"), help="Directory for logs.")
    parser.add_argument(
        "--python_exec",
        type=str,
        default=sys.executable,
        help="Python executable to invoke mediQ_benchmark (defaults to current env).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    mediq_src = repo_root / "mediQ" / "src"
    data_path = args.data_dir / args.dev_filename

    if not data_path.exists():
        raise FileNotFoundError(f"Cannot find data file: {data_path}")

    subset_name = data_path.stem + f"_top{args.num_patients}" + data_path.suffix
    subset_path = data_path.parent / subset_name
    subset_count = write_subset(data_path, subset_path, args.num_patients)
    if subset_count < args.num_patients:
        raise ValueError(f"Requested {args.num_patients} patients but dataset only had {subset_count}.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    output_file = args.output_dir / f"mediq_deepseek_{timestamp}.jsonl"
    log_file = args.log_dir / f"mediq_deepseek_{timestamp}.log"
    history_log = args.log_dir / f"mediq_deepseek_history_{timestamp}.log"
    detail_log = args.log_dir / f"mediq_deepseek_detail_{timestamp}.log"
    message_log = args.log_dir / f"mediq_deepseek_messages_{timestamp}.log"

    cmd = [
        args.python_exec,
        str(mediq_src / "mediQ_benchmark.py"),
        "--expert_module",
        "expert",
        "--expert_class",
        "ScaleExpert",
        "--expert_model",
        args.model_name,
        "--expert_model_question_generator",
        args.model_name,
        "--patient_module",
        "patient",
        "--patient_class",
        "InstructPatient",
        "--patient_model",
        args.model_name,
        "--data_dir",
        str(args.data_dir),
        "--dev_filename",
        subset_path.name,
        "--output_filename",
        str(output_file),
        "--log_filename",
        str(log_file),
        "--history_log_filename",
        str(history_log),
        "--detail_log_filename",
        str(detail_log),
        "--message_log_filename",
        str(message_log),
        "--max_questions",
        "10",
        "--temperature",
        "0.6",
        "--max_tokens",
        "1500",
        "--top_p",
        "0.9",
        "--self_consistency",
        "3",
        "--abstain_threshold",
        "3",
        "--use_api",
        "deepseek",
        "--api_account",
        "deepseek",
        "--api_base_url",
        "https://api.deepseek.com",
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Run complete. Results stored in {output_file}")


if __name__ == "__main__":
    main()
