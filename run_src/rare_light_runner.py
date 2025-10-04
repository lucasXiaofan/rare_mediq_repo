#!/usr/bin/env python3
"""Lightweight RARe runner for ad-hoc prompt experiments.

This script wraps the RARe helper used inside MediQ so that we can
quickly send a single prompt (with optional MediQ templating) into the
RARe reasoning stack and inspect the response.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from textwrap import indent
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
MEDIQ_SRC = REPO_ROOT / "mediQ" / "src"
if str(MEDIQ_SRC) not in sys.path:
    sys.path.append(str(MEDIQ_SRC))

from mediQ.src import prompts

PROMPT_SOURCES: tuple[str, ...] = (
    "medaq-problem",
    "scale-abstention",
    "question-generation",
)


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file '{path}' does not exist")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _format_history(interactions: List[Dict[str, str]]) -> str:
    history_lines = []
    for idx, qa in enumerate(interactions, start=1):
        history_lines.append(f"Doctor Q{idx}: {qa.get('question', '')}")
        history_lines.append(f"Patient A{idx}: {qa.get('answer', '')}")
    return "\n".join(history_lines)


def build_prompt(
    *,
    source: str,
    inquiry: str,
    patient_state: Dict[str, Any],
    options_dict: Dict[str, str],
    rationale_generation: bool,
    independent_modules: bool,
) -> str:
    if source == "medaq-problem":
        return inquiry

    patient_info = patient_state.get("initial_info", "None") or "None"
    conversation_history = _format_history(patient_state.get("interaction_history", []) or [])
    options_text = ", ".join([f"{letter}: {text}" for letter, text in options_dict.items()])

    if source == "scale-abstention":
        prompt_key = "scale_RG" if rationale_generation else "scale"
        task_suffix = prompts.expert_system[prompt_key]
        return prompts.expert_system["curr_template"].format(
            patient_info,
            conversation_history if conversation_history else "None",
            inquiry,
            options_text,
            task_suffix,
        )

    if source == "question-generation":
        task_suffix = prompts.expert_system["atomic_question_improved"]
        if independent_modules:
            return prompts.expert_system["curr_template"].format(
                patient_info,
                conversation_history if conversation_history else "None",
                inquiry,
                options_text,
                task_suffix,
            )
        # When modules are not independent we append the task as a new user turn.
        base_messages = prompts.expert_system["curr_template"].format(
            patient_info,
            conversation_history if conversation_history else "None",
            inquiry,
            options_text,
            "",
        )
        return base_messages.strip() + "\n\n" + task_suffix

    raise ValueError(f"Unsupported prompt source '{source}'")


def pretty_options(options_dict: Dict[str, str]) -> str:
    lines = [f"{key}: {value}" for key, value in options_dict.items()]
    return indent("\n".join(lines), prefix="    ")


def build_mediq_args(args: argparse.Namespace) -> SimpleNamespace:
    payload: Dict[str, Any] = {
        "rare_api": args.api,
        "rare_model_ckpt": args.model_ckpt,
        "rare_dataset_name": args.dataset_name,
        "rare_prompts_root": str(args.prompts_root),
        "rare_data_root": str(args.data_root),
        "rare_test_json_filename": args.test_json_filename,
        "rare_seed": args.seed,
        "seed": args.seed,
        "rare_temperature": args.temperature,
        "rare_top_p": args.top_p,
        "rare_max_tokens": args.max_tokens,
        "rare_verbose": args.verbose,
        "rare_num_rollouts": args.num_rollouts,
        "rare_num_subquestions": args.num_subquestions,
        "rare_num_queries": args.num_queries,
        "rare_num_retrieval": args.num_retrieval,
        "rare_num_votes": args.num_votes,
        "rare_max_depth_allowed": args.max_depth,
        "rare_mcts_discount_factor": args.mcts_discount_factor,
        "rare_mcts_exploration_weight": args.mcts_exploration_weight,
        "rare_mcts_weight_scheduler": args.mcts_weight_scheduler,
        "rare_mcts_num_last_votes": args.mcts_num_last_votes,
        "rare_search_query_weight": args.search_query_weight,
        "rare_combine_distributions": args.combine_distributions,
        "rare_majority_threshold": args.majority_threshold,
        "rare_enable_majority": args.enable_majority,
        "rare_retrieval_threshold": args.retrieval_threshold,
        "rare_enable_chat_template": args.enable_chat_template,
        "rare_enable_self_reward": args.enable_self_reward,
        "rare_enable_answer_checking": args.enable_answer_checking,
        "rare_modify_prompts_for_rephrasing": args.modify_prompts_for_rephrasing,
        "rare_disable_a1": args.disable_a1,
        "rare_disable_a3": args.disable_a3,
        "rare_disable_a4": args.disable_a4,
        "rare_disable_a5": args.disable_a5,
        "rare_disable_a6": args.disable_a6,
        "rare_disable_a7": args.disable_a7,
        "rare_disable_a8": args.disable_a8,
        "rare_enable_potential_score": args.enable_potential_score,
        "rare_disable_answer_selection": args.disable_answer_selection,
        "rare_save_generation": args.save_generation,
        "rare_use_triples": args.use_triples,
        "rare_retrieval_corpus": args.retrieval_corpus,
        "rare_tensor_parallel_size": args.tensor_parallel_size,
        "rare_model_parallel": args.model_parallel,
        "rare_half_precision": args.half_precision,
        "use_api": args.use_api,
        "use_vllm": args.use_vllm,
        "expert_model": args.fallback_expert_model,
    }

    if args.save_path:
        payload["rare_save_path"] = str(args.save_path)
    if args.run_outputs_root:
        payload["rare_run_outputs_root"] = str(args.run_outputs_root)
    if args.eval_outputs_root:
        payload["rare_eval_outputs_root"] = str(args.eval_outputs_root)

    return SimpleNamespace(**payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight RARe prompt runner")
    parser.add_argument(
        "--prompt-source",
        choices=("all",) + PROMPT_SOURCES,
        default="all",
        help="Which prompt template to run. Default runs all three variants sequentially.",
    )
    parser.add_argument("--problem-index", type=int, default=0, help="Index of the MedQA problem to load")
    parser.add_argument("--dataset-name", default="MedQA")
    parser.add_argument("--test-json-filename", default="test_all")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--prompts-root", type=Path, default=Path("prompts"))
    parser.add_argument("--run-outputs-root", type=Path, default=None)
    parser.add_argument("--eval-outputs-root", type=Path, default=None)
    parser.add_argument("--model-ckpt", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--fallback-expert-model", default=None, help="Optional expert model name used when RARe helper requires it")
    parser.add_argument("--api", default="vllm")
    parser.add_argument("--use-api", default=None)
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--model-parallel", action="store_true")
    parser.add_argument("--half-precision", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--num-subquestions", type=int, default=3)
    parser.add_argument("--num-queries", type=int, default=3)
    parser.add_argument("--num-retrieval", type=int, default=3)
    parser.add_argument("--num-votes", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--mcts-discount-factor", type=float, default=1.0)
    parser.add_argument("--mcts-exploration-weight", type=float, default=2.0)
    parser.add_argument("--mcts-weight-scheduler", default="const")
    parser.add_argument("--mcts-num-last-votes", type=int, default=None)
    parser.add_argument("--search-query-weight", type=float, default=0.5)
    parser.add_argument("--combine-distributions", default="add")
    parser.add_argument("--majority-threshold", type=float, default=0.5)
    parser.add_argument("--enable-majority", action="store_true")
    parser.add_argument("--retrieval-threshold", type=float, default=0.0)
    parser.add_argument("--enable-chat-template", action="store_true")
    parser.add_argument("--enable-self-reward", action="store_true")
    parser.add_argument("--enable-answer-checking", action="store_true")
    parser.add_argument("--modify-prompts-for-rephrasing", action="store_true")
    parser.add_argument("--disable-a1", action="store_true")
    parser.add_argument("--disable-a3", action="store_true")
    parser.add_argument("--disable-a4", action="store_true")
    parser.add_argument("--disable-a5", action="store_true")
    parser.add_argument("--disable-a6", action="store_true")
    parser.add_argument("--disable-a7", action="store_true")
    parser.add_argument("--disable-a8", action="store_true")
    parser.add_argument("--enable-potential-score", action="store_true")
    parser.add_argument("--disable-answer-selection", action="store_true")
    parser.add_argument("--save-generation", action="store_true")
    parser.add_argument("--enable-answer-elimination", action="store_true")
    parser.add_argument("--use-triples", action="store_true")
    parser.add_argument("--retrieval-corpus", default="medcorp")
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--independent-modules", action="store_true")
    parser.add_argument("--rationale-generation", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--initial-info", default="", help="Optional initial patient info override")
    parser.add_argument("--print-only", action="store_true", help="Skip RARe inference and only show the constructed prompt")

    args = parser.parse_args()

    dataset_path = args.data_root / args.dataset_name / f"{args.test_json_filename}.json"

    records = load_dataset(dataset_path)
    if not records:
        raise ValueError(f"Dataset '{dataset_path}' is empty")

    if args.problem_index < 0 or args.problem_index >= len(records):
        raise IndexError(
            f"problem-index {args.problem_index} is out of range for dataset of size {len(records)}"
        )

    sample = records[args.problem_index]
    sample_id = sample.get("id", f"idx-{args.problem_index}")
    inquiry = sample.get("question") or sample.get("problem") or ""
    if not inquiry:
        raise ValueError(f"Sample {sample_id} does not contain a question/problem field")

    options_dict = sample.get("options", {})
    if not options_dict:
        option_keys = [key for key in ("A", "B", "C", "D") if key in sample]
        if option_keys:
            options_dict = {key: sample[key] for key in option_keys}
        else:
            raise ValueError(f"Sample {sample_id} does not contain options")

    base_patient_state = {
        "initial_info": args.initial_info or sample.get("initial_info", ""),
        "interaction_history": sample.get("interaction_history", []),
    }

    print(f"Loaded problem {sample_id} (index {args.problem_index}) from '{dataset_path}'")
    print("\nQuestion:")
    print(indent(inquiry.strip(), "    "))
    print("\nOptions:")
    print(pretty_options(options_dict))

    if args.prompt_source == "all":
        chosen_sources: Iterable[str] = PROMPT_SOURCES
    else:
        chosen_sources = (args.prompt_source,)

    mediq_args = None
    helper = None
    if not args.print_only:
        from mediQ.src.expert_functions import get_or_create_rare_helper

        mediq_args = build_mediq_args(args)
        if getattr(mediq_args, "expert_model", None) is None:
            mediq_args.expert_model = mediq_args.rare_model_ckpt

        helper = get_or_create_rare_helper(mediq_args)

    for idx, source in enumerate(chosen_sources, start=1):
        heading = f"Prompt {idx}: {source}"
        underline = "=" * len(heading)
        print(f"\n{heading}")
        print(underline)

        prompt_text = build_prompt(
            source=source,
            inquiry=inquiry,
            patient_state=base_patient_state,
            options_dict=options_dict,
            rationale_generation=args.rationale_generation,
            independent_modules=args.independent_modules,
        )

        print("\nPrompt sent to RARe:")
        print(indent(prompt_text.strip(), "    "))

        if args.print_only:
            continue

        assert helper is not None
        patient_state = {
            "initial_info": base_patient_state.get("initial_info", ""),
            "interaction_history": list(base_patient_state.get("interaction_history", [])),
        }

        result = helper.run(
            inquiry=prompt_text,
            options_dict=options_dict,
            patient_state=patient_state,
        )

        print("\nRARe output:")
        if result.get("error"):
            print(indent(f"Error: {result['error']}", "    "))
            continue

        best_choice = result.get("best_choice")
        freq_choice = result.get("freq_choice")
        usage = result.get("usage", {})
        print(indent(f"Best choice: {best_choice}", "    "))
        print(indent(f"Most frequent choice: {freq_choice}", "    "))
        if usage:
            print(indent(f"Usage: {usage}", "    "))
        choice_info = result.get("choice_info")
        if choice_info:
            print(indent(f"Choice details: {choice_info}", "    "))


if __name__ == "__main__":
    main()
