import logging
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional

import prompts
import expert_basics

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from common.utils import fix_seeds
from run_src.MCTS_for_reasoning_plus import Generator, search_for_answers
from run_src.rstar_utils import GeneratorError
from eval_src import Evaluator as evaluator_module

PROB_THRESHOLD = 0.8
SCALE_THRESHOLD = 4.0

def answer_to_idx(answer):
    return ord(answer) - ord("A")

def log_info(message, logger="detail_logger", print_to_std=False):
    if isinstance(logger, str):
        logger = logging.getLogger(logger)
        if not logger.hasHandlers():
            # fallback handler to avoid AttributeError
            handler = logging.StreamHandler()
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
    try:
        logger.info(message)
    except Exception:
        print(message)
    if print_to_std:
        print(message + "\n")




def fixed_abstention_decision(max_depth, patient_state, inquiry, options_dict, **kwargs):
    """
    Fixed abstention strategy based on the current interaction length.
    If the interaction length is less than the max depth, abstain, otherwise answer.
    """
    # first get the model's abstention decision
    log_info(f"++++++++++++++++++++ Start of Fixed Abstention [expert_functions.py:fixed_abstention_decision()] ++++++++++++++++++++")
    abstain_decision = len(patient_state['interaction_history']) < max_depth
    conf_score = 1 if abstain_decision else 0
    log_info(f"[ABSTENTION RESPONSE]: {abstain_decision}\n")

    # second, no matter what the model's abstention decision is, get an intermediate answer for evaluation and analysis
    patient_info = patient_state["initial_info"]
    conv_log = '\n'.join([f"{prompts.expert_system['question_word']}: {qa['question']}\n{prompts.expert_system['answer_word']}: {qa['answer']}" for qa in patient_state["interaction_history"]])
    options_text = f'A: {options_dict["A"]}, B: {options_dict["B"]}, C: {options_dict["C"]}, D: {options_dict["D"]}'

    prompt_answer = prompts.expert_system["curr_template"].format(patient_info, conv_log if conv_log != '' else 'None', inquiry, options_text, prompts.expert_system["answer"])
    messages_answer = [
        {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
        {"role": "user", "content": prompt_answer}
    ]
    response_text, letter_choice, num_tokens = expert_basics.expert_response_choice(messages_answer, options_dict, **kwargs)

    log_info(f"[FIXED ABSTAIN RETURN]: abstain: {abstain_decision}, confidence: {conf_score}, letter_choice: {letter_choice}, usage: {num_tokens}\n")
    return {
        "abstain": abstain_decision,
        "confidence": conf_score,
        "usage": num_tokens,
        "messages": messages_answer,
        "letter_choice": letter_choice,
    }



def implicit_abstention_decision(patient_state, rationale_generation, inquiry, options_dict, **kwargs):
    """
    Implicit abstention strategy based on the current patient state.
    This function uses the expert system to make a decision on whether to abstain or not based on the current patient state.
    """
    # Get the response from the expert system
    prompt_key = "implicit_RG" if rationale_generation else "implicit"
    abstain_task_prompt = prompts.expert_system[prompt_key]

    patient_info = patient_state["initial_info"]
    conv_log = '\n'.join([f"{prompts.expert_system['question_word']}: {qa['question']}\n{prompts.expert_system['answer_word']}: {qa['answer']}" for qa in patient_state["interaction_history"]])
    options_text = f'A: {options_dict["A"]}, B: {options_dict["B"]}, C: {options_dict["C"]}, D: {options_dict["D"]}'
    
    # first get the model's abstention decision
    prompt_abstain = prompts.expert_system["curr_template"].format(patient_info, conv_log if conv_log != '' else 'None', inquiry, options_text, abstain_task_prompt)

    messages = [
        {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
        {"role": "user", "content": prompt_abstain}
    ]
    response_text, atomic_question, letter_choice, conf_score, top_logprobs, num_tokens = expert_basics.expert_response_choice_or_question(messages, options_dict, **kwargs)
    log_info(f"[ABSTENTION PROMPT]: {messages}")
    log_info(f"[ABSTENTION RESPONSE]: {response_text}\n")
    messages.append({"role": "assistant", "content": response_text})

    if atomic_question != None: abstain_decision = True  # if the model generates a question, it is abstaining from answering, therefore abstain decision is True
    elif letter_choice != None: abstain_decision = False  # if the model generates an answer, it is not abstaining from answering, therefore abstain decision is False
    else: abstain_decision = True  # if the model generates neither an answer nor a question, it is abstaining from answering, therefore abstain decision is True

    # second, no matter what the model's abstention decision is, get an intermediate answer for evaluation and analysis
    # note that we get this for free if implicit abstain already chooses an answer instead of a question
    if letter_choice == None:
        prompt_answer = prompts.expert_system["curr_template"].format(patient_info, conv_log if conv_log != '' else 'None', inquiry, options_text, prompts.expert_system["answer"])
        messages_answer = [
            {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
            {"role": "user", "content": prompt_answer}
        ]
        response_text, letter_choice, num_tokens_answer = expert_basics.expert_response_choice(messages_answer, options_dict, **kwargs)
        num_tokens["input_tokens"] += num_tokens_answer["input_tokens"]
        num_tokens["output_tokens"] += num_tokens_answer["output_tokens"]

    log_info(f"[IMPLICIT ABSTAIN RETURN]: abstain: {abstain_decision}, confidence: {conf_score}, letter_choice: {letter_choice}, usage: {num_tokens}, atomic_question: {atomic_question}\n")
    return {
        "abstain": abstain_decision,
        "confidence": conf_score,
        "usage": num_tokens,
        "messages": messages,
        "letter_choice": letter_choice,
        "atomic_question": atomic_question,
    }



def binary_abstention_decision(patient_state, rationale_generation, inquiry, options_dict, **kwargs):
    """
    Binary abstention strategy based on the current patient state.
    This function prompts the user to make a binary decision on whether to abstain or not based on the current patient state.
    """
    # Get the response from the expert system
    prompt_key = "binary_RG" if rationale_generation else "binary"
    abstain_task_prompt = prompts.expert_system[prompt_key]

    patient_info = patient_state["initial_info"]
    conv_log = '\n'.join([f"{prompts.expert_system['question_word']}: {qa['question']}\n{prompts.expert_system['answer_word']}: {qa['answer']}" for qa in patient_state["interaction_history"]])
    options_text = f'A: {options_dict["A"]}, B: {options_dict["B"]}, C: {options_dict["C"]}, D: {options_dict["D"]}'
    
    # first get the model's abstention decision
    prompt_abstain = prompts.expert_system["curr_template"].format(patient_info, conv_log if conv_log != '' else 'None', inquiry, options_text, abstain_task_prompt)

    messages = [
        {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
        {"role": "user", "content": prompt_abstain}
    ]
    response_text, abstain_decision, conf_score, log_probs, num_tokens = expert_basics.expert_response_yes_no(messages, **kwargs)
    abstain_decision = abstain_decision.lower() == 'no'
    log_info(f"[ABSTENTION PROMPT]: {messages}")
    log_info(f"[ABSTENTION RESPONSE]: {response_text}\n")
    messages.append({"role": "assistant", "content": response_text})

    # second, no matter what the model's abstention decision is, get an intermediate answer for evaluation and analysis
    prompt_answer = prompts.expert_system["curr_template"].format(patient_info, conv_log if conv_log != '' else 'None', inquiry, options_text, prompts.expert_system["answer"])
    messages_answer = [
        {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
        {"role": "user", "content": prompt_answer}
    ]
    response_text, letter_choice, num_tokens_answer = expert_basics.expert_response_choice(messages_answer, options_dict, **kwargs)
    num_tokens["input_tokens"] += num_tokens_answer["input_tokens"]
    num_tokens["output_tokens"] += num_tokens_answer["output_tokens"]

    log_info(f"[BINARY ABSTAIN RETURN]: abstain: {abstain_decision}, confidence: {conf_score}, letter_choice: {letter_choice}, usage: {num_tokens}\n")
    return {
        "abstain": abstain_decision,
        "confidence": conf_score,
        "usage": num_tokens,
        "messages": messages,
        "letter_choice": letter_choice,
    }



def numerical_abstention_decision(patient_state, rationale_generation, inquiry, options_dict, **kwargs):
    """
    Numerical abstention strategy based on the current patient state.
    This function prompts the model to produce a numerical confidence score of how confident it is in its decision, then ask whether it wants to proceed
    """

    # Get the response from the expert system
    prompt_key = "numerical_RG" if rationale_generation else "numerical"
    abstain_task_prompt = prompts.expert_system[prompt_key]

    patient_info = patient_state["initial_info"]
    conv_log = '\n'.join([f"{prompts.expert_system['question_word']}: {qa['question']}\n{prompts.expert_system['answer_word']}: {qa['answer']}" for qa in patient_state["interaction_history"]])
    options_text = f'A: {options_dict["A"]}, B: {options_dict["B"]}, C: {options_dict["C"]}, D: {options_dict["D"]}'
    
    # first get the model's abstention decision
    prompt_abstain = prompts.expert_system["curr_template"].format(patient_info, conv_log if conv_log != '' else 'None', inquiry, options_text, abstain_task_prompt)

    messages = [
        {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
        {"role": "user", "content": prompt_abstain}
    ]
    response_text, conf_score, log_probs, num_tokens = expert_basics.expert_response_confidence_score(messages, **kwargs)
    messages.append({"role": "assistant", "content": response_text})
    
    messages.append({"role": "user", "content": prompts.expert_system["yes_no"]})
    # third return is supposed to be the conf_score in the binary setup, but we don't use it here because has conf score from last turn.
    response_text, abstain_decision, _, log_probs, num_tokens_2 = expert_basics.expert_response_yes_no(messages, **kwargs)
    abstain_decision = abstain_decision.lower() == 'no'
    num_tokens["input_tokens"] += num_tokens_2["input_tokens"]
    num_tokens["output_tokens"] += num_tokens_2["output_tokens"]
    log_info(f"[ABSTENTION PROMPT]: {messages}")
    log_info(f"[ABSTENTION RESPONSE]: {response_text}\n")
    messages.append({"role": "assistant", "content": response_text})


    # second, no matter what the model's abstention decision is, get an intermediate answer for evaluation and analysis
    prompt_answer = prompts.expert_system["curr_template"].format(patient_info, conv_log if conv_log != '' else 'None', inquiry, options_text, prompts.expert_system["answer"])
    messages_answer = [
        {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
        {"role": "user", "content": prompt_answer}
    ]
    response_text, letter_choice, num_tokens_answer = expert_basics.expert_response_choice(messages_answer, options_dict, **kwargs)
    num_tokens["input_tokens"] += num_tokens_answer["input_tokens"]
    num_tokens["output_tokens"] += num_tokens_answer["output_tokens"]

    log_info(f"[NUMERICAL ABSTAIN RETURN]: abstain: {abstain_decision}, confidence: {conf_score}, letter_choice: {letter_choice}, usage: {num_tokens}\n")
    return {
        "abstain": abstain_decision,
        "confidence": conf_score,
        "usage": num_tokens,
        "messages": messages,
        "letter_choice": letter_choice,
    }



def numcutoff_abstention_decision(patient_state, rationale_generation, inquiry, options_dict, abstain_threshold, **kwargs):
    """
    Numcutoff abstention strategy based on the current patient state.
    This function prompts the model to produce a numerical confidence score of how confident it is in its decision, then decide abstention based on arbitrarily set threshold
    """
    if not abstain_threshold: abstain_threshold = PROB_THRESHOLD
    
    # Get the response from the expert system
    prompt_key = "numcutoff_RG" if rationale_generation else "numcutoff"
    abstain_task_prompt = prompts.expert_system[prompt_key]

    patient_info = patient_state["initial_info"]
    conv_log = '\n'.join([f"{prompts.expert_system['question_word']}: {qa['question']}\n{prompts.expert_system['answer_word']}: {qa['answer']}" for qa in patient_state["interaction_history"]])
    options_text = f'A: {options_dict["A"]}, B: {options_dict["B"]}, C: {options_dict["C"]}, D: {options_dict["D"]}'
    
    # first get the model's abstention decision
    prompt_abstain = prompts.expert_system["curr_template"].format(patient_info, conv_log if conv_log != '' else 'None', inquiry, options_text, abstain_task_prompt)

    messages = [
        {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
        {"role": "user", "content": prompt_abstain}
    ]
    response_text, conf_score, log_probs, num_tokens = expert_basics.expert_response_confidence_score(messages, abstain_threshold=abstain_threshold, **kwargs)
    abstain_decision = conf_score < abstain_threshold
    log_info(f"[ABSTENTION PROMPT]: {messages}")
    log_info(f"[ABSTENTION RESPONSE]: {response_text}\n")
    messages.append({"role": "assistant", "content": response_text})

    # second, no matter what the model's abstention decision is, get an intermediate answer for evaluation and analysis
    prompt_answer = prompts.expert_system["curr_template"].format(patient_info, conv_log if conv_log != '' else 'None', inquiry, options_text, prompts.expert_system["answer"])
    messages_answer = [
        {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
        {"role": "user", "content": prompt_answer}
    ]
    response_text, letter_choice, num_tokens_answer = expert_basics.expert_response_choice(messages_answer, options_dict, **kwargs)
    num_tokens["input_tokens"] += num_tokens_answer["input_tokens"]
    num_tokens["output_tokens"] += num_tokens_answer["output_tokens"]

    log_info(f"[NUMCUTOFF ABSTAIN RETURN]: abstain: {abstain_decision}, confidence: {conf_score}, letter_choice: {letter_choice}, usage: {num_tokens}\n")
    return {
        "abstain": abstain_decision,
        "confidence": conf_score,
        "usage": num_tokens,
        "messages": messages,
        "letter_choice": letter_choice,
    }



def scale_abstention_decision(patient_state, rationale_generation, inquiry, options_dict, abstain_threshold, **kwargs):
    """
    Likert abstention strategy based on the current patient state.
    This function prompts the model to produce a likert scale confidence score of how confident it is in its decision, then decide abstention based on a cutoff
    """
    if not abstain_threshold: abstain_threshold = SCALE_THRESHOLD

    # Get the response from the expert system
    prompt_key = "scale_RG" if rationale_generation else "scale"
    abstain_task_prompt = prompts.expert_system[prompt_key]

    patient_info = patient_state["initial_info"]
    conv_log = '\n'.join([f"{prompts.expert_system['question_word']}: {qa['question']}\n{prompts.expert_system['answer_word']}: {qa['answer']}" for qa in patient_state["interaction_history"]])
    options_text = f'A: {options_dict["A"]}, B: {options_dict["B"]}, C: {options_dict["C"]}, D: {options_dict["D"]}'
    
    # first get the model's abstention decision
    prompt_abstain = prompts.expert_system["curr_template"].format(patient_info, conv_log if conv_log != '' else 'None', inquiry, options_text, abstain_task_prompt)

    messages = [
        {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
        {"role": "user", "content": prompt_abstain}
    ]
    print(f"<problem feed to scale expert>: {messages}")
    response_text, conf_score, log_probs, num_tokens = expert_basics.expert_response_scale_score(messages, abstain_threshold=abstain_threshold, **kwargs)
    abstain_decision = conf_score < abstain_threshold
    log_info(f"[ABSTENTION PROMPT]: {messages}")
    log_info(f"[ABSTENTION RESPONSE]: {response_text}\n")
    messages.append({"role": "assistant", "content": response_text})

    # second, no matter what the model's abstention decision is, get an intermediate answer for evaluation and analysis
    prompt_answer = prompts.expert_system["curr_template"].format(patient_info, conv_log if conv_log != '' else 'None', inquiry, options_text, prompts.expert_system["answer"])
    messages_answer = [
        {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
        {"role": "user", "content": prompt_answer}
    ]
    response_text, letter_choice, num_tokens_answer = expert_basics.expert_response_choice(messages_answer, options_dict, **kwargs)
    num_tokens["input_tokens"] += num_tokens_answer["input_tokens"]
    num_tokens["output_tokens"] += num_tokens_answer["output_tokens"]

    log_info(f"[SCALE ABSTAIN RETURN]: abstain: {abstain_decision}, confidence: {conf_score}, letter_choice: {letter_choice}, usage: {num_tokens}\n")
    return {
        "abstain": abstain_decision,
        "confidence": conf_score,
        "usage": num_tokens,
        "messages": messages,
        "letter_choice": letter_choice,
    }



def question_generation(patient_state, inquiry, options_dict, messages, independent_modules, **kwargs):
    task_prompt = prompts.expert_system["atomic_question_improved"]

    if independent_modules:
        patient_info = patient_state["initial_info"]
        conv_log = '\n'.join([f"{prompts.expert_system['question_word']}: {qa['question']}\n{prompts.expert_system['answer_word']}: {qa['answer']}" for qa in patient_state["interaction_history"]])
        options_text = f'A: {options_dict["A"]}, B: {options_dict["B"]}, C: {options_dict["C"]}, D: {options_dict["D"]}'
        prompt = prompts.expert_system["curr_template"].format(patient_info, conv_log, inquiry, options_text, task_prompt)

        messages = [
            {"role": "system", "content": prompts.expert_system["meditron_system_msg"]},
            {"role": "user", "content": prompt}
        ]
    else:
        messages.append({"role": "user", "content": task_prompt})

    response_text, atomic_question, num_tokens = expert_basics.expert_response_question(messages, **kwargs)
    log_info(f"[ATOMIC QUESTION PROMPT]: {messages}")
    log_info(f"[ATOMIC QUESTION RESPONSE]: {atomic_question}\n")
    messages.append({"role": "assistant", "content": atomic_question})

    log_info(f"[ATOMIC QUESTION RETURN]: {atomic_question}, usage: {num_tokens}\n")
    return {
        "atomic_question": atomic_question,
        "messages": messages,
        "usage": num_tokens,
    }


_RARE_HELPER_CACHE: Dict[str, "RareHelper"] = {}


def _resolve_api_from_args(args) -> str:
    if getattr(args, "rare_api", None):
        return args.rare_api
    if getattr(args, "use_vllm", False):
        return "vllm"
    api_name = getattr(args, "use_api", None)
    if api_name == "openai":
        return "gpt3.5-turbo"
    return "huggingface"


def _build_helper_cache_key(args) -> str:
    model_ckpt = getattr(args, "rare_model_ckpt", getattr(args, "expert_model", None))
    dataset_name = getattr(args, "rare_dataset_name", "MedQA")
    prompts_root = str(getattr(args, "rare_prompts_root", REPO_ROOT / "prompts"))
    api = _resolve_api_from_args(args)
    return "::".join([str(model_ckpt), dataset_name, prompts_root, api])


class RareHelper:
    """Thin wrapper around RARe generator so MediQ can reuse the same model."""

    def __init__(self, mediq_args):
        self.repo_root = REPO_ROOT
        self.args = self._build_args(mediq_args)
        fix_seeds(self.args.seed)
        self.evaluator = self._load_evaluator(self.args.dataset_name)
        self.tokenizer, self.model = self._load_model()
        self.generator = Generator(self.args, self.tokenizer, self.model, self.evaluator)
        self.question_counter = 0

    def _build_args(self, mediq_args):
        dataset_name = getattr(mediq_args, "rare_dataset_name", "MedQA")
        model_ckpt = getattr(mediq_args, "rare_model_ckpt", getattr(mediq_args, "expert_model", None))
        if not model_ckpt:
            raise ValueError("RareExpert requires `expert_model` or `rare_model_ckpt` to be specified.")

        prompts_root = str(getattr(mediq_args, "rare_prompts_root", self.repo_root / "prompts"))
        data_root = str(getattr(mediq_args, "rare_data_root", self.repo_root / "data"))
        api = _resolve_api_from_args(mediq_args)

        defaults = {
            "note": "mediQ",
            "mode": "run",
            "api": api,
            "seed": getattr(mediq_args, "rare_seed", getattr(mediq_args, "seed", 42)),
            "verbose": getattr(mediq_args, "rare_verbose", False),
            "wandb_mode": "disabled",
            "model_ckpt": model_ckpt,
            "model_parallel": getattr(mediq_args, "rare_model_parallel", False),
            "half_precision": getattr(mediq_args, "rare_half_precision", False),
            "max_tokens": getattr(mediq_args, "rare_max_tokens", getattr(mediq_args, "max_tokens", 1024)),
            "temperature": getattr(mediq_args, "rare_temperature", getattr(mediq_args, "temperature", 0.8)),
            "top_k": getattr(mediq_args, "rare_top_k", 40),
            "top_p": getattr(mediq_args, "rare_top_p", getattr(mediq_args, "top_p", 0.95)),
            "num_beams": getattr(mediq_args, "rare_num_beams", 1),
            "max_num_worker": 3,
            "test_batch_size": 1,
            "tensor_parallel_size": getattr(mediq_args, "rare_tensor_parallel_size", 1),
            "prompts_root": prompts_root,
            "data_root": data_root,
            "dataset_name": dataset_name,
            "test_json_filename": getattr(mediq_args, "rare_test_json_filename", "test_all"),
            "start_idx": 0,
            "end_idx": float("inf"),
            "run_outputs_root": str(getattr(mediq_args, "rare_run_outputs_root", self.repo_root / "run_outputs")),
            "eval_outputs_root": str(getattr(mediq_args, "rare_eval_outputs_root", self.repo_root / "eval_outputs")),
            "num_rollouts": getattr(mediq_args, "rare_num_rollouts", 15),
            "num_subquestions": getattr(mediq_args, "rare_num_subquestions", 3),
            "num_queries": getattr(mediq_args, "rare_num_queries", 3),
            "num_retrieval": getattr(mediq_args, "rare_num_retrieval", 3),
            "num_votes": getattr(mediq_args, "rare_num_votes", 10),
            "max_depth_allowed": getattr(mediq_args, "rare_max_depth_allowed", 5),
            "mcts_discount_factor": getattr(mediq_args, "rare_mcts_discount_factor", 1.0),
            "mcts_exploration_weight": getattr(mediq_args, "rare_mcts_exploration_weight", 2.0),
            "mcts_weight_scheduler": getattr(mediq_args, "rare_mcts_weight_scheduler", "const"),
            "mcts_num_last_votes": getattr(mediq_args, "rare_mcts_num_last_votes", None),
            "save_tree": getattr(mediq_args, "rare_save_tree", False),
            "search_query_weight": getattr(mediq_args, "rare_search_query_weight", 0.5),
            "combine_distributions": getattr(mediq_args, "rare_combine_distributions", "add"),
            "majority_threshold": getattr(mediq_args, "rare_majority_threshold", 0.5),
            "enable_majority": getattr(mediq_args, "rare_enable_majority", False),
            "retrieval_threshold": getattr(mediq_args, "rare_retrieval_threshold", 0.0),
            "enable_chat_template": getattr(mediq_args, "rare_enable_chat_template", False),
            "enable_self_reward": getattr(mediq_args, "rare_enable_self_reward", False),
            "num_a1_steps": getattr(mediq_args, "rare_num_a1_steps", None),
            "disable_a1": getattr(mediq_args, "rare_disable_a1", False),
            "disable_a3": getattr(mediq_args, "rare_disable_a3", False),
            "disable_a4": getattr(mediq_args, "rare_disable_a4", False),
            "disable_a6": getattr(mediq_args, "rare_disable_a6", False),
            "disable_a7": getattr(mediq_args, "rare_disable_a7", False),
            "disable_a8": getattr(mediq_args, "rare_disable_a8", False),
            "enable_answer_checking": getattr(mediq_args, "rare_enable_answer_checking", False),
            "modify_prompts_for_rephrasing": getattr(mediq_args, "rare_modify_prompts_for_rephrasing", False),
            "disable_a5": getattr(mediq_args, "rare_disable_a5", False),
            "enable_potential_score": getattr(mediq_args, "rare_enable_potential_score", False),
            "disable_answer_selection": getattr(mediq_args, "rare_disable_answer_selection", False),
            "save_generation": getattr(mediq_args, "rare_save_generation", False),
            "save_path": str(getattr(mediq_args, "rare_save_path", self.repo_root / "save" / "mediQ_rare_results.json")),
            "retrieval_corpus": getattr(mediq_args, "rare_retrieval_corpus", "medcorp"),
            "use_triples": getattr(mediq_args, "rare_use_triples", False),
            "local_rank": 0,
            "world_size": 1,
            "expected_answer": "",
        }

        args = Namespace(**defaults)

        if args.mcts_num_last_votes is None:
            args.mcts_num_last_votes = 10 if args.enable_self_reward else 32
        if not args.disable_a1 and args.num_a1_steps is None:
            args.num_a1_steps = 3

        prompts_dir = os.path.join(args.prompts_root, args.dataset_name)
        args.fewshot_cot_prompt_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_cot_prompt.txt")
        args.fewshot_cot_config_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_cot_config.json")
        args.fewshot_cot_rag_prompt_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_cot_rag_prompt.txt")
        args.fewshot_cot_rag_config_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_cot_rag_config.json")
        args.fewshot_self_reward_prompt_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_self_reward_prompt.txt")
        args.fewshot_self_reward_config_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_self_reward_config.json")
        args.fewshot_ost_prompt_path = os.path.join(prompts_dir, "fewshot_ost", "fewshot_ost_prompt.txt")
        args.fewshot_ost_config_path = os.path.join(prompts_dir, "fewshot_ost", "fewshot_ost_config.json")
        args.decompose_template_path = os.path.join(prompts_dir, "decompose", "decompose_template.json")
        args.decompose_prompt_path = os.path.join(prompts_dir, "decompose", "decompose_prompt.txt")
        args.decompose_query_template_path = os.path.join(prompts_dir, "decompose", "decompose_query_template.json")
        args.decompose_query_prompt_path = os.path.join(prompts_dir, "decompose", "decompose_query_prompt.txt")

        if not args.disable_a5:
            args.rephrasing_prompt_template_path = os.path.join(prompts_dir, "rephrasing_prompt_template.txt")
            if args.modify_prompts_for_rephrasing:
                args.fewshot_cot_prompt_rephrased_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_cot_prompt_rephrased.txt")
                args.fewshot_ost_prompt_rephrased_path = os.path.join(prompts_dir, "fewshot_ost", "fewshot_ost_prompt_rephrased.txt")
                args.decompose_prompt_rephrased_path = os.path.join(prompts_dir, "decompose", "decompose_prompt_rephrased.txt")
            else:
                args.fewshot_cot_prompt_rephrased_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_cot_prompt.txt")
                args.fewshot_ost_prompt_rephrased_path = os.path.join(prompts_dir, "fewshot_ost", "fewshot_ost_prompt.txt")
                args.decompose_prompt_rephrased_path = os.path.join(prompts_dir, "decompose", "decompose_prompt.txt")

        os.makedirs(Path(args.save_path).parent, exist_ok=True)

        return args

    def _load_evaluator(self, dataset_name):
        evaluator_cls = getattr(evaluator_module, f"{dataset_name}Evaluator")
        return evaluator_cls()

    def _load_model(self):
        if self.args.api == "huggingface":
            from models.HuggingFace_API import load_HF_model

            return load_HF_model(self.args.model_ckpt)
        if self.args.api == "vllm":
            from models.vLLM_API import load_vLLM_model

            return load_vLLM_model(
                self.args.model_ckpt,
                self.args.seed,
                self.args.tensor_parallel_size,
                self.args.half_precision,
            )
        if self.args.api == "gpt3.5-turbo":
            from models.OpenAI_API import load_OpenAI_model

            return load_OpenAI_model(self.args.model_ckpt)
        if self.args.api == "llama":
            from models.Llama_API import load_OpenAI_model

            return load_OpenAI_model(self.args.model_ckpt)
        if self.args.api == "debug":
            # Debug mode does not need a model instance
            return None, None
        raise ValueError(f"Unsupported RARe API '{self.args.api}'")

    def _format_question(self, inquiry: str, patient_state: Dict[str, Any]) -> str:
        segments: List[str] = []
        initial_info = patient_state.get("initial_info")
        if initial_info:
            segments.append(f"Initial patient information: {initial_info}")
        history = patient_state.get("interaction_history", []) or []
        if history:
            history_lines = []
            for idx, qa in enumerate(history, start=1):
                history_lines.append(f"Doctor Q{idx}: {qa.get('question', '')}")
                history_lines.append(f"Patient A{idx}: {qa.get('answer', '')}")
            segments.append("Conversation history:\n" + "\n".join(history_lines))
        if segments:
            return f"{inquiry}\n\n" + "\n\n".join(segments)
        return inquiry

    def run(self, inquiry: str, options_dict: Dict[str, str], patient_state: Dict[str, Any]):
        self.question_counter += 1
        formatted_question = self._format_question(inquiry, patient_state)
        prev_calls = getattr(self.generator.io, "call_counter", 0)
        prev_tokens = getattr(self.generator.io, "token_counter", 0)

        try:
            best_choice, freq_choice, choice_info, solution_nodes, solutions = search_for_answers(
                args=self.args,
                user_question=formatted_question,
                question_id=self.question_counter,
                gt_answer="",
                generator=self.generator,
                options=options_dict,
            )
        except GeneratorError as exc:
            log_info(f"[RARE ERROR] {exc}", logger="detail_logger")
            return {
                "best_choice": None,
                "freq_choice": None,
                "choice_info": {},
                "solution_nodes": [],
                "solutions": [],
                "usage": {"input_tokens": 0, "output_tokens": 0, "call_count": 0},
                "error": str(exc),
            }

        usage = {
            "input_tokens": 0,
            "output_tokens": max(getattr(self.generator.io, "token_counter", 0) - prev_tokens, 0),
            "call_count": max(getattr(self.generator.io, "call_counter", 0) - prev_calls, 0),
        }

        return {
            "best_choice": best_choice,
            "freq_choice": freq_choice,
            "choice_info": choice_info or {},
            "solution_nodes": solution_nodes or [],
            "solutions": solutions or [],
            "usage": usage,
        }


def get_or_create_rare_helper(args) -> RareHelper:
    cache_key = _build_helper_cache_key(args)
    if cache_key not in _RARE_HELPER_CACHE:
        _RARE_HELPER_CACHE[cache_key] = RareHelper(args)
    return _RARE_HELPER_CACHE[cache_key]


def _extract_followup_questions(
    solution_nodes: List[Any],
    preferred_choice: Optional[str],
    asked_questions: Optional[set],
    limit: Optional[int] = None,
) -> List[str]:
    if asked_questions is None:
        asked_questions = set()

    filtered_nodes = []
    for node in solution_nodes:
        node_choice = getattr(node, "get_choice", lambda: None)()
        if preferred_choice and node_choice != preferred_choice:
            continue
        filtered_nodes.append(node)

    if not filtered_nodes and not preferred_choice:
        filtered_nodes = solution_nodes

    if not filtered_nodes:
        return []

    filtered_nodes.sort(key=lambda n: (getattr(n, "node_value", None) or float("-inf")), reverse=True)

    questions: List[str] = []
    for node in filtered_nodes:
        trace = getattr(node, "solution_trace", {}) or {}
        for key in sorted(trace.keys()):
            if key == 0:
                continue
            subquestion = trace[key].get("subquestion") if isinstance(trace[key], dict) else None
            if subquestion and subquestion not in asked_questions and subquestion not in questions:
                questions.append(subquestion)
                if limit is not None and len(questions) >= limit:
                    return questions
    return questions


def rare_abstention_decision(
    rare_helper: RareHelper,
    patient_state: Dict[str, Any],
    inquiry: str,
    options_dict: Dict[str, str],
    abstain_threshold: float,
    asked_questions: Optional[set] = None,
    max_followups: Optional[int] = None,
) -> Dict[str, Any]:
    rare_result = rare_helper.run(inquiry, options_dict, patient_state)

    choice_info = rare_result.get("choice_info", {}) or {}
    best_choice = rare_result.get("best_choice")
    freq_choice = rare_result.get("freq_choice")
    choice = best_choice or freq_choice

    confidence = 0.0
    if choice and choice in choice_info:
        scores = [info.get("score", 0) for info in choice_info.values() if info.get("score") is not None]
        score_sum = sum(scores)
        if score_sum > 0:
            confidence = choice_info[choice].get("score", 0) / score_sum
        else:
            total_counts = sum(info.get("count", 0) for info in choice_info.values())
            if total_counts > 0:
                confidence = choice_info[choice].get("count", 0) / total_counts

    abstain = choice is None or confidence < abstain_threshold

    followup_questions: List[str] = []
    if abstain:
        followup_questions = _extract_followup_questions(
            rare_result.get("solution_nodes", []),
            choice,
            asked_questions,
            limit=max_followups,
        )

    return {
        "abstain": abstain,
        "confidence": confidence,
        "letter_choice": choice,
        "usage": rare_result.get("usage", {"input_tokens": 0, "output_tokens": 0}),
        "followup_questions": followup_questions,
        "raw_result": rare_result,
    }
