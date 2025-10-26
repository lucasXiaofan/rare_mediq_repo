import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from eval_src import Evaluator as evaluator_module  # noqa: E402
from run_src.MCTS_for_reasoning_plus import Generator, search_for_answers  # noqa: E402
from run_src.rstar_utils import GeneratorError  # noqa: E402


def _build_deepseek_args(
    *,
    num_rollouts: int,
    dataset_name: str,
    model_ckpt: str,
    prompts_root: str,
    data_root: str,
    note: str = "deepseek_mcts_wrapper",
) -> argparse.Namespace:
    """
    Mirror the DeepSeek demo configuration while allowing a few knobs to be customized.
    """
    args = argparse.Namespace()

    # API / model configuration
    args.api = "deepseek"
    args.model_ckpt = model_ckpt

    # Dataset metadata
    args.dataset_name = dataset_name
    args.test_json_filename = "test_all"
    args.mode = "run"
    args.note = note

    # MCTS knobs
    args.num_rollouts = num_rollouts
    args.mcts_exploration_weight = 1.0
    args.mcts_weight_scheduler = "constant"
    args.mcts_discount_factor = 1.0
    args.max_depth_allowed = 2

    # Action toggles
    args.disable_a5 = False
    args.disable_a8 = True
    args.disable_a4 = False
    args.disable_a6 = True
    args.disable_a7 = True
    args.disable_a1 = False
    args.disable_a3 = False

    # Generation settings
    args.temperature = 0.8
    args.top_k = 40
    args.top_p = 0.95
    args.max_tokens = 1000

    # Generator internals
    args.num_subquestions = 3
    args.num_queries = 2
    args.num_a1_steps = 3
    args.num_votes = 2 # this is consistentcy 3 should be good good, make it 1 will causing error
    args.mcts_num_last_votes = 2
    args.search_query_weight = 0.5
    args.enable_potential_score = False
    args.enable_answer_checking = False
    args.combine_distributions = "add"
    args.enable_majority = False
    args.majority_threshold = 0.8
    args.enable_self_reward = False

    # Retrieval
    args.num_retrieval = 5
    args.retrieval_corpus = "pubmed"
    args.retrieval_threshold = 0.5
    args.use_triples = False

    # Prompting / IO
    args.enable_chat_template = True
    args.modify_prompts_for_rephrasing = False
    args.verbose = False
    args.prompts_root = prompts_root
    args.data_root = data_root

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

    return args


def _load_evaluator(dataset_name: str):
    evaluator_cls_name = f"{dataset_name}Evaluator"
    evaluator_cls = getattr(evaluator_module, evaluator_cls_name, None)
    if evaluator_cls is None:
        raise ValueError(f"Unsupported evaluator '{evaluator_cls_name}'.")
    return evaluator_cls()


class DeepseekMediqInferenceWrapper:
    """
    Thin utility that mirrors the DeepSeek demo configuration and exposes a
    `run` helper returning the tuple from `search_for_answers`.
    """

    def __init__(
        self,
        *,
        options_dict: Optional[Dict[str, str]] = None,
        num_rollouts: int = 2,
        dataset_name: str = "MedQA",
        model_ckpt: str = "deepseek-chat",
        prompts_root: Optional[str] = "prompts",
        data_root: Optional[str] = None,
        note: str = "deepseek_mediq_wrapper",
    ):
        self.options = options_dict or {}
        self.prompts_root = str(prompts_root) if prompts_root else str(REPO_ROOT / "prompts")
        self.data_root = str(data_root) if data_root else str(REPO_ROOT / "data")
        self.model_ckpt = model_ckpt
        self.note = note
        self.question_counter = 0
        self._initialize_components(num_rollouts=num_rollouts, dataset_name=dataset_name)

    def _initialize_components(self, *, num_rollouts: int, dataset_name: str):
        self.args = _build_deepseek_args(
            num_rollouts=num_rollouts,
            dataset_name=dataset_name,
            model_ckpt=self.model_ckpt,
            prompts_root=self.prompts_root,
            data_root=self.data_root,
            note=self.note,
        )
        self.evaluator = _load_evaluator(dataset_name)
        # DeepSeek API only needs the checkpoint name; tokenizer stays None.
        self.generator = Generator(self.args, tokenizer=None, model=self.args.model_ckpt, evaluator=self.evaluator)
        self.dataset_name = dataset_name

    def _ensure_configuration(self, *, num_rollouts: Optional[int], evaluator_name: Optional[str]):
        dataset_changed = evaluator_name and evaluator_name != self.dataset_name
        if dataset_changed:
            self._initialize_components(num_rollouts=num_rollouts or self.args.num_rollouts, dataset_name=evaluator_name)  # type: ignore[arg-type]
            return
        if num_rollouts is not None:
            self.args.num_rollouts = num_rollouts

    @staticmethod
    def format_question(inquiry: str, patient_state: Dict[str, Any],options_dict) -> str:
        """
        first role, patient_state, inquiry, option
        """
        role = """You are a medical doctor trying to reason through a real-life clinical case. Based on your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy, respond according to the task specified by the user. Base your response on the current and standard practices referenced in medical guidelines. A patient comes into the clinic presenting with a symptom as described in the conversation log below"""

        segments = [role,]
        initial_info = patient_state.get("initial_info")
        if initial_info:
            segments.append(f"Initial patient information: {initial_info}")
        segments.append(inquiry)
        options_text = f'A: {options_dict["A"]}, B: {options_dict["B"]}, C: {options_dict["C"]}, D: {options_dict["D"]}'
        segments.append(options_text)
        history = patient_state.get("interaction_history") or []
        if history:
            conv_lines = []
            for idx, qa in enumerate(history, start=1):
                conv_lines.append(f"Doctor Q{idx}: {qa.get('question', '')}")
                conv_lines.append(f"Patient A{idx}: {qa.get('answer', '')}")
            segments.append("Conversation history:\n" + "\n".join(conv_lines))
        return "\n\n".join(filter(None, segments))

    def run(
        self,
        *,
        user_prompt: str,
        options: Optional[Dict[str, str]] = None,
        num_rollouts: Optional[int] = None,
        evaluator_name: Optional[str] = None,
        gt_answer: str = "",
        question_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        
        Execute the DeepSeek-powered MCTS search and return the raw RARe tuple in a dict.
        """
        # Lucas this is the temporary way to include option in prompt 

        self._ensure_configuration(num_rollouts=num_rollouts, evaluator_name=evaluator_name)
        options_dict = options or self.options
        if not options_dict:
            raise ValueError("DeepseekInferenceWrapper requires multiple-choice options.")

        if question_id is None:
            self.question_counter += 1
            question_id = self.question_counter

        prev_calls = getattr(self.generator.io, "call_counter", 0)
        prev_tokens = getattr(self.generator.io, "token_counter", 0)

        try:
            best_choice, freq_choice, choice_info, solution_nodes, solutions = search_for_answers(
                args=self.args,
                user_question=user_prompt,
                question_id=question_id,
                gt_answer=gt_answer or "",
                generator=self.generator,
                options=options_dict,
            )
        except GeneratorError as exc:
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
            "error": None,
        }


def run_deepseek_mcts(
    user_prompt: str,
    *,
    options: Dict[str, str],
    num_rollouts: int = 1,
    evaluator_name: str = "MedQA",
    model_ckpt: str = "deepseek-chat",
    prompts_root: Optional[str] = None,
    data_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function for quick one-off calls outside the MediQ Expert stack.
    """
    wrapper = DeepseekMediqInferenceWrapper(
        options_dict=options,
        num_rollouts=num_rollouts,
        dataset_name=evaluator_name,
        model_ckpt=model_ckpt,
        prompts_root=prompts_root,
        data_root=data_root,
    )
    return wrapper.run(user_prompt=user_prompt)
