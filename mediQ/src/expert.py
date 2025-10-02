import random
import importlib
import logging
import expert_functions

class Expert:
    """
    Expert system skeleton
    """
    def __init__(self, args, inquiry, options):
        # Initialize the expert with necessary parameters and the initial context or inquiry
        self.args = args
        self.inquiry = inquiry
        self.options = options

    def respond(self, patient_state):
        # Decision-making based on the initial information, history of interactions, current inquiry, and options
        raise NotImplementedError
    
    def ask_question(self, patient_state, prev_messages):
        # Generate a question based on the current patient state
        kwargs = {
            "patient_state": patient_state,
            "inquiry": self.inquiry,
            "options_dict": self.options,
            "messages": prev_messages,
            "independent_modules": self.args.independent_modules,
            "model_name": self.args.expert_model_question_generator,
            "use_vllm": self.args.use_vllm,
            "use_api": self.args.use_api,
            "temperature": self.args.temperature,
            "max_tokens": self.args.max_tokens,
            "top_p": self.args.top_p,
            "top_logprobs": self.args.top_logprobs,
            "api_account": self.args.api_account
        }
        return expert_functions.question_generation(**kwargs)
    
    def get_abstain_kwargs(self, patient_state):
        kwargs = {
            "max_depth": self.args.max_questions,
            "patient_state": patient_state,
            "rationale_generation": self.args.rationale_generation,
            "inquiry": self.inquiry,
            "options_dict": self.options,
            "abstain_threshold": self.args.abstain_threshold,
            "self_consistency": self.args.self_consistency,
            "model_name": self.args.expert_model,
            "use_vllm": self.args.use_vllm,
            "use_api": self.args.use_api,
            "temperature": self.args.temperature,
            "max_tokens": self.args.max_tokens,
            "top_p": self.args.top_p,
            "top_logprobs": self.args.top_logprobs,
            "api_account": self.args.api_account
        }
        return kwargs


class RandomExpert(Expert):
    """
    Below is an example Expert system that randomly asks a question or makes a choice based on the current patient state.
    This should be replaced with a more sophisticated expert system that can make informed decisions based on the patient state.
    """

    def respond(self, patient_state):
        # Decision-making based on the initial information, history of interactions, current inquiry, and options
        initial_info = patient_state['initial_info']  # not use because it's random
        history = patient_state['interaction_history']  # not use because it's random

        # randomly decide to ask a question or make a choice
        abstain = random.random() < 0.5
        toy_question = "Can you describe your symptoms more?"
        toy_decision = self.choice(patient_state)
        conf_score = random.random()/2 if abstain else random.random()

        return {
            "type": "question" if abstain else "choice",
            "question": toy_question,
            "letter_choice": toy_decision,
            "confidence": conf_score,  # Optional confidence score
            "urgent": True,  # Example of another optional flag
            "additional_info": "Check for any recent changes."  # Any other optional data
        }

    def choice(self, patient_state):
        # Generate a choice or intermediate decision based on the current patient state
        # randomly choose an option
        return random.choice(list(self.options.keys()))


class BasicExpert(Expert):
    def respond(self, patient_state):
        kwargs = self.get_abstain_kwargs(patient_state)
        abstain_response_dict = expert_functions.implicit_abstention_decision(**kwargs)
        return {
            "type": "question" if abstain_response_dict["abstain"] else "choice",
            "question": abstain_response_dict["atomic_question"],
            "letter_choice": abstain_response_dict["letter_choice"],
            "confidence": abstain_response_dict["confidence"],
            "usage": abstain_response_dict["usage"]
        }


class FixedExpert(Expert):
    def respond(self, patient_state):
        # Decision-making based on the initial information, history of interactions, current inquiry, and options
        kwargs = self.get_abstain_kwargs(patient_state)
        abstain_response_dict = expert_functions.fixed_abstention_decision(**kwargs)
        if abstain_response_dict["abstain"] == False:
            return {
                "type": "choice",
                "letter_choice": abstain_response_dict["letter_choice"],
                "confidence": abstain_response_dict["confidence"],
                "usage": abstain_response_dict["usage"]
            }

        question_response_dict = self.ask_question(patient_state, abstain_response_dict["messages"])
        abstain_response_dict["usage"]["input_tokens"] += question_response_dict["usage"]["input_tokens"]
        abstain_response_dict["usage"]["output_tokens"] += question_response_dict["usage"]["output_tokens"]
        return {
            "type": "question",
            "question": question_response_dict["atomic_question"],
            "letter_choice": abstain_response_dict["letter_choice"],
            "confidence": abstain_response_dict["confidence"],
            "usage": abstain_response_dict["usage"]
        }
        

class BinaryExpert(Expert):
    def respond(self, patient_state):
        # Decision-making based on the initial information, history of interactions, current inquiry, and options
        kwargs = self.get_abstain_kwargs(patient_state)
        abstain_response_dict = expert_functions.binary_abstention_decision(**kwargs)
        if abstain_response_dict["abstain"] == False:
            return {
                "type": "choice",
                "letter_choice": abstain_response_dict["letter_choice"],
                "confidence": abstain_response_dict["confidence"],
                "usage": abstain_response_dict["usage"]
            }

        question_response_dict = self.ask_question(patient_state, abstain_response_dict["messages"])
        abstain_response_dict["usage"]["input_tokens"] += question_response_dict["usage"]["input_tokens"]
        abstain_response_dict["usage"]["output_tokens"] += question_response_dict["usage"]["output_tokens"]
        return {
            "type": "question",
            "question": question_response_dict["atomic_question"],
            "letter_choice": abstain_response_dict["letter_choice"],
            "confidence": abstain_response_dict["confidence"],
            "usage": abstain_response_dict["usage"]
        }


class NumericalExpert(Expert):
    def respond(self, patient_state):
        # Decision-making based on the initial information, history of interactions, current inquiry, and options
        kwargs = self.get_abstain_kwargs(patient_state)
        abstain_response_dict = expert_functions.numerical_abstention_decision(**kwargs)
        if abstain_response_dict["abstain"] == False:
            return {
                "type": "choice",
                "letter_choice": abstain_response_dict["letter_choice"],
                "confidence": abstain_response_dict["confidence"],
                "usage": abstain_response_dict["usage"]
            }

        question_response_dict = self.ask_question(patient_state, abstain_response_dict["messages"])
        abstain_response_dict["usage"]["input_tokens"] += question_response_dict["usage"]["input_tokens"]
        abstain_response_dict["usage"]["output_tokens"] += question_response_dict["usage"]["output_tokens"]
        return {
            "type": "question",
            "question": question_response_dict["atomic_question"],
            "letter_choice": abstain_response_dict["letter_choice"],
            "confidence": abstain_response_dict["confidence"],
            "usage": abstain_response_dict["usage"]
        }


class NumericalCutOffExpert(Expert):
    def respond(self, patient_state):
        # Decision-making based on the initial information, history of interactions, current inquiry, and options
        kwargs = self.get_abstain_kwargs(patient_state)
        abstain_response_dict = expert_functions.numcutoff_abstention_decision(**kwargs)
        if abstain_response_dict["abstain"] == False:
            return {
                "type": "choice",
                "letter_choice": abstain_response_dict["letter_choice"],
                "confidence": abstain_response_dict["confidence"],
                "usage": abstain_response_dict["usage"]
            }

        question_response_dict = self.ask_question(patient_state, abstain_response_dict["messages"])
        abstain_response_dict["usage"]["input_tokens"] += question_response_dict["usage"]["input_tokens"]
        abstain_response_dict["usage"]["output_tokens"] += question_response_dict["usage"]["output_tokens"]
        return {
            "type": "question",
            "question": question_response_dict["atomic_question"],
            "letter_choice": abstain_response_dict["letter_choice"],
            "confidence": abstain_response_dict["confidence"],
            "usage": abstain_response_dict["usage"]
        }


class ScaleExpert(Expert):
    def respond(self, patient_state):
        # Decision-making based on the initial information, history of interactions, current inquiry, and options
        kwargs = self.get_abstain_kwargs(patient_state)
        abstain_response_dict = expert_functions.scale_abstention_decision(**kwargs)
        if abstain_response_dict["abstain"] == False:
            return {
                "type": "choice",
                "letter_choice": abstain_response_dict["letter_choice"],
                "confidence": abstain_response_dict["confidence"],
                "usage": abstain_response_dict["usage"]
            }

        question_response_dict = self.ask_question(patient_state, abstain_response_dict["messages"])
        abstain_response_dict["usage"]["input_tokens"] += question_response_dict["usage"]["input_tokens"]
        abstain_response_dict["usage"]["output_tokens"] += question_response_dict["usage"]["output_tokens"]
        return {
            "type": "question",
            "question": question_response_dict["atomic_question"],
            "letter_choice": abstain_response_dict["letter_choice"],
            "confidence": abstain_response_dict["confidence"],
            "usage": abstain_response_dict["usage"]
        }


class RAREExpert(Expert):
    """
    Wraps the RARE pipeline behind the MediQ Expert interface.
    Requirements (must be importable on PYTHONPATH):
      - common.arguments (get_parser, post_process_args)
      - common.utils (fix_seeds)
      - eval_src.Evaluator (e.g., MedQAEvaluator)
      - MCTS_for_reasoning_plus (Generator, search_for_answers)
      - models.HuggingFace_API (load_HF_model) or models.vLLM_API if you prefer vLLM
    """
    def __init__(self, args, inquiry, options):
        super().__init__(args, inquiry, options)
        self.rare_ready = False
        self.rare_args = None
        self.rare_generator = None
        self.search_for_answers = None
        self._init_rare()

    def _init_rare(self):
        try:
            common_arguments = importlib.import_module("common.arguments")
            common_utils = importlib.import_module("common.utils")
            eval_evaluator = importlib.import_module("eval_src.Evaluator")
            mcts_mod = importlib.import_module("MCTS_for_reasoning_plus")
            hf_api = importlib.import_module("models.HuggingFace_API")

            # Build RARE args from its parser with defaults
            parser = common_arguments.get_parser()
            # parse empty list to keep defaults without touching sys.argv
            parsed_args = parser.parse_args(args=[])

            # Minimal required fields
            setattr(parsed_args, "api", "huggingface")
            setattr(parsed_args, "model_ckpt", getattr(self.args, "expert_model", "Qwen/Qwen3-0.6B"))
            # These are used by RARE but not strictly needed for single question mode
            setattr(parsed_args, "dataset_name", "MedQA")
            setattr(parsed_args, "prompts_root", "prompts")
            setattr(parsed_args, "data_root", "data")
            # Reasonable small defaults for interactive use
            if getattr(parsed_args, "num_rollouts", None) is None:
                setattr(parsed_args, "num_rollouts", 5)
            if getattr(parsed_args, "num_votes", None) is None:
                setattr(parsed_args, "num_votes", 5)

            parsed_args = common_arguments.post_process_args(parsed_args)
            # seed
            if hasattr(common_utils, "fix_seeds"):
                common_utils.fix_seeds(getattr(parsed_args, "seed", 42))

            # Evaluator (MedQAEvaluator present in eval_src/Evaluator.py)
            evaluator_class_name = "MedQAEvaluator"
            evaluator_class = getattr(eval_evaluator, evaluator_class_name, None)
            evaluator = evaluator_class() if evaluator_class else None

            # Load model/tokenizer via RARE HF API
            tokenizer, model = hf_api.load_HF_model(parsed_args.model_ckpt)

            Generator = getattr(mcts_mod, "Generator")
            self.search_for_answers = getattr(mcts_mod, "search_for_answers")

            self.rare_generator = Generator(parsed_args, tokenizer, model, evaluator)
            self.rare_args = parsed_args
            self.rare_ready = True
            logging.getLogger("detail_logger").info("[RAREExpert] Initialization successful.")
        except Exception as e:
            # Log and mark unavailable; respond() will fallback
            try:
                logging.getLogger("detail_logger").error(f"[RAREExpert] Initialization failed: {e}")
            except Exception:
                pass
            self.rare_ready = False

    def respond(self, patient_state):
        # If RARE stack isn't available, fall back to implicit abstention strategy
        if not self.rare_ready or self.search_for_answers is None or self.rare_generator is None:
            kwargs = self.get_abstain_kwargs(patient_state)
            abstain_response_dict = expert_functions.implicit_abstention_decision(**kwargs)
            return {
                "type": "question" if abstain_response_dict["abstain"] else "choice",
                "question": abstain_response_dict.get("atomic_question"),
                "letter_choice": abstain_response_dict.get("letter_choice"),
                "confidence": abstain_response_dict.get("confidence", 0.5),
                "usage": abstain_response_dict.get("usage", {"input_tokens": 0, "output_tokens": 0})
            }

        # Build a single user question for RARE from MediQ patient state
        patient_info = patient_state.get("initial_info", "")
        history = patient_state.get("interaction_history", [])
        conv_log = '\n'.join([f"Q: {qa.get('question','')}\nA: {qa.get('answer','')}" for qa in history])
        user_question = f"{self.inquiry}\n\nContext:\n{patient_info}\n\nHistory:\n{conv_log if conv_log else 'None'}"

        # Convert options dict (A-D) to the list RARE expects
        options_list = [self.options.get(letter, "") for letter in ["A", "B", "C", "D"]]

        try:
            best_choice, freq_choice, choice_info, all_solution_nodes, all_solutions = self.search_for_answers(
                args=self.rare_args,
                user_question=user_question,
                question_id=0,
                gt_answer="",
                generator=self.rare_generator,
                options=options_list
            )
            # Estimate confidence if choice_info carries voting distribution
            conf = 0.5
            try:
                if isinstance(choice_info, dict):
                    votes = choice_info.get("votes", {}) or choice_info.get("vote_counts", {})
                    total = sum(votes.values()) if votes else 0
                    conf = votes.get(best_choice, 0) / total if total > 0 else 0.5
            except Exception:
                pass

            return {
                "type": "choice",
                "letter_choice": best_choice,
                "confidence": float(conf),
                "usage": {"input_tokens": 0, "output_tokens": 0}
            }
        except Exception as e:
            # On any runtime error, degrade gracefully
            try:
                logging.getLogger("detail_logger").error(f"[RAREExpert] search_for_answers failed: {e}")
            except Exception:
                pass
            kwargs = self.get_abstain_kwargs(patient_state)
            abstain_response_dict = expert_functions.implicit_abstention_decision(**kwargs)
            return {
                "type": "question" if abstain_response_dict["abstain"] else "choice",
                "question": abstain_response_dict.get("atomic_question"),
                "letter_choice": abstain_response_dict.get("letter_choice"),
                "confidence": abstain_response_dict.get("confidence", 0.5),
                "usage": abstain_response_dict.get("usage", {"input_tokens": 0, "output_tokens": 0})
            }