"""
Minimal test: Verify IO_System works with DeepSeek API.
Set DEEPSEEK_API_KEY environment variable before running.
"""

import sys
sys.path.append(".")

import argparse
from models.IO_System import IO_System
from run_src.MCTS_backbone_verbose import MCTS_Searcher
from run_src.MCTS_for_reasoning_plus import Generator, Reasoning_MCTS_Node, Node_Type
from eval_src.Evaluator import Evaluator



def create_args_for_deepseek():
    """
    Create args object based on run_generate_medqa_qwen0_6b.sh settings,
    but using DeepSeek API instead of local model.
    """
    args = argparse.Namespace()

    # ============================================================================
    # API Configuration - Using DeepSeek API
    # ============================================================================
    args.api = "deepseek"  # Uses the DeepSeek API in IO_System
    args.model_ckpt = "deepseek-chat"  # DeepSeek model name

    # ============================================================================
    # Dataset & Mode (from shell script)
    # ============================================================================
    args.dataset_name = "MedQA"
    args.test_json_filename = "test_all"
    args.mode = "run"
    args.note = "deepseek_mcts_demo"

    # ============================================================================
    # MCTS Configuration (from shell script)
    # ============================================================================
    args.num_rollouts = 1  # Just 1 for demo (shell script uses 4)
    args.mcts_exploration_weight = 1.0
    args.mcts_weight_scheduler = "constant"
    args.mcts_discount_factor = 1.0
    args.max_depth_allowed = 3

    # ============================================================================
    # Action Flags (from shell script)
    # ============================================================================
    args.disable_a5 = False  # --disable_a5
    args.disable_a8 = True  # --disable_a8
    args.disable_a4 = False  # --disable_a4
    args.disable_a6 = True  # --disable_a6
    args.disable_a7 = True  # --disable_a7
    args.disable_a1 = False
    args.disable_a3 = False

    # ============================================================================
    # LLM Generation Parameters
    # ============================================================================
    args.temperature = 0.8
    args.top_k = 40
    args.top_p = 0.95
    args.max_tokens = 1000

    # ============================================================================
    # Generator Settings
    # ============================================================================
    args.num_subquestions = 3
    args.num_queries = 2
    args.num_a1_steps = 3
    args.num_votes = 3
    args.mcts_num_last_votes = 3
    args.search_query_weight = 0.5
    args.enable_potential_score = False
    args.enable_answer_checking = False
    args.combine_distributions = "add"
    args.enable_majority = False
    args.majority_threshold = 0.8
    args.enable_self_reward = False

    # ============================================================================
    # Retrieval Configuration (from shell script)
    # ============================================================================
    args.num_retrieval = 5  # --num_retrieval 5
    args.retrieval_corpus = "pubmed"  # --retrieval_corpus "pubmed"
    args.retrieval_threshold = 0.5
    args.use_triples = False

    # ============================================================================
    # Chat Template (from shell script)
    # ============================================================================
    args.enable_chat_template = True  # --enable_chat_template true
    args.modify_prompts_for_rephrasing = False
    # ============================================================================
    # Other Settings
    # ============================================================================
    args.verbose = True
    args.prompts_root = "prompts"
    import os
    # Paths to prompt template files (you'll need these to exist)
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
            args.fewshot_cot_prompt_rephrased_path = os.path.join(
                prompts_dir, "fewshot_cot", "fewshot_cot_prompt_rephrased.txt"
            )
            args.fewshot_ost_prompt_rephrased_path = os.path.join(
                prompts_dir, "fewshot_ost", "fewshot_ost_prompt_rephrased.txt"
            )
            args.decompose_prompt_rephrased_path = os.path.join(
                prompts_dir, "decompose", "decompose_prompt_rephrased.txt"
            )
        else:
            args.fewshot_cot_prompt_rephrased_path = os.path.join(prompts_dir, "fewshot_cot", "fewshot_cot_prompt.txt")
            args.fewshot_ost_prompt_rephrased_path = os.path.join(prompts_dir, "fewshot_ost", "fewshot_ost_prompt.txt")
            args.decompose_prompt_rephrased_path = os.path.join(prompts_dir, "decompose", "decompose_prompt.txt")
    return args


def main():
    print("=" * 80)
    print("MCTS Reasoning with DeepSeek API - Mini Demo")
    print("=" * 80)

    # ============================================================================
    # Question Setup - Using system/user message format
    # ============================================================================
    # NOTE: Can you set up questions in system/user message format?
    # YES! The system combines them into a single string for the Generator.
    # The IO_System with enable_chat_template=True will parse it back into messages.

    system_message = {
        'role': 'system',
        'content': 'You are a medical doctor trying to reason through a real-life clinical case. Based on your understanding of basic and clinical science, medical knowledge, and mechanisms underlying health, disease, patient care, and modes of therapy, respond according to the task specified by the user. Base your response on the current and standard practices referenced in medical guidelines.'
    }

    user_message = {
        'role': 'user',
        'content': '''A patient comes into the clinic presenting with a symptom as described in the conversation log below:

PATIENT INFORMATION: A 21-year-old sexually active male complains of fever, pain during urination, and inflammation and pain in the right knee.
CONVERSATION LOG:
None
QUESTION: The mechanism of action of the medication given blocks cell wall synthesis, which of the following was given?
OPTIONS: A: Gentamicin, B: Ciprofloxacin, C: Ceftriaxone, D: Trimethoprim
YOUR TASK: Medical conditions are complex, so you should seek to understand their situations across many features. First, consider which medical specialty is this patient's case; then, consider a list of necessary features a doctor would need to make the right medical judgment; finally, consider whether all necessary information is given in the conversation above. How confident are you to pick the correct option to the problem factually using the conversation log? Choose between the following ratings:
"Very Confident" - The correct option is supported by all evidence, and there is enough evidence to eliminate the rest of the answers, so the option can be confirmed conclusively.
"Somewhat Confident" - I have reasonably enough information to tell that the correct option is more likely than other options, more information is helpful to make a conclusive decision.
"Neither Confident or Unconfident" - There are evident supporting the correct option, but further evidence is needed to be sure which one is the correct option.
"Somewhat Unconfident" - There are evidence supporting more than one options, therefore more questions are needed to further distinguish the options.
"Very Unconfident" - There are not enough evidence supporting any of the options, the likelihood of picking the correct option at this point is near random guessing.

Think carefully step by step, respond with the chosen confidence rating ONLY and NOTHING ELSE.'''
    }

    # Combine system and user messages into formatted question
    # When enable_chat_template=True, the IO_System will parse this back into messages
    formatted_question = f"{system_message['content']}\n\n{user_message['content']}"

    print("\nCustom Question Setup:")
    print("-" * 80)
    print(f"System: {system_message['content'][:100]}...")
    print(f"User: {user_message['content'][:100]}...")
    print()

    # Multiple choice options
    options = {
        'A': 'Gentamicin',
        'B': 'Ciprofloxacin',
        'C': 'Ceftriaxone',
        'D': 'Trimethoprim'
    }

    # ============================================================================
    # Step 1: Create args and initialize components
    # ============================================================================
    print("Step 1: Initializing components...")
    args = create_args_for_deepseek()

    # For DeepSeek API, tokenizer and model are handled by IO_System
    # model should be the model name string
    tokenizer = None
    model = args.model_ckpt  # "deepseek-chat"

    # Initialize evaluator
    evaluator = Evaluator()

    # Initialize generator (handles LLM interactions via IO_System)
    generator = Generator(args, tokenizer, model, evaluator)
    print("   Generator initialized with DeepSeek API")

    # ============================================================================
    # Step 2: Initialize MCTS_Searcher
    # ============================================================================
    print("\nStep 2: Initializing MCTS_Searcher...")
    mcts_searcher = MCTS_Searcher(
        exploration_weight=args.mcts_exploration_weight,
        weight_scheduler=args.mcts_weight_scheduler,
        num_rollouts=args.num_rollouts,
        discount=args.mcts_discount_factor,
        verbose=args.verbose,
    )
    print("   MCTS_Searcher initialized")

    # ============================================================================
    # Step 3: Initialize root Reasoning_MCTS_Node
    # ============================================================================
    print("\nStep 3: Initializing root Reasoning_MCTS_Node...")
    root_node = Reasoning_MCTS_Node(
        parent=None,
        depth=0,
        node_type=Node_Type.USER_QUESTION,
        verbose=args.verbose,
        generator=generator,
        disable_a5=args.disable_a5,
        user_question=formatted_question,
        expected_answer="",
        max_depth_allowed=args.max_depth_allowed,
        disable_a1=args.disable_a1,
        disable_a3=args.disable_a3,
        disable_a4=args.disable_a4,
        disable_a6=args.disable_a6,
        disable_a7=args.disable_a7,
        disable_a8=args.disable_a8,
        options=options,
        search_query_answer_weight=args.search_query_weight,
        enable_potential_score=args.enable_potential_score,
        enable_answer_checking=args.enable_answer_checking,
        retrieval_corpus=args.retrieval_corpus
    )
    print("   Root node initialized")
    print(f"    - Node type: {root_node.node_type}")
    print(f"    - Depth: {root_node.depth}")
    print(f"    - Max depth allowed: {args.max_depth_allowed}")

    # ============================================================================
    # Step 4: Perform single rollout
    # ============================================================================
    print("\n" + "=" * 80)
    print("Step 4: Performing MCTS rollout...")
    print("=" * 80)
    for i in range(4):
        rollout_id = i
        try:
            rollout_node = mcts_searcher.do_rollout(root_node, rollout_id)

            print(f"\n Rollout {rollout_id} completed successfully!")
            print(f"  - Final node type: {rollout_node.node_type}")
            print(f"  - Node depth: {rollout_node.depth}")

            # ========================================================================
            # Step 5: Examine results
            # ========================================================================
            print("\n" + "=" * 80)
            print("Results:")
            print("=" * 80)

            if rollout_node.is_valid_solution_node():
                print(" Reached a valid solution node!")

                # Extract answer if available
                if hasattr(rollout_node, 'direct_answer') and rollout_node.direct_answer:
                    print(f"\nDirect Answer:")
                    print(f"  {rollout_node.direct_answer}")

                # Extract choice if available
                if hasattr(rollout_node, 'choice') and rollout_node.choice:
                    print(f"\nSelected Choice: {rollout_node.choice}")
                    print(f"Option Text: {options.get(rollout_node.choice, 'Unknown')}")

                # Show solution trace (reasoning path)
                print("\nSolution Trace:")
                for step_id, step_data in rollout_node.solution_trace.items():
                    print(f"  Step {step_id}: {list(step_data.keys())}")
            else:
                print("� Did not reach a valid solution node")
                print(f"  Current node type: {rollout_node.node_type}")

        except Exception as e:
            print(f"\n Error during rollout: {e}")
            print(f"  Make sure:")
            print(f"  1. DeepSeek API key is set in models/OpenAI_API.py")
            print(f"  2. All prompt template files exist")
            import traceback
            traceback.print_exc()

        print("\n" + "=" * 80)
        print("Demo Complete!")
        print("=" * 80)

        print("\n" + "=" * 80)
        print("NOTES ON SYSTEM/USER MESSAGE FORMAT:")
        print("=" * 80)
        print("Q: Can I set up questions in system/user message format?")
        print()
        print("A: YES! Here's what happens:")
        print("  1. You combine system + user messages into a formatted_question string")
        print("  2. The Generator receives this as a single string")
        print("  3. When enable_chat_template=True, the IO_System's OpenAI_API")
        print("     parses it back into message format before sending to DeepSeek")
        print("  4. The parsing happens in generate_with_OpenAI_model() using")
        print("     the ### Instruction: and ### Response: markers")
        print()
        print("  For your custom format, you may need to adjust the parsing logic")
        print("  in models/OpenAI_API.py lines 52-68 to match your message structure.")
        print("=" * 80)


if __name__ == "__main__":
    # First test DeepSeek integration
    # test_deepseek()

    # Run full MCTS demo
    main()
