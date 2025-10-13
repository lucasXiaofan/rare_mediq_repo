# MediQ MCTS Search Guide

This document explains how Rare-MediQ wires Monte Carlo Tree Search (MCTS) into its benchmarking flow, focusing on the `search_for_answers` entry point. It covers call structure, node actions, solution scoring, and guidelines for building new search routines for MediQ or other benchmarks.

## Entry Point: `do_rare.py`
- **Location:** `rare_mediq/run_src/do_rare.py:102`
- Loads dataset items, constructs a shared `Generator` with the requested model backend, and delegates each problem to `search_for_answers`.
- Persists the tuple returned by `search_for_answers` (best choice, frequency winner, per-choice stats, full solution traces) alongside accuracy counters and raw completions.

## `search_for_answers` API
- **Location:** `rare_mediq/run_src/MCTS_for_reasoning_plus.py:1821`
- **Signature:** `search_for_answers(args, user_question, question_id, gt_answer, generator, options)`
- **Inputs:** CLI args, raw question text, integer ID, gold solution (for logging), shared `Generator`, and optional multiple-choice `options`.
- **Outputs:**
  - `best_choice`: score-weighted winning answer (letter label when options exist).
  - `freq_choice`: most frequent answer across solution nodes.
  - `choice_info`: aggregated statistics per answer (scores, counts, completions).
  - `all_solution_nodes`: leaf nodes that qualify as candidate solutions.
  - `all_solutions`: raw completions extracted from those nodes.

## MCTS Backbone
- Implemented by `MCTS_Searcher` (`rare_mediq/run_src/MCTS_backbone.py`).
- Each rollout executes the classic four phases:
  1. `_select`: walk the tree via UCT until reaching an unexpanded node.
  2. `_expand`: ask the node to materialize children by calling its `find_children` helper.
  3. `_simulate`: perform a random playout among existing children until a terminal node is reached.
  4. `_backpropagate`: update cumulative value `Q` and visit count `N` along the visited path.
- Exploration weight scheduling (exponential, linear, constant) is controlled by CLI flags and modulates the UCT bonus over time.

## Reasoning Nodes
- Each state is represented by `Reasoning_MCTS_Node` (`rare_mediq/run_src/MCTS_for_reasoning_plus.py:1078`).
- Responsibilities:
  - Track depth-aware metadata: subquestion indices, retrieval queries, one-step thought history, and per-depth potential answers.
  - Enforce domain-specific termination checks: a node is terminal if it reaches maximum depth or produces a valid final answer/subquestion (`is_terminal`, `is_valid_leaf_node`).
  - Generate new children via `_create_children`, which dispatches the applicable actions (A1–A8) based on current node type and CLI disable flags.
  - Decide whether to skip reward propagation (paraphrase nodes do not backpropagate scores).

### Node Types and Actions
Node types are enumerated in `rare_mediq/run_src/rstar_utils.py:13`. The main actions are:

| Action | Description | Triggered From |
|--------|-------------|----------------|
| A1 | Generate one-step thought (chain-of-thought continuation). | USER_QUESTION, SUBQUESTION, RE_SUBANSWER, OST_STEP |
| A2 | Generate direct answer without retrieval. | Most node types except DIRECT_ANSWER variants |
| A3 | Spawn next subquestion plus provisional answer. | USER_QUESTION, SUBQUESTION |
| A4 | Re-answer the current subquestion. | SUBQUESTION |
| A5 | Rephrase the original user question. | USER_QUESTION |
| A6 | Produce direct answer with retrieval (RAG). | USER_QUESTION, SUBQUESTION, OST_STEP |
| A7 | Re-answer with retrieval support. | SUBQUESTION |
| A8 | Generate search queries and fetch documents. | USER_QUESTION, OST_STEP |

Each action forwards to a dedicated `Generator` method which returns new child nodes alongside model-derived confidence scores.

## Generator Responsibilities
- The `Generator` (`rare_mediq/run_src/MCTS_for_reasoning_plus.py:182`) centralizes all LLM/RAG prompting and scoring.
- Key methods:
  - `generate_direct_answers` / `_with_rag`: few-shot CoT answering with optional retrieved context; returns `(answers, scores, choice_labels)`.
  - `generate_subquestions`: decomposes the problem, answers subquestions, and optionally fuses retrieval-based verification when the subquestion aligns with the final answer.
  - `generate_queries`: synthesizes web/textbook queries, hits the retrieval API, and saves documents for later actions.
  - `generate_re_subanswers` / `_with_rag`: revisits subquestion answers to improve confidence or integrate new evidence.
  - `generate_rephrased_user_question`: paraphrases the user question to diversify reasoning paths.
  - `generate_ost_step`: emits a single reasoning step when building chain-of-thought traces.
- Return values typically include both natural-language completions and numeric confidences extracted by the dataset-specific evaluator.

## Solution Aggregation
- `stochastic_find_best_solution` (`rare_mediq/run_src/rstar_utils.py:719`) scans every node that satisfies `is_valid_solution_node` after each rollout.
- When multiple-choice options are present, it:
  - Extracts the letter choice from each node (`node.get_choice`).
  - Accumulates node values per choice to compute `best_choice` (highest confidence mass) and `freq_choice` (most visited).
- In free-form settings (`options=None`), the function simply returns the completion with the highest node value and includes raw solution strings for downstream metrics (e.g., ROUGE/BLEU).

## Adapting or Writing a New MCTS
1. **Define node semantics:** reuse `Reasoning_MCTS_Node` or subclass it to alter `find_children`, `is_valid_solution_node`, and reward propagation for your domain.
2. **Implement action generators:** add or modify methods on `Generator` that emit `(child_state, score)` tuples for new reasoning primitives (numeric solvers, structured APIs, etc.). Wire them into `_create_children` and expose CLI toggles if needed.
3. **Tune search policy:** adjust `_compute_uct` or add potential-based priors (`enable_potential_score`) to bias toward promising branches, or change simulation strategies to be deterministic instead of random sampling.
4. **Customize aggregation:** replace or extend `stochastic_find_best_solution` with domain-appropriate scoring (numeric error, logic validators, classifier checks) so the returned answer aligns with your benchmark.
5. **Integrate new benchmarks:** provide an evaluator compatible with your answer format (see `eval_src`) so `Generator` can normalize confidences and parse outputs regardless of whether the task is multiple-choice, free-form QA, or structured generation.

With these components, you can understand how MediQ’s MCTS iteratively explores reasoning actions and confidently adapt the framework to new datasets or bespoke planning algorithms.
