"""
Modified MCTS backbone with detailed step-by-step printing
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List
import math, random


node_cnt = 0


def verbose_print(s: str, verbose: bool):
    if verbose:
        print(s)


class MCTS_Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    def __init__(self) -> None:
        super().__init__()

        global node_cnt
        self.id = node_cnt
        node_cnt += 1

        self.rollout_id = None

    def set_rollout_id(self, rollout_id: int):
        self.rollout_id = rollout_id

    @abstractmethod
    def find_children(self, rollout_id: int):
        "All possible successors of this board state"
        raise NotImplementedError

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        raise NotImplementedError

    @abstractmethod
    def calculate_reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        raise NotImplementedError

    @abstractmethod
    def skip_backprop(self):
        "If True, the reward of this node will not be accumulated in the backpropagation step."
        raise NotImplementedError


class MCTS_Searcher:
    "Monte Carlo tree searcher with detailed step-by-step printing."

    def __init__(
        self,
        exploration_weight: float,
        weight_scheduler: str,
        num_rollouts: int,
        discount: float,
        verbose: bool = False,
    ):
        self.Q: Dict[MCTS_Node, float] = defaultdict(lambda: 0.0)  # total reward of each node
        self.N: Dict[MCTS_Node, int] = defaultdict(lambda: 0)  # total visit count for each node
        self.parent2children: Dict[MCTS_Node, List[MCTS_Node]] = dict()  # children of each node

        #! explored = expanded + simulated, i.e. has seen terminal at least once, i.e. we can calculate its UCT value, i.e. has Q and N
        self.explored_nodes = set()

        self.exploration_weight = exploration_weight
        self.weight_scheduler = weight_scheduler
        self.num_rollouts = num_rollouts
        self.discount = discount

        self.verbose = verbose

        global node_cnt
        node_cnt = 0

    def do_rollout(self, root_node: MCTS_Node, rollout_id: int):
        "Make the tree one layer better. (Train for one iteration.)"
        print("\n" + "=" * 80)
        print(f"ROLLOUT {rollout_id}: Starting")
        print("=" * 80)

        # STEP 1: Selection
        print(f"\n[STEP 1] SELECTION: Finding unexplored descendant...")
        path_1 = self._select(root_node, rollout_id)
        leaf = path_1[-1]
        print(f"  ✓ Selected path length: {len(path_1)}")
        print(f"  ✓ Leaf node: {leaf} (ID: {leaf.id}, Depth: {leaf.depth}, Type: {leaf.node_type})")

        # STEP 2: Expansion
        print(f"\n[STEP 2] EXPANSION: Expanding node {leaf.id}...")
        self._expand(leaf, rollout_id)
        if leaf in self.parent2children:
            print(f"  ✓ Created {len(self.parent2children[leaf])} children")
            for i, child in enumerate(self.parent2children[leaf]):
                print(f"    Child {i+1}: {child} (Type: {child.node_type})")
        else:
            print(f"  ✓ Node is terminal - no children created")

        # STEP 3: Simulation
        print(f"\n[STEP 3] SIMULATION: Simulating random path from node {leaf.id}...")
        path_2 = self._simulate(leaf, rollout_id)
        print(f"  ✓ Simulated path length: {len(path_2)}")
        if path_2:
            terminal_node = path_2[-1]
            print(f"  ✓ Terminal node: {terminal_node} (ID: {terminal_node.id}, Type: {terminal_node.node_type})")
        else:
            terminal_node = leaf
            print(f"  ✓ Terminal node is leaf: {terminal_node} (ID: {terminal_node.id})")

        # STEP 4: Backpropagation
        print(f"\n[STEP 4] BACKPROPAGATION: Propagating rewards...")
        full_path = path_1 + path_2
        reward = (path_2[-1] if path_2 else leaf).calculate_reward()
        print(f"  ✓ Terminal reward: {reward:.4f}")
        print(f"  ✓ Backpropagating through {len(full_path)} nodes:")
        self._backpropagate(full_path)

        # Summary
        print(f"\n[ROLLOUT {rollout_id} COMPLETE]")
        print(f"  Total nodes explored: {len(self.explored_nodes)}")
        print(f"  Final node: {terminal_node} (Type: {terminal_node.node_type})")
        print("=" * 80)

        try:
            return path_2[-1]
        except:
            return path_1[-1]

    def _select(self, node: MCTS_Node, rollout_id: int) -> List[MCTS_Node]:
        "Find an unexplored descendent of `node`"
        path = []
        step_num = 0
        while True:
            step_num += 1
            path.append(node)

            # case 1: a node does not have children, then select the node itself
            if node not in self.parent2children.keys():
                print(f"    Step {step_num}: Node {node.id} has no children yet → SELECT THIS NODE")
                return path

            # case 2: a node has children but not all children have been explored, then randomly select an unexplored child
            unexplored = [n for n in self.parent2children[node] if n not in self.explored_nodes]
            if unexplored:
                n = random.choice(unexplored)
                path.append(n)
                print(f"    Step {step_num}: Node {node.id} has {len(unexplored)} unexplored children → SELECT child {n.id}")
                return path

            # case 3: a node has children and all children have been explored, then select one child and go to the next layer
            print(f"    Step {step_num}: Node {node.id} - all children explored, using UCT...")
            node = self._uct_select(node, rollout_id)
            print(f"      → UCT selected child {node.id}")

    def _expand(self, node: MCTS_Node, rollout_id: int):
        "Update the `children` dict with the children of `node`"
        if node in self.explored_nodes:
            print(f"  ! Node {node.id} already explored - skipping expansion")
            return  # already expanded

        if node.is_terminal():
            self.explored_nodes.add(node)
            print(f"  ! Node {node.id} is terminal - no expansion needed")
            return  # terminal node is non-expandable

        print(f"  → Calling node.find_children() to generate actions...")
        print(f"  → Generating children for node type: {node.node_type}")
        self.parent2children[node] = node.find_children(rollout_id)

        # Print detailed info about what actions were used
        if hasattr(node, 'node_type'):
            action_map = {
                "OST_STEP": "A1 (One-Step Thought)",
                "DIRECT_ANSWER": "A2 (Direct Answer)",
                "SUBQUESTION": "A3 (Subquestion)",
                "RE_SUBANSWER": "A4 (Re-answer Subquestion)",
                "REPHRASED_USER_QUESTION": "A5 (Rephrase Question)"
            }
            child_types = {}
            for child in self.parent2children[node]:
                child_type = str(child.node_type).split('.')[-1]
                child_types[child_type] = child_types.get(child_type, 0) + 1

            print(f"  → Generated {len(self.parent2children[node])} children using actions:")
            for child_type, count in child_types.items():
                action_name = action_map.get(child_type, child_type)
                print(f"      • {action_name}: {count} children")

    def _simulate(self, node: MCTS_Node, rollout_id: int) -> List[MCTS_Node]:
        "Returns the reward for a random simulation (to completion) of `node`"
        path = []
        cur_node = node
        step_num = 0

        while True:
            step_num += 1

            if cur_node.is_terminal():
                self.explored_nodes.add(node)
                print(f"    Step {step_num}: Node {cur_node.id} is TERMINAL → END SIMULATION")
                return path

            if cur_node not in self.parent2children.keys():
                print(f"    Step {step_num}: Generating children for node {cur_node.id} (Type: {cur_node.node_type})...")
                self.parent2children[cur_node] = cur_node.find_children(rollout_id)

                # Show which actions were taken
                if hasattr(cur_node, 'node_type'):
                    action_map = {
                        "OST_STEP": "A1",
                        "DIRECT_ANSWER": "A2",
                        "SUBQUESTION": "A3",
                        "RE_SUBANSWER": "A4",
                        "REPHRASED_USER_QUESTION": "A5"
                    }
                    child_types = {}
                    for child in self.parent2children[cur_node]:
                        child_type = str(child.node_type).split('.')[-1]
                        child_types[child_type] = child_types.get(child_type, 0) + 1

                    action_summary = ", ".join([f"{action_map.get(ct, ct)}×{cnt}" for ct, cnt in child_types.items()])
                    print(f"      → Generated {len(self.parent2children[cur_node])} children ({action_summary})")

            selected = random.choice(self.parent2children[cur_node])  # randomly select a child
            selected_type = str(selected.node_type).split('.')[-1]
            action_map = {
                "OST_STEP": "A1",
                "DIRECT_ANSWER": "A2",
                "SUBQUESTION": "A3",
                "RE_SUBANSWER": "A4",
                "REPHRASED_USER_QUESTION": "A5"
            }
            action_used = action_map.get(selected_type, selected_type)
            print(f"    Step {step_num}: Randomly selected child {selected.id} using action {action_used}")
            cur_node = selected
            path.append(cur_node)

    def _backpropagate(self, path: List[MCTS_Node]):
        "Send the reward back up to the ancestors of the leaf"
        leaf = path[-1]
        reward = leaf.calculate_reward()

        for i, node in enumerate(reversed(path)):
            self.Q[node] += reward
            self.N[node] += 1
            self.explored_nodes.add(node)
            print(f"    Node {node.id}: Q={self.Q[node]:.4f}, N={self.N[node]}, Avg={self.Q[node]/self.N[node]:.4f}")

    def _get_weight(self, rollout_id: int):
        # start with exploration weight, end with 0.1 * exploration weight
        if self.weight_scheduler == "exp":
            return self.exploration_weight * (0.1 ** (rollout_id / self.num_rollouts))
        elif self.weight_scheduler == "lin":
            return self.exploration_weight * (1 - 0.9 * (rollout_id / self.num_rollouts))
        elif self.weight_scheduler == "const":
            return self.exploration_weight

    def _uct_select(self, node: MCTS_Node, rollout_id: int):
        "Select a child of node, balancing exploration & exploitation"

        # All children of the node should already be expanded
        assert all(n in self.explored_nodes for n in self.parent2children[node])

        # Calculate UCT for each child and print
        uct_values = []
        for child in self.parent2children[node]:
            uct = self._compute_uct(parent_node=node, node=child, rollout_id=rollout_id)
            uct_values.append((child, uct))
            print(f"        Child {child.id}: UCT={uct:.4f} (Q={self.Q[child]:.4f}, N={self.N[child]})")

        return max(uct_values, key=lambda x: x[1])[0]

    def _compute_uct(self, parent_node: MCTS_Node, node: MCTS_Node, rollout_id: int):
        "Upper confidence bound for trees"
        if parent_node is None:  # invalid UCT: the node is the root
            return 666
        else:
            if self.N[node] == 0:  # invalid UCT: the node has not been explored yet
                return 999
            else:
                weight = self._get_weight(rollout_id)
                return self.Q[node] / self.N[node] + weight * math.sqrt(math.log(self.N[parent_node]) / self.N[node])
