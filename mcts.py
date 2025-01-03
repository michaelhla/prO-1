import numpy as np
from collections import defaultdict
from reward import calculate_reward
from zymctrl import generate
import math

class MCTSNode:
    def __init__(self, sequence="", score=0, parent=None):
        self.sequence = sequence
        self.score = score
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_reward = 0.0

class MCTS:
    def __init__(self, model, tokenizer, special_tokens, device, exploration_weight=1.0):
        self.model = model
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens
        self.device = device
        self.exploration_weight = exploration_weight
        self.root = MCTSNode()

    def uct_score(self, node, parent_visits):
        """Upper Confidence Bound (UCT) calculation"""
        if node.visits == 0:
            return float('inf')
        
        exploitation = node.total_reward / node.visits
        exploration = math.sqrt(math.log(parent_visits) / node.visits)
        return exploitation + self.exploration_weight * exploration

    def select(self, node):
        """Select the most promising node to explore"""
        if not node.children:
            return node
            
        return self.select(max(
            node.children,
            key=lambda child: self.uct_score(child, node.visits)
        ))

    def expand(self, node, label):
        """Generate new children using ZymCTRL"""
        # Generate new sequences
        sequences = generate(
            label,
            self.model,
            self.special_tokens,
            self.device,
            self.tokenizer,
            num_sequences=5  # Reduced number for efficiency
        )
        
        # Create child nodes for each generated sequence
        for sequence, perplexity in sequences:
            child = MCTSNode(
                sequence=sequence,
                score=-perplexity,  # Using negative perplexity as initial score
                parent=node
            )
            node.children.append(child)
        
        return node.children[0] if node.children else node

    def simulate(self, node):
        """Calculate reward for a given sequence node using binding energy calculations
    
    Args:
        node: Node object containing sequence and other properties
        
    Returns:
        float: Reward score based on binding energies across reaction pathway
    """
    # Get sequence from node
    sequence = node.sequence
    
    # # Get reaction components (these should be defined/imported somewhere)
    # reagent = get_reagent_structure()  # RDKit Mol object
    # ts = get_transition_state()        # RDKit Mol object
    # product = get_product_structure()  # RDKit Mol object
    
    # Calculate reward using our BindingEnergyCalculator
    reward = calculate_reward(
        sequence=sequence,
        reagent=reagent,
        ts=ts,
        product=product
    )
    
    return reward

    def backpropagate(self, node, reward):
        """Update statistics for all nodes up to root"""
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def search(self, label, num_iterations=100):
        """Main MCTS loop"""
        for _ in range(num_iterations):
            # Selection
            node = self.select(self.root)
            
            # Expansion
            if node.visits > 0:
                node = self.expand(node, label)
            
            # Simulation
            reward = self.simulate(node)
            
            # Backpropagation
            self.backpropagate(node, reward)

        # Return best sequence found
        best_child = max(
            self.root.children,
            key=lambda child: child.total_reward / (child.visits + 1e-6)
        )
        return best_child.sequence, best_child.total_reward / best_child.visits

def run_mcts_optimization(label, model, tokenizer, special_tokens, device, 
                         num_iterations=100, exploration_weight=1.0):
    """Wrapper function to run MCTS optimization"""
    mcts = MCTS(model, tokenizer, special_tokens, device, exploration_weight)
    best_sequence, best_score = mcts.search(label, num_iterations)
    return best_sequence, best_score
