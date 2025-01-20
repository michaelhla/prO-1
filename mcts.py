import numpy as np
from collections import defaultdict
from reward import calculate_reward
from zymctrl import generate
import math

class MCTSNode:
    def __init__(self, sequence="", score=0, parent=None, mutation="", reasoning=""):
        self.sequence = sequence
        self.score = score
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_reward = 0.0
        self.mutation = mutation  # Store the mutation that led to this sequence
        self.reasoning = reasoning  # Store the reasoning for this mutation
        self.reward_calculated = False  # Flag to track if we've calculated the reward

class MCTS:
    def __init__(self, openai_client, base_sequence, device, exploration_weight=1.0):
        self.openai_client = openai_client
        self.base_sequence = base_sequence
        self.device = device
        self.exploration_weight = exploration_weight
        self.root = MCTSNode()
        self.pending_evaluations = []  # Store nodes that need reward calculation

    def parse_response(self, response_text):
        """Parse OpenAI response for mutations and reasoning"""
        mutations = []
        reasonings = []
        
        # Split response into lines
        lines = response_text.split('\n')
        
        for line in lines:
            if '%%MUTATION_' in line:
                mutation = line.split('%%MUTATION_')[1].split('%%')[0]
                mutations.append(mutation)
            if '%%REASONING_' in line:
                reasoning = line.split('%%REASONING_')[1].split('%%')[0]
                reasonings.append(reasoning)
                
        return list(zip(mutations, reasonings))

    def expand(self, node, label):
        """Generate new children using OpenAI API"""
        # Construct prompt for OpenAI
        prompt = f"Given the enzyme sequence: {self.base_sequence}\n"
        prompt += f"Suggest 5 mutations to improve {label}. Format each suggestion as:\n"
        prompt += "%%MUTATION_1%%[mutation]%%\n%%REASONING_1%%[reasoning]%%\n"
        
        # Call OpenAI API
        response = self.openai_client.chat.completions.create(
            model="your-fine-tuned-model",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        # Parse mutations and reasoning
        mutations_and_reasoning = self.parse_response(response.choices[0].message.content)
        
        # Create child nodes for each mutation
        for mutation, reasoning in mutations_and_reasoning:
            # Apply mutation to base sequence to get new sequence
            mutated_sequence = self.apply_mutation(self.base_sequence, mutation)
            
            child = MCTSNode(
                sequence=mutated_sequence,
                score=0,
                parent=node,
                mutation=mutation,
                reasoning=reasoning
            )
            node.children.append(child)
            self.pending_evaluations.append(child)
        
        return node.children[0] if node.children else node

    def apply_mutation(self, base_sequence, mutation):
        """Apply mutation string to base sequence"""
        # Implement mutation application logic here
        # Example: mutation might be "A123G" meaning replace A at position 123 with G
        return base_sequence  # Placeholder - implement actual mutation logic

    def simulate(self, node):
        """Placeholder simulation that returns 0 if reward not calculated"""
        if not node.reward_calculated:
            return 0
        return node.total_reward / max(node.visits, 1)

    def calculate_pending_rewards(self):
        """Calculate rewards for all pending nodes"""
        if not self.pending_evaluations:
            return

        # Calculate rewards for all pending nodes
        for node in self.pending_evaluations:
            reward = calculate_reward(sequence=node.sequence)
            node.reward_calculated = True
            self.backpropagate(node, reward)

        self.pending_evaluations = []  # Clear the pending list

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

    def backpropagate(self, node, reward):
        """Update statistics for all nodes up to root"""
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    def search(self, label, num_iterations=100):
        """Modified MCTS loop with batched reward calculations"""
        expansion_batch_size = 5  # Number of expansions before calculating rewards
        
        for i in range(num_iterations):
            # Selection
            node = self.select(self.root)
            
            # Expansion
            if node.visits > 0:
                node = self.expand(node, label)
            
            # Calculate rewards periodically or at the end
            if (i + 1) % expansion_batch_size == 0 or i == num_iterations - 1:
                self.calculate_pending_rewards()

        # Ensure all rewards are calculated
        self.calculate_pending_rewards()

        # Return best sequence found
        best_child = max(
            self.root.children,
            key=lambda child: child.total_reward / (child.visits + 1e-6)
        )
        return (
            best_child.sequence, 
            best_child.total_reward / best_child.visits,
            best_child.mutation,
            best_child.reasoning
        )

def run_mcts_optimization(label, openai_client, base_sequence, device, 
                         num_iterations=100, exploration_weight=1.0):
    """Wrapper function to run MCTS optimization"""
    mcts = MCTS(openai_client, base_sequence, device, exploration_weight)
    best_sequence, best_score, best_mutation, best_reasoning = mcts.search(label, num_iterations)
    return best_sequence, best_score, best_mutation, best_reasoning
