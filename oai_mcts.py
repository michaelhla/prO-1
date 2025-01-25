import numpy as np
from collections import defaultdict
from tqdm import tqdm
import math
from heapq import heappush, heappushpop, nlargest   
import openai
import dotenv
import os
import random
from reward import BindingEnergyCalculator, calculate_reward
dotenv.load_dotenv()

openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
        self.done_reached = False ## flag to check if we have terminated at this node before

    def get_path_history(self):
        """Get the history of mutations and reasoning leading to this node"""
        history = []
        current = self
        while current.parent is not None:  # Stop at root
            history.append({
                "mutation": current.mutation,
                "reasoning": current.reasoning
            })
            current = current.parent
        return list(reversed(history))  # Return in chronological order

class MCTS:
    def __init__(self, openai_client, initial_prompt, base_sequence, device, exploration_weight=1.0, passes=5, substrates=[], products=[], metal_ions=[], calculator=None, active_site_residues=None):
        self.openai_client = openai_client
        self.initial_prompt = initial_prompt
        self.base_sequence = base_sequence
        self.device = device
        self.exploration_weight = exploration_weight
        self.root = MCTSNode(sequence=base_sequence)
        self.sequence_to_node = {base_sequence: self.root}  # New dictionary to track all sequences
        self.passes = passes
        self.best_leaves = []  # Priority queue of (reward, node) tuples
        self.substrates = substrates
        self.products = products
        self.metal_ions = metal_ions
        self.calculator = calculator
        self.active_site_residues = [('A', idx) for idx, _ in active_site_residues] if active_site_residues else None  # Convert to ('A', idx) format

    def parse_response(self, response_text):
        """Parse OpenAI response for mutations and reasoning"""
        mutations = []
        reasonings = []
        
        # Split response into lines
        lines = response_text.split('\n')
        
        for line in lines:
            if '%%MUTATION_' in line:
                # Extract the actual mutation (e.g., "V55I")
                parts = line.split(':', 1) if '%%MUTATION_' in line else None
                if len(parts) == 2:
                    mutation_text = parts[1].strip()
                    # Extract position and residue from format like "154V"
                    import re
                    match = re.match(r'(\d+)([A-Z])', mutation_text)
                    if match:
                        position = match.group(1)
                        residue = match.group(2)
                        mutations.append(f"{position}{residue}")
            if '%%REASONING_' in line:
                # Extract the actual reasoning text
                parts = line.split(':', 1) if '%%REASONING_' in line else None
                if len(parts) == 2:
                    reasoning = parts[1].strip()
                    reasonings.append(reasoning)
                
        return list(zip(mutations, reasonings))

    def find_existing_node(self, sequence):
        """Find if a sequence already exists in the tree"""
        return self.sequence_to_node.get(sequence)

    def expand(self, node, label):
        """Generate a new child using OpenAI API with high temperature sampling"""
        # Get previous mutations in the path
        path_history = node.get_path_history()


        # Construct prompt for OpenAI
        active_site_info = f"ACTIVE SITE RESIDUES: {', '.join([f'{res}{idx}' for res, idx in self.active_site_residues])}"
        messages = [
            {"role": "system", "content": "You are an expert protein engineer with years of experience optimizing protein activity with rational design. \nWhenever the user inputs \"<CONTINUE>\", SELECT the next mutations to make and REASON in the FORMAT: \n\n%%MUTATION_i%%: [Position][New AA]\n%%REASONING_i%%: Reasoning\n\nKeep your response to under 100 words. select at most 1 mutation(s).  ****ALL REASONING MUST BE SPECIFIC TO THE ENZYME AND REACTION SPECIFIED IN THE PROMPT. CITE SCIENTFIC LITERATURE. CONSIDER SIMILAR ENZYMES AND REACTIONS.**** Keep your explanation concise and focused on the scientific reasoning. ONLY OUTPUT THE TEXT REASONING, NO OTHER TEXT OR LABELS. ONLY MODIFY SPECIFIED ACTIVE SITE RESIDUES. If there are no further optimizations to make, return <DONE> with no explanation."},
            {"role": "user", "content": f"{self.initial_prompt}\n{active_site_info}\nMutations are only allowed on the specified active site residues."}
        ]

        # Add previous steps as alternating assistant/user messages
        if path_history:
            for i, step in enumerate(path_history, 1):
                messages.append({
                    "role": "assistant", 
                    "content": f"%%MUTATION_{i}%%: {step['mutation']}\n%%REASONING_{i}%%: {step['reasoning']}"
                })
                messages.append({
                    "role": "user",
                    "content": "<CONTINUE>"
                })

        new_node = None
        while new_node is None: 
            # Call OpenAI API with high temperature for exploration
            response = self.openai_client.chat.completions.create(
                model="ft:gpt-4o-mini-2024-07-18:harvard-university::AruHbq4C",
                messages=messages,
                temperature=1.3,  # High temperature for more exploration
                max_tokens=200
            )
            
            response_content = response.choices[0].message.content
            print(response_content)
            
            # Check if we've reached a leaf node (model returns DONE)
            if "<DONE>" in response_content:
                print('node sequence: ', node.sequence)
                reward = calculate_reward(sequence=str(node.sequence), reagents=self.substrates, products=self.products, calculator=self.calculator)
                node.reward_calculated = True
                node.done_reached = True
                
                # Store in best_leaves if good enough
                if len(self.best_leaves) < self.passes:
                    heappush(self.best_leaves, (reward, node))
                elif reward > self.best_leaves[0][0]:  # Better than worst best leaf
                    heappushpop(self.best_leaves, (reward, node))
                    
                self.backpropagate(node, reward)
                print(f"Node {node.mutation} has reward {reward}")
                return node
                
            # Parse single mutation and reasoning
            mutations_and_reasoning = self.parse_response(response_content)
            
            # Create single child node
            if mutations_and_reasoning:
                mutation, reasoning = mutations_and_reasoning[0]
                try: 
                    mutated_sequence = self.apply_mutation(self.base_sequence, mutation)
                except Exception as e:
                    print(f"Mutation {mutation} failed")
                    continue
                
                # Check if this sequence already exists
                existing_node = self.find_existing_node(mutated_sequence)
                if existing_node:
                    if existing_node not in node.children:
                        node.children.append(existing_node)
                    return existing_node
                
                # Create new node if sequence doesn't exist
                child = MCTSNode(
                    sequence=mutated_sequence,
                    score=0,
                    parent=node,
                    mutation=mutation,
                    reasoning=reasoning
                )
                node.children.append(child)
                self.sequence_to_node[mutated_sequence] = child
                return child

    def apply_mutation(self, base_sequence, mutation):
        """Apply mutation string to base sequence"""
        # Extract original AA, position, and new AA from mutation string (e.g. "A123G")
        pos = int(mutation[0:-1]) - 1  # Convert to 0-based indexing
        new_aa = mutation[-1]

        # Verify original AA matches sequence and track failures
        if not hasattr(self, 'mutation_check_failures'):
            self.mutation_check_failures = 0
        if not hasattr(self, 'total_mutations'):
            self.total_mutations = 0

        self.total_mutations += 1
        
        # Check if position is valid
        if pos < 0 or pos >= len(base_sequence):
            print(mutation, pos)
            print(f"Warning: Position {pos} is out of bounds for sequence length {len(base_sequence)}")
            self.mutation_check_failures += 1
            return -1

        # if base_sequence[pos] != orig_aa:
        #     self.mutation_check_failures += 1
        #     print(f"Warning: Original AA {orig_aa} does not match sequence at position {pos} "
        #           f"({(self.mutation_check_failures/self.total_mutations)*100:.1f}% failure rate)")
        #     new_aa = base_sequence[pos]  # Use original AA instead
            
        # Create new sequence with mutation
        mutated_sequence = base_sequence[:pos] + new_aa + base_sequence[pos+1:]
        return mutated_sequence

    def simulate(self, node):
        """Placeholder simulation that returns 0 if reward not calculated"""
        if not node.reward_calculated:
            return 0
        return node.total_reward / max(node.visits, 1)
    
    def uct_score(self, node, parent_visits):
        """Upper Confidence Bound (UCT) calculation"""
        if node.visits == 0:
            return float('inf')
        
        exploitation = node.total_reward / node.visits
        exploration = math.sqrt(math.log(parent_visits) / node.visits)
        return exploitation + self.exploration_weight * exploration

    def select_next_move(self, node):
        possible_moves = []
        uct_scores = []
        
        # Option 1: Move to parent
        if node.parent:
            possible_moves.append(("parent", node.parent))
            uct_scores.append(self.uct_score(node.parent, node.visits))
        
        # Option 2: Move to existing child
        for child in node.children:
            possible_moves.append(("child", child))
            uct_scores.append(self.uct_score(child, node.visits))
        
        # Option 3: Expand new child (always an option now)
        possible_moves.append(("expand", None))
        expansion_score = float('inf') if not node.children else max(uct_scores) if uct_scores else float('inf')
        uct_scores.append(expansion_score)
        
        # If no moves possible (shouldn't happen), stay at current node
        if not possible_moves:
            return "stay", node
        
        # Select move with highest UCT score
        best_idx = max(range(len(uct_scores)), key=lambda i: uct_scores[i])
        move_type, next_node = possible_moves[best_idx]
        
        # If we chose to expand, do it now
        if move_type == "expand":
            new_node = self.expand(node, "label")  # You'll need to pass the appropriate label
            if new_node:  # Expansion succeeded
                return "expand", new_node
            # If expansion failed, choose the next best move
            possible_moves.pop(best_idx)
            uct_scores.pop(best_idx)
            if not possible_moves:
                return "stay", node
            best_idx = max(range(len(uct_scores)), key=lambda i: uct_scores[i])
            move_type, next_node = possible_moves[best_idx]
        
        return move_type, next_node

    def search(self, label, num_iterations=100):
        """Modified MCTS loop using single-step selection"""
        current_node = self.root
        
        for _ in tqdm(range(num_iterations), desc="MCTS iterations"):
            # Randomly restart from root occasionally
            if random.random() < 0.3:
                current_node = self.root
            
            # Select single next move
            move_type, next_node = self.select_next_move(current_node)
            
            # Handle the move
            if move_type == "expand":
                # Node has already been created and visited during expansion
                reward = self.simulate(next_node)
                self.backpropagate(next_node, reward)
                # print(f"New node {next_node.mutation} created")
            
            # Update current node
            current_node = next_node
        
        # Return k best leaves found during search
        if not self.best_leaves:
            return []
            
        best_k_leaves = nlargest(self.passes, self.best_leaves)
        return [(node.sequence, reward, node.mutation, node.reasoning) 
                for reward, node in best_k_leaves]

    def backpropagate(self, node, reward):
        """Update statistics for all nodes from the given node back to root"""
        current = node
        while current is not None:
            current.visits += 1  # Increment visit count
            current.total_reward += reward
            current = current.parent

def run_mcts_optimization(label, openai_client, device, 
                         num_iterations=100, exploration_weight=1.0, calculator=None, active_site_residues=None):
    """Wrapper function to run MCTS optimization"""
    sequence = "MGYARRVMDGIGEVAVTGAGGSVTGARLRHQVRLLAHALTEAGIPPGRGVACLHANTWRAIALRLAVQAIGCHYVGLRPTAAVTEQARAIAAADSAALVFEPSVEARAADLLERVSVPVVLSLGPTSRGRDILAASVPEGTPLRYREHPEGIAVVAFTSGTTGTPKGVAHSSTAMSACVDAAVSMYGRGPWRFLIPIPLSDLGGELAQCTLATGGTVVLLEEFQPDAVLEAIERERATHVFLAPNWLYQLAEHPALPRSDLSSLRRVVYGGAPAVPSRVAAARERMGAVLMQNYGTQEAAFIAALTPDDHARRELLTAVGRPLPHVEVEIRDDSGGTLPRGAVGEVWVRSPMTMSGYWRDPERTAQVLSGGWLRTGDVGTFDEDGHLHLTDRLQDIIIVEAYNVYSRRVEHVLTEHPDVRAAAVVGVPDPDSGEAVCAAVVVADGADPDPEHLRALVRDHLGDLHVPRRVEFVRSIPVTPAGKPDKVKVRTWFTD"
    ec_number = "6.3.1"
    name = "amide bond synthetase"
    general_information = "Amide bond synthetase is an enzyme that catalyzes the formation of amide bonds between a carbonyl group and an amine group. It is found in the liver of mammals and plays a crucial role in the synthesis of proteins."
    substrates = ["C1=CC(=CC=C1C(=O)O)Cl", "C1COCCN1CCN"]
    products = ["C1COCCN1CCNC(=O)C2=CC=C(C=C2)Cl", "[O-]P(=O)([O-])[O-]"]
    metal_ions = []
    known_mutations_text = "None"
    initial_prompt = f"""You are an expert protein engineer. You are working with an enzyme sequence given below, as well as other useful information regarding the enzyme/reaction: 

ENZYME NAME: {name}
EC NUMBER: {ec_number}
GENERAL INFORMATION: {general_information}
SUBSTRATES: {', '.join(substrates)}
PRODUCTS: {', '.join(products)}
METALS/IONS: {', '.join(metal_ions)}
{known_mutations_text}
ACTIVE SITE RESIDUES: {', '.join([f'{res}{idx}' for res, idx in active_site_residues])}

Propose a few mutations that will optimize enzymatic activity given the substrates and products above. For each proposed mutation, explain your reasoning and consider:
1. How the mutation affects protein structure and function
2. The chemical properties of the amino acids and substrates/products
3. The position's importance in the protein sequence

For each mutation you propose, provide clear, scientific reasoning for why the mutation would be beneficial, ****USE YOUR KNOWLEDGE OF THIS SPECIFIC ENZYME AND REACTION****. Keep your response to under 100 words."""

    mcts = MCTS(openai_client, initial_prompt, sequence, device, exploration_weight, passes=5, substrates=substrates, products=products, metal_ions=metal_ions, calculator=calculator, active_site_residues=active_site_residues)
    best_sequences = mcts.search(label, num_iterations)
    print(best_sequences)
    return best_sequences

if __name__ == "__main__":
    calculator = BindingEnergyCalculator(device="cuda")
    active_site_residues = [(154, 'V'), (197, 'I'), (300, 'A'), (407, 'R')]  # Example active site residues
    run_mcts_optimization("test", openai_client, device='cuda', calculator=calculator, active_site_residues=active_site_residues)