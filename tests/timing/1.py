import sys
import os
import time
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from esm import pretrained
from pyrosetta import init, pose_from_sequence
from deepseek import DeepSeekModel
from grpo_trainer import GRPOTrainer

## ESMfold can be 30 sec 
## Rosetta roughly 10 sec
## 2:10 for just one inference and backprop (amortized), total is 17 min for one GRPO step without reward function
## this is unreasonably slow 


def test_pipeline_timing():
    # Initialize PyRosetta
    init()
    
    # Test sequence
    sequence = "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQ"
    
    print("Testing pipeline timing...")
    
    # Time ESMFold
    print("\nTiming ESMFold prediction...")
    start = time.time()
    model, alphabet = pretrained.load_model_and_alphabet('esm2_t33_650M_UR50D')
    model = model.eval().cuda()
    batch_converter = alphabet.get_batch_converter()
    data = [("protein1", sequence)]
    _, _, batch_tokens = batch_converter(data)
    with torch.no_grad():
        results = model(batch_tokens.cuda(), repr_layers=[33])
    esm_time = time.time() - start
    print(f"ESMFold time: {esm_time:.2f} seconds")
    
    # Time Rosetta stability calculation
    print("\nTiming Rosetta stability calculation...")
    start = time.time()
    pose = pose_from_sequence(sequence)
    scorefxn = pyrosetta.get_fa_scorefxn()
    stability_score = scorefxn(pose)
    rosetta_time = time.time() - start
    print(f"Rosetta stability time: {rosetta_time:.2f} seconds")
    
    # Time DeepSeek inference
    print("\nTiming DeepSeek inference...")
    start = time.time()
    deepseek_model = DeepSeekModel()
    r1_output = deepseek_model.generate(sequence)
    deepseek_time = time.time() - start
    print(f"DeepSeek inference time: {deepseek_time:.2f} seconds")
    
    # Time GRPO training step
    print("\nTiming GRPO training step...")
    start = time.time()
    trainer = GRPOTrainer()
    loss = trainer.training_step(sequence)
    grpo_time = time.time() - start
    print(f"GRPO training step time: {grpo_time:.2f} seconds")
    
    # Print total time
    total_time = esm_time + rosetta_time + deepseek_time + grpo_time
    print(f"\nTotal pipeline time: {total_time:.2f} seconds")

if __name__ == "__main__":
    test_pipeline_timing()
