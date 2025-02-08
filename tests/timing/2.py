import time
import torch
from typing import List
import pytest
from esm.esmfold.v1 import ESMFoldingModel

def get_sample_sequences(n: int = 4) -> List[str]:
    """Generate sample protein sequences for testing with varying lengths."""
    sequences = [
        "MLKNVHVLI",                      # 9 residues
        "AGTKKLVSDPQRSTUVWY",            # 18 residues
        "MFKVGDKTNPQRSTUVWYAGTKKLVSD",   # 27 residues
        "MLKNVHVLIAGTKKLVSDMFKVGDKTN",    # 27 residues
        "PQRSTUVWYAGTKKLVSDMLKNVHVLI"    # 27 residues
    ]
    return sequences[:n]

@pytest.mark.slow
def test_batch_vs_sequential_prediction():
    # Load model
    model = ESMFoldingModel.from_pretrained()
    model.eval()
    
    # Get test sequences
    num_sequences = 4
    sequences = get_sample_sequences(num_sequences)
    
    # Measure sequential prediction time
    sequential_start = time.time()
    sequential_results = []
    with torch.no_grad():
        for seq in sequences:
            output = model.infer(seq)
            sequential_results.append(output)
    sequential_time = time.time() - sequential_start
    
    # Measure batched prediction time
    batch_start = time.time()
    with torch.no_grad():
        batched_results = model.infer(sequences)
    batch_time = time.time() - batch_start
    
    # Calculate speedup
    speedup = sequential_time / batch_time
    
    print(f"\nSequential prediction time: {sequential_time:.2f}s")
    print(f"Batched prediction time: {batch_time:.2f}s")
    print(f"Speedup factor: {speedup:.2f}x")
    
    # Assert that batching provides some speedup (at least 1.2x faster)
    assert speedup > 1.2, f"Expected batching to be at least 1.2x faster, but got {speedup:.2f}x"
    
    # Verify that results are similar
    for seq_result, batch_result in zip(sequential_results, batched_results):
        assert torch.allclose(seq_result['positions'], batch_result['positions'], atol=1e-5), \
            "Sequential and batched predictions produced different results"
