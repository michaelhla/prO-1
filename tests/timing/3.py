import time
import pytest
from stability_reward import StabilityRewardCalculator
from multiprocessing import Pool

def test_stability_calculation_speedup():
    """Test the speedup of parallel vs sequential stability calculations"""
    # Initialize calculator
    calculator = StabilityRewardCalculator()
    
    # Test sequences (using some simple example sequences)
    test_sequences = [
        "MKKLLVAVFVVLVLSTALPAQA",  # Signal peptide
        "MVKVGVNG",                 # Short enzyme fragment
        "MLSRAVCGTSRQLAPVLAYLG",   # Another signal peptide
        "MNIFEMLRIDEGLRLKIYK"      # Short protein sequence
    ]
    
    # First predict all structures to cache them
    print("\nPredicting structures first...")
    _ = calculator.predict_structure(test_sequences)
    
    # Test sequential processing
    print("\nTesting sequential processing...")
    start_time = time.time()
    sequential_scores = []
    for seq in test_sequences:
        score = calculator.calculate_stability(seq)
        sequential_scores.append(score)
    sequential_time = time.time() - start_time
    print(f"Sequential processing took {sequential_time:.2f} seconds")
    
    # Test parallel processing
    print("\nTesting parallel processing...")
    start_time = time.time()
    parallel_scores = calculator.calculate_stability(test_sequences)
    parallel_time = time.time() - start_time
    print(f"Parallel processing took {parallel_time:.2f} seconds")
    
    # Calculate speedup
    speedup = sequential_time / parallel_time
    print(f"\nSpeedup factor: {speedup:.2f}x")
    
    # Verify results are similar
    for seq_score, par_score in zip(sequential_scores, parallel_scores):
        assert abs(seq_score - par_score) < 1e-6, \
            "Sequential and parallel scores should be identical"
    
    # We expect some speedup with parallel processing
    assert speedup > 1.0, "Parallel processing should be faster than sequential"
    
    return {
        "sequential_time": sequential_time,
        "parallel_time": parallel_time,
        "speedup": speedup,
        "num_sequences": len(test_sequences)
    }

if __name__ == "__main__":
    # Run the test and print detailed results
    results = test_stability_calculation_speedup()
    print("\nDetailed Results:")
    print(f"Number of sequences processed: {results['num_sequences']}")
    print(f"Sequential processing time: {results['sequential_time']:.2f} seconds")
    print(f"Parallel processing time: {results['parallel_time']:.2f} seconds")
    print(f"Speedup factor: {results['speedup']:.2f}x")
