import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from stability_reward import StabilityRewardCalculator
import matplotlib.pyplot as plt

# Initialize calculator
calculator = StabilityRewardCalculator(device="cuda")

# Test sequence (a short peptide sequence)
test_sequence = "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNGGHFLRILPDGTVDGTRDRSDQHIQLQLSAESVGEVYIKSTETGQYLAMDTDGLLYGSQTPNEECLFLERLEENHYNTYISKKHAEKNWFVGLKKNGSCKRGPRTHYGQKAILFLPLPV"

# Calculate stability reward
pdb_file = calculator.predict_structure(test_sequence)
print(f"Structure prediction completed. PDB file saved at: {pdb_file}")

# Calculate stability score
stability_score = calculator.calculate_stability(test_sequence)
print(f"Stability score: {stability_score}")

# Plot the stability score
plt.figure(figsize=(10, 6))
plt.bar(['Stability Score'], [stability_score])
plt.title('Protein Stability Score')
plt.ylabel('Normalized Stability (0-1)')
plt.ylim(0, 1)
plt.savefig('stability_score.png')
plt.close()

print("Test completed and plot saved to stability_score.png")
