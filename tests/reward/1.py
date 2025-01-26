import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from activity_reward import calculate_reward

def test_calculate_reward():
    # Test protein sequence from 1HEA
    sequence = "MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSRTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK"
    
    # Test molecules
    reagent = "O=C=O"  # CO2
    product = "OC(=O)[O-]"  # Bicarbonate
    
    # Calculate reward without transition state
    reward = calculate_reward(
        sequence=sequence,
        reagent=reagent, 
        product=product,
        ts=None
    )
    
    # Basic sanity checks
    assert isinstance(reward, float), "Reward should be a float"
    assert reward < 0, "Reward should be negative for this test case"
    
    print(f"Calculated reward: {reward}")

if __name__ == "__main__":
    test_calculate_reward()
