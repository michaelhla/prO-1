import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# from reward import calculate_reward, BindingEnergyCalculator
import pandas as pd
from pathlib import Path

def process_mutation_files():
    # Base sequence from uniprot R4R1U5
    base_sequence = "MGYARRVMDGIGEVAVTGAGGSVTGARLRHQVRLLAHALTEAGIPPGRGVACLHANTWRAIALRLAVQAIGCHYVGLRPTAAVTEQARAIAAADSAALVFEPSVEARAADLLERVSVPVVLSLGPTSRGRDILAASVPEGTPLRYREHPEGIAVVAFTSGTTGTPKGVAHSSTAMSACVDAAVSMYGRGPWRFLIPIPLSDLGGELAQCTLATGGTVVLLEEFQPDAVLEAIERERATHVFLAPNWLYQLAEHPALPRSDLSSLRRVVYGGAPAVPSRVAAARERMGAVLMQNYGTQEAAFIAALTPDDHARRELLTAVGRPLPHVEVEIRDDSGGTLPRGAVGEVWVRSPMTMSGYWRDPERTAQVLSGGWLRTGDVGTFDEDGHLHLTDRLQDIIIVEAYNVYSRRVEHVLTEHPDVRAAAVVGVPDPDSGEAVCAAVVVADGADPDPEHLRALVRDHLGDLHVPRRVEFVRSIPVTPAGKPDKVKVRTWFTD"
    
    # Get mutation files from zero-shot predictions
    data_dir = Path("data/enzyme-activity")
    zero_shot_dir = data_dir / "zero-shot_predictions"
    mutation_files = list(zero_shot_dir.glob("*.csv"))
    
    # Get activity files from HSS directory
    hss_dir = data_dir / "HSS"
    
    # Dictionary to store drug-specific dataframes
    drug_data = {}
    
    # Process each drug's mutations and activities
    for mutation_file in mutation_files:
        drug_name = mutation_file.stem.replace('_zeroshot_EC', '')  # Get drug name without extension and suffix
        
        # Read mutations for this drug
        mutations_df = pd.read_csv(mutation_file)
        
        # Get activity file path
        activity_file = hss_dir / f"{drug_name}_train.csv"
        
        # Read activity data if file exists
        if activity_file.exists():
            activity_df = pd.read_csv(activity_file)
        else:
            print(f"No activity file found for {drug_name}")
            continue
        
        # Merge mutations with activities
        merged_df = pd.merge(
            mutations_df[['EVMutations', 'Mutations']],
            activity_df[['AminoAcid', 'Activity']],
            left_on='Mutations',
            right_on='AminoAcid',
            how='inner'
        )[['EVMutations', 'Activity']]
        
        # Store in dictionary
        drug_data[drug_name] = merged_df
        
        # Save to new CSV
        output_dir = data_dir / "processed_mutations"
        output_dir.mkdir(exist_ok=True)
        merged_df.to_csv(output_dir / f"{drug_name}_processed.csv", index=False)
    
    return base_sequence, data_dir

# def calculate_rewards(base_sequence, data_dir):
#     # Get processed mutation files
#     csv_files = list((data_dir / "processed_mutations").glob("*.csv"))
    
#     results = []
#     calculator = BindingEnergyCalculator()
    
#     for csv_file in csv_files:
#         print(f"Processing {csv_file}")
        
#         # Read mutations from CSV
#         df = pd.read_csv(csv_file)
        
#         # Process each mutation
#         for _, row in df.iterrows():
#             mutation_list = eval(row["EVMutations"])  # Convert string to list of tuples
#             activity = row["Activity"]
            
#             # Apply mutation to sequence
#             # Assuming mutation is always 4 letters where first is original, 
#             # Apply each mutation in sequence
#             mutated_sequence = base_sequence
#             for pos, orig, new in mutation_list:  # Changed mutations to mutation_list
#                 # Verify original amino acid matches
#                 if mutated_sequence[pos-1] != orig:
#                     print(f"Warning: Expected {orig} at position {pos} but found {mutated_sequence[pos-1]}")
#                     continue
#                 # Apply mutation
#                 mutated_sequence = mutated_sequence[:pos-1] + new + mutated_sequence[pos:]
            
#             # Calculate reward
#             reward = calculate_reward(
#                 sequence=mutated_sequence,
#                 reagent="",  # Left blank as per instructions
#                 product="",  # Left blank as per instructions
#                 ts=None,
#                 calculator=calculator
#             )
            
#             results.append({
#                 "file": csv_file.name,
#                 "mutation": mutation_list,
#                 "original_activity": activity,
#                 "calculated_reward": reward
#             })
    
#     # Convert results to DataFrame for analysis
#     results_df = pd.DataFrame(results)
#     print("\nResults summary:")
#     print(results_df.head())
    
#     return results_df

if __name__ == "__main__":
    base_sequence, data_dir = process_mutation_files()
    # results_df = calculate_rewards(base_sequence, data_dir)
