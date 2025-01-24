import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from reward import calculate_reward, BindingEnergyCalculator
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def process_mutation_files(base_sequence):
    # Base sequence from uniprot R4R1U5
    
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

def calculate_rewards(base_sequence, data_dir):
    common_reagents = {'ATP': 'C1=NC(=C2C(=N1)N(C=N2)[C@H]3[C@@H]([C@@H]([C@H](O3)COP(=O)(O)OP(=O)(O)OP(=O)(O)O)O)O)N', 'PPi': '[O-]P(=O)([O-])[O-]', 'H2O': 'O', 'ADP': 'C1=NC(=C2C(=N1)N(C=N2)[C@H]3[C@@H]([C@@H]([C@H](O3)COP(=O)(O)OP(=O)(O)O)O)O)N' }
    reaction_dict = {'moclobemide': {'reagents': ['C1=CC(=CC=C1C(=O)O)Cl', 'C1COCCN1CCN'], 'products': ['C1COCCN1CCNC(=O)C2=CC=C(C=C2)Cl', '[O-]P(=O)([O-])[O-]']}}

    # Get processed mutation files
    csv_files = list((data_dir / "processed_mutations").glob("*.csv"))
    
    results = []
    calculator = BindingEnergyCalculator()
    
    for csv_file in csv_files:
        drug_name = csv_file.stem.replace('_processed', '')
        if drug_name not in reaction_dict:
            continue
        print(f"Processing {csv_file}")
        
        # Read mutations from CSV
        df = pd.read_csv(csv_file)
        
        # Process each mutation
        for _, row in df.iterrows():
            mutation_list = eval(row["EVMutations"])  # Convert string to list of tuples
            activity = row["Activity"]
            id_active_site = [(pos, new) for pos, _, new in mutation_list]

        
            # Apply mutation to sequence
            # Assuming mutation is always 4 letters where first is original, 
            # Apply each mutation in sequence
            mutated_sequence = base_sequence
            for pos, orig, new in mutation_list:
                # Verify original amino acid matches
                if mutated_sequence[pos-1] != orig:
                    print(f"Warning: Expected {orig} at position {pos} but found {mutated_sequence[pos-1]}")
                    continue
                # Apply mutation
                mutated_sequence = mutated_sequence[:pos-1] + new + mutated_sequence[pos:]
            
            # Calculate reward
            reward = calculate_reward(
                sequence=mutated_sequence,
                reagents=reaction_dict[drug_name]['reagents'],  # Get first reagent for drug
                products=reaction_dict[drug_name]['products'],  # Get first product for drug
                ts=None,
                calculator=calculator, 
                id_active_site=id_active_site,
            )
            
            results.append({
                "file": csv_file.name,
                "mutation": mutation_list,
                "original_activity": activity,
                "calculated_reward": reward
            })

            print(f'MUTATION {mutation_list} - REWARD {reward} - ACTIVITY {activity}')
    
    # Group results by file and save each to separate CSV
    results_df = pd.DataFrame(results)
    for file_name, group_df in results_df.groupby('file'):
        output_path = data_dir / 'results' / f'{file_name}_results.csv'
        output_path.parent.mkdir(exist_ok=True)
        group_df.to_csv(output_path, index=False)
        print(f"\nSaved results for {file_name} to {output_path}")
        
        # Create scatter plot of activity vs reward
        plt.figure(figsize=(10,6))
        plt.scatter(group_df['original_activity'], group_df['calculated_reward'])
        plt.xlabel('Experimental Activity')
        plt.ylabel('Predicted Reward')
        plt.title(f'Activity vs Predicted Reward for {file_name}')
        
        # Save plot
        plot_path = data_dir / 'results' / f'{file_name}_correlation.png'
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved correlation plot to {plot_path}")
    
    print("\nResults summary:")
    print(results_df.head())
    
    return results_df

if __name__ == "__main__":
    base_sequence = 'MGYARRVMDGIGEVAVTGAGGSVTGARLRHQVRLLAHALTEAGIPPGRGVACLHANTWRAIALRLAVQAIGCHYVGLRPTAAVTEQARAIAAADSAALVFEPSVEARAADLLERVSVPVVLSLGPTSRGRDILAASVPEGTPLRYREHPEGIAVVAFTSGTTGTPKGVAHSSTAMSACVDAAVSMYGRGPWRFLIPIPLSDLGGELAQCTLATGGTVVLLEEFQPDAVLEAIERERATHVFLAPNWLYQLAEHPALPRSDLSSLRRVVYGGAPAVPSRVAAARERMGAVLMQNYGTQEAAFIAALTPDDHARRELLTAVGRPLPHVEVEIRDDSGGTLPRGAVGEVWVRSPMTMSGYWRDPERTAQVLSGGWLRTGDVGTFDEDGHLHLTDRLQDIIIVEAYNVYSRRVEHVLTEHPDVRAAAVVGVPDPDSGEAVCAAVVVADGADPDPEHLRALVRDHLGDLHVPRRVEFVRSIPVTPAGKPDKVKVRTWFTD'
    data_dir = Path("data/enzyme-activity")
    # _, data_dir = process_mutation_files(base_sequence)
    results_df = calculate_rewards(base_sequence, data_dir)
