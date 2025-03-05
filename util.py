import json
def print_top_performers(top_performers_json):
    """Print details from top performers JSON file including sequences and stability scores"""
    try:
        print("\n=== TOP PERFORMING SEQUENCES ===")
        print("-" * 50)
        for i, result in enumerate(top_performers_json, 1):
            print(f"\nResult #{i}")
            print(f"Stability Score: {result['stability_score']:.2f}")
            print(f"Sequence ({len(result['sequence'])} aa):")
            
            # Print sequence in chunks of 60 characters for readability
            sequence = result['sequence']
            for j in range(0, len(sequence), 60):
                print(sequence[j:j+60])
            
            # Print model output/reasoning
            print("\nModel Output:")
            print(result.get('response', 'No model output available'))
                
            print("-" * 50)
            
    except Exception as e:
        print(f"Error printing top performers: {e}")

def clean_response_field(top_performers):
    """Remove the system prompt from response fields in top performers data"""
    try:
        for result in top_performers:
            if 'response' in result:
                # Find the start of actual response content after system prompt
                response = result['response']
                prompt_end = response.find('.assistant')
                if prompt_end != -1:
                    # Remove everything before the user prompt
                    result['response'] = response[prompt_end + 11:]
        return top_performers
    except Exception as e:
        print(f"Error cleaning response fields: {e}")
        return top_performers

def read_top_performers_json(json_file):
    """Read and parse top performers JSON file"""
    try:
        with open(json_file, 'r') as f:
            top_performers = json.load(f)
        # Clean the response fields before returning
        return clean_response_field(top_performers)
    except Exception as e:
        print(f"Error reading top performers JSON: {e}")
        return None


from stability_reward import StabilityRewardCalculator

def calculate_stability_for_pdb():
    """Calculate stability score for 1hea.pdb structure"""
    try:
        calculator = StabilityRewardCalculator()
        stability_score = calculator.calculate_stability(sequence=None, pdb_file_path="1hea.pdb")
        print(f"\nStability score for 1hea.pdb: {stability_score:.2f}")
        return stability_score
    except Exception as e:
        print(f"Error calculating stability score: {e}")
        return None

calculate_stability_for_pdb()

# Stability score for 1hea.pdb: 194.98 (Is this right?)


# print_top_performers(read_top_performers_json('top_performers.json'))