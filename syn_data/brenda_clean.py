import json
import requests
from typing import Dict, List, Optional
from tqdm import tqdm 
import os
def get_uniprot_sequence(accession: str, db: str = 'uniprot') -> Optional[str]:
    """Fetch protein sequence from UniProt or SwissProt API
    
    Args:
        accession: Protein accession ID
        db: Database to query - either 'uniprot' or 'swiss' (default: 'uniprot')
    """
    if db.lower() == 'swiss':
        url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta?format=fasta"
    else:
        url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
        
    response = requests.get(url)
    if response.status_code == 200:
        # Skip the header line and join sequence lines
        sequence = ''.join(response.text.split('\n')[1:])
        return sequence
    return None

def extract_reaction_components(reaction_list: List[Dict], protein_id: str) -> Dict[str, List[str]]:
    """Extract unique substrates and products from reaction list for specific protein"""
    reactions = []
    for reaction in reaction_list:
        # Only include reactions associated with this protein
        if protein_id in reaction.get('proteins', []):
            reactions.append({
                'substrates': reaction.get('educts', []),
                'products': reaction.get('products', [])
            })
    return reactions

def extract_kinetics(entry: Dict, protein_id: str) -> Dict:
    """Extract kinetics-related information for specific protein"""
    kinetics = {}
    
    if 'km_value' in entry:
        km_values = []
        for km in entry['km_value']:
            if protein_id in km.get('proteins', []):
                km_values.append({
                    'substrate': km['value'],
                    'value': km['num_value'],
                    'comment': km.get('comment', '')
                })
        if km_values:
            kinetics['km_values'] = km_values
    
    if 'specific_activity' in entry:
        specific_activities = []
        for sa in entry['specific_activity']:
            if protein_id in sa.get('proteins', []):
                specific_activities.append({
                    'value': sa['num_value'],
                })
        if specific_activities:
            kinetics['specific_activity'] = specific_activities
        
    return kinetics

def transform_brenda_data(data: Dict) -> Dict:
    """Transform BRENDA JSON data into new structure"""
    transformed_data = {}

    # Try to load brenda sequence data if available
    try:
        import pandas as pd
        if os.path.exists('syn_data/brenda_seq.xlsx'):
            brenda_seq_df = pd.read_excel('syn_data/brenda_seq.xlsx')
        print(brenda_seq_df.head())
    except Exception as e:
        brenda_seq_df = None
        print(e)
    
    for ec_number, entry in tqdm(data.items()):
        # Get protein accessions if available
        proteins = entry.get('proteins', {})
        if not proteins:  # Skip if no proteins
            continue
            
        for protein_id, protein_info in proteins.items():
            # Get UniProt/SwissProt accession from the protein info
            # Get first accession from list of accessions
            if "accessions" not in protein_info[0].keys():
                continue
            accession = (protein_info[0]['accessions'][0], protein_info[0]['source'])
                
            if brenda_seq_df is not None:
                sequence = brenda_seq_df.loc[brenda_seq_df['From'] == accession[0], 'Sequence'].iloc[0] if not brenda_seq_df.loc[brenda_seq_df['From'] == accession[0], 'Sequence'].empty else None
            else:
                sequence = get_uniprot_sequence(accession[0], accession[1])
            if not sequence:
                continue
                
            # Create new entry structure
            transformed_entry = {
                'ec_number': ec_number,
                'name': entry['name'],
                'sequence': sequence,
                'reaction': extract_reaction_components(entry.get('reaction', []), protein_id),
            }
            
            # Optional fields - only include if they exist and are associated with this protein
            try: 
                transformed_entry['kinetics'] = extract_kinetics(entry, protein_id)
            except Exception as e:
                pass
                
            # Add engineering data if available and associated with this protein
            if 'engineering' in entry:
                engineering_data = []
                for eng in entry['engineering']:
                    if protein_id in eng.get('proteins', []):
                        comment = eng['comment']
                        # Find the specific comment for this protein
                        start = comment.find(f"#{protein_id}#")
                        if start != -1:
                            # Find the next colon after the protein ID
                            colon_pos = comment.find('<', start)
                            if colon_pos == -1:  # If no < found, look for period
                                colon_pos = comment.find('.', start)
                            if colon_pos == -1:  # If still no delimiter, take whole comment
                                parsed_comment = comment[start + len(f"#{protein_id}#"):].strip()
                            else:
                                parsed_comment = comment[start + len(f"#{protein_id}#"):colon_pos].strip()
                            engineering_data.append({
                                'mutation': eng['value'],
                                'effect': parsed_comment
                            })
                if engineering_data:
                    transformed_entry['engineering'] = engineering_data
                    
            if 'general_information' in entry:
                general_info_data = []
                for info in entry['general_information']:
                    if protein_id in info.get('proteins', []):
                        comment = info['comment']
                        # Find the specific comment for this protein
                        start = comment.find(f"#{protein_id}#")
                        if start != -1:
                            # Find the next colon after the protein ID
                            colon_pos = comment.find(':', start)
                            if colon_pos != -1:
                                # Extract text between protein ID and colon
                                parsed_comment = comment[start + len(f"#{protein_id}#"):colon_pos].strip()
                                general_info_data.append(parsed_comment)
                            
                if general_info_data:
                    transformed_entry['general_information'] = general_info_data

            if 'metals_ions' in entry:
                metals_ions_data = [
                    metal['value'] for metal in entry['metals_ions']
                    if protein_id in metal.get('proteins', [])
                ]
                if metals_ions_data:
                    transformed_entry['metals_ions'] = metals_ions_data

            if 'cofactors' in entry:
                cofactors_data = [
                    cof['value'] for cof in entry['cofactors']
                    if protein_id in cof.get('proteins', [])
                ]
                if cofactors_data:
                    transformed_entry['metals_ions'] = metals_ions_data
                    
            transformed_data[accession[0]] = transformed_entry
            
    return transformed_data


input_file = 'data/brenda_2023_1.json'
# with open(input_file) as f:
#     data = json.load(f)['data']
#     # Get first 50 EC IDs
#     test_ec_ids = list(data.keys())[:50]
#     test_data = {ec_id: data[ec_id] for ec_id in test_ec_ids}

data = json.load(open(input_file))['data']

transformed = transform_brenda_data(data)

# Save test data to file
with open('test_transformed_brenda.json', 'w') as f:
    json.dump(transformed, f, indent=2)