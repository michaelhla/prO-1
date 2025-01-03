import torch
from transformers import GPT2LMHeadModel, AutoTokenizer
import os
from tqdm import tqdm
import math

def remove_characters(sequence, char_list):
    "This function removes special tokens used during training."
    columns = sequence.split('<sep>')
    seq = columns[1]
    for char in char_list:
        seq = seq.replace(char, '')
    return seq

def calculatePerplexity(input_ids,model,tokenizer):
    "This function computes perplexities for the generated sequences"
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return math.exp(loss)
        
def generate(label, model, special_tokens, device, tokenizer, num_sequences=20, max_length=1024):
    """
    Generate and score sequences for a given label.
    Returns a list of tuples (sequence, perplexity) sorted by perplexity.
    """
    # Generating sequences
    input_ids = tokenizer.encode(label, return_tensors='pt').to(device)
    outputs = model.generate(
        input_ids,
        top_k=9,
        repetition_penalty=1.2,
        max_length=max_length,
        eos_token_id=1,
        pad_token_id=0,
        do_sample=True,
        num_return_sequences=num_sequences
    )
    
    # Filter out truncated sequences
    valid_outputs = [output for output in outputs if output[-1] == 0]
    if not valid_outputs:
        return []  # Return empty list if no valid sequences

    # Calculate perplexities and clean sequences
    sequences_with_scores = [
        (remove_characters(tokenizer.decode(output), special_tokens),
         calculatePerplexity(output, model, tokenizer))
        for output in valid_outputs
    ]
    
    # Sort by perplexity (lower is better)
    sequences_with_scores.sort(key=lambda x: x[1])
    
    return sequences_with_scores

if __name__=='__main__':
    device = torch.device("cuda") # Replace with 'cpu' if you don't have a GPU - but it will be slow
    print('Reading pretrained model and tokenizer')
    tokenizer = AutoTokenizer.from_pretrained('/path/to/zymCTRL/') # change to ZymCTRL location
    model = GPT2LMHeadModel.from_pretrained('/path/to/zymCTRL').to(device) # change to ZymCTRL location
    special_tokens = ['<start>', '<end>', '<|endoftext|>','<pad>',' ', '<sep>']

    # change to the appropriate EC classes
    labels=['3.5.5.1'] # nitrilases. You can put as many labels as you want.

    for label in tqdm(labels):
        # We'll run 100 batches per label. 20 sequences will be generated per batch.
        for i in range(0,100): 
            sequences = generate(label, model, special_tokens, device, tokenizer)
            for index, (sequence, perplexity) in enumerate(sequences):
                # Sequences will be saved with the name of the label followed by the batch index,
                # and the order of the sequence in that batch.           
                fn = open(f"/path/to/folder/{label}_{i}_{index}.fasta", "w")
                fn.write(f'>{label}_{i}_{index}\t{perplexity}\n{sequence}')
                fn.close()
