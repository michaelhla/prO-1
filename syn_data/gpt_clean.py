## clean the synthetically generated data into the format for openai finetuning api.
# we pass in reasoning tuples with mutation details, reasoning, and sequence
# we want the prompt (from our brenda json), and then the steps taken by the model on that mutation
# get the model to output "the next mutation" with reasoning
## randomly sample for the reasoning step to take
## allow for preference finetuning and SFT modes 

import openai
import json
import os

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def calculate_lm_reward(reasonings: list[str]) -> float:
    """Calculate reward for a list of reasonings using language model
    
    Args:
        reasonings: List of reasoning strings to evaluate
        
    Returns:
        
    """
    # Construct prompt for LM to evaluate reasonings
    prompt = "You are an expert enzyme engineer. Please evaluate these mutation reasonings based ONLY on their scientific accuracy and validity. Do not consider writing style or clarity. Rank these reasonings from most to least scientifically valid. DO NOT EXPLAIN YOUR RANKING. RESPOND ONLY IN THE FORMAT '1, 2, 3, 4' where Reasoning 1 is the most scientifically valid reasoning and Reasoning 4 is the least scientifically valid reasoning.\n\n"
    
    for i, reasoning in enumerate(reasonings, 1):
        prompt += f"Reasoning {i}:\n{reasoning}\n\n"
        
    # Get LM response
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert enzyme engineer evaluating mutation rationales."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    # Extract ranking from response
    ranking_text = response.choices[0].message.content
    
    # Parse ranking into ordered list of reasonings from best to worst
    try:
        # Extract just the numbers from ranking text (e.g. "1, 2, 3, 4")
        ranking_nums = [int(x.strip()) for x in ranking_text.split(',')]
        
        # Create list ordered from best to worst reasoning
        ordered_reasonings = []
        for rank in ranking_nums:
            ordered_reasonings.append(reasonings[rank-1])
            
        return ordered_reasonings
        
    except:
        return []  # Return empty list if parsing fails