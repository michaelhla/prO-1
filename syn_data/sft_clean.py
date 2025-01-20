## clean the synthetically generated data into the format for openai finetuning api.
# we pass in reasoning tuples with mutation details, reasoning, and sequence
# we want the prompt (from our brenda json), and then the steps taken by the model on that mutation
# get the model to output "the next mutation" with reasoning
## randomly sample for the reasoning step to take
## allow for preference finetuning and SFT modes 

import openai
import json
import os
import random
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def format_sft_data(json_file: str) -> list:
    """Format mutation traces into OpenAI's fine-tuning conversation format."""
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    formatted_data = []
    k = 1
    for trace in data["traces"]:
        enzyme_prompt = trace["prompt"]
        messages = [
            {
                "role": "system",
                "content": f"""You are an expert protein engineer with years of experience optimizing protein activity with rational design. 
Whenever the user inputs "<CONTINUE>", SELECT the next mutations to make and REASON in the FORMAT: 

%%MUTATION_i%%: [Original AA][Position][New AA]
%%REASONING_i%%: Reasoning

Keep your response to under 100 words. select at most {k} mutation(s). If there are no further optimizations to make, return <DONE> with no explanation."""
            },
            {
                "role": "user",
                "content": enzyme_prompt
            }
        ]
        
        # Add each step as part of the conversation
        for i, step in enumerate(trace["steps"]):
            curr_mut = step["correct_mutation"]
            mutation_response = f"%%MUTATION_{i}%%: {curr_mut['from_aa']}{curr_mut['position']}{curr_mut['to_aa']}\n%%REASONING_{i}%%   : {curr_mut['reasoning']}"
            
            # Add the mutation response
            messages.append({
                "role": "assistant",
                "content": mutation_response
            })
            
            # Add the continue prompt for next step
            messages.append({
                "role": "user",
                "content": "<CONTINUE>"
            })
        
        # Add final <DONE> response
        messages.append({
            "role": "assistant",
            "content": "<DONE>"
        })
        
        formatted_data.append({"messages": messages})
    
    return formatted_data

def sft(json_file: str, output_file: str):
    """Create and validate supervised fine-tuning dataset from mutation traces."""
    # Format the data
    formatted_data = format_sft_data(json_file)
    
    # Write JSONL file
    with open(output_file, 'w') as file:
        for item in formatted_data:
            file.write(json.dumps(item) + '\n')
            
    response = client.files.create(
        file=open(output_file, "rb"),
        purpose="fine-tune"
    )
    
    # Create fine-tuning job
    job = client.fine_tuning.jobs.create(
        training_file=response.id,
        model="gpt-4o-mini-2024-07-18",
        method={
            "type": "supervised",
        }
    )
    
    print(f"Created fine-tuning job: {job.id}")
    return job.id

if __name__ == "__main__":
    job_id = sft("data/mutation_traces.json", "data/mutation_traces.jsonl")
    print(f"Fine-tuning job created: {job_id}")




