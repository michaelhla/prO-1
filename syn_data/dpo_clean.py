import openai
import json
import random
import os
import dotenv

dotenv.load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def format_dpo_data(json_file: str) -> list:
    """Format mutation traces into OpenAI's DPO fine-tuning format."""
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    formatted_data = []
    
    for trace in data["traces"]:
        enzyme_prompt = trace["prompt"]
        previous_steps = []
        
        for step in trace["steps"]:
            # Build conversation history
            messages = [
                {
                    "role": "system", 
                    "content": f"""You are an expert protein engineer with years of experience optimizing protein activity with rational design. 
Whenever the user inputs "<CONTINUE>", SELECT the next mutations to make and REASON in the FORMAT: 

%%MUTATION_i%%: [Original AA][Position][New AA]
%%REASONING_i%%: Reasoning

Keep your response to under 100 words. select at most 1 mutation(s). If there are no further optimizations to make, return <DONE> with no explanation."""
                },
                {
                    "role": "user",
                    "content": enzyme_prompt
                }
            ]
            
            # Add previous steps to conversation history
            for i, prev_step in enumerate(previous_steps):
                prev_mut = prev_step["correct_mutation"]
                messages.append({
                    "role": "assistant",
                    "content": f"%%MUTATION_{i}%%: {prev_mut['from_aa']}{prev_mut['position']}{prev_mut['to_aa']}\n%%REASONING_{i}%%   : {prev_mut['reasoning']}"
                })
                messages.append({
                    "role": "user",
                    "content": "<CONTINUE>"
                })
            
            # Format current correct mutation
            curr_mut = step["correct_mutation"]
            i = len(previous_steps)
            preferred_output = [{
                "role": "assistant",
                "content": f"%%MUTATION_{i}%%: {curr_mut['from_aa']}{curr_mut['position']}{curr_mut['to_aa']}\n%%REASONING_{i}%%   : {curr_mut['reasoning']}"
            }]
            
            # Randomly select one incorrect mutation
            incorrect_mut = random.choice(step["incorrect_mutations"])
            example = {
                "input": {
                    "messages": messages + [{"role": "user", "content": "<CONTINUE>"}],
                    "tools": [],
                    "parallel_tool_calls": True
                },
                "preferred_output": preferred_output,
                "non_preferred_output": [{
                    "role": "assistant",
                    "content": f"%%MUTATION_{i}%%: {incorrect_mut['from_aa']}{incorrect_mut['position']}{incorrect_mut['to_aa']}\n%%REASONING_{i}%%   : {incorrect_mut['reasoning']}"
                }]
            }
            formatted_data.append(example)
            
            previous_steps.append(step)
    
    return formatted_data

def dpo(json_file: str, output_file: str):
    """Create and validate DPO fine-tuning dataset from mutation traces."""
    # Format the data
    formatted_data = format_dpo_data(json_file)
    
    # Write JSONL file
    with open(output_file, 'w') as file:
        for item in formatted_data:
            file.write(json.dumps(item) + '\n')
    
    # Upload file to OpenAI
    client = openai.OpenAI()
    with open(output_file, 'rb') as file:
        response = client.files.create(
            file=file,
            purpose='fine-tune'
        )
    
    # Create DPO fine-tuning job
    job = client.fine_tuning.jobs.create(
        training_file=response.id,
        model="gpt-4o-mini-2024-07-18",
        method={
            "type": "dpo",
            "dpo": {
                "hyperparameters": {"beta": 0.1}
            }
        }
    )
    
    print(f"Created DPO fine-tuning job: {job.id}")
    return job.id
