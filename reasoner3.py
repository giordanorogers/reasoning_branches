"""
A script to use nnsight to call the R1 model to answer prompts.
"""

import os
# Set tokenizers parallelism to false to avoid tokenizer deadlock
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import yaml
import pandas as pd
from dotenv import load_dotenv
import torch
import nnsight
from nnsight import LanguageModel, CONFIG
import time

# Load environment variables from .env file
load_dotenv()

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set up model configuration
CONFIG.APP.REMOTE_LOGGING = config['model']['remote_logging']
CONFIG.API.APIKEY = os.getenv("NDIF_API_KEY")

# Load special tokens from config
BOS = config['tokens']['bos']
USER = config['tokens']['user']
ASSISTANT = config['tokens']['assistant']
NEWLINE = config['tokens']['newline']
THINK_START = config['tokens']['think_start']
THINK_END = config['tokens']['think_end']
EOS = config['tokens']['eos']

model = LanguageModel(config['model']['name'])
tokenizer = model.tokenizer

def custom_encoding(user_message: str, thinking_message: str = ""):
    """ Encode the user message and thinking message. """
    user_tokens = tokenizer.encode(user_message, add_special_tokens=False)
    thinking_tokens = tokenizer.encode(thinking_message, add_special_tokens=False)
    return [[BOS] + user_tokens + [NEWLINE] + [THINK_START] + thinking_tokens]

def generate_text(input_ids, model, tokenizer):
    """ Generate text. """
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.ones_like(input_ids)
    
    gen_config = config['generation']
    with model.generate(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        max_new_tokens=gen_config['max_length'],
        do_sample=gen_config['do_sample'],
        temperature=gen_config['temperature'],
        top_p=gen_config['top_p'],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        remote=True
    ):
        outputs = nnsight.list().save()

        with model.lm_head.all():
            outputs.append(model.lm_head.output[0][-1].argmax(dim=-1))

    # Decode and return the generated text
    input_text = tokenizer.decode(input_ids[0])
    # Check if outputs has a 'value' attribute or is a list directly
    if hasattr(outputs, 'value'):
        generated_tokens = outputs.value
    else:
        generated_tokens = outputs
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    return input_text + "\n" + generated_text

def generate_olymMATH_responses(problem, n=None):
    n = n or config['experiment']['num_responses']
    responses = []
    for i in range(n):
        problem_enc = custom_encoding(problem)
        r1_answer = generate_text(problem_enc, model, tokenizer)
        if r1_answer is None:
            print(f"Failed to generate response {i+1}")
            continue
        responses.append(r1_answer)
        # No need for additional wait since generation takes ~30 seconds
        print(f"Response {i+1} complete")
    return responses

if __name__ == "__main__":
    # Load the JSON file containing the GSM8K problems
    df = pd.read_json(config['experiment']['input_file'], orient="records")
    
    if config['experiment']['num_problems']:
        df = df[:config['experiment']['num_problems']]

    # Initialize or load existing responses
    output_file = config['experiment']['output_file']
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_json(output_file, orient="records")
            # Merge existing responses with current dataframe
            df = df.merge(existing_df[['problem', 'responses']], on='problem', how='left')
            print(f"Loaded {len(existing_df)} existing responses")
        except Exception as e:
            print(f"Error loading existing responses: {str(e)}")
            df['responses'] = None

    # Generate responses for each question with progress saving
    for idx, row in df.iterrows():
        problem = row['problem']
        current_responses = row.get('responses', []) if pd.notna(row.get('responses')) else []
        
        # Skip if we already have all responses for this problem
        if len(current_responses) >= config['experiment']['num_responses']:
            print(f"Skipping problem {idx + 1}/{len(df)} - already complete")
            continue
            
        print(f"Processing problem {idx + 1}/{len(df)}")
        try:
            # Generate only the missing responses
            num_needed = config['experiment']['num_responses'] - len(current_responses)
            new_responses = generate_olymMATH_responses(problem, n=num_needed)
            
            # Combine existing and new responses
            df.at[idx, 'responses'] = current_responses + new_responses
            
            # Save progress after each problem
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                # Save with error handling
                df.to_json(output_file, orient="records", indent=2)
                print(f"Successfully saved progress after problem {idx + 1} to {output_file}")
                print(f"Current file size: {os.path.getsize(output_file)} bytes")
            except Exception as save_error:
                print(f"Error saving progress: {str(save_error)}")
                print("Attempting to save to backup file...")
                backup_file = f"{output_file}.backup"
                df.to_json(backup_file, orient="records", indent=2)
                print(f"Saved backup to {backup_file}")
            
            # Wait 10 seconds between problems
            print("Waiting 10 seconds before next problem...")
            time.sleep(10)
            
        except Exception as e:
            print(f"Error processing problem {idx + 1}: {str(e)}")
            # Save progress even if there was an error
            try:
                df.to_json(output_file, orient="records", indent=2)
                print(f"Saved progress after error on problem {idx + 1}")
            except Exception as save_error:
                print(f"Error saving progress after error: {str(save_error)}")
            # Still wait between problems even if there was an error
            print("Waiting 10 seconds before next problem...")
            time.sleep(10)

    print("Completed all problems!")