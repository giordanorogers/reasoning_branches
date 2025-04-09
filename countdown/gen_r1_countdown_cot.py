import os
# Set tokenizers parallelism to false to avoid tokenizer deadlock
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import yaml
from pathlib import Path
from dotenv import load_dotenv
import torch
import nnsight
from nnsight import LanguageModel, CONFIG
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Load config
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    logging.error("config.yaml not found. Please create it.")
    exit(1)
except yaml.YAMLError as e:
    logging.error(f"Error parsing config.yaml: {e}")
    exit(1)

# Set up model configuration
CONFIG.APP.REMOTE_LOGGING = config['model'].get('remote_logging', False)
api_key = os.getenv("NDIF_API_KEY")
if not api_key:
    logging.warning("NDIF_API_KEY environment variable not set. Remote execution might fail.")
else:
    CONFIG.API.APIKEY = api_key
    logging.info("API KEY GOOD")

# Load special tokens from config
try:
    tokens_config = config['tokens']
    BOS = tokens_config['bos']
    USER = tokens_config['user']
    ASSISTANT = tokens_config['assistant']
    NEWLINE = tokens_config['newline']
    THINK_START = tokens_config['think_start']
    THINK_END = tokens_config['think_end']
    EOS = tokens_config['eos']
except KeyError as e:
    logging.error(f"Missing token configuration in config.yaml: {e}")
    exit(1)

# Load model
model = None
tokenizer = None
model_name = None # Initialize model_name
try:
    model_name = config['model']['name']
    logging.info(f"Loading model: {model_name}")
    # Using device_map='auto' for better resource management if multiple GPUs are available
    model = LanguageModel(model_name)
    tokenizer = model.tokenizer
except Exception as e:
    logging.error(f"Error loading model '{model_name}': {e}")
    exit(1) # Exit if model loading fails

def format_prompt(numbers: list, target: int, tokenizer):
    """Formats the prompt for the Countdown problem and encodes it."""
    numbers_str = ", ".join(map(str, numbers))
    # Construct the prompt text ending exactly where the model should start generating
    prefix_text = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Using the numbers {numbers_str}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
Assistant: Let me solve this step by step.
<think>""" # Include the think start tag

    # Encode this prefix text. add_special_tokens=True generally adds BOS if configured in tokenizer.
    input_ids = tokenizer.encode(prefix_text, add_special_tokens=True)

    # Manually check and add BOS based on config if tokenizer didn't add it or added a different one.
    if not input_ids or input_ids[0] != BOS:
        # Check if tokenizer *has* a BOS token defined and if it matches config
        if tokenizer.bos_token_id == BOS:
             if not input_ids or input_ids[0] != tokenizer.bos_token_id:
                 input_ids.insert(0, BOS) # Add configured BOS if tokenizer didn't
        elif tokenizer.bos_token_id is not None:
             logging.warning(f"Tokenizer BOS ({tokenizer.bos_token_id}) doesn't match config BOS ({BOS}). Using config BOS.")
             input_ids.insert(0, BOS) # Prioritize config BOS
        else:
             # Tokenizer has no BOS, add the one from config
             input_ids.insert(0, BOS)

    return [input_ids] # Return as list of lists for nnsight compatibility

def generate_text(input_ids, model, tokenizer):
    """ Generate cot text. """
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.ones_like(input_ids)

    gen_config = config['generation']
    token_logits = []

    with model.generate(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        max_new_tokens=gen_config['max_length'],
        do_sample=gen_config['do_sample'],
        temperature=gen_config['temperature'],
        top_p=gen_config['top_p'],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        remote=True
    ):
        generated_tokens = nnsight.list().save()

        if get_logits:
            # Capture logits for each token generation step
            with model.lm_head.all():
                # Save the logits from each generation step
                step_logits = model.lm_head.output[0][-1].save()
                token_logits.append(step_logits)

        with model.lm_head.all():
            next_token = model.lm_head.output[0][-1].argmax(dim=-1)
            generated_tokens.append(next_token)

    # Decode the generated text
    input_text = tokenizer.decode(input_ids[0])

    # Extract the generated tokens
    if hasattr(generated_tokens, 'value'):
        gen_tokens = generated_tokens.value
    else:
        gen_tokens = generated_tokens

    generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=False)

    # Process the logits if requested
    processed_logits = None
    if get_logits and token_logits:
        # Convert the tensor list to a Python list structure
        processed_logits = [logit.value.tolist() if hasattr(logit, 'value') else logit.tolist() for logit in token_logits]

    return {
        "text": input_text + "\n" + generated_text,
        "tokens": gen_tokens,
        "logits": processed_logits
    }

def generate_cot(input_ids, model, tokenizer):
    """Generates the Chain-of-Thought text using nnsight, stopping at </think>."""
    input_ids_tensor = torch.tensor(input_ids)
    attention_mask = torch.ones_like(input_ids_tensor)

    gen_config = config['generation']

    generated_text = ""

    try:
        with model.generate(
            {"input_ids": input_ids_tensor, "attention_mask": attention_mask},
            max_new_tokens=gen_config.get('cot_max_length', 4000),
            do_sample=gen_config.get('do_sample', True),
            temperature=gen_config.get('temperature', 0.6),
            top_p=gen_config.get('top_p', None),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=False,
            remote=True
        ):
            generated_tokens = nnsight.list().save()

            with model.lm_head.all():
                next_token = model.lm_head.output[0][-1].argmax(dim=-1)
                generated_tokens.append(next_token)

            # Extract the generated tokens
        if hasattr(generated_tokens, 'value'):
            gen_tokens = generated_tokens.value
        else:
            gen_tokens = generated_tokens

        generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=False)

        return generated_text
    except Exception as e:
        logging.exception(f"Error during nnsight generation: {e}") # Log full traceback
        return "" # Return empty string on error

if __name__ == "__main__":
    if model is None or tokenizer is None:
        logging.error("Exiting because model or tokenizer failed to load.")
        exit(1)

    input_dir = Path("countdown_data")
    output_dir = Path("countdown_data_with_cot")
    output_dir.mkdir(exist_ok=True)

    files_to_process = ["train.jsonl", "validation.jsonl", "test_ood_1e3.jsonl"]

    for filename in files_to_process:
        input_path = input_dir / filename
        output_path = output_dir / filename
        logging.info(f"Processing {input_path} -> {output_path}")

        if not input_path.exists():
            logging.warning(f"Input file not found: {input_path}. Skipping.")
            continue

        processed_count = 0
        error_count = 0
        start_time = time.time()

        start_line = 0
        if output_path.exists():
             logging.warning(f"Output file {output_path} already exists.")
             # Simple resume: count lines and skip already processed ones.
             try:
                 with open(output_path, 'r') as f:
                     start_line = sum(1 for _ in f)
                 logging.info(f"Resuming from line {start_line + 1}...")
                 # Open in append mode
                 outfile_mode = 'a'
             except Exception as e:
                 logging.error(f"Could not read existing output file {output_path} to resume: {e}. Starting from scratch (overwrite).")
                 outfile_mode = 'w' # Overwrite if resume fails
        else:
             outfile_mode = 'w' # Create new file

        try:
            with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
                for i, line in enumerate(infile):
                    line = line.strip()

                    try:
                        data = json.loads(line)
                        numbers = data.get('numbers')
                        target = data.get('target')

                        if numbers is None or target is None:
                            logging.warning(f"Skipping line {i+1} in {filename}: Missing 'numbers' or 'target'. Data: {line[:100]}...")
                            error_count += 1
                            continue

                        # Format the prompt
                        input_ids = format_prompt(numbers, target, tokenizer)

                        # Generate CoT
                        cot_start_time = time.time()
                        generated_cot = generate_cot(input_ids, model, tokenizer)
                        cot_end_time = time.time()

                        print(generated_cot)
                        time.sleep(10)
                    except Exception as e:
                        print(f"Uh oh: {e}")
        except Exception as e:
            print(f"big uh oh...: {e}")
