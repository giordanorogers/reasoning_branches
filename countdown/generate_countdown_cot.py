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
CONFIG.API.APIKEY = api_key

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
try:
    model_name = config['model']['name']
    logging.info(f"Loading model: {model_name}")
    # Using device_map='auto' for better resource management if multiple GPUs are available
    model = LanguageModel(model_name, device_map='auto')
    tokenizer = model.tokenizer
    # Ensure pad token is set if not present, default to EOS
    if tokenizer.pad_token_id is None:
        logging.info(f"Tokenizer pad_token_id not set. Setting to eos_token_id ({tokenizer.eos_token_id})")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # Also update the model's config object for consistency during generation
        model.config.pad_token_id = tokenizer.eos_token_id
    logging.info("Model and tokenizer loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model '{model_name}': {e}")
    exit(1) # Exit if model loading fails

def format_prompt(numbers: list, target: int, tokenizer) -> list[list[int]]:
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

def generate_cot(input_ids: list[list[int]], model: LanguageModel, tokenizer) -> str:
    """Generates the Chain-of-Thought text using nnsight, stopping at </think>."""
    if not model or not tokenizer:
        logging.error("Model or tokenizer not loaded properly.")
        return ""

    input_ids_tensor = torch.tensor(input_ids).to(model.device)
    attention_mask = torch.ones_like(input_ids_tensor)

    gen_config = config['generation']
    # Allow overriding cot_max_length via config, otherwise default
    cot_max_length = gen_config.get('cot_max_length', 256)
    temperature = gen_config.get('temperature', 0.6)
    do_sample = gen_config.get('do_sample', True)
    top_p = gen_config.get('top_p', None) # Pass None if not specified

    # Define EOS tokens: stop at </think> or the model's natural EOS
    eos_tokens = [THINK_END] # Primary stop token is </think>
    if EOS not in eos_tokens: # Add model's standard EOS from config
        eos_tokens.append(EOS)
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id not in eos_tokens: # Add tokenizer's specific EOS
        eos_tokens.append(tokenizer.eos_token_id)

    generated_text = ""
    outputs_proxy = None # Define proxy variable outside try block

    try:
        # Use the nnsight generate context manager for remote execution
        with model.generate(
            {"input_ids": input_ids_tensor, "attention_mask": attention_mask}, # Pass as dictionary
            max_new_tokens=cot_max_length,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_tokens, # List of tokens to stop generation
            return_dict_in_generate=False, # Set to False if we just need the sequence
            remote=True
        ) as generator:
            # Create a proxy to save the generated token sequence.
            # This relies on nnsight internals associating the proxy with the generation output.
            outputs_proxy = nnsight.list.tracking().save()
            # No explicit invocation or loop needed here if using the dictionary input format.
            # The generate context should handle the process.

        # --- Post-generation processing (after context manager exits) ---
        if outputs_proxy is not None and hasattr(outputs_proxy, 'value'):
            gen_tokens_list = outputs_proxy.value

            # Process the collected tokens (might be list of tensors, tensor, etc.)
            gen_tokens = []
            if isinstance(gen_tokens_list, list):
                 if all(isinstance(t, torch.Tensor) for t in gen_tokens_list):
                     gen_tokens = [t.item() for t in gen_tokens_list] # Convert tensors to ints
                 elif all(isinstance(t, int) for t in gen_tokens_list):
                     gen_tokens = gen_tokens_list # Already list of ints
                 else:
                      logging.warning(f"Generated token list contains unexpected types: {[type(t) for t in gen_tokens_list]}. Attempting conversion.")
                      try:
                          gen_tokens = [int(t) for t in gen_tokens_list]
                      except (ValueError, TypeError) as conv_err:
                          logging.error(f"Could not convert token list to ints: {conv_err}")
            elif isinstance(gen_tokens_list, torch.Tensor):
                 gen_tokens = gen_tokens_list.tolist() # Convert tensor to list
            else:
                 logging.warning(f"Unexpected type for generated tokens proxy value: {type(gen_tokens_list)}. Decoding might fail.")

            if gen_tokens:
                # Decode the sequence, skipping special tokens like EOS, THINK_END
                generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                generated_text = generated_text.strip() # Clean whitespace
            else:
                logging.warning("Failed to extract valid tokens from generation proxy.")

        else:
            logging.warning("Failed to retrieve generated tokens from proxy. Generation might have failed or proxy not populated.")

    except Exception as e:
        logging.exception(f"Error during nnsight generation: {e}") # Log full traceback
        return "" # Return empty string on error

    return generated_text

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

        # Check if output file exists and ask user if they want to overwrite or resume
        # Basic resume logic: count lines in output and skip that many in input.
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
            with open(input_path, 'r') as infile, open(output_path, outfile_mode) as outfile:
                for i, line in enumerate(infile):
                    if i < start_line: # Skip lines already processed if resuming
                        continue

                    line = line.strip()
                    if not line:
                        continue

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

                        if not generated_cot:
                             logging.warning(f"Failed to generate CoT for line {i+1} in {filename}. Input: numbers={numbers}, target={target}. Skipping output.")
                             error_count += 1
                             # Optional: Write the original data without CoT? For now, skip.
                             continue

                        # Add generated CoT to the data
                        data['generated_cot'] = generated_cot

                        # Write updated data to output file
                        outfile.write(json.dumps(data) + '\n')
                        processed_count += 1

                        if processed_count % 50 == 0: # Log progress
                            current_time = time.time()
                            elapsed_total = current_time - start_time
                            # Calculate rate based only on newly processed items if resuming
                            items_in_this_run = i + 1 - start_line
                            avg_time_per_item = elapsed_total / items_in_this_run if items_in_this_run > 0 else 0
                            logging.info(f"Processed line {i+1} ({processed_count} in this run) from {filename}. Avg time/item (this run): {avg_time_per_item:.2f}s. Last CoT time: {cot_end_time - cot_start_time:.2f}s")

                        # Optional: Add a small delay between API calls if rate limits are a concern
                        # time.sleep(0.5)

                    except json.JSONDecodeError:
                        logging.error(f"Skipping line {i+1} in {filename}: Invalid JSON.")
                        error_count += 1
                    except KeyboardInterrupt:
                         logging.info("Keyboard interrupt received. Stopping processing.")
                         raise # Re-raise to stop the outer loop
                    except Exception as e:
                        logging.error(f"Unexpected error processing line {i+1} in {filename}: {e}", exc_info=True) # Log traceback info
                        error_count += 1
                        # Optional: Add a longer sleep/retry logic here?
                        time.sleep(2) # Wait a bit after an unexpected error

        except KeyboardInterrupt:
             logging.info(f"Stopped processing {filename} due to keyboard interrupt.")
        except Exception as e:
             logging.error(f"Failed to process file {input_path}: {e}", exc_info=True)

        end_time = time.time()
        total_processed_in_run = processed_count # Already counts only items processed in this run
        logging.info(f"Finished processing {filename}. Processed in this run: {total_processed_in_run}, Errors/Skipped: {error_count}. Time: {end_time - start_time:.2f}s")

    logging.info("All files processed.") 