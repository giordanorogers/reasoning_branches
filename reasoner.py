"""
A script to use nnsight to call the R1 model to answer prompts.
"""

import os
# Set tokenizers parallelism to false to avoid tokenizer deadlock
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
import torch
import nnsight
from nnsight import LanguageModel, CONFIG

load_dotenv()

# Token IDs for R1
BOS = 128000
USER = 128011
ASSISTANT = 128012
NEWLINE = 198
THINK_START = 128013
THINK_END = 128014
EOS = 128001

CONFIG.APP.REMOTE_LOGGING = False
CONFIG.API.APIKEY = os.getenv("NDIF_API_KEY")

model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
model = LanguageModel(model_name)
tokenizer = model.tokenizer

def custom_encoding(user_message: str, thinking_message: str = ""):
    """ Encode the user message and thinking message. """
    if len(thinking_message) > 1:
        user_tokens = tokenizer.encode(user_message, add_special_tokens=False)
        thinking_tokens = tokenizer.encode(thinking_message, add_special_tokens=False)
        return [[BOS] + user_tokens + [NEWLINE] + [THINK_START] + thinking_tokens]
    else:
        user_tokens = tokenizer.encode(user_message, add_special_tokens=False)
        return [[BOS] + user_tokens + [NEWLINE] + [THINK_START]]

def generate_text(input_ids, model, tokenizer, max_length=100):
    """ Generate text. """
    # Convert our input tokens to a pytorch tensor
    input_ids = torch.tensor(input_ids)
    # Create a tensor of ones the same length as our input tensor
    attention_mask = torch.ones_like(input_ids)
    with model.generate(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        max_new_tokens=max_length,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        remote=True
    ):
        outputs = nnsight.list().save()

        with model.lm_head.all():
            outputs.append(model.lm_head.output[0][-1].argmax(dim=-1))

    # Decode the input tokens
    input_text = tokenizer.decode(input_ids[0])

    # Get the generated tokens
    if hasattr(outputs, 'value'):
        generated_tokens = outputs.value
    else:
        generated_tokens = outputs

    # Decode the generated tokens
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    return input_text + "\n" + generated_text

def g_n_c(input_ids, model, tokenizer, max_length=100):
    """ Generate text and classify every sentence automatically. """
    # Convert the input tokens to a pytorch tensor
    input_ids = torch.tensor(input_ids)
    print(input_ids)
    attention_mask = torch.ones_like(input_ids)
    print(attention_mask)
    with model.generate(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        max_new_tokens=max_length,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        remote=True
    ):
        # Create a persistent list to save the output
        outputs = nnsight.list().save()

        end_token_values = [12, 382, 627, 0, 30]

        with model.lm_head.all():
            # This gets the FIRST GENERATED token, not the last input token
            next_token = model.lm_head.output[0][0].argmax(dim=-1)
            outputs.append(next_token)

        while True:
            if next_token.item() in end_token_values:
                model.lm_head.output.stop()
                break
            else:
                next_token = model.lm_head.next().output[0][-1].argmax(dim=-1)

    return outputs



def gen_and_classify(input_ids, model, tokenizer, max_length=100):
    """ Generate text and classify every sentence automatically. """
    # Convert the input tokens to a pytorch tensor
    input_ids = torch.tensor(input_ids)
    print(input_ids)
    attention_mask = torch.ones_like(input_ids)
    print(attention_mask)
    with model.generate(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        max_new_tokens=max_length,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        remote=True
    ):

        # Create a persistent list to save the output
        outputs = nnsight.list().save()

        # Period, exclamation, and question mark tokens respectively
        end_tokens = [
            [13], [382], [627], # ".", ".\n\n", ".\n"
            [0], # "!"
            [30], # "?"
        ]
        end_token_values = [12, 382, 627, 0, 30]

        end_token_tensors = [torch.tensor(token[0]) for token in end_tokens]
        print(end_token_tensors)

        # Start the token generation.
        with model.lm_head.all():
            next_token = model.lm_head.output[0][-1].argmax(dim=-1)
            outputs.append(next_token)
            if next_token.item() in end_token_values:
                model.lm_head.output.stop()

    print(tokenizer.decode(outputs))

    for token in outputs:
        print(token)
        print(tokenizer.decode(token))
        print()

    print("Failed to exit context successfully :(")
    return outputs
    
    """# Decode the input tokens
    input_text = tokenizer.decode(input_ids[0])

    # Get the generated tokens
    if hasattr(outputs, 'value'):
        generated_tokens = outputs.value
    else:
        generated_tokens = outputs

    # Decode the generated tokens
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    return input_text + "\n" + generated_text"""

def gen_and_classif(input_ids, model, tokenizer, max_length=100):
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.ones_like(input_ids)
    
    end_token_values = [13, 382, 627, 0, 30]
    
    with model.generate(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        max_new_tokens=max_length,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        remote=True
    ) as tracer:
        # Create a list to save all outputs
        outputs = nnsight.list().save()
        
        # Generate the first token
        with model.lm_head.all():
            token = model.lm_head.output[0][-1].argmax(dim=-1).save()
            outputs.append(token)
        
        # Generate subsequent tokens with no conditional logic
        for i in range(1, max_length):
            with model.lm_head.all():
                token = model.lm_head.next().output[0][-1].argmax(dim=-1).save()
                outputs.append(token)
    
    # AFTER the context completes, now we can access values
    result_tokens = []
    for token in outputs:
        # Now token.value is available
        result_tokens.append(token.value.item())
        
        # Check if we've reached an end token
        if token.value.item() in end_token_values:
            break
    
    return torch.tensor(result_tokens)

if __name__ == "__main__":

    # Create a query to send the model.
    query = "What is 1 + 1?"

    # Encode the query to induce distinct CoT.
    input = custom_encoding(query)

    """# Get the model's response as text.
    model_output = generate_text(input, model, tokenizer)

    print(model_output)"""

    outputs = gen_and_classif(input, model, tokenizer)

    print("Exited context successfully!")
    print(outputs)
    print(tokenizer.decode(outputs))
