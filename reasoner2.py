"""
A script to use nnsight to call the R1 model to answer prompts.
"""

import os
from dotenv import load_dotenv
import torch
import nnsight
from nnsight import LanguageModel, CONFIG

# Load environment variables from .env file
load_dotenv()

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
    user_tokens = tokenizer.encode(user_message, add_special_tokens=False)
    thinking_tokens = tokenizer.encode(thinking_message, add_special_tokens=False)
    return [[BOS] + user_tokens + [NEWLINE] + [THINK_START] + thinking_tokens]

def generate_text(input_ids, model, tokenizer, max_length=100):
    """ Generate text. """
    input_ids = torch.tensor(input_ids)
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

    # Decode and return the generated text
    input_text = tokenizer.decode(input_ids[0])
    # Check if outputs has a 'value' attribute or is a list directly
    if hasattr(outputs, 'value'):
        generated_tokens = outputs.value
    else:
        generated_tokens = outputs
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    return input_text + "\n" + generated_text

if __name__ == "__main__":

    query = "What is 1 + 1?"
    input = custom_encoding(query)
    model_output = generate_text(input, model, tokenizer)

    print(model_output)