"""
A class file for doing analysis of DeepSeek-R1 responses to mathematical question-answering tasks.
"""

import os
import json
import re
from typing import List, Dict, Optional, Callable, TypeVar, Generic, cast
from dataclasses import dataclass, field
from openai import OpenAI
from dotenv import load_dotenv

class R1Analyzer(Generic[T]):
    """ A class to analyze R1 responses to different mathematical benchmarks. """

    def __init__(self, parser: Optional[Callable[[Dict], T]] = None):
        # Initialize OpenAI client for response annotation.
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # Initialize attribute to store responses
        self.responses: List[T] = []
        # Initialize the benchmark data parser
        self.parser = parser or self._math_parser
        # Set thinking tokens
        self.think_start = "<think>\n<think>"
        self.think_end = "</think>"

    def _math_parser(self, item: Dict) -> MATHResponse:
        """ Parser for MATH dataset format. """
        # Create a copy of the item to avoid modifying the original
        parsed_item = item.copy()
        
        # Check if responses is a list of dictionaries (from reasoner4.py format)
        if "responses" in parsed_item and isinstance(parsed_item["responses"], list):
            # If responses exist and have the new format (dicts with 'text', 'tokens', 'logits')
            if parsed_item["responses"] and all(isinstance(resp, dict) and "text" in resp for resp in parsed_item["responses"]):
                # Extract just the text from each response
                parsed_item["responses"] = [resp["text"] for resp in parsed_item["responses"]]
        
        return MATHResponse(**parsed_item)

    def _gsm8k_parser(self, item: Dict) -> GSM8KResponse:
        """ Parser for GSM8K dataset format. """
        return GSM8KResponse(**item)

    def responses_to_dict(self) -> List[Dict]:
        """Convert response objects to JSON-serializable dictionaries."""
        result = []
        for resp in self.responses:
            # Convert dataclass to dict
            resp_dict = {}
            # Add all fields from the dataclass
            for field_name, field_value in resp.__dict__.items():
                # Handle special cases
                if field_name == 'responses':
                    resp_dict[field_name] = field_value
                elif field_name in ['chains_of_thought', 'annotated_chains']:
                    # Convert None values to empty strings for better JSON representation
                    resp_dict[field_name] = [item if item is not None else "" for item in field_value]
                else:
                    resp_dict[field_name] = field_value
            result.append(resp_dict)
        return result

    def save_to_json(self, output_file: str):
        """Save the current state of responses to a JSON file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Convert responses to serializable format and save
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.responses_to_dict(), f, indent=2, ensure_ascii=False)
                
            print(f"Successfully saved data to {output_file}")
        except Exception as e:
            print(f"Error saving to JSON: {str(e)}")

    def annotate_chains_of_thought(self, output_file: Optional[str] = None):
        """ Call the OpenAI API to annotate R1's chain-of-thought into distinct
        behavioral patterns and return the annotated chain-of-thought.
        
        Args:
            output_file: Optional path to save progress to a JSON file.
                         If provided, results will be saved after each annotation.
        """
        annotation_prompt = """
        Please split the following reasoning chain of an LLM into
        annotated parts using labels and the following format ["label"]...["end-section"].
        A sequence should be split into multiple parts if it incorporates multiple
        distinct behaviours indicated by the labels.

        Available labels:
        0. initializing -> The model is rephrasing the given task and states initial thoughts.
        1. deduction -> The model is performing a deduction step based on its current approach and assumptions.
        2. adding-knowledge -> The model is enriching the current approach with recalled facts.
        3. example-testing -> The model generates examples to test its current approach.
        4. uncertainty-estimation -> The model is stating its own uncertainty.
        5. backtracking -> The model decides to change its approach.

        IMPORTANT: You must preserve ALL the original text exactly without skipping any words or characters. 
        Do not summarize, paraphrase, or truncate the text. Simply add the label tags around appropriate sections.

        The reasoning chain to analyze:
        {thinking_process}

        Answer only with the annotated text. Only use the labels outlined above.
        If there is a tail that has no annotation, leave it out.
        """
        annotated_chains_all = []
        
        # First ensure we have chains of thought extracted
        if not any(hasattr(item, 'chains_of_thought') and item.chains_of_thought for item in self.responses):
            print("Extracting chains of thought first...")
            self.extract_chains_of_thought()
        
        for i, item in enumerate(self.responses):
            print(f"Processing problem {i+1}/{len(self.responses)}")
            annotated_chains = []
            for j, cot in enumerate(item.chains_of_thought):
                if cot is not None:
                    formatted_prompt = annotation_prompt.format(thinking_process=cot)
                    try:
                        print(f"--- Calling OpenAI API for response {j+1} cot annotation ---")
                        api_response = self.client.chat.completions.create(
                            model='deepseek-chat',
                            messages=[
                                {'role': 'system', 'content': 'You are a helpful expert at annotating language model reasoning patterns.'},
                                {'role': 'user', 'content': formatted_prompt}
                            ],
                            temperature=0
                        )

                        # Check if response content exists before accessing
                        if api_response and api_response.choices and api_response.choices[0].message:
                            annotated_cot = api_response.choices[0].message.content
                            if annotated_cot:
                                annotated_cot = annotated_cot.strip()
                                annotated_chains.append(annotated_cot)
                            else:
                                print("Warning! Empty API response content!")
                                annotated_chains.append(None)
                        else:
                            print("Warning! Unexpected API response format!")
                            annotated_chains.append(None)
                    except Exception as e:
                        print(f"Error annotating chain-of-thought: {str(e)}")
                        annotated_chains.append(None)
                else:
                    annotated_chains.append(None)
            
            # Add the annotated chains to the response data
            item.annotated_chains = annotated_chains
            
            # Save progress after each problem if output_file is provided
            if output_file:
                self.save_to_json(output_file)
                print(f"Saved progress after problem {i+1}/{len(self.responses)}")
            
            annotated_chains_all.append(annotated_chains)
        
        # Final save after all processing is complete
        if output_file:
            self.save_to_json(output_file)
            print(f"Completed annotation for all {len(self.responses)} problems")
            
        return annotated_chains_all

    def extract_chains_of_thought(self) -> None:
        """ Extract chains of thought from all responses to all questions. """
        if not self.responses:
            raise ValueError("No responses loaded. Call load_responses first.")

        for r in self.responses:
            r.chains_of_thought = []  # Reset existing chains
            for response in r.responses:
                # Extract text between thinking tags
                pattern = f"{self.think_start}(.*?){self.think_end}"
                match = re.search(pattern, response, re.DOTALL)

                if match:
                    # Extract and clean the chain of thought
                    chain_of_thought = match.group(1).strip()
                    r.chains_of_thought.append(chain_of_thought)
                else:
                    # No chain of thought found
                    r.chains_of_thought.append(None)

if __name__ == "__main__":

    # Initialize analyzer with MATH dataset format
    math_analyzer = R1Analyzer[MATHResponse]()

    # Load response data from file
    math_analyzer.load_responses(
        "MATH-500_123_r1.json",
        label="test/precalculus/807.json"
    )

    # Print the number of responses for each problem
    for i, response in enumerate(math_analyzer.responses):
        print(f"Problem {i+1} has {len(response.responses)} responses")
    
    # Extract the chains of thought
    math_analyzer.extract_chains_of_thought()
    print(f"Extracted chains of thought from {len(math_analyzer.responses)} problems")
    
    # Annotate the chains of thought and save progress to a JSON file
    output_file = "annotated_responses.json"
    annotated_chains = math_analyzer.annotate_chains_of_thought(output_file=output_file)
    
    print(f"Annotation complete! Results saved to {output_file}") 