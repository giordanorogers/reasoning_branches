"""
A class file for doing analysis of DeepSeek-R1 responses to mathematical question-answering tasks.

- Framework implemented as the object instance of a class.
- Load and register 8 or more documents.
- Each text file preprocesses (e.g., cleaned, stripped, etc.). At minimum should
include a word-count (but Rachlin said my sankey can be of behavioral pattern count.)
- Implement a generic parser and pre-processor for handling simple unstructured
text files.
- Implement the ability to specify a custom domain-specific parser. Now, when
registering the file, you can specify a custom parsing function that will carry
out the parsing and pre-processing of your unique files! Must implement support
for this even if I don't need it for my documents.
- Three insightful visualizations.
    - Text-to-Word Sankey diagram. Given the loaded r1 responses and a set of
    behavioral patterns of interest, generate a sankey diagram from text name
    to pattern, where the thickness of the connection represents the coung of
    that pattern in the specified text.
    - Histogram of r1 responses to different questions where the bars are fraction
    of the text devoted to each pattern.
    - Graph of trajectories across all responses.
"""

import os
import json
import re
from typing import List, Dict, Optional, Callable, TypeVar, Generic, cast
from dataclasses import dataclass, field
from openai import OpenAI
from dotenv import load_dotenv

# Generic type for different response formats
T = TypeVar('T', bound='BaseResponse')

@dataclass
class BaseResponse:
    """ A base data class for responses to all benchmark dataset types. """
    responses: List[str] = field(default_factory=list)
    chains_of_thought: List[Optional[str]] = field(default_factory=list)
    annotated_chains: List[Optional[str]] = field(default_factory=list)

@dataclass
class MATHResponse(BaseResponse):
    """ MATH dataset specific response format. """
    problem: str = field(default="")
    solution: str = field(default="")
    answer: str = field(default="")
    subject: str = field(default="")
    level: int = field(default=0)
    unique_id: str = field(default="")

@dataclass
class GSM8KResponse(BaseResponse):
    """ GSM8K dataset specific response format. """
    question: str = field(default="")
    answer: str = field(default="")

class R1Analyzer(Generic[T]):
    """ A class to analyze R1 responses to different mathematical benchmarks. """

    def __init__(self, parser: Optional[Callable[[Dict], T]] = None):
        # Initialize DeepSeek API for response annotation.
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
        # Initialize attribute to store responses
        self.responses: List[T] = []
        # Initialize the benchmark data parser
        self.parser = parser or self._math_parser
        # Set thinking tokens
        self.think_start = "<think>\n<think>"
        self.think_end = "</think>"

    def _math_parser(self, item: Dict) -> MATHResponse:
        """ Parser for MATH dataset format. """
        return MATHResponse(**item)

    def _gsm8k_parser(self, item: Dict) -> GSM8KResponse:
        """ Parser for GSM8K dataset format. """
        return GSM8KResponse(**item)

    def load_responses(
            self,
            filename: str,
            label: Optional[str] = None,
            id_field: str = 'unique_id'
    ):
        """
        Load responses from JSON with optional filtering by label/identifier.

        Args:
            filename: Path to the JSON file.
            label: Optional identifier to filter responses.
            id_field: Name of the field containing the identifier (default: 'unique_id' for MATH)
        """
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Parse the data
        responses = [cast(T, self.parser(item)) for item in data]

        # Filter by label if provided
        if label:
            responses = [r for r in responses if getattr(r, id_field) == label]

        self.responses = responses

    def extract_chains_of_thought(self) -> None:
        """ Extract chains of thought from all responses to all questions. """
        if not self.responses:
            raise ValueError("No responses loaded. Call load_responses first.")

        for r in self.responses:
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

    def annotate_chains_of_thought(self):
        """ Call the deepseek API to annotate R1's chain-of-thought into distinct
        behavioral patterns and return the annotated chain-of-thought. """
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
        for item in self.responses:
            annotated_chains = []
            for cot in item.chains_of_thought:
                if cot is not None:
                    formatted_prompt = annotation_prompt.format(thinking_process=cot)
                    try:
                        print("--- Calling DeepSeek API for cot annotation ---")
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
            annotated_chains_all.append(annotated_chains)
            return annotated_chains_all


if __name__ == "__main__":

    """    # Initialize analyzer with MATH dataset format
    math_analyzer = R1Analyzer[MATHResponse]()

    # Load one question's response data from file
    math_analyzer.load_responses(
        "ds4200_final/MATH-500_r1_7.json",
        label="test/precalculus/807.json"
    )"""

    #math_analyzer.extract_chains_of_thought()

    #annotated_chains = math_analyzer.annotate_chains_of_thought()


    """    for i, annot_cot in enumerate(annotated_chains):
        print("\n", annot_cot, "\n")"""

    """    # Print number of repsonses
    print(math_analyzer.responses[0].chains_of_thought[0])"""

