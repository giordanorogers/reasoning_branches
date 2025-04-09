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

    def chunk_text(self, text: str, max_chunk_length: int = 3000) -> List[str]:
        """
        Split a long text into chunks of max_chunk_length characters.
        
        Args:
            text: The text to split
            max_chunk_length: Maximum length of each chunk
            
        Returns:
            List of text chunks
        """
        # If the text is shorter than the maximum length, return it as is
        if not text or len(text) <= max_chunk_length:
            return [text]
        
        # Split the text into chunks of max_chunk_length characters
        chunks = []
        for i in range(0, len(text), max_chunk_length):
            chunks.append(text[i:i + max_chunk_length])
        
        return chunks

    def cot_annotator(self):
        """ Call gpt-4o from the openai api to annotate an r1 chain-of-thought into distinct
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
        for response in self.responses:
            annotated_chains = []
            for chain in response.chains_of_thought:
                if chain is not None:
                    # Check if the chain needs to be processed in chunks
                    chain_chunks = self.chunk_text(chain)
                    
                    if len(chain_chunks) == 1:
                        # Process the chain normally if it's just one chunk
                        formatted_prompt = annotation_prompt.format(thinking_process=chain)
                        try:
                            api_response = self.client.chat.completions.create(
                                model="gpt-4-turbo",  # Use gpt-4-turbo for better handling of longer texts
                                messages=[
                                    {"role": "system", "content": "You are an expert at analyzing language model reasoning patterns."},
                                    {"role": "user", "content": formatted_prompt}
                                ],
                                temperature=0.1
                            )
                            # Check if the content exists before accessing it
                            if api_response and api_response.choices and api_response.choices[0].message:
                                annotated_text = api_response.choices[0].message.content
                                if annotated_text:
                                    annotated_text = annotated_text.strip()
                                    annotated_chains.append(annotated_text)
                                else:
                                    print("Warning: Empty response content")
                                    annotated_chains.append(None)
                            else:
                                print("Warning: Unexpected API response structure")
                                annotated_chains.append(None)
                        except Exception as e:
                            print(f"Error annotating chain of thought: {str(e)}")
                            annotated_chains.append(None)
                    else:
                        # Process the chain in chunks and combine the results
                        print(f"Processing chain in {len(chain_chunks)} chunks...")
                        chunk_annotations = []
                        
                        for i, chunk in enumerate(chain_chunks):
                            chunk_prompt = f"""
                            This is chunk {i+1} of {len(chain_chunks)} from a longer reasoning chain.
                            Please annotate this chunk with the same labels as before.
                            
                            {annotation_prompt.format(thinking_process=chunk)}
                            """
                            
                            try:
                                api_response = self.client.chat.completions.create(
                                    model="gpt-4-turbo",
                                    messages=[
                                        {"role": "system", "content": "You are an expert at analyzing language model reasoning patterns."},
                                        {"role": "user", "content": chunk_prompt}
                                    ],
                                    temperature=0.1
                                )
                                
                                if api_response and api_response.choices and api_response.choices[0].message:
                                    chunk_annotation = api_response.choices[0].message.content
                                    if chunk_annotation:
                                        chunk_annotations.append(chunk_annotation.strip())
                                    else:
                                        print(f"Warning: Empty response content for chunk {i+1}")
                                else:
                                    print(f"Warning: Unexpected API response structure for chunk {i+1}")
                            except Exception as e:
                                print(f"Error annotating chunk {i+1}: {str(e)}")
                        
                        # Combine all chunk annotations
                        if chunk_annotations:
                            combined_annotation = "\n\n".join(chunk_annotations)
                            annotated_chains.append(combined_annotation)
                        else:
                            print("Warning: No successful annotations for any chunks")
                            annotated_chains.append(None)
                else:
                    annotated_chains.append(None)
            # Add the annotated chains to the response data
            response.annotated_chains = annotated_chains
            annotated_chains_all.append(annotated_chains)
        
        return annotated_chains_all

if __name__ == "__main__":
    # Initialize analyzer with MATH dataset format
    math_analyzer = R1Analyzer[MATHResponse]()

    # Load one question's response data from file
    math_analyzer.load_responses(
        "MATH-500_123_r1.json",
        label="test/precalculus/807.json"
    )

    # Extract chains of thought from responses
    math_analyzer.extract_chains_of_thought()

    # Print the original chains of thought
    print("=== Original Chains of Thought ===")
    for idx, cot in enumerate(math_analyzer.responses[0].chains_of_thought):
        print(f"\nCoT {idx}:")
        print(cot)
    
    """# Annotate the chains of thought
    print("\n=== Annotating Chains of Thought ===")
    annotated_chains = math_analyzer.cot_annotator()
    
    # Print the annotated chains of thought
    print("\n=== Annotated Chains of Thought ===")
    for idx, annotated in enumerate(annotated_chains[0]):
        print(f"\nAnnotated CoT {idx}:")
        print(annotated)"""
