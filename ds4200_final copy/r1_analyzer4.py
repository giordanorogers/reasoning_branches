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


A thought I'm having:
    Input: Annotated chains-of-thought
    Stopfile: Behavioral pattern markers (names + end_sections)
    load_stop_words: Will load in the stopfile
    load_text: loads in all the annotated chains-of-thought
    parser: splits the input into a dictionary where the keys are patterns and
        the values are dictionaries with counts and lists of the occurrences of\
        the pattern in the text
    pattern_sankey: 

"""

import os
import json
import re

class R1Analyzer():
    """
    A class to analyze R1 responses to different mathematical benchmark
    question & answer tasks.
    """

    def __init__(self, custom_parser=None, stop_words=None):
        # Initialize attribute to store response data for multiple files
        self.response_data = {}
        # Initialize custom parser
        self.custom_parser = custom_parser
        # Initialize stop words
        self.stop_words = stop_words
        # Define the pattern types
        self.pattern_types = [
            "initializing", "deduction", "adding-knowledge", 
            "example-testing", "uncertainty-estimation", "backtracking"
        ]

    def load_patterns(self, patternfile_path):
        """ Load the pattern types from the file containing pattern names. """
        pattern_types = []
        with open(patternfile_path, 'r', encoding='utf-8') as f:
            # Read each line, strip newline characters, and add to the list
            pattern_types = [line.strip() for line in f.readlines() if line.strip()]

        self.pattern_types = pattern_types

        print(f"Loaded pattern types: {self.pattern_types}")

    def load_responses(self, directory_path):
        """
        Loads all text files from a directory and processes them.
        
        Args:
            directory_path (str): Path to the directory containing text files.
        """
        # Check if directory exists
        if not os.path.isdir(directory_path):
            print(f"Directory {directory_path} does not exist.")
            return
        
        # Get all text files in the directory
        text_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
        
        # Process each file
        for file_name in text_files:
            file_path = os.path.join(directory_path, file_name)
            self.parse_file(file_path, self._pattern_parser)
            
        print(f"Loaded {len(text_files)} files from {directory_path}")

    def _pattern_parser(self, text, stop_words):
        """
        Parses a file to organize behavioral pattern data. 
        
        Args:
            text (str): The text content to parse.
            stop_words (list): List of stop words to filter out.

        Returns:
            dict: Dictionary with the pattern data.
        """
        # Initialize pattern data for this file
        pattern_data = {pattern: {"chars": 0, "count": 0} for pattern in self.pattern_types}
        
        # Initialize variables for looping through the text
        current_pattern = None
        current_start = 0

        try:
            i = 0
            while i < len(text):
                # Look for pattern start labels
                if text[i:i+2] == '["' and i + 2 < len(text):
                    # Find the end bracket of the label
                    end_bracket = text.find('"]', i)
                    if end_bracket != -1:
                        # Extract the pattern label type
                        pattern_type = text[i+2:end_bracket]

                        # If we were already in a pattern, calculate its characters
                        if current_pattern is not None and current_pattern in pattern_data:
                            pattern_text = text[current_start:i]
                            pattern_data[current_pattern]["chars"] += len(pattern_text)

                        # Start a new section
                        if pattern_type in pattern_data:
                            current_pattern = pattern_type
                            pattern_data[current_pattern]["count"] += 1
                            # Move the start to the end of the closing bracket
                            current_start = end_bracket + 2

                        i = end_bracket + 2
                        continue

                # Look for section end markers
                if text[i:i+13] == '["end_section"]':
                    end_bracket = text.find('"]', i)
                    if end_bracket != -1 and current_pattern is not None:
                        # Calculate the characters for the pattern
                        pattern_text = text[current_start:i]
                        pattern_data[current_pattern]["chars"] += len(pattern_text)
                        current_pattern = None
                        i = end_bracket + 2
                        continue

                i += 1

            print("Parsed text successfully")
            return pattern_data
        except Exception as e:
            print(f"Error parsing data: {str(e)}")
            return pattern_data

    def parse_file(self, textfile, specific_parser=None):
        """ 
        A generic parser to parse a text file.
        
        Args:
            textfile (str): Path to the text file.
            specific_parser (function, optional): Custom parser function.
            
        Returns:
            dict: Parsed data from the file.
        """
        # Get the filename without extension
        file_name = os.path.basename(textfile)
        
        # Read the file
        with open(textfile, 'r', encoding='utf-8') as file:
            text = file.read()

        # Initialize parsed_text
        parsed_text = {}

        if specific_parser:
            parsed_text = specific_parser(text, self.stop_words)
        else:
            # Use the default pattern parser if no specific parser is provided
            parsed_text = self._pattern_parser(text, self.stop_words)

        # Store the parsed data with the filename as the key
        self.response_data[file_name] = parsed_text
        
        return parsed_text


if __name__ == "__main__":

    # Initialize analyzer
    math_analyzer = R1Analyzer()

    # Load the pattern types from the patternfile
    patternfile_path = "/Users/giordanorogers/Documents/Code/reasoning_branches/ds4200_final/patternfile.txt"
    math_analyzer.load_patterns(patternfile_path)

   # Load all files from the data directory
    data_dir = "/Users/giordanorogers/Documents/Code/reasoning_branches/ds4200_final/data"
    math_analyzer.load_responses(data_dir)

    # Print the response data for all files
    for file_name, data in math_analyzer.response_data.items():
        print(f"\nFile: {file_name}")
        print(data)
