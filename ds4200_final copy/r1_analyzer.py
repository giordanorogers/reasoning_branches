"""
A class file for doing analysis of DeepSeek-R1 responses to mathematical question-answering tasks.
"""

import yaml
import json
import re
import os
from openai import OpenAI
from dotenv import load_dotenv
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from graphviz import Digraph

class R1Analyzer:

    def __init__(self):
        # Load stop tokens by default
        self.stop_tokens = self.load_stop_tokens('stopfile.txt')
        # Initialize OpenAI client
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # Initialize responses attribute
        self.responses = None

    def load_stop_tokens(self, stopfile):
        """ A list of special tokens we're looking for. In particular, the thinking
        tokens, the final answer tokens, and the begin and end tokens. """
        with open(stopfile, 'r', encoding='utf-8') as f:
            stop_tokens = [line.strip() for line in f.readlines()]
        return stop_tokens

    def load_responses(self, filename, num_questions=None, num_responses=None, label=None, parser=None):
        """ Load and process responses from a JSON file.
        
        Args:
            filename (str): Path to the JSON file
            num_questions (int, optional): Number of questions to load. If None, loads all.
            num_responses (int, optional): Number of responses per question to load. If None, loads all.
            label (str, optional): Specific question label to load
            parser (callable, optional): Custom parser function for responses
            
        Returns:
            dict: Processed responses with structure:
                {
                    'questions': [
                        {
                            'name': str,
                            'question': str,
                            'answer': str,
                            'responses': [str],
                            'processed_responses': [dict]  # If parser provided
                        }
                    ]
                }
        """
        with open(filename, 'r', encoding='utf-8') as f:
            response_json = json.load(f)

        processed_data = {'questions': []}

        # If a label is provided, filter for the question with the corresponding name.
        if label:
            questions = [q for q in response_json if q['name'] == label]
        else:
            questions = response_json

        # Get a specific number of questions if specified
        if num_questions:
            questions = questions[:num_questions]

        # Organize the questions into a dictionary
        for question_data in questions:
            processed_question = {
                'name': question_data['name'],
                'question': question_data['question'],
                'answer': question_data['answer'],
                'responses': question_data['responses']
            }

            # Get a specific number of responses if specified
            if num_responses:
                processed_question['responses'] = processed_question['responses'][:num_responses]

            # Apply the custom parser if provided
            if parser:
                processed_question['processed_responses'] = [
                    parser(response) for response in processed_question['responses']
                ]

            processed_data['questions'].append(processed_question)

        # Store the processed data in the instance
        self.responses = processed_data
        return processed_data

    def pre_processor(self, response):
        """ Each R1 response will be stripped of everything but the CoT between
        think tage and the final answer. I'll call the openai api to use gpt-4o
        to automatically annotate the response and categorize it into behavioral
        patterns. I'll probably create a second method to create a dictionary that
        stores both a count of how many times each pattern appears in the response,
        as well as a list that indicates the order of the patterns. """
        # Get think tokens from stop_tokens list
        think_start = re.escape(self.stop_tokens[0])  # First token is <think>
        think_end = re.escape(self.stop_tokens[1])    # Second token is </think>

        # Create pattern using the tokens
        pattern = f'{think_start}(.*?){think_end}'

        # Find the chain of thought
        match = re.search(pattern, response, re.DOTALL)

        if match:
            # Extract and clean the chain of thought
            chain_of_thought = match.group(1).strip()
            return chain_of_thought
        else:
            return None

    def generic_parser(self, text):
        """ A generic parser for handling simple unstructured text files. Can
        optionally specify a custom domain-specific parser. Now, when registering
        the file, you can specify custom parsing function that will carry out the
        parsing and pre-processing of your unique files. You should implement
        support for this even if you might not need it for your documents. """
        pass

    def behavior_annotator(self, chain_of_thought):
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

        The reasoning chain to analyze:
        {thinking_process}

        Answer only with the annotated text. Only use the labels outlined above.
        If there is a tail that has no annotation, leave it out.
        """
        
        try:
            # Format the prompt with the chain of thought
            formatted_prompt = annotation_prompt.format(thinking_process=chain_of_thought)
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes LLM reasoning patterns."},
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.1  # Low temperature for consistent annotations
            )
            
            # Extract the annotated text from the response
            annotated_text = response.choices[0].message.content.strip()
            return annotated_text
            
        except Exception as e:
            print(f"Error in behavior_annotator: {str(e)}")
            return None

    def behavior_sankey(self, behavior_list=None, k=5):
        """ Map each R1 response to behavioral patterns using a Sankey diagram,
        where the thickness of the line is the proportion of the response taken
        up by that behavioral pattern. Users can specify a particular behavior, or
        the behaviors can be the union of the k most common behaviors across each
        text file. """
        # Define the available behaviors
        available_behaviors = [
            "initializing",
            "deduction",
            "adding-knowledge",
            "example-testing",
            "uncertainty-estimation",
            "backtracking"
        ]
        
        # If no specific behaviors provided, use the k most common ones
        if behavior_list is None:
            behavior_list = available_behaviors[:k]
            
        # Initialize lists for Sankey diagram
        source = []
        target = []
        value = []
        label = []
        
        # Process each question's responses
        for question in self.responses['questions']:
            for response in question['processed_responses']:
                # Extract behaviors from annotated text
                behaviors = re.findall(r'\[(.*?)\](.*?)\[end-section\]', response, re.DOTALL)
                
                # Calculate total length of response
                total_length = sum(len(behavior[1].strip()) for behavior in behaviors)
                
                # Add edges for each behavior
                for behavior, text in behaviors:
                    if behavior in behavior_list:
                        # Add edge from question to behavior
                        source.append(question['name'])
                        target.append(behavior)
                        value.append(len(text.strip()) / total_length)
                        
                        # Add labels if not already present
                        if question['name'] not in label:
                            label.append(question['name'])
                        if behavior not in label:
                            label.append(behavior)
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node = dict(
                pad = 15,
                thickness = 20,
                line = dict(color = "black", width = 0.5),
                label = label,
                color = "blue"
            ),
            link = dict(
                source = [label.index(s) for s in source],
                target = [label.index(t) for t in target],
                value = value
            )
        )])
        
        # Update layout
        fig.update_layout(
            title_text="Behavioral Pattern Flow in R1 Responses",
            font_size=10
        )
        
        return fig

    def behavior_hist(self, mode='single', num_subplots=3):
        """ Create histograms showing the proportion of behavioral patterns.
        
        Args:
            mode (str): Either 'single' (show individual responses) or 'average' (show averaged proportions)
            num_subplots (int): Number of subplots to display
        """
        # Define section markers
        sections = {
            'initializing': '[initializing]',
            'deduction': '[deduction]',
            'adding-knowledge': '[adding-knowledge]',
            'example-testing': '[example-testing]',
            'uncertainty-estimation': '[uncertainty-estimation]',
            'backtracking': '[backtracking]'
        }

        def compute_text_fraction(text):
            """Compute fraction of total text for each section."""
            markers = list(sections.values())
            pattern = '(' + '|'.join([re.escape(m) for m in markers]) + ')'

            matches = list(re.finditer(pattern, text))
            lengths = {key: 0 for key in sections.keys()}

            for i, match in enumerate(matches):
                marker_found = match.group()
                section = next((sec for sec, val in sections.items() if val == marker_found), None)
                if section:
                    start = match.end()
                    end = matches[i+1].start() if i + 1 < len(matches) else len(text)
                    lengths[section] += len(text[start:end])

            total_length = len(text)
            return {
                sec: (length / total_length if total_length > 0 else 0)
                for sec, length in lengths.items()
            }

        # Section names with line breaks for better display
        section_names = [
            "initializing",
            "deduction",
            "adding<br>knowledge",
            "example<br>testing",
            "uncertainty<br>estimation",
            "backtracking"
        ]

        if mode == 'single':
            # Create subplots for individual responses
            fig = make_subplots(
                rows=num_subplots, cols=1,
                subplot_titles=[f"Response {i+1}" for i in range(num_subplots)]
            )

            # Process first question's responses
            question = self.responses['questions'][0]
            responses = question['processed_responses'][:num_subplots]

            for i, response in enumerate(responses, 1):
                fractions = compute_text_fraction(response)
                values = [fractions[sec.replace("<br>", " ")] for sec in section_names]
                
                fig.add_trace(
                    go.Bar(
                        x=section_names,
                        y=values,
                        text=[f"{val:.1%}" for val in values],
                        textposition='outside',
                        name=f"Response {i}",
                        showlegend=False,
                        marker_color=px.colors.qualitative.Safe[i % len(px.colors.qualitative.Safe)]
                    ),
                    row=i, col=1
                )

            fig.update_layout(
                title=f"Behavioral Patterns in Responses to: {question['name']}",
                height=300 * num_subplots,
                showlegend=False
            )

        else:  # mode == 'average'
            # Create subplots for averaged responses per question
            questions = self.responses['questions'][:num_subplots]
            fig = make_subplots(
                rows=num_subplots, cols=1,
                subplot_titles=[q['name'] for q in questions]
            )

            for i, question in enumerate(questions, 1):
                # Calculate average fractions across all responses
                all_fractions = []
                for response in question['processed_responses']:
                    all_fractions.append(compute_text_fraction(response))
                
                # Average the fractions
                avg_fractions = {
                    sec: sum(f[sec] for f in all_fractions) / len(all_fractions)
                    for sec in sections.keys()
                }
                
                values = [avg_fractions[sec.replace("<br>", " ")] for sec in section_names]
                
                fig.add_trace(
                    go.Bar(
                        x=section_names,
                        y=values,
                        text=[f"{val:.1%}" for val in values],
                        textposition='outside',
                        name=question['name'],
                        showlegend=False,
                        marker_color=px.colors.qualitative.Safe[i % len(px.colors.qualitative.Safe)]
                    ),
                    row=i, col=1
                )

            fig.update_layout(
                title="Average Behavioral Patterns Across Questions",
                height=300 * num_subplots,
                showlegend=False
            )

        # Common layout updates
        fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            shapes=[dict(
                type='rect',
                xref='paper',
                yref='paper',
                x0=0, y0=0,
                x1=1, y1=1,
                line=dict(color='black', width=1)
            )],
            margin=dict(l=80, r=40, t=60, b=80)
        )

        # Configure axes
        fig.update_xaxes(
            tickangle=0,
            showline=True,
            linecolor='black'
        )

        fig.update_yaxes(
            title="Fraction of Total Text (%)",
            range=[0, 1],
            showline=True,
            linecolor='black',
            showgrid=True,
            gridcolor='lightgray',
            tickformat='.0%'
        )

        return fig

    def behavior_graph(self, mode='single', question_idx=0, response_idx=0):
        """ Create a weighted graph showing the reasoning trajectory.
        
        Args:
            mode (str): Either 'single' (show one response) or 'aggregate' (show weighted paths across all responses)
            question_idx (int): Index of the question to analyze
            response_idx (int): Index of the response to analyze (only used in 'single' mode)
            
        Returns:
            graphviz.Digraph: A directed graph showing the reasoning trajectory
        """
        # Create a new directed graph
        dot = Digraph(comment='Reasoning Trajectory')
        dot.attr(rankdir='TB')  # Top to Bottom layout
        
        # Define node shapes
        dot.attr('node', shape='box')  # Default shape for behavior nodes
        dot.attr('node', style='filled')  # Filled nodes for better visibility
        
        # Get the question and its responses
        question = self.responses['questions'][question_idx]
        responses = question['processed_responses']
        
        if mode == 'single':
            # Process a single response
            response = responses[response_idx]
            behaviors = re.findall(r'\[(.*?)\](.*?)\[end-section\]', response, re.DOTALL)
            
            # Add nodes and edges for the single response
            prev_node = 'question'
            for i, (behavior, _) in enumerate(behaviors):
                node_name = f"{behavior}_{i}"
                dot.node(node_name, behavior)
                dot.edge(prev_node, node_name, label='1')
                prev_node = node_name
            
            # Add final answer node
            dot.node('answer', 'answer', shape='oval')
            dot.edge(prev_node, 'answer', label='1')
            
            # Update graph title
            dot.attr(label=f"Reasoning Trajectory for Response {response_idx + 1}")
            
        else:  # mode == 'aggregate'
            # Process all responses to create weighted edges
            behavior_sequences = []
            for response in responses:
                behaviors = re.findall(r'\[(.*?)\](.*?)\[end-section\]', response, re.DOTALL)
                behavior_sequences.append([b[0] for b in behaviors])
            
            # Create nodes and count edge weights
            edge_weights = {}  # (from_node, to_node) -> count
            node_counts = {}   # node_name -> count
            
            for sequence in behavior_sequences:
                prev_node = 'question'
                for i, behavior in enumerate(sequence):
                    node_name = f"{behavior}_{i}"
                    edge_key = (prev_node, node_name)
                    edge_weights[edge_key] = edge_weights.get(edge_key, 0) + 1
                    node_counts[node_name] = node_counts.get(node_name, 0) + 1
                    prev_node = node_name
                
                # Add edge to answer
                edge_key = (prev_node, 'answer')
                edge_weights[edge_key] = edge_weights.get(edge_key, 0) + 1
            
            # Add nodes and edges with weights
            for (from_node, to_node), weight in edge_weights.items():
                if from_node == 'question':
                    dot.node(from_node, question['name'], shape='oval')
                elif from_node != 'answer':
                    dot.node(from_node, from_node.split('_')[0])
                
                if to_node != 'answer':
                    dot.node(to_node, to_node.split('_')[0])
                else:
                    dot.node(to_node, 'answer', shape='oval')
                
                dot.edge(from_node, to_node, label=str(weight))
            
            # Update graph title
            dot.attr(label=f"Aggregated Reasoning Trajectories for {question['name']}")
        
        # Set graph attributes
        dot.attr('graph', 
                fontname='Arial',
                fontsize='12',
                splines='ortho')  # Orthogonal edges for cleaner layout
        
        dot.attr('node',
                fontname='Arial',
                fontsize='10',
                fillcolor='lightblue')
        
        dot.attr('edge',
                fontname='Arial',
                fontsize='8')
        
        return dot

if __name__ == "__main__":
    # Initialize the class
    r1a = R1Analyzer()

    # Load the stop words
    r1_stop_words = r1a.load_stop_tokens('stopfile.txt')
    print(r1_stop_words)
    print(r1_stop_words[1])

    # Load the responses
    r1_responses = r1a.load_responses(
        "gsm8k_50_r1.json",
        label="Natalia"
    )

    # Process the first response from the first question
    first_response = r1_responses['questions'][0]['responses'][0]
    print(first_response)
    cot = r1a.pre_processor(first_response)
    print("Chain of Thought:")
    print(cot)

    annotated_cot = r1a.behavior_annotator(cot)
    print("Annotated Chain of Thought:")
    print(annotated_cot)
