"""
Script for classifying behavioral patterns in chain-of-thought sequences.
Credit to Chris Wendler's GSM8K simple steering vector experiment for the
inspiration for how to implement the zero-shot classification.
Credit to [Venhoff et al](https://openreview.net/pdf?id=OwhVWNOBcz) for the
idea of what behavioral patterns to use for the classification.
"""

from typing import List
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline
import reasoner2

def classify_text_segments(
        texts: List[str],
        behavioral_patterns: List[str] = [
            "Initializing",
            "Deduction",
            "Adding Knowledge",
            "Example Testing",
            "Uncertainty Estimation",
            "Backtracking"
        ]
) -> List[dict]:
    """
    Classify the a sentence of Chain-of-Thought as one of a given set of
    behavioral reasoning patterns.

    Args:
        texts: List of text segments to classify.
        behavioral_patterns: Labels to classify between.
    
    Returns:
        List of classification results dictionaries.
    """
    hypothesis_template = "This is an example of {}"

    # Initialize zero-shot classification pipeline
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

    results = classifier(
        texts,
        behavioral_patterns,
        hypothesis_template=hypothesis_template,
        multi_label=False
    )

    # Ensure results is a list even when a single item is classified
    if isinstance(results, dict):
        results = [results]

    processed_results = []
    for result in results:
        ordered_scores = []
        ordered_labels = []
        # Iterate through the behavioral patterns
        for pattern in behavioral_patterns:
            idx = result['labels'].index(pattern)
            ordered_scores.append(result['scores'][idx])
            ordered_labels.append(pattern)
        result['scores'] = ordered_scores
        result['labels'] = ordered_labels
        processed_results.append(result)
    return processed_results

if __name__ == "__main__":
    

    # Setup some example strings for testing the classifier.
    classification_examples = {
        "Initializing": "I am now initializing.",
        "Deduction": "I am now using deduction.",
        "Adding Knowledge": "I am now adding knowledge.",
        "Example Testing": "I am now testing examples.",
        "Uncertainty Estimation": "I am now estimating my uncertainty.",
        "Backtracking": "I am now backtracking."
    }

    # Classify each test example and print the results
    for label, text in classification_examples.items():
        print(f"\nTesting text: '{text}'")
        print(f"Expected label: {label}")
        
        result = classify_text_segments([text])[0]
        
        # Print top classification and confidence
        top_label_index = result['scores'].index(max(result['scores']))
        top_label = result['labels'][top_label_index]
        top_score = result['scores'][top_label_index]
        
        print(f"Predicted label: {top_label}")
        print(f"Confidence: {top_score:.4f}")
        
        # Print all labels and scores for comparison
        print("\nAll predictions:")
        for i, (label, score) in enumerate(zip(result['labels'], result['scores'])):
            print(f"{label}: {score:.4f}")