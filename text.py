import re
import classifier

#

with open("r1_response.txt", 'r', encoding='utf-8') as f:
    text = f.read()

# Extract content between <think> and </think> tags
think_pattern = r'<think>(.*?)</think>'
think_match = re.search(think_pattern, text, re.DOTALL)  # re.DOTALL allows matching across newlines

if think_match:
    # Get just the thinking content
    thinking_text = think_match.group(1)

    # Split the thinking text into sentences
    sentences = re.split(r'([.!?])', thinking_text)

    # Combine each sentence with its delimiter
    sentences_with_delimiters = [
        ''.join(sentences[i:i+2]) for i in range(0, len(sentences)-1, 2)
    ]

    # Strip whitespace from each sentence for cleaner output
    sentences_with_delimiters = [s.strip() for s in sentences_with_delimiters if s.strip()]

    print(sentences_with_delimiters)

    # Get classification results for all sentences
    results = classifier.classify_text_segments(sentences_with_delimiters)

    print("\nClassification Results:")
    for i, result in enumerate(results):
        sentence = sentences_with_delimiters[i]

        # Now access the dictionary for each individual result
        top_label_index = result['scores'].index(max(result['scores']))
        top_label = result['labels'][top_label_index]
        top_score = result['scores'][top_label_index]

        print(f"\nSentence: {sentence}")
        print(f"Predicted label: {top_label}")
        print(f"Confidence: {top_score:.4f}")
else:
    print("No <think> tags found in the text.")
