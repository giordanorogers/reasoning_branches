import re
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Define section markers
sections = {
    'initializing': '[“initializing”]',
    'deduction': '[“deduction”]',
    'adding knowledge': '[“adding-knowledge”]',
    'example testing': '[“example-testing”]',
    'uncertainty estimation': '[“uncertainty-estimation”]',
    'backtracking': '[“backtracking”]'
}

def compute_behavioral_patterns(text):
    """Count occurrences of each marker."""
    return {key: text.count(marker) for key, marker in sections.items()}

def compute_text_fraction(text):
    """Compute fraction of total text (by character count) for each section."""
    markers = list(sections.values())
    pattern = '(' + '|'.join([re.escape(m) for m in markers]) + ')'

    matches = list(re.finditer(pattern, text))
    lengths = {key: 0 for key in sections.keys()}

    for i, match in enumerate(matches):
        marker_found = match.group()
        section = next((sec for sec, val in sections.items() if val == marker_found), None)
        start = match.end()
        end = matches[i+1].start() if i + 1 < len(matches) else len(text)
        lengths[section] += len(text[start:end])

    total_length = len(text)
    return {
        sec: (length / total_length if total_length > 0 else 0)
        for sec, length in lengths.items()
    }

# Add line breaks to long section names for nicer x-axis labels
section_names = [
    "initializing",
    "deduction",
    "adding<br>knowledge",
    "example<br>testing",
    "uncertainty<br>estimation",
    "backtracking"
]

# Compute counts and fractions for each text
counts_list = []
fractions_list = []
for t in texts:
    c = compute_behavioral_patterns(t)
    f = compute_text_fraction(t)
    # Keep them in the same section order as section_names
    counts_list.append([c[sec.replace("<br>", " ")] for sec in section_names])
    fractions_list.append([f[sec.replace("<br>", " ")] for sec in section_names])

# Color-blind friendly palette
color_palette = px.colors.qualitative.Safe

def create_grouped_bar_chart(
    y_values_list,          # list of lists: one list of y-values per text
    annotations_list=None,  # list of lists of annotation strings per text (optional)
    title="",
    yaxis_title="",
    yaxis_range=None,
    is_fraction=False       # whether the y-values are fractions (0 to 1)
):
    fig = go.Figure()

    for i, label in enumerate(labels):
        # Decide what annotation text to show
        if annotations_list:
            text_vals = annotations_list[i]
        else:
            # Default to string version of the y-values
            text_vals = [str(v) for v in y_values_list[i]]

        fig.add_trace(go.Bar(
            x=section_names,
            y=y_values_list[i],
            text=text_vals,
            textposition='outside',       # place labels above the bars
            name=label,
            marker_color=color_palette[i % len(color_palette)]
        ))

    # Common layout updates
    fig.update_layout(
        title=title,
        barmode='group',
        paper_bgcolor='white',
        plot_bgcolor='white',
        # Add a black border around the entire plot area
        shapes=[dict(
            type='rect',
            xref='paper',
            yref='paper',
            x0=0, y0=0,
            x1=1, y1=1,
            line=dict(color='black', width=1)
        )],
        # Legend inside, with border
        legend=dict(
            x=0.7,
            y=0.95,
            bordercolor='black',
            borderwidth=1
        ),
        margin=dict(l=80, r=40, t=60, b=80)
    )

    # Configure x-axis (horizontal labels, line break support already in section_names)
    fig.update_xaxes(
        tickangle=0,            # keep labels horizontal
        showline=True,
        linecolor='black'
    )

    # Configure y-axis
    yaxis_dict = dict(
        title=yaxis_title,
        range=yaxis_range,
        showline=True,
        linecolor='black',
        showgrid=True,
        gridcolor='lightgray'
    )
    # If plotting fractions, show them as percentages on the axis
    if is_fraction:
        yaxis_dict['tickformat'] = '.0%'
    fig.update_yaxes(**yaxis_dict)

    # Ensure bar text isn't clipped at the top
    fig.update_traces(cliponaxis=False)

    fig.show()

# ------------------------------------------------------------------------------
# 1. Combined chart (counts as bar height, annotation shows both counts & fraction)
combined_annotations = []
for i in range(len(texts)):
    row_annotations = []
    for j in range(len(section_names)):
        count_val = counts_list[i][j]
        frac_val = fractions_list[i][j]
        row_annotations.append(f"C: {count_val}, F: {frac_val:.1%}")
    combined_annotations.append(row_annotations)

create_grouped_bar_chart(
    y_values_list=counts_list,
    annotations_list=combined_annotations,
    title="Combined: Section Occurrences with Text Fractions",
    yaxis_title="Occurrence Count",
    yaxis_range=None,   # or e.g. (0, 35) if you need to limit
    is_fraction=False   # bar heights are counts here
)

# ------------------------------------------------------------------------------
# 2. Counts-only chart
counts_annotations = []
for i in range(len(texts)):
    counts_annotations.append([str(v) for v in counts_list[i]])

create_grouped_bar_chart(
    y_values_list=counts_list,
    annotations_list=counts_annotations,
    title="Section Occurrence Counts",
    yaxis_title="Count",
    is_fraction=False
)

# ------------------------------------------------------------------------------
# 3. Fractions-only chart (plot actual fractions 0..1, display as %)
# Convert each fraction to a string with a percent sign for annotations
fractions_annotations = []
for i in range(len(texts)):
    fractions_annotations.append([f"{val:.1%}" for val in fractions_list[i]])

create_grouped_bar_chart(
    y_values_list=fractions_list,        # still 0..1
    annotations_list=fractions_annotations,
    title="Section Text Fractions",
    yaxis_title="Fraction of Total Text (%)",
    is_fraction=True                     # triggers .0% axis format
)