from graphviz import Digraph
from IPython.display import display

# Create a new directed graph
dot = Digraph(comment='Tree Diagram')
dot.attr(rankdir='TB') # Arrange nodes from Top to Bottom

# Add nodes with rectangular shape (default for boxes)
dot.node('question', 'question', shape='oval')
dot.node('initializing', 'intializing', shape='box')
dot.node('adding knowledge 1', 'adding knowledge', shape='box')
dot.node('deduction 1', 'deduction', shape='box')
dot.node('adding knowledge 2', 'adding knowledge', shape='box')
dot.node('backtracking', 'backtracking', shape='box')
dot.node('deduction 2', 'deduction', shape='box')

# Add edges to form a tree structure
dot.edge('question', 'initializing', label='3', weight='3')
dot.edge('initializing', 'adding knowledge 1', label='3', weight='3')
dot.edge('adding knowledge 1', 'deduction 1', label='3', weight='3')
dot.edge('deduction 1', 'backtracking', label='1', weight='1')
dot.edge('deduction 1', 'adding knowledge 2', label='2', weight='2')
dot.edge('adding knowledge 2', 'deduction 2', label='2', weight='1')
dot.edge('backtracking', 'deduction 2', label='1', weight='1')

# Rander the diagram (this will create a file and open it if possible )
display(dot)
