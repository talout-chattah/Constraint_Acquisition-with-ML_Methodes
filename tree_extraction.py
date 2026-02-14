# ============================================================
# Constraint Extraction from Decision Trees
# ============================================================

import numpy as np
from sklearn.tree import _tree
from config import N_VARS


def extract_constraints_from_tree(tree_model, feature_names):
    """
    Extract constraints from a decision tree in readable format.
    Returns a list of constraint rules (VALID paths only).
    """
    tree_ = tree_model.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    constraints = []
    
    def recurse(node, depth, path_constraints):
        indent = "  " * depth
        
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            
            # Left branch (<=)
            left_constraint = f"{name} <= {threshold:.1f}"
            recurse(tree_.children_left[node], depth + 1, 
                   path_constraints + [left_constraint])
            
            # Right branch (>)
            right_constraint = f"{name} > {threshold:.1f}"
            recurse(tree_.children_right[node], depth + 1, 
                   path_constraints + [right_constraint])
        else:
            # Leaf node
            class_value = np.argmax(tree_.value[node][0])
            if class_value == 1:  # VALID (constraints satisfied)
                constraints.append({
                    'path': path_constraints.copy(),
                    'decision': 'VALID'
                })
    
    recurse(0, 0, [])
    return constraints


def print_learned_constraints(tree_model, model_name):
    """
    Print learned constraints from a tree model.
    """
    feature_names = [f"X{i}" for i in range(N_VARS)]
    constraints = extract_constraints_from_tree(tree_model, feature_names)
    
    print(f"\n{'='*60}")
    print(f"Learned Constraints from {model_name}")
    print(f"{'='*60}")
    
    if len(constraints) == 0:
        print("No valid constraint paths found")
        return
    
    print(f"Found {len(constraints)} valid constraint paths:\n")
    
    for i, constraint in enumerate(constraints[:10], 1):  # Show first 10
        print(f"Rule {i}:")
        print(f"  IF: {' AND '.join(constraint['path'])}")
        print(f"  THEN: {constraint['decision']}\n")
    
    if len(constraints) > 10:
        print(f"... and {len(constraints) - 10} more valid constraint paths")
