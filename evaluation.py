# ============================================================
# Model Evaluation and Testing
# ============================================================

import numpy as np
from sklearn.metrics import accuracy_score
from constraints import check_constraints
from config import N_VARS, DOMAIN


def evaluate_model(model, X, y_true, name):
    """
    Evaluate a model's accuracy against ground truth labels.
    """
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    print(f"{name} accuracy vs oracle: {acc}")
    return acc


def test_new_examples(models_dict, hidden_constraints, n_examples=5):
    """
    Generate new random examples and test all models.
    Shows predictions and constraint violations.
    """
    print("\n" + "="*60)
    print("Testing Models with New Examples")
    print("="*60)
    
    test_examples = np.random.randint(1, DOMAIN + 1, size=(n_examples, N_VARS))
    
    for idx, example in enumerate(test_examples, 1):
        print(f"\nExample {idx}: {example}")
        
        # Check against oracle
        oracle_result = check_constraints(example, hidden_constraints)
        print(f"  Oracle (True constraints): {'✓ VALID' if oracle_result else '✗ INVALID'}")
        
        # Check which hidden constraints are violated
        if oracle_result == 0:
            print("  Violated constraints:")
            for c in hidden_constraints:
                if c[0] == "neq":
                    _, i, j = c
                    if example[i] == example[j]:
                        print(f"    - X{i} != X{j} violated ({example[i]} == {example[j]})")
                
                elif c[0] == "sum_leq":
                    _, i, j, k = c
                    if example[i] + example[j] > k:
                        print(f"    - X{i} + X{j} <= {k} violated ({example[i]} + {example[j]} = {example[i] + example[j]})")
                
                elif c[0] == "if_pos_then_lt":
                    _, z, i, j = c
                    if example[z] > DOMAIN // 2 and not (example[i] < example[j]):
                        print(f"    - IF X{z} > {DOMAIN//2} THEN X{i} < X{j} violated (X{z}={example[z]}, X{i}={example[i]}, X{j}={example[j]})")
        
        # Test each model
        print("\n  Model Predictions:")
        for model_name, model in models_dict.items():
            prediction = model.predict([example])[0]
            symbol = '✓' if prediction == 1 else '✗'
            status = 'VALID' if prediction == 1 else 'INVALID'
            match = '✓' if prediction == oracle_result else '✗ MISMATCH'
            print(f"    {model_name:25s}: {symbol} {status:7s} {match}")
