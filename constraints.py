# ============================================================
# Constraint Generation and Checking
# ============================================================

import random
from config import N_VARS, DOMAIN


def generate_hidden_constraints(n_vars):
    """
    Create a list of hidden constraints.
    These simulate the unknown constraints we want to learn.
    """
    constraints = []

    # X_i != X_j
    for _ in range(5):
        i, j = random.sample(range(n_vars), 2)
        constraints.append(("neq", i, j))

    # X_i + X_j <= c
    for _ in range(5):
        i, j = random.sample(range(n_vars), 2)
        c = random.randint(DOMAIN + 2, DOMAIN + 8)
        constraints.append(("sum_leq", i, j, c))

    # Conditional constraints
    for _ in range(3):
        z = random.randint(0, n_vars - 1)
        i, j = random.sample(range(n_vars), 2)
        constraints.append(("if_pos_then_lt", z, i, j))

    return constraints


def check_constraints(x, constraints):
    """
    Check if a given assignment satisfies all constraints.
    Returns 1 if valid, 0 if invalid.
    """
    for c in constraints:
        if c[0] == "neq":
            _, i, j = c
            if x[i] == x[j]:
                return 0

        elif c[0] == "sum_leq":
            _, i, j, k = c
            if x[i] + x[j] > k:
                return 0

        elif c[0] == "if_pos_then_lt":
            _, z, i, j = c
            if x[z] > DOMAIN // 2 and not (x[i] < x[j]):
                return 0

    return 1


def print_original_constraints(constraints):
    """
    Print the original hidden constraints in readable format.
    """
    print(f"\n{'='*60}")
    print(f"Original Hidden Constraints (Ground Truth)")
    print(f"{'='*60}")
    print(f"Total: {len(constraints)} constraints\n")
    
    neq_count = sum(1 for c in constraints if c[0] == "neq")
    sum_count = sum(1 for c in constraints if c[0] == "sum_leq")
    cond_count = sum(1 for c in constraints if c[0] == "if_pos_then_lt")
    
    print(f"Constraint types:")
    print(f"  - Inequality (!=): {neq_count}")
    print(f"  - Sum (<=): {sum_count}")
    print(f"  - Conditional: {cond_count}\n")
    
    print("Detailed constraints:")
    
    rule_num = 1
    for c in constraints:
        if c[0] == "neq":
            _, i, j = c
            print(f"  {rule_num}. X{i} != X{j}")
            
        elif c[0] == "sum_leq":
            _, i, j, k = c
            print(f"  {rule_num}. X{i} + X{j} <= {k}")
            
        elif c[0] == "if_pos_then_lt":
            _, z, i, j = c
            print(f"  {rule_num}. IF X{z} > {DOMAIN//2} THEN X{i} < X{j}")
        
        rule_num += 1
    print()
