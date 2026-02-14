# ============================================================
# Constraint Acquisition with Trees and Neural Networks
# Main Script
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.tree import export_text

# Import from our modules
from config import N_VARS, DOMAIN, N_SAMPLES, SEED
from constraints import (
    generate_hidden_constraints,
    print_original_constraints
)
from data_generation import generate_dataset, sample_points
from models import (
    train_decision_tree,
    train_neural_network,
    train_xgboost,
    distill_to_tree
)
from evaluation import evaluate_model, test_new_examples
from tree_extraction import print_learned_constraints


def main():
    """
    Main execution function for the constraint acquisition system.
    """
    
    # ============================================================
    # 1. Generate hidden constraints
    # ============================================================
    print("Generating hidden constraints...")
    hidden_constraints = generate_hidden_constraints(N_VARS)
    
    print("Hidden constraints:")
    for c in hidden_constraints:
        print("  ", c)
    
    
    # ============================================================
    # 2. Generate dataset
    # ============================================================
    print(f"\nGenerating dataset with {N_SAMPLES} samples...")
    X, y = generate_dataset(N_VARS, DOMAIN, hidden_constraints, N_SAMPLES)
    
    print("\nDataset size:", len(X))
    print("Valid examples:", y.sum())
    print("Invalid examples:", len(y) - y.sum())
    
    
    # ============================================================
    # 3. Train / test split
    # ============================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED
    )
    
    
    # ============================================================
    # 4. Train Decision Tree
    # ============================================================
    print("\n" + "="*60)
    print("Training Decision Tree...")
    print("="*60)
    
    tree = train_decision_tree(X_train, y_train)
    tree_acc = evaluate_model(tree, X_test, y_test, "Decision Tree")
    
    print("\n================ Decision Tree =================")
    print("Tree accuracy:", tree_acc)
    #print(export_text(tree, feature_names=[f"X{i}" for i in range(N_VARS)]))
    
    
    # ============================================================
    # 5. Train Neural Network
    # ============================================================
    print("\n" + "="*60)
    print("Training Neural Network...")
    print("="*60)
    
    nn = train_neural_network(X_train, y_train)
    nn_acc = evaluate_model(nn, X_test, y_test, "Neural Network")
    
    print("\n================ Neural Network =================")
    print("Neural Network accuracy:", nn_acc)
    
    
    # ============================================================
    # 6. Train XGBoost
    # ============================================================
    print("\n" + "="*60)
    print("Training XGBoost...")
    print("="*60)
    
    xgb = train_xgboost(X_train, y_train)
    xgb_acc = evaluate_model(xgb, X_test, y_test, "XGBoost")
    
    print("\n================ XGBoost =================")
    print("XGBoost accuracy:", xgb_acc)
    
    
    # ============================================================
    # 7. Distillation: Neural Net → Decision Tree
    # ============================================================
    print("\n" + "="*60)
    print("Distilling Neural Network to Decision Tree...")
    print("="*60)
    
    distill_X = sample_points(50_000, N_VARS, DOMAIN)
    distilled_tree = distill_to_tree(nn, distill_X)
    
    #print("\n========== Distilled Tree (NN → Rules) ==========")
    #print(export_text(distilled_tree,feature_names=[f"X{i}" for i in range(N_VARS)]))
    
    
    # ============================================================
    # 8. Distillation: XGBoost → Decision Tree
    # ============================================================
    print("\n" + "="*60)
    print("Distilling XGBoost to Decision Tree...")
    print("="*60)
    
    distill_X_xgb = sample_points(50_000, N_VARS, DOMAIN)
    distilled_tree_xgb = distill_to_tree(xgb, distill_X_xgb)
    
    #print("\n========== Distilled Tree (XGBoost → Rules) ==========")
    #print(export_text(distilled_tree_xgb, feature_names=[f"X{i}" for i in range(N_VARS)]))
    
    
    # ============================================================
    # 9. Final evaluation
    # ============================================================
    print("\n" + "="*60)
    print("Final Evaluation Against Oracle")
    print("="*60)
    
    evaluate_model(tree, X_test, y_test, "Decision Tree")
    evaluate_model(nn, X_test, y_test, "Neural Network")
    evaluate_model(xgb, X_test, y_test, "XGBoost")
    evaluate_model(distilled_tree, X_test, y_test, "Distilled Tree (NN)")
    evaluate_model(distilled_tree_xgb, X_test, y_test, "Distilled Tree (XGBoost)")
    
    
    # ============================================================
    # 10. Print original and learned constraints
    # ============================================================
    print_original_constraints(hidden_constraints)
    print_learned_constraints(tree, "Decision Tree")
    print_learned_constraints(distilled_tree, "Distilled Tree (NN → Tree)")
    print_learned_constraints(distilled_tree_xgb, "Distilled Tree (XGBoost → Tree)")
    
    
    # ============================================================
    # 11. Test with new examples
    # ============================================================
    models_to_test = {
        'Decision Tree': tree,
        'Neural Network': nn,
        'XGBoost': xgb,
        'Distilled Tree (NN)': distilled_tree,
        'Distilled Tree (XGB)': distilled_tree_xgb
    }
    
    test_new_examples(models_to_test, hidden_constraints, n_examples=10)


if __name__ == "__main__":
    main()
