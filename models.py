# ============================================================
# Machine Learning Models
# ============================================================

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from config import SEED


def train_decision_tree(X_train, y_train, max_depth=8, min_samples_leaf=200):
    """
    Train a decision tree classifier.
    """
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=SEED
    )
    tree.fit(X_train, y_train)
    return tree


def train_neural_network(X_train, y_train, hidden_layers=(128, 64), max_iter=3000):
    """
    Train a neural network classifier.
    """
    nn = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        max_iter=max_iter,
        random_state=SEED
    )
    nn.fit(X_train, y_train)
    return nn


def train_xgboost(X_train, y_train, n_estimators=100, max_depth=6, learning_rate=0.1):
    """
    Train an XGBoost classifier.
    """
    xgb = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=SEED,
        eval_metric='logloss'
    )
    xgb.fit(X_train, y_train)
    return xgb


def distill_to_tree(teacher_model, distill_X, max_depth=8, min_samples_leaf=300):
    """
    Distill a complex model (teacher) into a decision tree (student).
    """
    distill_y = teacher_model.predict(distill_X)
    
    distilled_tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=SEED
    )
    distilled_tree.fit(distill_X, distill_y)
    
    return distilled_tree
