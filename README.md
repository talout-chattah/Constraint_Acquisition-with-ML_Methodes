# Constraint Acquisition with Trees and Neural Nets

## Usage

Simply run the main script:

```bash
python main.py
```

## Workflow

1. Generate hidden constraints (ground truth)
2. Create training dataset by sampling and labeling
3. Split into train/test sets
4. Train multiple models:
   - Decision Tree (interpretable)
   - Neural Network (high accuracy)
   - XGBoost (ensemble method)
5. Distill complex models into interpretable trees
6. Evaluate all models against oracle
7. Extract and compare learned vs. true constraints
8. Test on new random examples

## Customization

To modify parameters, edit `config.py`:
- `N_VARS`: Number of variables
- `DOMAIN`: Domain size for each variable
- `N_SAMPLES`: Training dataset size
- `SEED`: Random seed for reproducibility

## Dependencies

- numpy
- scikit-learn
- xgboost

Install with:
```bash
pip install numpy scikit-learn xgboost
```
