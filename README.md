# Scalable Constraint Acquisition with Trees and Neural Nets

## Project Structure

This project has been organized into modular files for better maintainability and reusability:

### Files Overview

1. **`config.py`**
   - Contains all configuration parameters (N_VARS, DOMAIN, N_SAMPLES, SEED)
   - Initializes random seeds for reproducibility

2. **`constraints.py`**
   - Functions for generating hidden constraints
   - Constraint checking oracle
   - Printing and formatting constraint information

3. **`data_generation.py`**
   - Dataset generation from constraints
   - Random sampling functions for distillation

4. **`models.py`**
   - Training functions for all models:
     - Decision Tree
     - Neural Network
     - XGBoost
   - Distillation function (teacher → student)

5. **`evaluation.py`**
   - Model evaluation against ground truth
   - Testing new examples
   - Violation reporting

6. **`tree_extraction.py`**
   - Extract interpretable constraints from decision trees
   - Print learned rules in human-readable format

7. **`main.py`**
   - Main orchestration script
   - Executes the entire workflow
   - Coordinates all modules

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
