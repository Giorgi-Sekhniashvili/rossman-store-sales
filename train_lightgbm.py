import json
from functools import partial

import lightgbm
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error


def objective_fn(train_df, val_df, feature_columns, target_column, trial):
    """Train a model using XGBoost.

    Args:
        train_df (pandas.DataFrame): Training data.
        val_df (pandas.DataFrame): Validation data.
        feature_columns (list): List of feature columns to use.
        target_column (str): Name of the target column.
        trial (optuna.trial.Trial): Current trial.

    Returns:
        float: RMSE of the model.
    """
    y_train = np.log1p(train_df[target_column])
    y_valid = np.log1p(val_df[target_column])
    x_train = train_df[feature_columns]
    x_valid = val_df[feature_columns]

    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "learning_rate": trial.suggest_float("learning_rate", 1e-8, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-8, 100.0),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100.0),
        "max_depth": trial.suggest_int("max_depth", 1, 20),
        "min_split_gain": trial.suggest_float("min_split_gain", 1e-8, 100.0),
    }

    train_data = lightgbm.Dataset(x_train, label=y_train)

    model = lightgbm.train(params, train_data)
    y_pred = model.predict(x_valid)

    rmse = mean_squared_error(y_valid, y_pred, squared=False)
    return rmse


def train_and_evaluate(n_trials=2):
    with open('data/for_training/features.json') as f:
        column_names = json.load(f)

    features = column_names['features']
    target = column_names['target']

    train_df = pd.read_csv('data/for_training/train.csv')
    val_df = pd.read_csv('data/for_training/valid.csv')

    objective = partial(objective_fn, train_df, val_df, features, target)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    return study


if __name__ == "__main__":
    whole_study = train_and_evaluate(n_trials=100)
    print("Number of finished trials: ", len(whole_study.trials))
    print("Best hyper parameters:", whole_study.best_trial.params)
    print("Best value:", whole_study.best_value)
    print("Done!")
