# Rossman Store Sales

This is a project for the [Kaggle Rossman Store Sales competition](https://www.kaggle.com/c/rossmann-store-sales). The
goal is to predict the daily sales of stores.

## Data

Data preparation
We extract several features from the dataset

1. CompetitionDistance - distance to the nearest competitor store
2. Promo - whether the store is running a promo on that day
3. Promo2 - whether the store is running a promo2 on that day
4. SchoolHoliday - whether the (Store, Date) was affected by the closure of public schools
5. StoreType - one-hot encoding of store type
6. Assortment - one-hot encoding of store assortment
7. StateHoliday - one-hot encoding of state holiday
8. DayOfWeek - one-hot encoding of day of week
9. Month - one-hot encoding of month
10. Year
11. IsPromoMonth - whether the month is a promo month for the store

all feature names are in [here](./data/for_training/features.json)

## Model

model is lightgbm. We try to find the best hyperparameters using optuna.
We try several hyperparameters and choose the best one at the end.

## Results

Best hyper parameters:

```python
best_params = {
    'num_leaves': 128,
    'learning_rate': 0.9142244642559713,
    'min_child_samples': 42,
    'min_child_weight': 82.67196585530857,
    'subsample': 0.8484409608430301,
    'subsample_freq': 10,
    'colsample_bytree': 0.7233264088685998,
    'reg_alpha': 21.228348804552645,
    'reg_lambda': 87.17459207514409,
    'max_depth': 14,
    'min_split_gain': 0.053099560774982534
}
```

Best MSR: 0.47350310197843565 on validation set after running 100 trials

## Future Ideas

1. Try other models, or combination of models.
2. Create more features for example - lagged values of the sales.
3. We could also use different type of models for different types of stores. 


