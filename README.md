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

```json
{
  num_leaves: 77,
  learning_rate: 0.897019124796627,
  min_child_samples: 79,
  min_child_weight: 78.36835995132282,
  subsample: 0.30122196311141236,
  subsample_freq: 3,
  colsample_bytree: 0.3047873793369096,
  reg_alpha: 75.55574706480405,
  reg_lambda: 24.99995435901818,
  max_depth: 19,
  min_split_gain: 52.58780312409352
}
```

Best MSR: 0.7999518396713539 on validation set after running 10 trials only

