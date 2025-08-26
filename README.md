# Airbnb Reviews per Month Predictor

---

## Project Overview

This project builds and evaluates regression models to predict the number of reviews per month for New York City Airbnb listings. We explore feature relationships, engineer new predictors, compare multiple algorithms, tune hyperparameters, and interpret the final model with feature importances and SHAP values.

---

## Dataset

The dataset is `AB_NYC_2019.csv`, containing:

- Listing identifiers and metadata (`id`, `name`, `host_id`, `host_name`)
- Location details (`neighbourhood_group`, `neighbourhood`, `latitude`, `longitude`)
- Listing characteristics (`room_type`, `price`, `minimum_nights`, `availability_365`)
- Review statistics (`number_of_reviews`, `reviews_per_month`, `last_review`)
- Host activity (`calculated_host_listings_count`)

Target variable:  
- `reviews_per_month` - a proxy for listing popularity.

---

## Data Preprocessing & Feature Engineering

- **Missing values**: Drop rows missing `reviews_per_month`.  
- **Scaling**: Standardize continuous features (`price`, `number_of_reviews`, `calculated_host_listings_count`, `availability_365`).  
- **Discretization**:  
  - Latitude & longitude binned into 20 intervals  
  - Minimum nights binned into 3 intervals  
- **Text vectorization**: CountVectorizer on listing names.  
- **One-hot encoding**: Categorical features (`neighbourhood_group`, `neighbourhood`, `room_type`).  
- **New feature**:  
  - `reviews_per_listing` = number_of_reviews / calculated_host_listings_count

---

## Modeling

### Baseline

- **DummyRegressor** (strategy=`mean`)  
  - Training R² = 0.00

### Algorithms and Cross-Validation Scores

| Model                     | Mean CV R² | Mean Fit Time (s) |
|---------------------------|------------|-------------------|
| Ridge Regression          |      0.37  |            0.06   |
| KNeighborsRegressor       |      0.32  |            0.08   |
| DecisionTreeRegressor     |      0.08  |            0.02   |
| RandomForestRegressor     |      0.51  |            0.07   |

---

### Hyperparameter Tuning

| Model                   | Tuned Params                         | Tuned CV R² |
|-------------------------|--------------------------------------|-------------|
| Ridge                   | α = 74.25                            |       0.41  |
| KNeighborsRegressor     | n_neighbors = 31                     |       0.37  |
| DecisionTreeRegressor   | max_depth = 2 (with RFECV)           |       0.30  |
| RandomForestRegressor   | n_estimators = 105, max_depth = 33   |       0.51  |

---

### Final Model Performance

- **RandomForestRegressor** (n_estimators=105, max_depth=33)  
  - Mean CV R² = 0.51  
  - Test R² = 0.46  

---

## Interpretation

### Feature Importances (Random Forest)

| Rank | Feature                   | Importance |
|-----:|---------------------------|------------|
|    1 | number_of_reviews         |      0.384 |
|    2 | availability_365          |      0.118 |
|    3 | minimum_nights_2.0        |      0.051 |
|    4 | reviews_per_listing       |      0.041 |
|    5 | price                     |      0.038 |

Remaining features each contribute less than 0.035.

---

### SHAP Insights

SHAP waterfall plots for two test examples reveal that:

- `number_of_reviews`, `availability_365`, and `minimum_nights_2.0` are the top drivers.
- The direction and magnitude of impact vary by listing, reflecting feature interactions.

---

## Future Work
 
- Increase cross-validation folds for RFECV to improve feature selection.  
- Explore additional hyperparameter tuning for RandomForestRegressor (e.g., `min_samples_leaf`, `max_features`).  
- Test gradient boosting models (e.g., XGBoost, LightGBM) for potential performance gains.