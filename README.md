# Rainfall Prediction Classifier

This project builds a machine learning classifier to predict whether it will rain based on weather observations from the Australian weather dataset. The work is implemented in the Jupyter notebook [Rainfall_Prediction_Classifier.ipynb](Rainfall_Prediction_Classifier.ipynb).

## Project Overview

The notebook follows a full classification workflow:

- Load the weather dataset from a remote CSV source.
- Remove rows with missing values.
- Focus on a subset of locations: Melbourne, MelbourneAirport, and Watsonia.
- Engineer a seasonal feature from the date column.
- Split the data into train and test sets with stratified sampling.
- Preprocess numeric and categorical features separately.
- Train and tune a Random Forest classifier with grid search.
- Compare the tuned model with a Logistic Regression baseline.

## Dataset

The model uses the weatherAUS dataset, which contains daily weather measurements such as temperature, rainfall, humidity, pressure, wind direction, and cloud cover. The target variable is `RainToday`, recast from the original `RainTomorrow` label after the feature engineering step in the notebook.

After cleaning and filtering, the final working dataset contains 7,557 rows and 23 columns.

## Features Used

Numeric features:

- MinTemp
- MaxTemp
- Rainfall
- Evaporation
- Sunshine
- WindGustSpeed
- WindSpeed9am
- WindSpeed3pm
- Humidity9am
- Humidity3pm
- Pressure9am
- Pressure3pm
- Cloud9am
- Cloud3pm
- Temp9am
- Temp3pm

Categorical features:

- Location
- WindGustDir
- WindDir9am
- WindDir3pm
- RainYesterday
- Season

## Modeling Approach

The notebook uses scikit-learn pipelines and a column transformer so preprocessing stays tied to the model:

- Numeric features are scaled with `StandardScaler`.
- Categorical features are one-hot encoded with `OneHotEncoder(handle_unknown='ignore')`.
- A `RandomForestClassifier` is tuned with `GridSearchCV` and `StratifiedKFold`.

The Random Forest parameter search explores:

- `n_estimators`: 50, 100
- `max_depth`: None, 10, 20
- `min_samples_split`: 2, 5

The notebook also trains a Logistic Regression baseline for comparison.

## Results

The notebook reports the following scores:

- Random Forest best cross-validation score: 0.86
- Random Forest test score: 0.84
- Logistic Regression test accuracy: 0.83

For the tuned Random Forest model, the test-set classification report shows stronger performance on the negative class (`No`) than the positive class (`Yes`), which is expected given the class imbalance in the data.

## Visualizations

The notebook includes:

- A confusion matrix for the tuned model.
- A feature-importance plot for the Random Forest classifier.

## Requirements

The notebook installs and uses:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## How to Run

1. Open [Rainfall_Prediction_Classifier.ipynb](Rainfall_Prediction_Classifier.ipynb) in Jupyter or VS Code.
2. Run the cells from top to bottom.
3. If needed, install the required packages listed above.

## Notes

- The notebook loads the dataset directly from an online CSV URL, so an internet connection is required.
- The data is imbalanced, so accuracy alone is not enough to judge model quality.
- Stratified splitting is used to preserve class proportions across train and test sets.

## Repository Structure

- [Rainfall_Prediction_Classifier.ipynb](Rainfall_Prediction_Classifier.ipynb) - main notebook with data preparation, modeling, and evaluation
- [README.md](README.md) - project documentation
