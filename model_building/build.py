import arviz as az
import pandas as pd
import numpy as np
import pymc as pm
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

categorical_cols = ['Weather_Intensity', 'Weather_Obscuration', 'Weather_Precipitation']
numerical_cols = ['Wind_Direction', 'Wind_Gusts', 'Wind_Speed', 'Visibility', 'Hour', 'DayOfWeek', 'Month']

import numpy as np

### Incorporate features that represent uncertainty in the weather data, such as the forecasted vs. actual weather conditions.
def add_forecast(X):
    # 0-100% that this weather is accurate
    # each row should have another value
    X = X.copy()
    X['Weather_Uncertainty'] = np.random.normal(0, 100, X.shape[0])
    return X

def add_noise_to_features(X, noise_level=0.05):
    X_noisy = X.copy()

    # Only apply noise to numerical columns
    for col in X_noisy.select_dtypes(include=[np.number]).columns:
        # Calculate the standard deviation of the column
        std_dev = X_noisy[col].std()
        # Generate noise for each entry in the column
        noise = np.random.normal(0, noise_level * std_dev, X_noisy[col].shape)
        # Add the noise to the column
        X_noisy[col] += noise

    return X_noisy

### Prepare for model building
def preprocessor():
    return ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),      # Handle missing values in numerical features
            ('scaler', StandardScaler())                      # Standardize numerical features
        ]), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols) # One-hot encode categorical features
    ])

def prepare_data(X_train, X_test):
    preprocessor_pipeline = preprocessor()
    X_train_processed = preprocessor_pipeline.fit_transform(X_train)
    X_test_processed = preprocessor_pipeline.transform(X_test)

    # Convert to numpy arrays
    X_train_processed = X_train_processed.toarray() if hasattr(X_train_processed, 'toarray') else X_train_processed
    X_test_processed = X_test_processed.toarray() if hasattr(X_test_processed, 'toarray') else X_test_processed

    return X_train_processed, X_test_processed

## load the data and remove some attributes
def process_data():
    # Load the data
    data = pd.read_csv('../merged_flight_weather_data.csv', low_memory=False)
    # Drop columns with all missing values
    data = data.loc[:, data.notna().any()]
    # Create target variable
    data['IsDelayed'] = (data['DepDelayMin'] > 0).astype(int)
    # Create a binary target variable based on DepDelayMin
    data['Delayed'] = np.where(data['DepDelayMin'] > 15, 1, 0)
    # Convert FlightDepDateTime to datetime and extract relevant features
    data['FlightDepDateTime'] = pd.to_datetime(data['FlightDepDateTime'])
    data['Hour'] = data['FlightDepDateTime'].dt.hour
    data['DayOfWeek'] = data['FlightDepDateTime'].dt.dayofweek
    data['Month'] = data['FlightDepDateTime'].dt.month
    # Drop columns not needed for prediction
    data = data.drop(['DepDelayMin', 'FlightDepDateTime', 'Date', 'Dest', 'WeatherDelay', 'Origin'], axis=1)

    # Define features and target
    X = data.drop('Delayed', axis=1)
    y = data['Delayed']

    return X, y

def get_model(model_name):
    if model_name == 'logistic_regression':
        return LogisticRegression(max_iter=1000)
    elif model_name == 'decision_tree':
        return DecisionTreeClassifier()
    elif model_name == 'random_forest':
        return RandomForestClassifier()
    elif model_name == 'xgboost':
        return XGBClassifier(eval_metric='logloss')
    else:
        raise ValueError("Model not recognized. Choose from 'logistic_regression', 'decision_tree', 'random_forest', 'xgboost'.")

# Build a classification model (e.g., logistic regression, decision tree, random forest, or XGBoost) to predict whether a flight will be delayed.
def build_models(X, y, test_size=0.3, random_state=42, suffix=""):
    # List of model names to evaluate
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model_names = ['logistic_regression', 'decision_tree', 'random_forest', 'xgboost']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    #set title of the plot
    fig.suptitle(f"Model Evaluation{suffix}")
    axes = axes.flatten()
    for i, model_name in enumerate(model_names):
        print(f"\nEvaluating Model: {model_name}")

        # Build the pipeline for the chosen model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor()),
            ('classifier', get_model(model_name))
        ])

        # Fit the pipeline
        pipeline.fit(X_train, y_train)

        # Make predictions and probabilities
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        # Print metrics
        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Confusion Matrix:")
        print(pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted']))
        print(f"----------------------------------")

        # Plot ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        axes[i].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        axes[i].plot([0, 1], [0, 1], 'k--')
        axes[i].set_xlabel('False Positive Rate')
        axes[i].set_ylabel('True Positive Rate')
        axes[i].set_title(f'{model_name} ROC Curve')
        axes[i].legend(loc='lower right')
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

# Bayesian Logistic Regression Model with PyMC
def bayesian_logistic_regression(X, y, test_size=0.3, random_state=42, num_samples=50, suffix=""):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_test = prepare_data(X_train, X_test)  # Ensure all categorical data is encoded and standardized
    y_train = y_train.values if hasattr(y_train, 'values') else y_train  # Ensure y_train is a numpy array

    # Bayesian logistic regression with PyMC
    with pm.Model() as logistic_model:
        # Priors for the coefficients and intercept
        intercept = pm.Normal('intercept', mu=0, sigma=10)
        coeffs = pm.Normal('coeffs', mu=0, sigma=10, shape=(X_train.shape[1],))

        # Logistic regression model
        p = pm.invlogit(intercept + pm.math.dot(X_train, coeffs))

        # Likelihood (Bernoulli with observed data)
        y_obs = pm.Bernoulli('y_obs', p=p, observed=y_train)

        # Inference: sample from the posterior using NUTS
        trace = pm.sample(num_samples, return_inferencedata=True, tune=1000)

    return trace


def plot_trace(suffix=""):
    az.plot_trace(trace, var_names=['intercept', 'coeffs'])
    plt.title(f"Trace Plot for Bayesian Logistic Regression{suffix}")
    plt.show()


if __name__ == "__main__":
    trace = None
    X, y = process_data()

    suffix=""
    build_models(X, y)
    #trace = bayesian_logistic_regression(X, y, suffix=suffix)
    if trace is not None:
        plot_trace()

    suffix = "_with_forecast"
    X = add_forecast(X)
    build_models(X, y, suffix=suffix)
    #trace = bayesian_logistic_regression(X, y, suffix=suffix)
    if trace is not None:
        plot_trace(suffix=suffix)


    suffix = "_with_forecast_and_noise"
    X = add_noise_to_features(X)
    build_models(X, y, suffix=suffix)
    #trace = bayesian_logistic_regression(X, y, suffix=suffix)
    if trace is not None:
        plot_trace(suffix=suffix)