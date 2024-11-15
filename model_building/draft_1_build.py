import pandas as pd
import numpy as np
import pymc as pm
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier


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
    data = data.drop(['DepDelayMin', 'FlightDepDateTime', 'Date', 'Dest', 'WeatherDelay'], axis=1)
    return data

def predict_with_uncertainty(_pipeline, user_input):
    # Convert input data into a DataFrame
    input_df = pd.DataFrame([user_input])

    # Predict probability of delay
    probabilities = _pipeline.predict_proba(input_df)
    delay_probability = probabilities[0][1]  # Probability of 'Delayed' class

    # Output the prediction and confidence
    confidence_percentage = delay_probability * 100
    print(f"Prediction: There is a {confidence_percentage:.2f}% chance of delay.")

def bayesian_method(X_train, X_test, y_train, y_test):
    # Standardize numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to DataFrames for PyMC3
    X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    y_train_array = y_train.values

    # Bayesian Logistic Regression with PyMC3
    with pm.Model() as logistic_model:
        # Priors for coefficients and intercept
        intercept = pm.Normal('intercept', mu=0, sigma=10)
        coefficients = pm.Normal('coefficients', mu=0, sigma=10, shape=X_train_df.shape[1])

        # Logistic regression model
        linear_combination = intercept + pm.math.dot(X_train_df, coefficients)
        probability = pm.Deterministic('probability', pm.math.sigmoid(linear_combination))

        # Likelihood
        likelihood = pm.Bernoulli('likelihood', p=probability, observed=y_train_array)

        # Inference
        trace = pm.sample(2000, cores=2, return_inferencedata=True)  # MCMC sampling

    # Summary of the model parameters
    print(pm.summary(trace))

    # Predictive samples for X_test
    with logistic_model:
        posterior_predictive = pm.sample_posterior_predictive(trace, var_names=['probability'], random_seed=42)

    # Get mean probabilities and confidence intervals
    mean_probs = posterior_predictive['probability'].mean(axis=0)
    lower_bound = np.percentile(posterior_predictive['probability'], 2.5, axis=0)
    upper_bound = np.percentile(posterior_predictive['probability'], 97.5, axis=0)

    # Display probabilistic predictions with uncertainty bounds
    for i in range(5):  # Display first 5 predictions
        print(f"Prediction: {mean_probs[i]:.3f}, 95% CI: [{lower_bound[i]:.3f}, {upper_bound[i]:.3f}]")

df = process_data()

# Define features and target
X = df.drop('Delayed', axis=1)
y = df['Delayed']

categorical_cols = ['Origin', 'Weather_Intensity', 'Weather_Obscuration', 'Weather_Precipitation']
numerical_cols = ['Wind_Direction', 'Wind_Gusts', 'Wind_Speed', 'Visibility', 'Hour', 'DayOfWeek', 'Month']

# Create a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),      # Handle missing values in numerical features
            ('scaler', StandardScaler())                      # Standardize numerical features
        ]), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols) # One-hot encode categorical features
    ])

# Create a pipeline with preprocessing and model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(eval_metric='logloss'))
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

pipeline.fit(X_train, y_train)

# Evaluate the model
predict_with_uncertainty(pipeline, user_input = {
        'Origin': 'BOS',
        'Weather_Intensity': 'FG',  # Example condition indicating fog
        'Weather_Obscuration': 'BR',  # Add other weather conditions as needed
        'Weather_Precipitation': '',
        'Wind_Direction': 150.0,
        'Wind_Gusts': 47.0,
        'Wind_Speed': 36.0,
        'Visibility': 1.0,
        'Hour': 10,
        'DayOfWeek': 4,
        'Month': 11,
        'IsFog': 1 if 'FG' in ['FG'] else 0  # Automatically set based on condition
    })

y_pred_deterministic = pipeline.predict(X_test)
y_pred_prob = pipeline.predict_proba(X_test)[:, 1]  # Probability of delay
accuracy = accuracy_score(y_test, y_pred_deterministic)
precision = precision_score(y_test, y_pred_deterministic)
recall = recall_score(y_test, y_pred_deterministic)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

## ROC: Depending on the threshold, the model can have different trade-offs between precision and recall.
## I might want to adjust the threshold based on the specific use case. In certain use cases, I accept more false positives
## to catch more true positives (e.g., predicting flight delays to avoid missing a flight). In other cases, I might want to
## minimize false positives at the expense of missing some true positives (e.g., predicting flight delays to avoid unnecessary
## cancellations).


# Create a copy of X_test to add noise to weather features
X_test_noisy = X_test.copy()

# Define the columns that represent weather features
weather_features = ['Wind_Direction', 'Wind_Gusts', 'Wind_Speed', 'Visibility']

# Add random noise to weather features (simulating uncertainty)
np.random.seed(42)
noise_factor = 0.1  # Adjust this to control the amount of noise
for col in weather_features:
    noise = np.random.normal(0, noise_factor * X_test_noisy[col].std(), X_test_noisy[col].shape)
    X_test_noisy[col] += noise

# Predictions with noisy data
y_pred_noisy = pipeline.predict(X_test_noisy)
y_prob_noisy = pipeline.predict_proba(X_test_noisy)[:, 1]

# Recompute metrics with noisy data
accuracy_noisy = accuracy_score(y_test, y_pred_noisy)
precision_noisy = precision_score(y_test, y_pred_noisy)
recall_noisy = recall_score(y_test, y_pred_noisy)
roc_auc_noisy = roc_auc_score(y_test, y_prob_noisy)

print("\nWith Uncertain Weather Data (Noisy Features):")
print(f"Accuracy: {accuracy_noisy:.4f}")
print(f"Precision: {precision_noisy:.4f}")
print(f"Recall: {recall_noisy:.4f}")
print(f"ROC AUC: {roc_auc_noisy:.4f}")

# Adding noise to a specific weather feature to simulate uncertainty
weather_feature = 'Wind_Speed'  # Choose a weather feature to vary
noises = np.linspace(-5, 5, 50)  # Range of noise to add (adjust for your data range)

mean_preds = []
upper_bounds = []
lower_bounds = []

for noise in noises:
    X_test_noisy = X_test.copy()
    X_test_noisy[weather_feature] += noise  # Vary the chosen feature
    y_prob_noisy = pipeline.predict_proba(X_test_noisy)[:, 1]  # Probability predictions
    mean_preds.append(y_prob_noisy.mean())
    lower_bounds.append(np.percentile(y_prob_noisy, 2.5))
    upper_bounds.append(np.percentile(y_prob_noisy, 97.5))

# Plotting the mean predictions with 95% confidence intervals
plt.figure(figsize=(10, 6))
plt.plot(noises, mean_preds, label='Mean Prediction (Probability of Delay)', color='b')
plt.fill_between(noises, lower_bounds, upper_bounds, color='b', alpha=0.2, label='95% Confidence Interval')
plt.xlabel(f'{weather_feature} Uncertainty (Added Noise)')
plt.ylabel('Predicted Probability of Delay')
plt.title(f'Impact of {weather_feature} Uncertainty on Flight Delay Prediction')
plt.legend()
plt.show()


# Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)  # One-hot encode categorical variables
    ])
X_processed = preprocessor.fit_transform(X)
X_processed = pd.DataFrame(X_processed, columns=preprocessor.get_feature_names_out())
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.3, random_state=42)
bayesian_method(X_train, X_test, y_train, y_test)