import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

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
data = data.drop(['DepDelayMin', 'FlightDepDateTime', 'Date'], axis=1)

# Define features and target
X = data.drop('Delayed', axis=1)
y = data['Delayed']

# Define categorical and numerical columns
categorical_cols = ['Origin', 'Dest', 'Weather_Intensity', 'Weather_Obscuration', 'Weather_Precipitation']
numerical_cols = ['WeatherDelay', 'Wind_Direction', 'Wind_Gusts', 'Wind_Speed', 'Visibility', 'Hour', 'DayOfWeek', 'Month']

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
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", report)
