# How to run


## Get the Data
see the data folder for more information

## Install the requirements
```bash
pip install -r requirements.txt
```

## Preprocessing
After you have optained the data, preprocess the data with the following script.

```bash
python preprocessing.py
```

## Merge the Data (Optional)
Certain visualizations require the data to be merged. This can be done with the following script.

```bash
python merge_with_minimum_features.py
```

As the name suggests, this script only merges the data with the minimum features required for the visualizations. So most features are not included in the merged data.


## Visualizations
This is done for the preprocessed aswell as for the merged data. It shows first correlations and then the distribution of delays.

```bash
python visualize.py
```

# Data sources

* Flight Delay Data: https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr
  * January 2020
* Weather Data: https://www.ncei.noaa.gov/access/search/data-search/global-hourly?pageNum=1&startDate=2020-01-01T00:00:00&endDate=2020-01-31T23:59:59&stations=72530094846
  *  for ten most listed airports

# Limitations

* The weather data from NOAA needs to be downloaded manually for each station. Hence, we have focused on the ten most listed airports in the flight data.
* The weather data has a granularity of one hour. This means that we can only compare the weather data to the flight data on an hourly basis. This might not be the most accurate representation of the weather at the time of the flight and certain weather events might be missed.
* The flight data includes all domestic flights in the US. This means that the data is very large and might not be representative of all flights worldwide.
* We limit ourselves to the entire year of 2020. 
  * All flights from one Airport only, since the weather data is only available for individual airports and gathering the data for all airports would be too time-consuming.

# Task
Copy from task description in ORTUS: https://estudijas.rtu.lv/mod/assign/view.php?id=4750923
## Background
Flight delays are a common issue for travelers and airlines, and they can be caused by a variety of factors including weather conditions, air traffic, and airport logistics. In this task, you will use historical weather and flight data to predict whether a flight will be delayed. The goal is to build a predictive model that incorporates uncertainty due to changing weather conditions.

## Objective
Build a machine learning model that predicts whether a flight will be delayed based on historical weather data.
Use uncertainty quantification techniques to account for unpredictable weather patterns.
## Dataset
* Flight Data: Use real flight data from sources like the U.S. Department of Transportation's Bureau of Transportation Statistics (BTS) Flight Delays Dataset.
* Weather Data: Use weather data from the National Oceanic and Atmospheric Administration (NOAA), which includes factors such as temperature, precipitation, wind speed, and visibility.

## Steps
1. Data Collection:
- [ x ] Gather flight data, including departure times, arrival times, delay status (delayed or on-time), and flight numbers.
- [ x ] Gather historical weather data for the departure and destination airports at the relevant times.
2. Data Preprocessing:
- [ x ] Clean the data to handle missing values, anomalies, and outliers.
- [ x ] Feature engineering: Create new features such as the difference between scheduled and actual departure times, or weather conditions at the time of departure.
3. Exploratory Data Analysis (EDA):
- [ x ] Visualize the distribution of delays and how they relate to weather conditions.
- [ x ] Analyze correlations between weather features (e.g., temperature, wind speed, precipitation) and flight delays.
4. Model Building:
- [ ] Build a classification model (e.g., logistic regression, decision tree, random forest, or XGBoost) to predict whether a flight will be delayed.
- [ ] Incorporate features that represent uncertainty in the weather data, such as the forecasted vs. actual weather conditions.
5. Incorporating Uncertainty:
- [ ] Use probabilistic models such as Bayesian methods to handle uncertainty in weather forecasts.
- [ ] Alternatively, use Monte Carlo simulations to quantify uncertainty in weather predictions.
- [ ] Evaluate the model's performance by comparing deterministic predictions (certain) with probabilistic predictions (uncertain).
6. Model Evaluation:
- [ ] Evaluate the model using accuracy, precision, recall, and the ROC curve.
- [ ] Additionally, evaluate the impact of uncertainty in the weather data on the model's performance.
- [ ] Discuss the trade-off between accuracy and certainty in predictions.
7. Task Deliverables:
- [ ] A Jupyter notebook (or Python script) that includes the full analysis.
- [ ] Visualizations that show how uncertainty in weather data impacts flight delay predictions.
- [ ] A report discussing the model's performance, the role of uncertainty, and potential improvements.

## Bonus (Optional):
* Implement ensemble methods (e.g., bagging or boosting) to improve predictions under uncertain conditions.
* Use real-time weather data from an API (e.g., OpenWeather) to make live predictions for upcoming flights.
* Compare different uncertainty quantification techniques (e.g., Bayesian vs Monte Carlo) and their impact on model robustness.

## Example of Uncertainty Quantification:
Incorporating uncertainty in weather data could involve techniques like:

Bayesian Neural Networks (BNNs): These models output a distribution of possible outcomes rather than a single prediction. In the context of flight delays, this allows the model to capture the uncertainty of weather forecasts and predict a range of delay probabilities.
Monte Carlo Simulations: Simulate multiple scenarios of possible weather conditions and predict how each scenario would impact the likelihood of a delay. The result would be a probabilistic forecast of flight delays.
By focusing on real data with uncertain conditions, students can gain hands-on experience with predictive modeling, uncertainty quantification, and data-driven decision-making.