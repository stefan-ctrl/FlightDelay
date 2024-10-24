import pandas as pd

from plot_scripts.cloud_cover_with_flights import plot_cloud_cover_with_flight_delays
from plot_scripts.numerical_weather_with_flight import visualize_overlaid_flight_and_weather
from plot_scripts.weather_effects_and_flight_delay import plot_weather_obscuration_flight_delay, \
    plot_weather_precipitation_flight_delay
from util.read_latest_flight_and_weather import read_latest_files


def remove_flight_without_weather_delay(flight_data):
    # Filter for WeatherDelay == 0.0 or WeatherDelay is NaN (None)
    filtered_data = flight_data[
        (flight_data['WeatherDelay'] != 0.0) & (~flight_data['WeatherDelay'].isna())
    ]
    return filtered_data

def visualize_with_data_range(directory, date_range=None, shouldFilter=True):
    flight_data, weather_data = read_latest_files(directory, date_range)

    if shouldFilter:
        flight_data = remove_flight_without_weather_delay(flight_data)
    if not flight_data.empty and not weather_data.empty:
        visualize_overlaid_flight_and_weather(flight_data, weather_data)
        plot_cloud_cover_with_flight_delays(weather_data, flight_data)
    else:
        print("No flight or weather data found.")


# Visualize both and layover the flight and weather data
directory = './'  # Set the path to your directory containing the CSV files
date_range = (pd.to_datetime('2020-01-01 00:00'), pd.to_datetime('2020-01-31 23:59'))  # Define a date range to filter data
visualize_with_data_range(directory, date_range)

# Merged data, has less features than the original data
merged_data = pd.read_csv('merged_flight_weather_data.csv')
plot_weather_obscuration_flight_delay(merged_data)
plot_weather_precipitation_flight_delay(merged_data)
