import pandas as pd
import os
import glob

from plot_scripts.cloud_cover_with_flights import plot_cloud_cover_with_flight_delays
from plot_scripts.numerical_weather_with_flight import visualize_overlaid_flight_and_weather

def read_latest_files(directory, date_range=None):
    """
    Read the latest flight and weather CSV files from the specified directory,
    filtering data based on the specified date range.

    Parameters:
        directory (str): The directory path to search for CSV files.
        date_range (tuple): A tuple containing start and end date strings (YYYY-MM-DD).

    Returns:
        DataFrame, DataFrame: DataFrames for flight and weather data.
    """
    # Search for flight and weather files in the directory
    flight_files = glob.glob(os.path.join(directory, '*_flight_data.csv'))
    weather_files = glob.glob(os.path.join(directory, '*_weather_data_.csv'))

    # Get the latest flight and weather files based on modification time
    latest_flight_file = max(flight_files, key=os.path.getmtime) if flight_files else None
    latest_weather_file = max(weather_files, key=os.path.getmtime) if weather_files else None

    # Read and filter the flight data
    flight_data = pd.read_csv(latest_flight_file) if latest_flight_file else pd.DataFrame()
    print(f"Flight data count: {flight_data['Year'].count()}\n")

    if not flight_data.empty and date_range:
        flight_data['FlightDepDateTime'] = pd.to_datetime(flight_data['FlightDepDateTime'])
        flight_data = flight_data[(flight_data['FlightDepDateTime'].between(date_range[0], date_range[1]))]

    # Read and filter the weather data
    weather_data = pd.read_csv(latest_weather_file) if latest_weather_file else pd.DataFrame()
    print(f"Weather data count: {weather_data['Date'].count()}\n")

    if not weather_data.empty and date_range:
        weather_data['Date'] = pd.to_datetime(weather_data['Date'])
        weather_data = weather_data[(weather_data['Date'].between(date_range[0], date_range[1]))]

    print(f"Flight data count: {flight_data['Year'].count()}, Weather data count: {weather_data['Date'].count()}\n")
    return flight_data, weather_data

def remove_flight_without_weather_delay(flight_data):
    # Filter for WeatherDelay == 0.0 or WeatherDelay is NaN (None)
    filtered_data = flight_data[
        (flight_data['WeatherDelay'] != 0.0) & (~flight_data['WeatherDelay'].isna())
    ]
    return filtered_data

def visualize_with_data_range(directory, date_range=None):
    """
    Process flight and weather data while displaying progress.

    Parameters:
        directory (str): The directory path to search for CSV files.
        date_range (tuple): A tuple containing start and end date strings (YYYY-MM-DD).

    Returns:
        None
    """
    flight_data, weather_data = read_latest_files(directory, date_range)

    flight_data = remove_flight_without_weather_delay(flight_data)
    if not flight_data.empty and not weather_data.empty:
        visualize_overlaid_flight_and_weather(flight_data, weather_data)
        plot_cloud_cover_with_flight_delays(weather_data, flight_data)
    else:
        print("No flight or weather data found.")


# Example usage
directory = './'  # Set the path to your directory containing the CSV files
date_range = (pd.to_datetime('2020-01-01 00:00'), pd.to_datetime('2020-12-31 23:59'))  # Define a date range to filter data
visualize_with_data_range(directory, date_range)
