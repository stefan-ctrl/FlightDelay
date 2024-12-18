import glob
import os
import re

import pandas as pd

def read_latest_files(directory, date_range=None):
    # Search for flight and weather files in the directory
    flight_files = glob.glob(os.path.join(directory, '*_flight_data.csv'))
    weather_files = glob.glob(os.path.join(directory, '*_weather_data_.csv'))

    # Get the latest flight and weather files based on modification time
    latest_flight_file = max(flight_files, key=os.path.getmtime) if flight_files else None
    latest_weather_file = max(weather_files, key=os.path.getmtime) if weather_files else None

    # Read and filter the flight data
    flight_data = pd.read_csv(latest_flight_file) if latest_flight_file else pd.DataFrame()

    if not flight_data.empty and date_range:
        flight_data['FlightDepDateTime'] = pd.to_datetime(flight_data['FlightDepDateTime'])
        flight_data = flight_data[(flight_data['FlightDepDateTime'].between(date_range[0], date_range[1]))]

    # Read and filter the weather data
    weather_data = pd.read_csv(latest_weather_file) if latest_weather_file else pd.DataFrame()

    if not weather_data.empty and date_range:
        weather_data['Date'] = pd.to_datetime(weather_data['Date'])
        weather_data = weather_data[(weather_data['Date'].between(date_range[0], date_range[1]))]

    print(f"Files successfully read. Flight data count: {flight_data['Year'].count()}, Weather data count: {weather_data['Date'].count()}\n Files: {latest_flight_file}, {latest_weather_file}")
    return flight_data, weather_data

def airport_latest_files(directory):
    flight_files = glob.glob(os.path.join(directory, '*_flight_data.csv'))
    latest_flight_file = max(flight_files, key=os.path.getmtime) if flight_files else None
    match = re.search(r"\d+_(\w+)_flight_data\.csv", latest_flight_file)
    airport_code = None
    if match:
        airport_code = match.group(1)
    return airport_code