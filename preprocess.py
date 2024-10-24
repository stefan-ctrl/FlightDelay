import os
import time

import pandas as pd

from feature_scripts.flight_feature_engineering import flight_feature_engineering
from feature_scripts.weater_metar_feature_engineering import metar_extraction
from plot_scripts.flight import visualize_delays

timestamp = time.strftime('%Y%m%d%H%M%S')
INVALID_VALUE = "999,9,9,9999,9" # indicates missing or invalid data from the weather station

def weather_preprocessing(file_path, airport_filter):
    weather_data = pd.read_csv(file_path, low_memory=False)

    # Drop rows where the 'REM' column starts with 'SYN' or 'SO' (these are synthetic reports)
    weather_data = weather_data[~weather_data['REM'].str.startswith('SYN')]
    weather_data = weather_data[~weather_data['REM'].str.startswith('SO')]
    selected_weather_features = ['STATION', 'DATE', 'REM']
    weather_data = weather_data[selected_weather_features]
    weather_data.rename(columns={
        'STATION': 'Station',
        'DATE': 'Date',
    }, inplace=True)
    weather_data['Station'] = airport_filter

    # alternative way to parse weather data
    weather_data['Date'] = pd.to_datetime(weather_data['Date'])
    weather_data['Year'] = weather_data['Date'].dt.year
    weather_data['Month'] = weather_data['Date'].dt.month
    weather_data['Day'] = weather_data['Date'].dt.day

    # Loop through each row in the DataFrame and extract features
    #for index, row in weather_data.iterrows():
    #    # Call metar_extraction once per row
    #    extracted_features = metar_extraction(row['REM'], day=row['Day'], month=row['Month'], year=row['Year'])
    #    # Assign the extracted features to new columns in the DataFrame
    #    if extracted_features is not None:
    #        for feature, value in extracted_features.items():
    #            weather_data.at[index, feature] = value

    # merge works, since data is sorted by date and time
    weather_data = weather_data.merge(
        weather_data.apply(
            lambda row: pd.Series(metar_extraction(row['REM'], day=row['Day'], month=row['Month'], year=row['Year'])),
            axis=1
        ), left_index=True, right_index=True)

    # save to csv
    weather_data.to_csv(timestamp+'_'+airport_filter + '_weather_data_'+'.csv', index=False)

def flight_preprocessing(file_path, airport_filter, save_mode: str = 'append'):
    flight_data = pd.read_csv(file_path, low_memory=False)
    # Drop rows
    flight_data.fillna({'DepDelay': 0, 'ArrDelay': 0}, inplace=True)
    flight_data.dropna(subset=['DepTime', 'ArrTime'], inplace=True)

    # apply airport filter if applied
    if airport_filter != '' or airport_filter is not None:
        flight_data = flight_data[(flight_data['Origin'] == airport_filter) | (flight_data['Dest'] == airport_filter)]

    # Drop columns
    selected_flight_features = ['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'FlightDate', 'Reporting_Airline', 'Flight_Number_Reporting_Airline', 'Origin', 'Dest', 'CRSDepTime', 'DepTime', 'DepDelay', 'DepDel15', 'CRSArrTime', 'ArrTime', 'ArrDelay', 'ArrDel15', 'WeatherDelay']
    flight_data = flight_data[selected_flight_features]

    # feature engineering
    flight_data = flight_feature_engineering(flight_data)

    save_to_csv_file(airport_filter, file_path, flight_data, save_mode)
    visualize_delays(flight_data)
    print(f'File {file_path} processed successfully')


def save_to_csv_file(airport_filter, file_path, flight_data, save_mode):
    core_name = timestamp + '_' + airport_filter + '_flight_data'
    if save_mode == 'append':
        saved_file_name = core_name + '.csv'
        if not os.path.exists(saved_file_name):
            flight_data.to_csv(saved_file_name, index=False)
        else:
            flight_data.to_csv(saved_file_name, mode='a', index=False, header=False)
    else:
        # get last str after '/'
        file_name = file_path.split('/')[-1].split('.')[0]
        saved_file_name = core_name + '_' + file_name + '.csv'
        flight_data.to_csv(saved_file_name, index=False)


def flight_multi_preprocessing(flight_data_dir, airport_filter, save_mode='append'):
    filenames = sorted([f for f in os.listdir(flight_data_dir) if f.endswith('.csv')])
    for filename in filenames:
        flight_preprocessing(flight_data_dir + filename, airport_filter, save_mode=save_mode)

port = 'LAX'
flight_dir = './data/flight/'
weather_preprocessing('./data/weather/2020/'+port.lower()+'_airport.csv', port)
#flight_preprocessing(flight_dir + '2020_01.csv', port)
flight_multi_preprocessing(flight_dir, port, save_mode='append')