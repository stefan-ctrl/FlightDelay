import os
import time

import pandas as pd
import numpy as np

timestamp = time.strftime('%Y%m%d%H%M%S')

def weather_preprocessing(file_path, airport_filter):
    weather_data = pd.read_csv(file_path, low_memory=False)
    selected_weather_features = ['STATION', 'DATE', 'WND', 'CIG', 'VIS', 'TMP', 'DEW', 'SLP']
    weather_data = weather_data[selected_weather_features]
    weather_data.rename(columns={
        'STATION': 'Station',
        'DATE': 'Date',
        'WND': 'Wind',
        'CIG': 'Ceiling',
        'VIS': 'Visibility',
        'TMP': 'Temperature',
        'DEW': 'DewPoint',
        'SLP': 'SeaLevelPressure'
    }, inplace=True)
    weather_data['Station'] = airport_filter
    weather_data.to_csv(timestamp+'_'+airport_filter + '_weather_data_'+'.csv', index=False)

def flight_preprocessing(file_path, airport_filter, save_mode: str = 'append'):
    flight_data = pd.read_csv(file_path, low_memory=False)
    flight_data.fillna({'DepDelay': 0, 'ArrDelay': 0}, inplace=True)
    flight_data.dropna(subset=['DepTime', 'ArrTime'], inplace=True)

    if airport_filter != '' or airport_filter is not None:
        flight_data = flight_data[(flight_data['Origin'] == airport_filter) | (flight_data['Dest'] == airport_filter)]

    selected_flight_features = ['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'FlightDate', 'Reporting_Airline', 'Flight_Number_Reporting_Airline', 'Origin', 'Dest', 'CRSDepTime', 'DepTime', 'DepDelay', 'DepDel15', 'CRSArrTime', 'ArrTime', 'ArrDelay', 'ArrDel15', 'WeatherDelay']
    flight_data = flight_data[selected_flight_features]


    core_name = timestamp+'_'+airport_filter + '_flight_data'
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

    print(f'File {file_path} processed successfully')

def flight_multi_preprocessing(flight_data_dir, airport_filter, save_mode='append'):
    filenames = sorted([f for f in os.listdir(flight_data_dir) if f.endswith('.csv')])
    for filename in filenames:
        flight_preprocessing(flight_data_dir + filename, airport_filter, save_mode=save_mode)

port = 'LAX'
flight_dir = './data/flight/'
weather_preprocessing('./data/weather/2020/'+port.lower()+'_airport.csv', port)
flight_multi_preprocessing(flight_dir, port, save_mode='append')