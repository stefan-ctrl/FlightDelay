import os
import time

import pandas as pd
import numpy as np



from weather_preprocessor import parse_wind, parse_ceiling
from weather_rem_based_preprocessor import metar_extraction

timestamp = time.strftime('%Y%m%d%H%M%S')
INVALID_VALUE = "999,9,9,9999,9" # indicates missing or invalid data from the weather station

def weather_preprocessing(file_path, airport_filter):
    weather_data = pd.read_csv(file_path, low_memory=False)

    # Drop rows where the Wind column equals "999,9,9,9999,9"
    #weather_data = weather_data[(weather_data['WND'] != INVALID_VALUE) & (weather_data['CIG'] != INVALID_VALUE)]

    #selected_weather_features = ['STATION', 'DATE', 'WND', 'CIG', 'VIS', 'TMP', 'DEW', 'SLP', 'REM']
    # Drop rows where the 'REM' column starts with 'SYN'
    weather_data = weather_data[~weather_data['REM'].str.startswith('SYN')]
    selected_weather_features = ['STATION', 'DATE', 'REM']
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

    # alternative way to parse weather data
    weather_data['Date'] = pd.to_datetime(weather_data['Date'])
    weather_data['Year'] = weather_data['Date'].dt.year
    weather_data['Month'] = weather_data['Date'].dt.month
    weather_data['Day'] = weather_data['Date'].dt.day

    weather_data = weather_data.merge(
        weather_data.apply(
            lambda row: pd.Series(metar_extraction(row['REM'], day=row['Day'], month=row['Month'], year=row['Year'])),
            axis=1
        ),
        left_index=True, right_index=True
    )

    #wind parsing
   # weather_data[['Wind_Direction', 'Wind_Speed', 'Wind_Condition', 'Wind_Gust_Speed', 'Wind_Additional_Code']] = weather_data[
   #     'Wind'].apply(lambda x: pd.Series(parse_wind(x)))
    #weather_data.drop(columns=['Wind'], inplace=True)

    #Ceiling parsing
    #weather_data[['Ceiling_Height', 'Ceiling_Condition', 'Ceiling_Additional_Code', 'Celling_Visibility_Condition']] = weather_data['Ceiling'].apply(lambda x: pd.Series(parse_ceiling(x)))
    #weather_data.drop(columns=['Ceiling'], inplace=True)


    # save to csv
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
#flight_multi_preprocessing(flight_dir, port, save_mode='append')