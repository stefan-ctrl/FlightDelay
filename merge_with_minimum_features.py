import pandas as pd

from util.read_latest_flight_and_weather import read_latest_files

def drop_to_minimum_weather(weather_df):
    return weather_df[['Date', 'Weather_Intensity','Weather_Obscuration','Weather_Other','Weather_Precipitation', 'Wind_Direction', 'Wind_Gusts','Wind_Speed', 'Visibility']]

def drop_to_minimum_flight(flight_df):
    return flight_df[['FlightDepDateTime', 'DepDelayMin', 'Origin', 'Dest', 'WeatherDelay']]

def merge_with_minimum_features():
    flight_data, weather_data = read_latest_files('./')

    flight_data = drop_to_minimum_flight(flight_data)
    weather_data = drop_to_minimum_weather(weather_data)

    flight_data['FlightDepDateTime'] = pd.to_datetime(flight_data['FlightDepDateTime'])
    weather_data['Date'] = pd.to_datetime(weather_data['Date'])


    # Initialize variables
    weather_idx = 0  # Index to track the position in the weather dataset
    merged_rows = []  # List to store merged results

    # Iterate through each flight entry
    for idx, flight in flight_data.iterrows():
        flight_time = flight['FlightDepDateTime']

        # Find the closest preceding weather entry
        # Find the closest preceding weather entry without using abs
        while weather_idx < len(weather_data) - 1 and \
                (weather_data.iloc[weather_idx + 1]['Date'] <= flight_time and
                 flight_time - weather_data.iloc[weather_idx + 1]['Date'] < flight_time - weather_data.iloc[weather_idx][
                     'Date']):
            weather_idx += 1

        # Merge the flight data with the current weather entry
        merged_row = {**flight.to_dict(), **weather_data.iloc[weather_idx].to_dict()}  # Merge flight and weather data row
        merged_rows.append(merged_row)

    # Convert the list of merged rows to a DataFrame
    merged_data = pd.DataFrame(merged_rows)

    # Save the merged result to a new CSV file
    merged_data.to_csv('merged_flight_weather_data.csv', index=False)

merge_with_minimum_features()
