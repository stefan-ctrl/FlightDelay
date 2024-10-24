# Function to extract hour and minute features from time columns
import pandas as pd


def extract_time_feature(flight_data, time_column, new_time_column):
    flight_data[new_time_column] = flight_data[time_column].apply(lambda x: f"{int(x // 100):02}:{int(x % 100):02}")

# New function to handle feature engineering
def flight_feature_engineering(flight_data):
    # Extract time features
    extract_time_feature(flight_data, 'DepTime', 'DepTimeClock')
    extract_time_feature(flight_data, 'ArrTime', 'ArrTimeClock')
    extract_time_feature(flight_data, 'CRSDepTime', 'ScheduledDepTimeClock')
    extract_time_feature(flight_data, 'CRSArrTime', 'ScheduledArrTimeClock')

    # Rename DepDelay and ArrDelay with "Min" suffix
    flight_data.rename(columns={'DepDelay': 'DepDelayMin', 'ArrDelay': 'ArrDelayMin'}, inplace=True)

    # Create FlightDepDateTime and FlightArrDateTime in ISO 8601 format
    flight_data['FlightDate'] = pd.to_datetime(flight_data['FlightDate'])
    flight_data['FlightDepDateTime'] = pd.to_datetime(
        flight_data['FlightDate'].dt.strftime('%Y-%m-%d') + 'T' + flight_data['DepTimeClock'],
        format='%Y-%m-%dT%H:%M', errors='coerce'
    )
    flight_data['FlightArrDateTime'] = pd.to_datetime(
        flight_data['FlightDate'].dt.strftime('%Y-%m-%d') + 'T' + flight_data['ArrTimeClock'],
        format='%Y-%m-%dT%H:%M', errors='coerce'
    )


    return flight_data