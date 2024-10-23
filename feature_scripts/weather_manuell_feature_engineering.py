###
# This script contains functions to parse and interpret weather data columns manuell before the knowledge of METAR data
# Hence, it is no longer used
###

import pandas as pd


# Function to parse and interpret the Ceiling column
def parse_ceiling(ceiling_str):
    if pd.isna(ceiling_str) or ceiling_str == '0':
        return None, None, None, None  # Return None for missing data or no significant ceiling

    # Split the ceiling string by commas
    ceiling_parts = ceiling_str.split(',')

    # Extract components
    ceiling_height = int(ceiling_parts[0])  # Height in feet
    condition_code = ceiling_parts[1]  # Condition code
    additional_code = ceiling_parts[2]  # Additional code
    visibility_condition = ceiling_parts[3]  # Visibility condition

    # Interpret the condition code
    if condition_code == '1':
        ceiling_condition = 'Good ceiling'
    elif condition_code == '2':
        ceiling_condition = 'Variable ceiling'
    elif condition_code == '3':
        ceiling_condition = 'Poor ceiling'
    else:
        ceiling_condition = 'Unknown ceiling condition'

    return ceiling_height, ceiling_condition, additional_code, visibility_condition

# Function to interpret the wind direction into compass directions
def interpret_wind_direction(degree):
    if pd.isna(degree) or degree == '999':
        return 'Variable'  # Variable wind direction
    degree = float(degree)  # Convert to float for comparison
    if degree >= 337.5 or degree < 22.5:
        return 'N'
    elif 22.5 <= degree < 67.5:
        return 'NE'
    elif 67.5 <= degree < 112.5:
        return 'E'
    elif 112.5 <= degree < 157.5:
        return 'SE'
    elif 157.5 <= degree < 202.5:
        return 'S'
    elif 202.5 <= degree < 247.5:
        return 'SW'
    elif 247.5 <= degree < 292.5:
        return 'W'
    elif 292.5 <= degree < 337.5:
        return 'NW'
    else:
        return 'Unknown'

def parse_wind(wind_str):
    if pd.isna(wind_str):
        return None, None, None, None, None  # Return None for missing data

    # Split the wind string by commas
    wind_parts = wind_str.split(',')

     # Extract wind components
    wind_direction = wind_parts[0]  # e.g., 999: Variable wind direction
    wind_speed = float(wind_parts[1])  # e.g., 9: Wind speed (units depend on dataset)
    wind_condition = wind_parts[2]  # e.g., C: Calm or condition code
    gust_speed = float(wind_parts[3])  # e.g., 0000: Gust speed
    additional_code = wind_parts[4]  # e.g., 1: Additional code

    # Interpret the wind condition code
    if wind_condition == 'C':
        wind_condition_desc = 'Calm'
    elif wind_condition == 'G':
        wind_condition_desc = 'Gusty'
    elif wind_condition == 'V':
        wind_condition_desc = 'Variable'
    elif wind_condition == 'N':
        wind_condition_desc = 'No_Wind'
    else:
        wind_condition_desc = 'Unknown: ' + wind_condition

    # Interpret the wind direction
    wind_direction = interpret_wind_direction(wind_direction)

    return wind_direction, wind_speed, wind_condition_desc, gust_speed, additional_code
