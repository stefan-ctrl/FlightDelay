from metar import Metar

def metar_extraction(metar: str, day: int = 1, month: int = 1, year: int = 2024):

    if not metar.startswith('MET'):
        return None, None, None, None, None, None, None, None, None, None, None, None, None, None

    # Remove the extra information (like MET10812/31/19) and keep the METAR report
    metar_report = ' '.join(metar.split()[2:])



    # Define the regular expression pattern for extracting the METAR features
    # Parse the METAR string using the Metar library
    try:
        parsed_metar = Metar.Metar(metar_report, month=month, year=year)
    except Exception as e:
        print(f"Error parsing METAR: {e}")
        return None
        # Extract the features

    features = {}

    # 1. Station
    features['Station'] = parsed_metar.station_id

    # 2. Day
    features['Metar_Day'] = parsed_metar.time.day if parsed_metar.time else None

    # 3. Hour
    features['Hour'] = parsed_metar.time.hour if parsed_metar.time else None

    # 4. Minute
    features['Minute'] = parsed_metar.time.minute if parsed_metar.time else None

    # 5. Wind Direction
    features['Wind_Direction'] = parsed_metar.wind_dir.value() if parsed_metar.wind_dir else None

    # 6. Wind Speed
    features['Wind_Speed'] = parsed_metar.wind_speed.value('KT') if parsed_metar.wind_speed else None

    # 7. Wind Gusts
    features['Wind_Gusts'] = parsed_metar.wind_gust.value('KT') if parsed_metar.wind_gust else None

    # 8. Visibility
    features['Visibility'] = parsed_metar.vis.value('SM') if parsed_metar.vis else None

    # 9. Sky Condition (Clouds)
    clouds = parsed_metar.sky

    cloud_buckets = {
        '0-5000': None,  # 0 - 5,000 feet
        '5000-10000': None,  # 5,000 - 10,000 feet
        '10000-15000': None,  # 10,000 - 15,000 feet
        '15000-20000': None,  # 15,000 - 20,000 feet
        '20000-25000': None,  # 20,000 - 25,000 feet
        '25000+': None  # 25,000+ feet
    }

    if any(cloud[0] == 'CLR' for cloud in clouds):
        cloud_buckets = {key: 'CLR' for key in cloud_buckets}  # Set all buckets to "CLR"
    else:
        # Loop through each cloud layer and assign it to the correct altitude bucket
        for cloud in clouds:
            cloud_type = cloud[0]  # e.g., 'FEW', 'SCT', 'BKN'
            if cloud[1] is None:
                print(f"Cloud layer {cloud_type} has no altitude information. Assume previous value")
            else:
                cloud_altitude = cloud[1].value('FT')  # Altitude in feet

            # Categorize the cloud altitude into the specified buckets
            if cloud_altitude < 5000:
                cloud_buckets['0-5000'] = cloud_type
            elif 5000 <= cloud_altitude < 10000:
                cloud_buckets['5000-10000'] = cloud_type
            elif 10000 <= cloud_altitude < 15000:
                cloud_buckets['10000-15000'] = cloud_type
            elif 15000 <= cloud_altitude < 20000:
                cloud_buckets['15000-20000'] = cloud_type
            elif 20000 <= cloud_altitude <= 25000:
                cloud_buckets['20000-25000'] = cloud_type
            elif cloud_altitude >= 25001:
                cloud_buckets['25000+'] = cloud_type

    # Assign the cloud buckets to the features dictionary
    features['Clouds_0-5000'] = cloud_buckets['0-5000']
    features['Clouds_5000-10000'] = cloud_buckets['5000-10000']
    features['Clouds_10000-15000'] = cloud_buckets['10000-15000']
    features['Clouds_20000-25000'] = cloud_buckets['20000-25000']
    features['Clouds_25000+'] = cloud_buckets['25000+']

    # 10. Temperature
    features['Temperature'] = parsed_metar.temp.value('C') if parsed_metar.temp else None

    # 11. Dew Point
    features['Dew_Point'] = parsed_metar.dewpt.value('C') if parsed_metar.dewpt else None

    # 12. Altimeter (Pressure)
    features['Altimeter'] = parsed_metar.press.value('IN') if parsed_metar.press else None

    # 13. Sea-Level Pressure
    features['Sea_Level_Pressure'] = parsed_metar.press_sea_level.value(
        'MB') if parsed_metar.press_sea_level else None

    # 14. Snow Depth
    features['Snow_Depth'] = parsed_metar.snowdepth.value('M') if parsed_metar.snowdepth else None

    # 15. Weather Conditions
    if parsed_metar.weather:
        weather_conditions = parsed_metar.weather
        found_weather_intensity = None
        found_weather_descriptor = None
        found_weather_phenomenon = None
        features['Weather_Intensity'] = None
        features['Weather_Precipitation'] = None
        features['Weather_Obscuration'] = None
        features['Weather_Other'] = None

        # Iterate through the weather conditions and check for keywords
        for condition in weather_conditions:
            code = condition[0]  # Extract the code
            # Check for intensity
            if code.startswith('+'):
                features['Weather_Intensity'] = '+'
            elif code.startswith('-'):
                features['Weather_Intensity'] = '-'
            else:
                features['Weather_Intensity'] = 'o'

            features['Weather_Precipitation'] = condition[2]
            features['Weather_Obscuration'] = condition[3]  # Directly use the code as descriptor
            features['Weather_Other'] = condition[4]



    return features
    #return features['Station'], features['Day'], features['Hour'], features['Minute'], features['Wind_Direction'], features['Wind_Speed'], features['Wind_Gusts'], features['Visibility'], features['Sky_Condition'], features['Temperature'], features['Dew_Point'], features['Altimeter'], features['Sea_Level_Pressure'], features['Snow_Depth'], features['Clouds_150'], features['Clouds_250']

