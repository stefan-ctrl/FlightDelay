import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def cloud_cover_color(value):
    """
    Map cloud cover numerical values to colors.

    Parameters:
        value (float): The numerical value of cloud cover.

    Returns:
        color (str): The corresponding color based on the value.
    """
    if value == 0:  # SKC (Clear)
        return 0, 0, 1, 0.3  # Transparent blue (RGBA)
    elif value == 1:  # FEW
        return 0, 0, 1, 0.5  # More opaque blue
    elif value == 3:  # SCT
        return 0, 0, 1, 0.7  # Less opaque blue
    elif value == 5:  # BKN
        return 0.5, 0.5, 0.5, 0.8  # Gray
    elif value == 8:  # OVC
        return 0.33, 0.33, 0.33, 1.0  # Dark gray
    else:
        return 0, 0, 0, 0  # Default (transparent)


def prepare_cloud_cover_data(weather_data):
    """
    Prepare data for stacked bar chart based on METAR definitions.

    Parameters:
        weather_data (DataFrame): The weather data DataFrame.

    Returns:
        cloud_cover_df (DataFrame): DataFrame suitable for stacked bar plotting.
    """
    # Create a new DataFrame for cloud cover
    cloud_cover_df = pd.DataFrame()

    # Map each cloud cover category to numerical values
    metar_mapping = {
        'SKC': 0,
        'FEW': 1,
        'SCT': 3,
        'BKN': 5,
        'OVC': 8
    }

    # Map the cloud categories to their numerical values and add to the new DataFrame
    for cloud_column in ['Clouds_0-5000', 'Clouds_5000-10000', 'Clouds_10000-15000', 'Clouds_20000-25000']:
        if cloud_column in weather_data.columns:
            cloud_cover_df[cloud_column] = weather_data[cloud_column].map(metar_mapping).fillna(0)

    return cloud_cover_df


def plot_cloud_cover_with_flight_delays(weather_data, flight_data):
    """
    Plot cloud cover as horizontal lines and overlay flight delays.

    Parameters:
        weather_data (DataFrame): The weather data DataFrame.
        flight_data (DataFrame): The flight data DataFrame.

    Returns:
        None
    """
    cloud_cover_df = prepare_cloud_cover_data(weather_data)

    plt.figure(figsize=(14, 8))
    plt.title('Cloud Cover Levels and Flight Delays', fontsize=16)

    # Set y-ticks for cloud levels
    y_levels = np.arange(1, 6)  # y=1 for 0-5000, y=2 for 5000-10000, ..., y=5 for 20000-25000
    cloud_lines = ['Clouds_0-5000', 'Clouds_5000-10000', 'Clouds_10000-15000', 'Clouds_20000-25000', 'Clouds_25000+']

    # Draw horizontal lines for each cloud category
    for i, cloud_column in enumerate(cloud_lines):
        if cloud_column in cloud_cover_df.columns:
            last_value = None  # To hold the last value for color change
            for idx, value in enumerate(cloud_cover_df[cloud_column]):
                color = cloud_cover_color(value)
                # Only draw a line when the value changes
                if value != last_value:
                    plt.hlines(y=y_levels[i], xmin=cloud_cover_df.index[idx], xmax=cloud_cover_df.index[idx + 1] if idx + 1 < len(cloud_cover_df) else cloud_cover_df.index[idx],
                               color=color, linewidth=3, alpha=color[3])
                    last_value = value

    # Overlay flight delays as vertical lines
    delay_times = flight_data[flight_data['DepDelayMin'] > 0].index

    for flight_time in delay_times:
        plt.axvline(x=flight_time, color='red', linestyle='--', alpha=0.5,
                    label='Flight Delay' if 'Flight Delay' not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.xlabel('Date and Time', fontsize=14)
    plt.ylabel('Cloud Cover Levels', fontsize=14)
    plt.yticks(ticks=y_levels, labels=['Clouds 0-5000', 'Clouds 5000-10000', 'Clouds 10000-15000', 'Clouds 20000-25000', 'Clouds 25000+'])
    plt.legend(title='Cloud Cover Categories')
    latest_flight_time = flight_data.index.max()
    plt.xlim(flight_data.index.min(), latest_flight_time)  # Set x-axis limit
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()
