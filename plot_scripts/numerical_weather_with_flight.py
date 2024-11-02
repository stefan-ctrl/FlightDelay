import matplotlib.pyplot as plt
import seaborn as sns


def visualize_overlaid_flight_and_weather(flight_data, weather_data, airport, delay_type='', title_suffix='', save_base_name=''):
    """
    Overlay flight delays and weather conditions in a single plot.

    Parameters:
        flight_data (DataFrame): The flight data DataFrame.
        weather_data (DataFrame): The weather data DataFrame.

    Returns:
        None
    """
    # Set the date as index for easier plotting
    flight_data.set_index('FlightDepDateTime', inplace=True)
    latest_flight_time = flight_data.index.max()
    weather_data.set_index('Date', inplace=True)

    # Create a figure
    plt.figure(figsize=(14, 8))
    title = f"[{airport}]{delay_type} Delays and Weather Conditions ({title_suffix})"
    plt.title(title)

    # Plot flight delays
    # Draw vertical lines where DepDelayMin > 0
    delay_times = flight_data[flight_data['DepDelayMin'] > 0].index
    for flight_time in delay_times:
        plt.axvline(x=flight_time, color='red', linestyle='--', alpha=0.5)

    # Overlay temperature
    # sns.lineplot(data=weather_data['Temperature'], color='blue', label='Temperature (°C)', linewidth=2, alpha=0.7)

    # Overlay wind speed
    sns.lineplot(data=weather_data['Wind_Speed'], color='green', label='Wind Speed (km/h)', linewidth=2, alpha=0.7)

    # Overlay wind gusts
    if 'Wind_Gusts' in weather_data.columns:
        sns.lineplot(data=weather_data['Wind_Gusts'], color='orange', label='Wind Gusts (km/h)', linewidth=2, alpha=0.7)

    # Overlay visibility
    if 'Visibility' in weather_data.columns:
        sns.lineplot(data=weather_data['Visibility'], color='purple', label='Visibility (miles)', linewidth=2,
                     alpha=0.7)

    # Add labels and legend
    plt.xlabel('Date and Time', fontsize=14)
    plt.ylabel('Values', fontsize=14)
    plt.xlim(flight_data.index.min(), latest_flight_time)  # Set x-axis limit
    plt.legend()

    # Show the plot
    #plt.show()
    if save_base_name:
        plt.savefig(f'{airport}_{save_base_name}_{title_suffix}.png', dpi=1000)
